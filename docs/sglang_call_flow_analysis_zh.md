# SGLang 调用逻辑深度分析（从入口函数到一次 decode 迭代）

> 本文不是“模块介绍”，而是“可执行追踪图”：从入口函数开始，按真实调用顺序追到 scheduler event loop、模型 executor 与 detokenizer 回包。

---

## 0. 先给结论：你应该盯住的主链

无论是 HTTP/OpenAI 接口，还是 Python `Engine` 直调，最终都会汇合到同一条主链：

1. 入口层（CLI/FastAPI/Engine）构造 `ServerArgs`，拉起子进程拓扑。
2. 请求进入 `TokenizerManager`，完成参数规范化、tokenize、多模态预处理、请求注册。
3. `Scheduler` 从 IPC 收到 tokenized 请求，执行 admission + batching + prefill/decode 调度。
4. 模型 executor 执行前向，输出 token ids / embedding / score。
5. `DetokenizerManager` 把 token ids 增量解码成文本，裁剪 stop 条件后回传。
6. `TokenizerManager` 把回包路由到 rid 对应的异步生成器，HTTP 层再封装 SSE 或一次性 JSON。

---

## 1. 入口一：CLI 到 HTTP Server

文件：`python/sglang/launch_server.py`

启动路径非常短，但它定义了“真正入口在哪”：

```text
if __name__ == "__main__":
  server_args = prepare_server_args(sys.argv[1:])
  launch_server(server_args)
  finally: kill_process_tree(os.getpid(), include_parent=False)
```

所以真正需要深追的是：

- `python/sglang/srt/entrypoints/http_server.py::launch_server`

### 1.1 `http_server.launch_server` 做了什么

可以拆成 5 步：

1. `_launch_subprocesses(server_args)`：创建 runtime 进程图（最关键）。
2. `set_global_state(...)`：把 `tokenizer_manager/template_manager/scheduler_info` 放到全局。
3. 注册中间件（API key、Prometheus）。
4. 准备 warmup thread。
5. `uvicorn.run(...)`：开始监听。

### 1.2 `lifespan(...)` 是 OpenAI 适配层初始化点

FastAPI 生命周期里会创建：

- `OpenAIServingChat`
- `OpenAIServingCompletion`
- `OpenAIServingEmbedding`
- `OpenAIServingScore`
- `OpenAIServingRerank`

并执行 warmup 请求（如果启用）。因此 `/v1/*` 路由不是“直接调用 tokenizer manager”，而是先经过 OpenAI serving 对象做协议转换。

---

## 2. 入口二：Engine 直调与 HTTP 的同构关系

文件：`python/sglang/srt/entrypoints/engine.py`

`Engine.__init__` 同样调用 `_launch_subprocesses(...)`，`Engine.generate(...)` 最终同样走 `tokenizer_manager.generate_request(...)`。

所以 HTTP 与 Engine 只是在“入口协议层”不同：

- HTTP：JSON/SSE + FastAPI
- Engine：Python 方法调用 + 迭代器/字典返回

但**推理执行路径（tokenizer/scheduler/detokenizer）是共用的**。这点对排查问题非常关键：HTTP 复现不了时可以直接拿 Engine 缩短链路做最小复现。

---

## 3. `_launch_subprocesses`：SRT 进程拓扑真正装配点

文件：`python/sglang/srt/entrypoints/engine.py`

这是整个系统的“总装函数”。

### 3.1 启动前配置

先做基础环境与守护逻辑：

- `configure_logger(server_args)`
- `server_args.check_server_args()`
- `_set_envs_and_config(server_args)`

其中 `_set_envs_and_config` 典型动作：

- 设置 CUDA/NCCL/内核相关环境变量
- `set_ulimit()`
- 内核/依赖检查（如 flashinfer）
- 安装 `SIGCHLD` 等处理器，子进程异常时触发进程树清理
- `mp.set_start_method("spawn", force=True)`

### 3.2 通信端口和模型路径

- `PortArgs.init_new(server_args)`：统一生成 IPC 地址/端口。
- `prepare_model_and_tokenizer(...)`：模型路径规范化（可能触发下载/重写路径）。

### 3.3 按并行策略拉起 scheduler 侧

- `dp_size == 1`：按 TP/PP rank 启动一个或多个 `run_scheduler_process(...)`。
- `dp_size > 1`：先起 `run_data_parallel_controller_process(...)`，由其驱动 DP 逻辑。

每个 scheduler 子进程通过 `mp.Pipe` 给父进程返回 ready 信息（包含容量等元信息）。

### 3.4 node_rank 分支

多节点下通常只有 `node_rank==0` 承担 tokenizer/detokenizer 与 HTTP 协议层；其余节点主要运行 scheduler 计算分片。

### 3.5 主节点补齐 Tokenizer/Detokenizer

主节点继续完成：

1. `run_detokenizer_process(...)`
2. 主进程内创建 `TokenizerManager(...)`
3. `TemplateManager.initialize_templates(...)`
4. 等待 scheduler ready，并把 `max_req_input_len` 等回填到 tokenizer manager

到这一步，系统才进入“可接单状态”。

---

## 4. HTTP 路由到 runtime：两条主入口

文件：`python/sglang/srt/entrypoints/http_server.py`

### 4.1 原生入口 `/generate`

`generate_request(obj, request)` 按 `obj.stream` 分叉：

- `stream=True`：返回 `StreamingResponse`，内部迭代 `tokenizer_manager.generate_request(...)` 并封装 SSE。
- `stream=False`：直接 `__anext__()` 等到最终结果一次性返回。

### 4.2 OpenAI 入口 `/v1/chat/completions`

路由层调用 `OpenAIServingChat.handle_request(...)`：

1. 校验模型与参数。
2. 把 OpenAI 请求对象转换成内部 `GenerateReqInput`。
3. 后续执行链与 `/generate` 完全一致。

**关键理解**：OpenAI 层是“请求重写层”，不是“另一套推理引擎”。

---

## 5. TokenizerManager 深入：请求生命周期起点

文件：`python/sglang/srt/managers/tokenizer_manager.py`

`TokenizerManager` 是主进程中的“请求编排器 + IPC 汇聚器”。

### 5.1 初始化中的关键通道

通常会建立：

- `send_to_scheduler`（向 scheduler 下发）
- `recv_from_detokenizer`（收 detokenizer 回包）
- 控制面 communicator（abort/flush/update_weights/profile 等）
- `rid_to_state`（请求状态索引）

### 5.2 `generate_request(...)` 的分层逻辑

主逻辑可以理解成 4 层：

1. **并发闸门**：若正在更新权重/内部状态，等待 `_updating` 结束。
2. **参数标准化**：`normalize_batch_and_arguments()`，统一单条/批量输入、默认参数。
3. **输入变换**：`_tokenize_one_request(...)`（或批量分支）把 text/input_ids/input_embeds 统一成 tokenized 请求结构。
4. **下发与等待**：`_send_one_request(...)` + `_wait_one_response(...)`，通过 rid 关联异步输出。

### 5.3 `_tokenize_one_request(...)` 的关键细节

输入源优先级通常是：

- `input_embeds`（跳过文本 tokenize）
- `input_ids`
- `text`（常规 tokenizer 编码）

如果存在多模态字段，会触发 `mm_processor` 异步处理，产生可供 scheduler/executor 使用的多模态特征载荷。

最后产物是 `TokenizedGenerateReqInput`（包含 sampling params、logprob 选项、stop 条件、多模态上下文等），通过 IPC 发给 scheduler。

### 5.4 后台回包循环

TokenizerManager 还会持续从 `recv_from_detokenizer` 收消息，按 rid 更新状态并唤醒对应请求迭代器：

- 流式请求：逐 chunk 推送
- 非流式请求：等完成后一次产出
- 异常/abort：传递错误并清理 `rid_to_state`

---

## 6. Scheduler 深入：从请求入队到一次 decode 迭代

文件：`python/sglang/srt/managers/scheduler.py`

`run_scheduler_process(...)` 的骨架：

1. 初始化进程上下文（日志、CPU 亲和、设备等）。
2. 构造 `Scheduler(...)`。
3. 通过 pipe 回传 ready。
4. 进入某个 `event_loop_*`。

### 6.1 为什么有多个 `event_loop_*`

SGLang 会根据并行模式/优化路径选择不同循环：

- normal
- overlap
- pipeline parallel (pp)
- prefill/decode disaggregation 等

虽然函数不同，但核心阶段一致：

```text
recv_requests
  -> process_input_requests
  -> schedule_next_batch (prefill/decode)
  -> model_executor.forward
  -> postprocess + send_to_detokenizer
```

### 6.2 `recv_requests` / `process_input_requests`

这两步负责“控制面 + 数据面”统一收敛：

- 生成请求（`TokenizedGenerateReqInput`）入等待队列。
- embedding/score/rerank 请求走对应分支。
- abort/flush/profile/update_weights 属于控制消息，改变 scheduler 内部状态机。

### 6.3 prefill 与 decode 的调度核心

实践中可把 scheduler 看成双阶段状态机：

- **prefill**：新请求第一次计算 KV cache，成本高、吞吐敏感。
- **decode**：已有 cache 上逐 token 增量生成，延迟敏感。

scheduler 每轮会在显存预算、批大小、并发限制和公平性策略之间取平衡，决定：

- 哪些请求进入本轮 prefill
- 哪些请求继续 decode
- 是否发生抢占、延后或提前终止

### 6.4 executor 前向与输出

模型执行后，scheduler 会把结果打包成批输出对象（如 token ids 批次或 embedding 批次），发往 detokenizer。对于生成请求，它通常携带：

- 每个 rid 本轮新 token id（或结束标记）
- 对应 logprob/top-logprob（如果开启）
- hidden states（若请求要求）

---

## 7. DetokenizerManager 深入：为什么它能稳定增量输出

文件：`python/sglang/srt/managers/detokenizer_manager.py`

### 7.1 event loop 分发

`event_loop()` 从 scheduler IPC 读对象后，按类型进入：

- `BatchTokenIDOut`（最常见，文本生成）
- `BatchEmbeddingOut`
- `BatchMultimodalDecodeReq`

### 7.2 `handle_batch_token_id_out(...)` 的增量逻辑

核心是维护每个 rid 的 `decode_status`：

1. 追加本轮 token ids。
2. 批量 decode 为字符串（降低 tokenizer 调用开销）。
3. 计算“新增长度差分”，只输出本轮新增文本片段。
4. 应用 stop string / stop token 裁剪（`trim_matched_stop(...)`）。
5. 打包 `BatchStrOut` 回发 tokenizer manager。

这个“状态差分 + stop 裁剪”设计，是流式输出不重复、不漏字、不越界的关键。

---

## 8. OpenAI 适配层深入：`messages` 到 `prompt_ids`

相关文件：

- `python/sglang/srt/entrypoints/openai/serving_base.py`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`

### 8.1 统一模板：`handle_request`

`OpenAIServingBase` 提供通用序列：

1. `_validate_request`
2. `_convert_to_internal_request`
3. streaming/non-streaming 输出封装

### 8.2 chat 请求转换关键点

`OpenAIServingChat._convert_to_internal_request(...)` 主要做：

- 处理 `messages`、system/developer/user/assistant 历史
- 套 conversation template 或 jinja template
- 构造 sampling params
- tools/tool_choice 约束转内部结构化约束
- 生成 `GenerateReqInput`

也就是说，OpenAI 接口“复杂”主要在请求改写，而不是执行路径。

---

## 9. 一次完整请求的“函数级时序图”

### 9.1 非流式 `/generate`

```text
Client
  -> http_server.generate_request(stream=False)
  -> tokenizer_manager.generate_request(...).__anext__()
  -> _tokenize_one_request
  -> _send_one_request (IPC -> scheduler)
  -> scheduler.event_loop_* : recv/process/schedule/forward
  -> send BatchTokenIDOut (IPC -> detokenizer)
  -> detokenizer.handle_batch_token_id_out
  -> send BatchStrOut (IPC -> tokenizer)
  -> tokenizer _wait_one_response 完成
  -> HTTP 返回最终 JSON
```

### 9.2 流式 `/v1/chat/completions`

```text
Client
  -> OpenAIServingChat.handle_request(stream=True)
  -> _convert_to_internal_request -> GenerateReqInput
  -> tokenizer_manager.generate_request (async generator)
  -> scheduler 多轮 decode
  -> detokenizer 每轮输出 BatchStrOut
  -> FastAPI StreamingResponse 持续输出 SSE chunk
  -> [DONE]
```

---

## 10. 深读代码建议：从“主链”切到“旁路”

如果你想在“可改代码”层面继续深入，建议顺序：

1. **主链必读**：
   - `launch_server.py`
   - `http_server.py`
   - `engine.py`
   - `tokenizer_manager.py`
   - `scheduler.py`
   - `detokenizer_manager.py`
2. **协议扩展**：
   - `entrypoints/openai/*`
3. **调度策略细节**：
   - scheduler 内部 admission/批处理策略与控制消息处理分支
4. **性能关键点**：
   - prefill/decode overlap
   - 多模态 preprocess 路径
   - detokenizer batch decode 与 stop 裁剪

---

## 11. 最终总结（一句话）

SGLang 的核心不是“某个 API 函数”，而是一个跨进程流水线状态机：

**入口协议层（HTTP/OpenAI/Engine）→ TokenizerManager（规范化/派发）→ Scheduler（批调度/前向）→ Detokenizer（增量解码/裁剪）→ 入口层回包。**

理解这条链后，再看任何功能（tool calling、多模态、权重热更新、并行策略）都只是挂在主链上的分支。
