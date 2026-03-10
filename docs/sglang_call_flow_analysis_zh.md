# SGLang 代码调用逻辑深度解析（从入口函数出发）

> 目标：从**入口函数**一路追到**模型前向与回包**，把关键模块、关键函数、关键数据结构串成可执行的“读码地图”。

---

## 1. 总体架构先看结论

SGLang Runtime（SRT）在服务模式下是一个**1 主进程 + 多子进程**的结构：

- 主进程：FastAPI + `TokenizerManager`
- 子进程 A：`Scheduler`（可能 1 个或多个，取决于 TP/PP/DP）
- 子进程 B：`DetokenizerManager`

核心思想：
1. 主进程负责协议接入（HTTP/OpenAI）、请求规范化、tokenize、多模态预处理。
2. Scheduler 负责 batching + 调度 + 调用模型 executor 前向。
3. Detokenizer 负责把 token ids 增量解码成文本再回传主进程。

进程间通信统一通过 ZMQ IPC（`PUSH/PULL` 或 `DEALER`）完成。

---

## 2. 入口函数与启动调用链

### 2.1 CLI 入口
文件：`python/sglang/launch_server.py`

入口代码路径非常短：

```text
if __name__ == "__main__":
  server_args = prepare_server_args(sys.argv[1:])
  launch_server(server_args)
  finally: kill_process_tree(...)
```

这意味着你要追“真实启动逻辑”，应直接进入：
- `python/sglang/srt/entrypoints/http_server.py::launch_server`

### 2.2 `http_server.launch_server` 的职责切分
`launch_server(server_args)` 做了五件关键事：

1. `_launch_subprocesses(server_args)`：拉起引擎进程拓扑（最关键）。
2. `set_global_state(...)`：把 `TokenizerManager/TemplateManager/scheduler_info` 注入全局。
3. 条件注册中间件：API key 与 Prometheus。
4. 准备 warmup 线程（延后在 FastAPI lifespan 阶段触发）。
5. `uvicorn.run(...)` 启动 HTTP 监听。

### 2.3 FastAPI 生命周期钩子
`lifespan(...)` 在服务“可用前”初始化 OpenAI 适配 handler：

# SGLang 代码调用逻辑总览（从入口函数出发）

本文从**启动入口**开始，梳理 SGLang 在 `python/sglang` 目录下的主要调用链路，重点覆盖：

- HTTP 服务模式（`launch_server.py`）
- Python Engine 模式（`Engine`）
- 请求在 Tokenizer/Scheduler/Detokenizer 三段流水线中的流转
- OpenAI 兼容接口如何适配为内部请求
- `sglang` DSL（`api.py`）如何连接到 runtime

---

## 1. 启动入口（CLI）

### 1.1 文件入口
- 入口文件：`python/sglang/launch_server.py`
- 主流程：
  1. `prepare_server_args(sys.argv[1:])` 解析启动参数。
  2. `launch_server(server_args)` 启动 HTTP + 引擎。
  3. `finally` 中调用 `kill_process_tree(...)`，确保子进程收敛退出。

对应调用顺序：

```text
python -m sglang.launch_server / sglang.launch_server.py
  -> prepare_server_args(...)
  -> sglang.srt.entrypoints.http_server.launch_server(server_args)
  -> (退出时) kill_process_tree(...)
```

---

## 2. HTTP 服务主干（FastAPI）

### 2.1 `http_server.launch_server` 核心职责
`python/sglang/srt/entrypoints/http_server.py::launch_server` 会：

1. 调用 `_launch_subprocesses(server_args)` 拉起 runtime 子系统。
2. 把 `TokenizerManager / TemplateManager / scheduler_info` 写入全局状态。
3. 注册 API Key 中间件、Prometheus 中间件（可选）。
4. 启动 warmup 线程。
5. `uvicorn.run(app, ...)` 对外提供 HTTP 服务。

### 2.2 FastAPI 生命周期
`lifespan(...)` 在服务就绪阶段初始化 OpenAI handler：

 main
- `OpenAIServingCompletion`
- `OpenAIServingChat`
- `OpenAIServingEmbedding`
- `OpenAIServingScore`
- `OpenAIServingRerank`

并按 `server_args.warmups` 执行预热请求，最后启动 `warmup_thread`。

---

## 3. `_launch_subprocesses` 深入：引擎如何被拼起来

文件：`python/sglang/srt/entrypoints/engine.py`

`_launch_subprocesses(server_args, port_args=None)` 是 runtime 的“总装函数”。

### 3.1 初始化阶段
顺序是：

1. `configure_logger(server_args)`
2. `server_args.check_server_args()`
3. `_set_envs_and_config(server_args)`

`_set_envs_and_config` 会做：
- 设置 NCCL/CUDA/Triton 环境（如 `NCCL_NVLS_ENABLE`, `CUDA_MODULE_LOADING`）
- `set_ulimit()`
- flashinfer / sgl-kernel 版本校验
- 注册 `SIGCHLD` / `SIGQUIT` 处理：子进程异常时主进程执行 `kill_process_tree`
- `mp.set_start_method("spawn", force=True)`

### 3.2 IPC 与模型路径
- 若未给 `port_args`，调用 `PortArgs.init_new(server_args)` 分配通信端口。
- `prepare_model_and_tokenizer(...)` 处理模型路径（含 modelscope 下载/解析场景）。

### 3.3 Scheduler 进程拓扑
按 `dp_size` 分支：

#### 分支 A：`dp_size == 1`
按 `TP x PP` 组合循环创建多个 `run_scheduler_process(...)`：
- 每个进程有独立 `gpu_id/tp_rank/pp_rank`
- 通过 `mp.Pipe` 给主进程回传 `ready` 状态

#### 分支 B：`dp_size > 1`
先启动 `run_data_parallel_controller_process(...)`，由其控制数据并行行为。

### 3.4 非 0 节点特殊路径
多节点时，`node_rank >= 1` 节点一般不跑 tokenizer/detokenizer：
- 等待 scheduler ready
- 可能启动 dummy health server
- 阻塞等待子进程结束

### 3.5 主节点补齐剩余组件
`node_rank == 0` 时继续：
1. 启动 detokenizer 子进程：`run_detokenizer_process(...)`
2. 主进程初始化 `TokenizerManager(server_args, port_args)`
3. 初始化 `TemplateManager.initialize_templates(...)`
4. 等待 scheduler 的 pipe 回传 `{"status":"ready", ...}`
5. 将 `max_req_input_len` 回填到 tokenizer manager

这一步结束后，引擎才算“具备接单能力”。

---

## 4. HTTP 路由层：请求从哪里进入

文件：`python/sglang/srt/entrypoints/http_server.py`

可以粗分三类 API：

1. **原生 SRT API**
   - `/generate`
   - `/encode`
   - `/classify`
   - `/flush_cache`
   - `/update_weights_from_*`
   - `/set_internal_state` 等

2. **OpenAI 兼容 API**
   - `/v1/chat/completions`
   - `/v1/completions`
   - `/v1/embeddings`
   - `/v1/score`
   - `/v1/rerank`

3. **健康与元数据 API**
   - `/health`
   - `/health_generate`
   - `/get_model_info`
   - `/get_server_info`
   - `/get_load`

---

## 5. `/generate` 精确调用链（最核心）

### 5.1 HTTP 层行为
`generate_request(obj: GenerateReqInput, request: Request)`：

- `obj.stream == True`：
  - 返回 `StreamingResponse`
  - 内部异步迭代 `tokenizer_manager.generate_request(...)`
  - 每个 chunk 包装为 SSE：`data: {json}\n\n`
  - 最后发送 `data: [DONE]`

- `obj.stream == False`：
  - 直接 `await ...generate_request(...).__anext__()`
  - 一次性返回最终结果

### 5.2 TokenizerManager：请求规范化与下发
文件：`python/sglang/srt/managers/tokenizer_manager.py`

#### 关键成员（初始化时建立）
- `recv_from_detokenizer`（ZMQ PULL）
- `send_to_scheduler`（ZMQ PUSH）
- `rid_to_state`：请求状态表
- 多种 communicator（权重更新、内部状态、LoRA 等控制面）

#### `generate_request(...)` 主流程
1. 等待 `_updating` 结束（防止模型更新期并发冲突）
2. `obj.normalize_batch_and_arguments()`
3. 单请求：`_tokenize_one_request(obj)`
4. `_send_one_request(...)` 发给 scheduler
5. `_wait_one_response(...)` 等待 detokenizer 回包并 `yield`
6. 批请求则走 `_handle_batch_request(...)`

#### `_tokenize_one_request(...)` 细节
输入来源分三种：
- `input_embeds`
- `input_ids`
- `text`（需要 tokenizer 编码）

多模态请求（图/音频/视频）额外走：
- `mm_processor.process_mm_data_async(...)`

最后产出 tokenized request（包含采样参数、多模态特征、日志概率配置等）发送到 scheduler。

### 5.3 Scheduler：调度与前向
文件：`python/sglang/srt/managers/scheduler.py`

`run_scheduler_process(...)` 会：
1. 设置进程标题、日志、亲和性等
2. `scheduler = Scheduler(...)`
3. `pipe_writer.send({status: ready, ...})`
4. 按模式进入事件循环

#### 事件循环入口
- `event_loop_normal`
- `event_loop_overlap`
- `event_loop_pp`
- disaggregation prefill/decode 对应 loop

#### 每轮循环共性
```text
recv_requests()
  -> process_input_requests(recv_reqs)
  -> 形成 prefill/decode batch
  -> model forward
  -> send_to_detokenizer（token ids / embedding / mm decode req）
```

`process_input_requests(...)` 通过 dispatcher 将输入分流：
- `TokenizedGenerateReqInput` -> 生成类请求入队
- `TokenizedEmbeddingReqInput` -> embedding/reward 请求
- Abort/Flush/Profile/UpdateWeights 等控制消息

### 5.4 Detokenizer：增量解码与 stop 裁剪
文件：`python/sglang/srt/managers/detokenizer_manager.py`

`event_loop()`：
1. `recv_from_scheduler.recv_pyobj()`
2. 基于类型分发：
   - `BatchEmbeddingOut`
   - `BatchTokenIDOut`
   - `BatchMultimodalDecodeReq`
3. `send_to_tokenizer.send_pyobj(output)`

`handle_batch_token_id_out(...)` 的关键点：
- 维护 `decode_status[rid]`（增量解码状态）
- 先做 `batch_decode` 得到 `surr_texts/read_texts`
- 按增量差分生成输出
- 应用 `trim_matched_stop(...)`（stop string / stop token 裁剪）
- 返回 `BatchStrOut`（含 tokens、logprobs、hidden states 等）

### 5.5 回到 TokenizerManager 并最终返回
TokenizerManager 的后台 loop 持续从 `recv_from_detokenizer` 接收消息：
- 更新 `rid_to_state`
- 将 chunk 推入请求对应的异步生成器
- 完成时清理状态、统计指标

HTTP 层随后把结果以 JSON（非流）或 SSE（流式）返回给客户端。

---

## 6. `/v1/chat/completions` 详细链路（OpenAI 适配）

相关文件：
- `python/sglang/srt/entrypoints/openai/serving_base.py`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`

调用顺序：

1. 路由函数调用 `OpenAIServingChat.handle_request(...)`
2. 基类 `OpenAIServingBase.handle_request(...)` 执行统一模板：
   - `_validate_request`
   - `_convert_to_internal_request`
   - 按 `stream` 走 streaming/non-streaming
3. `OpenAIServingChat._convert_to_internal_request(...)`：
   - `_process_messages(...)`（messages/tool/template）
   - `_build_sampling_params(...)`
   - 构造内部 `GenerateReqInput`
4. 之后进入与 `/generate` 完全一致的 runtime 通路

### 6.1 messages 如何变成 prompt
`_process_messages(...)` 的两种路径：
- 有模板名：`_apply_conversation_template(...)`
- 无模板名：`_apply_jinja_template(...)`

`_apply_jinja_template(...)` 会：
- 处理 multimodal content
- 处理 tools 参数格式（含异常兜底）
- `tokenizer.apply_chat_template(..., tokenize=True)` 得到 `prompt_ids`
- 如需 `continue_final_message`，追加 assistant prefix token ids

### 6.2 tool calling 约束
当 `request.tools` 存在且 `tool_choice != "none"`：
- 构建 `FunctionCallParser`
- 生成结构化约束 `tool_call_constraint`
- 注入采样参数，约束模型输出格式

---

## 7. Python Engine 模式：绕过 HTTP 的同构路径

文件：`python/sglang/srt/entrypoints/engine.py`

`Engine` 的关键结论：
- `Engine.__init__` 同样调用 `_launch_subprocesses(...)`
- `Engine.generate(...)` 同样创建 `GenerateReqInput`
- 实际执行仍是 `tokenizer_manager.generate_request(...)`

区别仅在入口协议层：
- HTTP 模式：FastAPI + JSON/SSE
- Engine 模式：Python 直接调用，返回 dict 或 iterator

因此调试模型行为时，HTTP 与 Engine 大多数“推理内核”问题是复现一致的。

---

## 8. DSL (`sglang.api`) 与 Runtime 的衔接点

文件：`python/sglang/api.py`

`api.py` 同时提供：

1. DSL 构造能力：`gen/select/function/system/user/assistant...`
2. runtime 入口：
   - `Runtime(...)` -> `lang.backend.runtime_endpoint.Runtime`
   - `Engine(...)` -> `srt.entrypoints.engine.Engine`

也就是说 DSL 负责“表达任务”，Runtime 负责“执行任务”。

补一句定位：
- `lang/*` 更偏前端编排语义层
- `srt/*` 更偏后端高性能执行层

---

## 9. 关键数据结构（读码抓手）

为了更快读懂调用链，建议先盯住这些类型：

1. `ServerArgs` / `PortArgs`
   - 决定了进程拓扑、通信端口、并行参数、运行模式

2. `GenerateReqInput` / `EmbeddingReqInput`
   - 请求统一载体（text/input_ids/mm_input/sampling_params/stream...）

3. Scheduler 内部 `Req` / batch 结构
   - 决定 prefill/decode 调度、缓存复用、并行分发

4. `BatchTokenIDOut` / `BatchStrOut`
   - detokenizer 前后核心交换结构

从这些结构反向搜引用，能快速定位关键路径。

---

## 10. 两张“实战追踪图”

### 10.1 非流式 `/generate`

```text
Client POST /generate(stream=false)
  -> http_server.generate_request
  -> tokenizer_manager.generate_request(...).__anext__()
  -> _tokenize_one_request
  -> send_to_scheduler
  -> scheduler.recv_requests/process/batch/forward
  -> send_to_detokenizer
  -> detokenizer.decode -> BatchStrOut
  -> tokenizer_manager 收包并完成该 rid
  -> HTTP 返回最终 JSON
```

### 10.2 流式 `/v1/chat/completions`

```text
Client POST /v1/chat/completions(stream=true)
  -> OpenAIServingChat.handle_request
  -> _convert_to_internal_request(GenerateReqInput)
  -> tokenizer_manager.generate_request (async generator)
  -> scheduler 循环多轮 decode
  -> detokenizer 每轮输出增量文本
  -> HTTP SSE: data: {...}\n\n
  -> data: [DONE]
```

---

## 11. 建议的深度读码顺序（比“总览版”更具体）

1. `python/sglang/launch_server.py`
2. `python/sglang/srt/entrypoints/http_server.py`
   - 重点：`launch_server`, `lifespan`, `/generate`, `/v1/*`
3. `python/sglang/srt/entrypoints/engine.py`
   - 重点：`_launch_subprocesses`, `Engine.generate`
4. `python/sglang/srt/managers/tokenizer_manager.py`
   - 重点：`__init__`, `generate_request`, `_tokenize_one_request`
5. `python/sglang/srt/managers/scheduler.py`
   - 重点：`run_scheduler_process`, `event_loop_*`, `recv_requests`, `process_input_requests`
6. `python/sglang/srt/managers/detokenizer_manager.py`
   - 重点：`event_loop`, `handle_batch_token_id_out`
7. `python/sglang/srt/entrypoints/openai/serving_base.py`
8. `python/sglang/srt/entrypoints/openai/serving_chat.py`
9. `python/sglang/api.py`

---

## 12. 一句话总结

SGLang 的主干调用逻辑本质是：

**入口层（CLI/HTTP/Python） -> 协议适配层（原生/OpenAI/DSL） -> TokenizerManager（规范化与分发） -> Scheduler（调度与前向） -> Detokenizer（增量解码） -> 回传。**

只要按这条主线抓住关键数据结构与 event loop，就能从“能跑”走到“能改”。
