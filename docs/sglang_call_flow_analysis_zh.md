# SGLang Runtime 调用链深度解析（从入口函数出发）

> 基于当前主干（main）代码结构整理，目标是给出一条可落地的“追调用链”路径：
> `launch_server.py` → HTTP/OpenAI 入口 → TokenizerManager → Scheduler → Detokenizer → 返回。

---

## 0. 快速导读（先建立全局图）

SGLang 在线服务是一个多进程流水线：

- **主进程**：FastAPI（协议层）+ `TokenizerManager`（请求编排层）
- **子进程组**：`Scheduler`（调度/执行层，数量由 TP/PP/DP 决定）
- **子进程**：`DetokenizerManager`（解码回包层）

一句话：
**入口层负责接请求，TokenizerManager 负责规范化与收发，Scheduler 负责算，Detokenizer 负责把 token ids 变回文本。**

---

## 1. 入口：`launch_server.py`

文件：`python/sglang/launch_server.py`

启动顺序：

```text
__main__
  -> prepare_server_args(sys.argv[1:])
  -> sglang.srt.entrypoints.http_server.launch_server(server_args)
  -> finally: kill_process_tree(...)
```

关键点：
- 真正“重逻辑”不在这个文件，而在 `http_server.launch_server`。
- `finally` 里统一清理进程树，避免子进程残留。

---

## 2. HTTP 服务主干：`http_server.launch_server`

文件：`python/sglang/srt/entrypoints/http_server.py`

`launch_server(server_args)` 的职责是：

1. 调 `_launch_subprocesses(server_args)` 组装运行时拓扑。
2. 设置全局状态（`TokenizerManager` / `TemplateManager` / `scheduler_info`）。
3. 注入中间件（API Key、Prometheus，按参数开启）。
4. 准备 warmup 线程。
5. `uvicorn.run(app, ...)` 启动服务。

### 2.1 lifespan 阶段

在 FastAPI `lifespan(...)` 中，会初始化 OpenAI 适配 handler：
- `OpenAIServingCompletion`
- `OpenAIServingChat`
- `OpenAIServingEmbedding`
- `OpenAIServingScore`
- `OpenAIServingRerank`

并按配置执行 warmups。

---

## 3. 引擎总装：`_launch_subprocesses`

文件：`python/sglang/srt/entrypoints/engine.py`

函数：`_launch_subprocesses(server_args, port_args=None)`

### 3.1 初始化

顺序：
1. `configure_logger(server_args)`
2. `server_args.check_server_args()`
3. `_set_envs_and_config(server_args)`

`_set_envs_and_config` 包含：
- CUDA/NCCL/Triton 环境变量
- `set_ulimit()`
- flashinfer / sgl-kernel 版本检查
- `SIGCHLD` / `SIGQUIT` 处理（子进程故障触发清理）
- `mp.set_start_method("spawn", force=True)`

### 3.2 端口与模型准备

- `PortArgs.init_new(server_args)` 分配 IPC 端口
- `prepare_model_and_tokenizer(...)` 准备模型/分词器路径

### 3.3 调度进程创建

- `dp_size == 1`：按 TP×PP 组合创建多个 `run_scheduler_process(...)`
- `dp_size > 1`：走 `run_data_parallel_controller_process(...)`

每个 scheduler 通过 `Pipe` 回传 ready 信息（如 `max_req_input_len`）。

### 3.4 节点差异

- `node_rank >= 1`：通常不启动 tokenizer/detokenizer，等待调度层。
- `node_rank == 0`：继续启动完整链路：
  1. `run_detokenizer_process(...)`
  2. `TokenizerManager(...)`
  3. `TemplateManager.initialize_templates(...)`
  4. 汇总 ready 信息并回填 `max_req_input_len`

---

## 4. 路由入口分类（HTTP）

文件：`python/sglang/srt/entrypoints/http_server.py`

1. **原生 SRT 路由**：`/generate`, `/encode`, `/classify`, `/flush_cache`, `/update_weights_*`...
2. **OpenAI 兼容路由**：`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/score`, `/v1/rerank`
3. **健康与元数据**：`/health`, `/health_generate`, `/get_model_info`, `/get_server_info`, `/get_load`

---

## 5. 核心链路一：`/generate`

### 5.1 HTTP 层

函数：`generate_request(obj: GenerateReqInput, request: Request)`

- `obj.stream=True`：返回 `StreamingResponse`（SSE 分片 + `[DONE]`）
- `obj.stream=False`：`await ...generate_request(...).__anext__()` 一次返回

### 5.2 TokenizerManager 层

文件：`python/sglang/srt/managers/tokenizer_manager.py`

关键通信端：
- `send_to_scheduler`（PUSH）
- `recv_from_detokenizer`（PULL）

主流程：

```text
generate_request(...)
  -> normalize_batch_and_arguments()
  -> _tokenize_one_request(...)（或 batch 分支）
  -> _send_one_request(...) -> scheduler
  -> _wait_one_response(...) <- detokenizer
  -> yield chunk / final
```

`_tokenize_one_request(...)` 支持：
- `input_embeds`
- `input_ids`
- `text`（调用 tokenizer 编码）

多模态额外进入 `mm_processor.process_mm_data_async(...)`。

### 5.3 Scheduler 层

文件：`python/sglang/srt/managers/scheduler.py`

`run_scheduler_process(...)`：
1. 初始化 `Scheduler(...)`
2. pipe 回传 ready
3. 进入 `event_loop_normal / overlap / pp / disagg_*`

单轮循环共性：

```text
recv_requests()
  -> process_input_requests(...)
  -> 组 prefill/decode batch
  -> forward
  -> send_to_detokenizer
```

### 5.4 Detokenizer 层

文件：`python/sglang/srt/managers/detokenizer_manager.py`

`event_loop()`：
1. 收 scheduler 输出
2. 分发到 `BatchTokenIDOut / BatchEmbeddingOut / BatchMultimodalDecodeReq` 处理函数
3. 回传 tokenizer manager

`handle_batch_token_id_out(...)` 负责：
- `rid` 级增量状态维护
- `batch_decode` + 差分增量输出
- stop string / stop token 裁剪
- 产出 `BatchStrOut`

### 5.5 回传客户端

TokenizerManager 汇总回包并更新请求状态，最终由 HTTP 层返回：
- 非流式：最终 JSON
- 流式：SSE chunk + `[DONE]`

---

## 6. 核心链路二：`/v1/chat/completions`（OpenAI 适配）

相关文件：
- `python/sglang/srt/entrypoints/openai/serving_base.py`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`

调用序列：

```text
/v1/chat/completions 路由
  -> OpenAIServingChat.handle_request(...)
  -> OpenAIServingBase.handle_request(...)
      -> _validate_request
      -> _convert_to_internal_request
      -> streaming / non-streaming
  -> TokenizerManager.generate_request(...)
```

`_convert_to_internal_request(...)` 的作用：
1. 处理 messages（模板、工具调用、多模态）
2. 构建采样参数
3. 生成内部 `GenerateReqInput`

结论：
- OpenAI 层是协议适配壳；执行内核仍是 `/generate` 同一管线。

---

## 7. Python Engine 模式（不走 HTTP）

文件：`python/sglang/srt/entrypoints/engine.py`

- `Engine.__init__` 也走 `_launch_subprocesses(...)`
- `Engine.generate(...)` 也封装 `GenerateReqInput` 后调用 `tokenizer_manager.generate_request(...)`

因此：
- HTTP 与 Engine 的核心推理链路同构
- 区别主要在入口协议与返回封装

---

## 8. DSL 与 Runtime 的连接点

文件：`python/sglang/api.py`

- DSL 表达：`gen/select/function/system/user/...`
- 运行时入口：
  - `Runtime(...)`（endpoint 模式）
  - `Engine(...)`（本地引擎模式）

可以理解为：
- `lang/*`：描述任务
- `srt/*`：执行任务

---

## 9. 推荐“先看这些类型”

1. `ServerArgs` / `PortArgs`：并行拓扑、端口与模式开关。
2. `GenerateReqInput` / `EmbeddingReqInput`：请求统一输入结构。
3. Scheduler 内部请求/批结构：prefill/decode 调度关键。
4. `BatchTokenIDOut` / `BatchStrOut`：detokenize 前后核心交换结构。

---

## 10. 两张实战追踪图

### 10.1 非流式 `/generate`

```text
Client -> /generate(stream=false)
  -> http_server.generate_request
  -> tokenizer_manager.generate_request(...).__anext__()
  -> _tokenize_one_request
  -> send_to_scheduler
  -> scheduler循环forward
  -> send_to_detokenizer
  -> detokenizer -> BatchStrOut
  -> tokenizer_manager收包
  -> HTTP返回JSON
```

### 10.2 流式 `/v1/chat/completions`

```text
Client -> /v1/chat/completions(stream=true)
  -> OpenAIServingChat.handle_request
  -> _convert_to_internal_request(GenerateReqInput)
  -> tokenizer_manager.generate_request(async generator)
  -> scheduler多轮decode
  -> detokenizer增量解码
  -> SSE分片输出
  -> [DONE]
```

---

## 11. 最短阅读路径（建议）

1. `python/sglang/launch_server.py`
2. `python/sglang/srt/entrypoints/http_server.py`
3. `python/sglang/srt/entrypoints/engine.py`（重点 `_launch_subprocesses`）
4. `python/sglang/srt/managers/tokenizer_manager.py`
5. `python/sglang/srt/managers/scheduler.py`
6. `python/sglang/srt/managers/detokenizer_manager.py`
7. `python/sglang/srt/entrypoints/openai/serving_base.py`
8. `python/sglang/srt/entrypoints/openai/serving_chat.py`
9. `python/sglang/api.py`

---

## 12. 一句话总结

**入口（CLI/HTTP/Python） -> 协议适配（原生/OpenAI/DSL） -> TokenizerManager -> Scheduler -> Detokenizer -> 回传。**

沿这条主线追函数和数据结构，就能快速定位 SGLang runtime 的关键行为与调试切入点。
