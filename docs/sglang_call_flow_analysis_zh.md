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
- `OpenAIServingCompletion`
- `OpenAIServingChat`
- `OpenAIServingEmbedding`
- `OpenAIServingScore`
- `OpenAIServingRerank`

并可执行 warmup 请求。

---

## 3. 引擎子系统拉起流程（关键）

`python/sglang/srt/entrypoints/engine.py::_launch_subprocesses` 是服务端核心编排函数。

### 3.1 环境初始化
- `configure_logger`
- `server_args.check_server_args`
- `_set_envs_and_config`：
  - NCCL/Triton/CUDA 环境变量
  - ulimit
  - 版本校验（如 `flashinfer_python` / `sgl-kernel`）
  - 信号处理（子进程异常 -> 主进程清理树）

### 3.2 端口与模型
- `PortArgs.init_new(server_args)` 分配 IPC 端口
- `prepare_model_and_tokenizer(...)` 处理模型/tokenizer 路径（含 modelscope 场景）

### 3.3 进程拓扑
根据 `dp_size` 分两类：

1. **`dp_size == 1`**：直接拉起若干 `Scheduler` 进程（按 TP/PP 拆分）
2. **`dp_size > 1`**：先拉起 `DataParallelController`，再由其管理并行

之后主节点（`node_rank == 0`）继续：
- 启动 `DetokenizerManager` 子进程
- 在主进程创建 `TokenizerManager`
- 初始化 `TemplateManager`
- 等待 scheduler pipe 返回 `status=ready`，拿到 `max_req_input_len` 等能力边界

> 非 0 rank 节点通常不跑 tokenizer/detokenizer，而是等待并维护健康检查行为。

---

## 4. 一次请求的完整调用链

下面以 `/generate` 为例描述 E2E 链路。

### 4.1 HTTP 层
`/generate` endpoint：
- 流式：`StreamingResponse`，持续迭代 `TokenizerManager.generate_request(...)`
- 非流式：只取异步生成器首个结果 `.__anext__()` 返回

### 4.2 TokenizerManager（主进程）
`TokenizerManager.generate_request(...)` 主要步骤：

1. 参数归一化（`normalize_batch_and_arguments`）
2. 对单请求执行 `_tokenize_one_request`：
   - 文本 tokenizer 编码（或直接使用 `input_ids` / `input_embeds`）
   - 多模态时调用 `mm_processor.process_mm_data_async`
3. 将 tokenized 请求通过 ZMQ `send_to_scheduler` 发给 scheduler
4. 在 `_wait_one_response`（或 batch 分支）中等待 detokenizer 回流结果
5. 把增量/最终结果 yield 给 HTTP 层

### 4.3 Scheduler（子进程）
`run_scheduler_process(...)`：

1. 构造 `Scheduler(...)`
2. pipe 回传 `status=ready`
3. 按模式进入事件循环：
   - `event_loop_normal`
   - `event_loop_overlap`
   - `event_loop_pp`
   - disaggregation prefill/decode 变体

事件循环主体模式：

```text
recv_requests()
  -> process_input_requests(...)
  -> 调度批次/前向计算/缓存管理
  -> 输出 token/id/logprob 等到 detokenizer IPC
```

### 4.4 Detokenizer（子进程）
`DetokenizerManager.event_loop()`：

1. 从 scheduler 拉取 `BatchTokenIDOut / BatchEmbeddingOut / BatchMultimodalDecodeReq`
2. 对 token-id 结果增量解码（含 stop 条件裁剪）
3. 产出 `BatchStrOut`（或其他输出）
4. 通过 ZMQ 推回 TokenizerManager

### 4.5 回传用户
TokenizerManager 收到 detokenizer 结果后：
- 更新 rid 状态
- 组装对外 schema（文本、logprob、token 使用量等）
- 流式请求持续推送 chunk，结束发送 `[DONE]`

---

## 5. OpenAI 兼容接口调用链

以 `/v1/chat/completions` 为例：

1. FastAPI endpoint 调用 `OpenAIServingChat.handle_request(...)`
2. 基类 `OpenAIServingBase.handle_request` 执行通用模板：
   - 校验请求
   - `_convert_to_internal_request`
   - 按 `stream` 分流
3. `OpenAIServingChat._convert_to_internal_request`：
   - 处理 messages、template、tools/function call 约束
   - 构造内部 `GenerateReqInput`
4. 后续调用回到 `TokenizerManager.generate_request`，与原生 `/generate` 共享同一执行引擎

即：

```text
OpenAI协议层 -> 协议适配(GenerateReqInput) -> 统一运行时管线(Tokenizer/Scheduler/Detokenizer)
```

---

## 6. Python API 入口（非 HTTP）

`python/sglang/srt/entrypoints/engine.py::Engine` 提供本地 Python 直连接口：

- `Engine.__init__`：同样走 `_launch_subprocesses` 建 runtime
- `Engine.generate(...)`：包装 `GenerateReqInput` 后直接调用 `tokenizer_manager.generate_request`
- 可同步（内部跑 event loop）或异步（`async_generate`）

因此 HTTP 只是“外壳”；真正执行栈与 Python Engine 基本一致。

---

## 7. DSL 层（`sglang.api`）与 runtime 的关系

`python/sglang/api.py` 暴露两类能力：

1. **语言 DSL 构造**（`function/gen/select/system/user/...`）
   - 生成 IR（`SglGen/SglSelect/SglFunction...`），用于程序式 prompt 编排
2. **runtime 连接入口**
   - `Runtime(...)`：连接运行时 endpoint
   - `Engine(...)`：直接构造本地 Engine

所以可以理解为：

```text
DSL层(描述“要做什么”) + Runtime层(负责“如何执行推理”)
```

---

## 8. 全局调用图（简版）

```text
[入口]
launch_server.py::__main__
  -> prepare_server_args
  -> http_server.launch_server
      -> engine._launch_subprocesses
          -> run_scheduler_process (1..N 子进程)
          -> run_detokenizer_process (1 子进程)
          -> TokenizerManager (主进程)
          -> TemplateManager.init
      -> uvicorn.run(FastAPI)

[请求]
HTTP /generate or /v1/chat/completions
  -> (OpenAI层可选) OpenAIServing* 适配
  -> TokenizerManager.generate_request
      -> tokenize/mm preprocess
      -> ZMQ -> Scheduler
      -> batch schedule + model forward
      -> ZMQ -> Detokenizer
      -> incremental detokenize
      -> ZMQ -> TokenizerManager
  -> FastAPI 返回(流式/非流式)
```

---

## 9. 阅读代码时建议的主线顺序

建议按以下顺序追代码，最快建立整体心智模型：

1. `python/sglang/launch_server.py`
2. `python/sglang/srt/entrypoints/http_server.py`
3. `python/sglang/srt/entrypoints/engine.py::_launch_subprocesses`
4. `python/sglang/srt/managers/tokenizer_manager.py`
5. `python/sglang/srt/managers/scheduler.py`
6. `python/sglang/srt/managers/detokenizer_manager.py`
7. `python/sglang/srt/entrypoints/openai/*`（协议适配层）
8. `python/sglang/api.py` 与 `python/sglang/lang/*`（DSL层）

