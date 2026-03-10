# SGLang 调用链深度解析（从入口函数出发）

> 本文给出一条“可追踪、可定位、可调试”的主线：从 `launch_server.py` 入口，到 HTTP/OpenAI 请求进入，再到 `TokenizerManager -> Scheduler -> DetokenizerManager` 全链路返回。

---

## 1. 先看全局：谁负责什么

SGLang Runtime（SRT）服务模式本质是：

- **主进程**：FastAPI + `TokenizerManager`
- **子进程（一个或多个）**：`Scheduler`
- **子进程（一个）**：`DetokenizerManager`

职责分工：

1. **TokenizerManager**：接收请求、参数归一化、tokenize/多模态预处理、发送给 scheduler、汇总回包。
2. **Scheduler**：接收 tokenized 请求，做批调度与前向执行，输出 token ids/embedding 等中间结果。
3. **DetokenizerManager**：将 token ids 增量解码为文本并回传。

进程间通信通过 ZMQ IPC（PUSH/PULL、DEALER）。

---

## 2. 启动入口与第一跳

### 2.1 CLI 入口

入口文件：`python/sglang/launch_server.py`

启动顺序：

```text
__main__
  -> prepare_server_args(sys.argv[1:])
  -> launch_server(server_args)
  -> finally: kill_process_tree(os.getpid(), include_parent=False)
```

第一跳关键函数是：
- `python/sglang/srt/entrypoints/http_server.py::launch_server`

### 2.2 `http_server.launch_server` 做了什么

`launch_server(server_args)` 关键步骤：

1. 调 `engine._launch_subprocesses(...)` 组装运行时进程拓扑。
2. 将 `tokenizer_manager/template_manager/scheduler_info` 写入全局状态。
3. 按参数注入中间件（API Key、Prometheus）。
4. 准备 warmup 线程。
5. `uvicorn.run(app, ...)` 对外监听端口。

### 2.3 FastAPI 生命周期

`lifespan(...)` 会在服务可用前初始化 OpenAI handler：
- `OpenAIServingCompletion`
- `OpenAIServingChat`
- `OpenAIServingEmbedding`
- `OpenAIServingScore`
- `OpenAIServingRerank`

并在需要时执行 warmups。

---

## 3. `_launch_subprocesses`：引擎总装函数

文件：`python/sglang/srt/entrypoints/engine.py`

函数：`_launch_subprocesses(server_args, port_args=None)`

#### 3.1 初始化阶段

执行顺序：
1. `configure_logger(server_args)`
2. `server_args.check_server_args()`
3. `_set_envs_and_config(server_args)`

其中 `_set_envs_and_config` 负责：
- 设置 CUDA/NCCL/Triton 相关环境变量
- `set_ulimit()`
- 关键依赖版本检查（如 `flashinfer_python`、`sgl-kernel`）
- 注册 `SIGCHLD` / `SIGQUIT` 信号处理
- 设置 multiprocessing start method 为 `spawn`

#### 3.2 端口与模型路径

- `PortArgs.init_new(server_args)` 分配 IPC 端口
- `prepare_model_and_tokenizer(...)` 解析/准备模型与 tokenizer 路径

#### 3.3 Scheduler 进程创建分支

- `dp_size == 1`：按 TP/PP 组合创建多个 `run_scheduler_process(...)`
- `dp_size > 1`：创建 `run_data_parallel_controller_process(...)`

每个 scheduler 就绪后通过 `Pipe` 回传 `status=ready` 与容量信息。

#### 3.4 主节点与非主节点差异

- `node_rank >= 1`：通常不启动 tokenizer/detokenizer，等待调度进程并提供健康检查能力。
- `node_rank == 0`：继续启动完整链路：
  1. `run_detokenizer_process(...)`
  2. `TokenizerManager(...)`
  3. `TemplateManager.initialize_templates(...)`
  4. 汇总 scheduler 就绪信息并回填 `max_req_input_len`

---

## 4. 请求入口层（HTTP 路由）

文件：`python/sglang/srt/entrypoints/http_server.py`

路由可分三组：

1. **原生 SRT 路由**：`/generate`, `/encode`, `/classify`, `/flush_cache`, `/update_weights_*`...
2. **OpenAI 兼容路由**：`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/score`, `/v1/rerank`
3. **健康与元信息**：`/health`, `/health_generate`, `/get_model_info`, `/get_server_info`, `/get_load`

---

## 5. `/generate` 端到端调用链（核心）

### 5.1 HTTP 层

函数：`generate_request(obj: GenerateReqInput, request: Request)`

- `obj.stream=True`：返回 `StreamingResponse`，按 SSE 格式持续输出，末尾 `[DONE]`。
- `obj.stream=False`：取异步生成器第一项并直接返回。

### 5.2 TokenizerManager

文件：`python/sglang/srt/managers/tokenizer_manager.py`

关键通信成员：
- `send_to_scheduler`（PUSH）
- `recv_from_detokenizer`（PULL）

主流程：

```text
generate_request(...)
  -> normalize_batch_and_arguments()
  -> _tokenize_one_request(...) / batch分支
  -> _send_one_request(...) 发送给scheduler
  -> _wait_one_response(...) 等待detokenizer回包
  -> yield chunk或最终结果
```

`_tokenize_one_request(...)` 支持三种输入：
- `input_embeds`
- `input_ids`
- `text`（调用 tokenizer 编码）

多模态会进入：`mm_processor.process_mm_data_async(...)`。

### 5.3 Scheduler

文件：`python/sglang/srt/managers/scheduler.py`

`run_scheduler_process(...)`：
1. 初始化调度器 `Scheduler(...)`
2. 通过 pipe 回传 ready
3. 进入事件循环（`event_loop_normal/overlap/pp/...`）

单轮循环共性：

```text
recv_requests()
  -> process_input_requests(...)
  -> 形成prefill/decode batch
  -> forward
  -> 输出到detokenizer
```

其中 `process_input_requests(...)` 还会处理控制类消息（abort、flush、profile、权重更新等）。

### 5.4 Detokenizer

文件：`python/sglang/srt/managers/detokenizer_manager.py`

`event_loop()`：
1. 从 scheduler 接收对象
2. 按类型分发处理：`BatchTokenIDOut` / `BatchEmbeddingOut` / `BatchMultimodalDecodeReq`
3. 回传 tokenizer manager

`handle_batch_token_id_out(...)` 关键逻辑：
- 维护每个 `rid` 的增量解码状态
- `batch_decode` 后做差分增量输出
- 应用 stop 字符串/stop token 裁剪
- 生成 `BatchStrOut`

### 5.5 回传客户端

TokenizerManager 接收 detokenizer 回包后更新 `rid_to_state`，将输出写入对应异步流；HTTP 层再返回 JSON 或 SSE。

---

## 6. OpenAI 兼容链路（`/v1/chat/completions`）

相关文件：
- `python/sglang/srt/entrypoints/openai/serving_base.py`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`

调用顺序：

```text
v1_chat路由
  -> OpenAIServingChat.handle_request(...)
  -> OpenAIServingBase.handle_request(...)
      -> _validate_request
      -> _convert_to_internal_request
      -> streaming/non-streaming分支
  -> 进入TokenizerManager.generate_request(...)
```

`_convert_to_internal_request(...)` 会：
1. 处理 messages（含模版、多模态内容、tool配置）
2. 生成采样参数
3. 构造内部 `GenerateReqInput`

即：OpenAI 协议层只是适配壳，执行内核与 `/generate` 一致。

---

## 7. Python Engine 模式（非 HTTP）

文件：`python/sglang/srt/entrypoints/engine.py`

`Engine` 入口同样走 `_launch_subprocesses(...)`，`Engine.generate(...)` 也是封装 `GenerateReqInput` 后调用 `tokenizer_manager.generate_request(...)`。

结论：
- HTTP 模式与 Engine 模式在推理主干上是同构的；差别主要在入口协议与返回封装。

---

## 8. DSL 层与 Runtime 层关系

文件：`python/sglang/api.py`

- DSL 构造：`gen/select/function/system/user/...`
- Runtime 连接：
  - `Runtime(...)`（endpoint 模式）
  - `Engine(...)`（本地引擎模式）

简化理解：
- `lang/*` 负责“描述要做什么”
- `srt/*` 负责“如何高效执行”

---

## 9. 建议抓手：先看这些数据结构

1. `ServerArgs` / `PortArgs`：决定拓扑、并行与端口。
2. `GenerateReqInput` / `EmbeddingReqInput`：请求统一载体。
3. Scheduler 内部请求/批结构：决定 prefill/decode 调度行为。
4. `BatchTokenIDOut` / `BatchStrOut`：detokenizer 前后关键交换结构。

---

## 10. 两张实战追踪图

#### 10.1 非流式 `/generate`

```text
Client -> /generate(stream=false)
  -> http_server.generate_request
  -> tokenizer_manager.generate_request(...).__anext__()
  -> _tokenize_one_request
  -> send_to_scheduler
  -> scheduler循环处理并forward
  -> send_to_detokenizer
  -> detokenizer解码为BatchStrOut
  -> tokenizer_manager收包
  -> HTTP返回JSON
```

#### 10.2 流式 `/v1/chat/completions`

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

## 11. 推荐阅读顺序（最短路径）

1. `python/sglang/launch_server.py`
2. `python/sglang/srt/entrypoints/http_server.py`
3. `python/sglang/srt/entrypoints/engine.py`（尤其 `_launch_subprocesses`）
4. `python/sglang/srt/managers/tokenizer_manager.py`
5. `python/sglang/srt/managers/scheduler.py`
6. `python/sglang/srt/managers/detokenizer_manager.py`
7. `python/sglang/srt/entrypoints/openai/serving_base.py`
8. `python/sglang/srt/entrypoints/openai/serving_chat.py`
9. `python/sglang/api.py`

---

## 12. 一句话总括

**入口层（CLI/HTTP/Python） -> 协议适配层（原生/OpenAI/DSL） -> TokenizerManager -> Scheduler -> Detokenizer -> 回传。**

沿着这条主线追函数与数据结构，基本就能覆盖 SGLang runtime 的关键行为与调试入口。
