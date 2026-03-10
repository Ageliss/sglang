# SGLang `w4afp8`：代码调用链与支持情况（深入版）

> 目标：不只说明“支持/不支持”，而是把 **从启动参数到 kernel dispatch 的完整调用链** 写清楚，便于定位性能、精度和兼容性问题。

---

## 1. 先给结论（架构层面）

`w4afp8` 在 SGLang 里不是“全模型同一种算子”，而是 **按层类型分流**：

- 普通线性层（`LinearBase`）→ `Fp8LinearMethod`
- MoE 层（`FusedMoE` / EP MoE）→ `W4AFp8MoEMethod` + `cutlass_w4a8_moe`

也就是说，它是一个“**Linear走FP8，MoE专家走W4A8 kernel**”的混合路径。

---

## 2. 入口调用链（参数/配置如何落到 `w4afp8`）

### 2.1 显式指定路径

```text
CLI: --quantization w4afp8
  └─ ServerArgs.parse()
      └─ ModelConfig._verify_quantization()
          └─ quantization = "w4afp8"
```

关键点：

- `server_args.py` 中 `--quantization` choices 包含 `w4afp8`。
- 因此这是“一级公民参数”，不是 hidden flag。

### 2.2 自动识别路径（ModelOpt MIXED_PRECISION）

```text
ModelConfig._parse_quant_hf_config()
  ├─ 读取 hf_config.quantization_config / compression_config
  └─ 若本地或远端存在 hf_quant_config.json:
       quant_algo == "MIXED_PRECISION"
         → quant_cfg = {"quant_method": "w4afp8"}

ModelConfig._verify_quantization()
  └─ 将 quantization 统一到 w4afp8 并做合法性检查
```

这条链路的意义是：你不传 `--quantization w4afp8`，只要 checkpoint 元数据满足条件，也会自动命中 `w4afp8`。

---

## 3. 量化配置对象如何下沉到具体层（类分派链）

```text
quantization string: "w4afp8"
  └─ layers/quantization/__init__.py
      └─ QUANTIZATION_METHODS["w4afp8"] = W4AFp8Config

W4AFp8Config.get_quant_method(layer, prefix)
  ├─ if LinearBase: return Fp8LinearMethod(self)
  ├─ if FusedMoE:   return W4AFp8MoEMethod(self)
  └─ else:          return None
```

`W4AFp8Config` 的关键约束：

- `get_supported_act_dtypes()` 返回 `[torch.bfloat16, torch.float8_e4m3fn]`
- `get_min_capability()` 返回 `90`

即：至少 SM90（Hopper）能力级别。

---

## 4. MoE 权重加载调用链（这是最容易被忽略的一段）

`w4afp8` 的 MoE 不是仅仅“前向时调用 kernel”，它对权重结构和 scale 组织有一整套加载规约。

### 4.1 从模型权重加载入口进入

```text
DeepseekV2ForCausalLM.load_weights(...)
  ├─ make_expert_params_mapping(...)
  └─ if quant == w4afp8:
       + make_expert_input_scale_params_mapping(...)
```

当 `quant_config.get_name() == "w4afp8"` 时，会额外注入 input scale 的映射规则，这一步对后面 `a1_scale / a2_scale` 至关重要。

### 4.2 EP MoE 内部的 shard 装配

```text
EPMoE.weight_loader(...)
  └─ _weight_loader_physical(...)
      ├─ 非 scale 权重: 按 w1/w2/w3 拼到 w13/w2 对应分片
      └─ scale 权重: _load_fp8_scale(...)
           ├─ input_scale: w1/w3 分别落到 w13_input_scale 的两个槽位
           └─ weight_scale: 按 w1/w3/w2 拼接到 *_weight_scale_inv
```

### 4.3 load 完成后的后处理（`W4AFp8MoEMethod.process_weights_after_loading`）

```text
w13_weight_scale_inv / w2_weight_scale_inv
  └─ _interleave_scales(): 每4个元素做交织重排

w13_input_scale / w2_input_scale
  └─ 取 max -> 压缩成单元素张量
```

这一步是为了匹配底层 cutlass 路径的数据布局预期，不是可省略的“清洗动作”。

---

## 5. 前向执行调用链（从 EP MoE 到 CUDA kernel）

下面是核心“跑起来时”的调用链：

```text
EPMoE.forward_normal(...)
  └─ if self.use_w4afp8:
      └─ cutlass_w4a8_moe(...)
          ├─ run_cutlass_moe_ep_preproess(...)
          ├─ pre_reorder_triton_kernel_for_cutlass_moe(...)
          ├─ get_cutlass_w4a8_moe_mm_data(...)
          ├─ cutlass_w4a8_moe_mm(...)   # GEMM1: gate+up
          ├─ silu_and_mul(...)
          ├─ sgl_per_tensor_quant_fp8(...)
          ├─ cutlass_w4a8_moe_mm(...)   # GEMM2: down
          └─ post_reorder_triton_kernel(...)
```

可见它是“**两次 grouped GEMM + 中间激活 + 再量化 + 回排**”的 pipeline，不是单 kernel。

---

## 6. `cutlass_w4a8_moe_mm` 再向下的 C++/CUDA 调用链

Python 侧（`sgl_kernel/cutlass_moe.py`）通过 `torch.ops.sgl_kernel.*` 进入 C++ extension：

```text
torch.ops.sgl_kernel.cutlass_w4a8_moe_mm
  └─ csrc/common_extension.cc 注册项
      └─ csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu
          └─ cutlass_w4a8_moe_mm_sm90(...)
              └─ w4a8_grouped_mm_c3x.cu::dispatch_w4a8_moe_mm_sm90(...)
```

`dispatch_w4a8_moe_mm_sm90` 会按 `(n, k, m)` 分支，选择不同 tile/cluster 配置：

- `n==4096 && k==7168`（group gemm1）多档分支
- `n==7168 && k==2048`（group gemm2）多档分支
- 其他形状走 fallback 配置

这说明它不是“一个通吃配置”，而是为典型 DeepSeek MoE 形状做了显式调优。

---

## 7. 支持情况（基于真实代码路径，而非口号）

## 7.1 已支持

1. **CLI 显式指定**：`--quantization w4afp8`
2. **模型配置自动识别**：`MIXED_PRECISION -> w4afp8`
3. **量化方法注册**：`QUANTIZATION_METHODS` 中有 `w4afp8`
4. **MoE 专用执行链**：`EPMoE.forward_normal -> cutlass_w4a8_moe -> cutlass_w4a8_moe_mm`
5. **内核测试存在**：`sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py`

## 7.2 约束与边界

1. **最小硬件能力**：`get_min_capability() == 90`
2. **ROCm 列表限制**：`_verify_quantization` 的 `rocm_supported_quantization` 未包含 `w4afp8`
3. **实现指向 SM90 路径**：`scaled_mm_entry.cu` 直接调用 `cutlass_w4a8_moe_mm_sm90`
4. **激活 dtype 路径固定性较强**：`cutlass_w4a8_moe` 中关键中间量使用 `float8_e4m3fn`/`half` 组合

---

## 8. 排障建议（按调用链倒查）

### 8.1 没命中 `w4afp8`

按这个顺序查：

1. CLI 是否传了 `--quantization w4afp8`
2. 若未传，`hf_quant_config.json` 的 `quant_algo` 是否是 `MIXED_PRECISION`
3. `_verify_quantization` 是否因不兼容报错或回退

### 8.2 命中了但性能/行为异常

按这条链查：

1. 是否走入 `EPMoE.forward_normal` 的 `self.use_w4afp8` 分支
2. `cutlass_w4a8_moe` 里 `w1/w2 scale` 形状断言是否满足
3. `dispatch_w4a8_moe_mm_sm90` 是否落在预期 `(n,k,m)` 分支
4. 是否在非 SM90 / ROCm 环境误用该路径

---

## 9. 最小复现实例

```bash
python -m sglang.launch_server \
  --model <your_model_or_path> \
  --quantization w4afp8 \
  --dtype bfloat16
```

> 说明：若模型自带 `MIXED_PRECISION` 元数据，也可先不传 `--quantization`，确认自动识别结果。

