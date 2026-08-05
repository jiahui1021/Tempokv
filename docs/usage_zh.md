# TempoKV 中文使用指南

TempoKV 负责对注意力层的 K/V cache 做参数拟合、量化和重建。仓库提供 codec、PyTorch 参考解码器、可选的 Triton CUDA 解码器，以及 `self-forcing/` 下的首个模型适配器。其他模型仍需要在自己的 cache 写入与读取位置接入。

## 1. 安装与配置

CPU 或 PyTorch 参考路径：

```bash
pip install -e ".[test]"
```

使用 NVIDIA GPU 上的 Triton 解码器：

```bash
pip install -e ".[cuda,test]"
```

仓库提供两份配置：

- `configs/int2.json`：K/V 残差为 INT2，首个 anchor unit 保持 INT4。
- `configs/int4.json`：K/V 残差和首个 anchor unit 都为 INT4。

JSON 字段与 `tempokv.config.CodecConfig` 一一对应，可直接加载：

```python
import json

from tempokv import CodecConfig

with open("configs/int2.json", encoding="utf-8") as file:
    config = CodecConfig(**json.load(file))
```

如果模型的 head dimension 不是 64 的倍数，必须先把 `group_size` 改为能够整除 head dimension 的正整数。

## 2. 准备 clean K/V trace

### 两种布局

运行时，单层 codec 接收的 K 和 V 都是：

```text
[B, H, S, D]
```

- `B`：batch size。
- `H`：KV head 数。
- `S`：cache 的 sequence 长度。
- `D`：每个 head 的维度。

参数拟合使用：

```text
[L, ..., T, H, D]
```

- `L`：层数。
- `...`：可合并统计的样本维度，例如样本、batch、unit 内位置。
- `T`：按时间排列的 cache unit；必须至少为 `history_length + 1`。
- `H`、`D` 与运行时一致。

K 和 V 必须形状相同、位于同一设备、使用浮点 dtype，并且只包含有限数值。`group_size` 必须整除 `D`。

### trace 文件 key

`scripts/fit_params.py` 读取关闭 pickle 的 `.npz` 文件。默认 key 为：

```text
keys       clean K，[L,...,T,H,D]
values     clean V，[L,...,T,H,D]
layer_ids  可选；每层对应的真实模型层号，一维整数数组
```

其中固定使用 `keys` 和 `values`，避免把 K/V 顺序写反。也可以通过 `--keys-name` 和 `--values-name` 读取已有数据集的其他 key。

如果采集结果是 `[L, N, H, S, D]`，其中 `N` 是样本数，并且运行时按 `stride` 切分 sequence，可以把完整 unit 转为拟合布局：

```python
import torch


def cache_to_fit_trace(cache: torch.Tensor, stride: int) -> torch.Tensor:
    # [L,N,H,S,D] -> [L,N,U,T,H,D]
    layers, samples, heads, sequence, width = cache.shape
    units = sequence // stride
    if units < 2:
        raise ValueError("sequence 至少要包含两个完整 unit")
    cache = cache[..., : units * stride, :]
    return (
        cache.reshape(layers, samples, heads, units, stride, width)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
    )


clean_k = cache_to_fit_trace(captured_k, config.stride)
clean_v = cache_to_fit_trace(captured_v, config.stride)
```

保存为拟合脚本可读取的文件：

```python
import numpy as np

np.savez(
    "clean_kv_trace.npz",
    keys=clean_k.cpu().numpy(),
    values=clean_v.cpu().numpy(),
    layer_ids=np.asarray(model_layer_ids, dtype=np.int32),
)
```

这里的 `U` 是 unit 内对齐位置，会作为额外样本维度；`T` 是 unit 顺序。末尾不足一个 `stride` 的部分可不参与拟合，运行时 codec 仍可处理 partial final unit。采集多个 prompt、输入或生成样本通常比只采集一个样本更可靠，但训练 trace 应与实际部署模型、层和输入分布一致。

## 3. 拟合并保存参数

`fit_predictors` 会以 clean K/V 为目标，闭环重建 K，再用最终重建 K 拟合 V。不要用 clean K 代替运行时可见的重建 K 去单独拟合 V。

直接运行拟合脚本：

```bash
python scripts/fit_params.py clean_kv_trace.npz \
  --config configs/int2.json \
  --output-dir fit_output \
  --history-length 3 \
  --iterations 2 \
  --temporal-ridge 0.1 \
  --cross-ridge 0.1 \
  --max-gain 1.0
```

输出目录包含：

```text
fit_output/k_temporal.npz  K 时序预测参数
fit_output/v_cross_kv.npz  V 的 K-to-V 预测参数
fit_output/fit_report.json 输入、配置、拟合参数和参数形状摘要
```

命令行参数会覆盖 JSON 中的同名配置。例如先保持其他 INT2 设置不变，只把 K 改为 INT4：

```bash
python scripts/fit_params.py clean_kv_trace.npz \
  --config configs/int2.json \
  --output-dir fit_k4_v2 \
  --history-length 3 \
  --k-bits 4
```

需要在 Python 中拟合时，可以从同一 `.npz` 读取数据：

```python
import numpy as np
import torch

from tempokv import fit_predictors

with np.load("clean_kv_trace.npz", allow_pickle=False) as trace:
    clean_k = torch.from_numpy(trace["keys"].copy())
    clean_v = torch.from_numpy(trace["values"].copy())
    layer_ids = (
        tuple(int(item) for item in trace["layer_ids"])
        if "layer_ids" in trace
        else None
    )

k_parameters, v_parameters = fit_predictors(
    clean_k,
    clean_v,
    config=config,
    history_length=3,
    iterations=2,
    temporal_ridge=0.1,
    cross_ridge=0.1,
    max_gain=1.0,
    layer_ids=layer_ids,
)

k_parameters.save("k_temporal.npz")
v_parameters.save("v_cross_kv.npz")
```

参数文件只保存数值数组，加载时关闭 pickle。拟合时记录的 `stride` 必须与运行时配置相同；不匹配时 codec 会报错。

常用拟合参数：

- `history_length`：K 预测使用的历史 unit 数。更大可能利用更长时序关系，也会增加拟合量和运行时预测工作；必须小于 `T`。
- `iterations`：K 闭环重建与重拟合轮数。建议从 2 开始，用保留 trace 比较误差后再增加。
- `temporal_ridge`、`cross_ridge`：线性拟合的正则强度。数据少或拟合不稳定时适当增大。
- `max_gain`：限制 K 历史系数的绝对值总和。较小更保守；默认 1.0。
- `layer_ids`：参数层与真实模型层号的映射。传入非连续层号时，运行时 `layer=` 必须使用这些真实层号。

## 4. 用参数量化与重建

### 对导出的 cache 做完整往返检查

`scripts/generate.py` 接收一个 `.npz` cache 文件，默认读取 `keys` 和 `values`。两者必须使用相同的浮点布局：单层为 `[B,H,S,D]`，多层为 `[L,B,H,S,D]`。

```bash
python scripts/generate.py exported_cache.npz reconstructed_int2 \
  --config configs/int2.json \
  --k-params fit_output/k_temporal.npz \
  --v-params fit_output/v_cross_kv.npz \
  --device cpu \
  --no-triton
```

输出目录包含：

```text
reconstructed_int2/reconstructed.npz  重建后的 keys、values、layer_ids
reconstructed_int2/report.json         每层 NMSE、存储字节数和压缩比摘要
```

多层输入默认使用参数文件保存的 `layer_ids`。需要显式指定时使用 `--layer-ids 3,7,11`。输入文件使用其他 key 时，可分别传 `--k-key` 和 `--v-key`。

这个脚本只对已经导出的 K/V cache 做量化和重建检查。它不会加载生成模型，也不会独立生成图片或视频。

### 在 Python 中使用 codec

```python
from tempokv import CrossKVParameters, TempoKVCodec, TemporalParameters

k_parameters = TemporalParameters.load("k_temporal.npz")
v_parameters = CrossKVParameters.load("v_cross_kv.npz")
codec = TempoKVCodec(k_parameters, v_parameters, config)

# key_cache/value_cache: 当前层的 [B,H,S,D] 浮点张量
packed = codec.encode(key_cache, value_cache, layer=layer_index)

# 省略 output_dtype 时，使用 config.reconstruction_dtype。
key_rebuilt, value_rebuilt = codec.decode(packed)
```

编码与解码顺序固定：K 先量化并闭环重建，V 再从这份重建 K 预测残差；解码也先恢复 K，再恢复 V。`decode(..., output_dtype=...)` 如果显式传值，必须与 `reconstruction_dtype` 一致。

`PackedKV` 是量化后的 cache 状态。实际节省多少显存，取决于调用方是否用它替换原始 K/V，并在注意力计算需要时只保留必要的重建结果。

## 5. 参数怎么调

建议用与部署输入分布一致的保留 trace，同时记录 K/V 重建误差、下游质量、显存和耗时。一次只改一组参数。

| 参数 | 可选值与约束 | 调整影响 |
| --- | --- | --- |
| `k_bits` / `v_bits` | 2、4、8 | 越低越省残差存储，误差风险越高。K 与 V 可分别选择。 |
| `anchor_bits` | 2、4、8 | 控制第一个 unit；它还是后续 K 闭环预测的起点。低 bit 可能把误差传播到后续 unit。 |
| `group_size` | 正整数，且整除 `D` | 小 group 通常量化更细，但 scale/zero-point 开销更高；大 group 元数据较少，误差可能增加。 |
| `stride` | 正整数 | 决定 `[B,H,S,D]` 沿 `S` 的 unit 长度。小 stride 会产生更多预测 unit；大 stride 会让 anchor 占比更高。它必须与拟合参数完全一致。 |
| `residual_mode` | `symmetric`、`asymmetric` | asymmetric 多存 zero point，适合分布偏移的残差；symmetric 格式更简单。应以实测误差和开销选择。 |
| `scale_dtype` | `float32`、`float16`、`bfloat16` | float32 的 scale 精度最高、占用更大；16-bit dtype 更省。 |
| `reconstruction_dtype` | `float32`、`float16`、`bfloat16` | 决定闭环重建和 decode 输出 dtype。float32 更精细但重建张量占用更大；应匹配模型计算策略。 |

推荐起点是 `configs/int2.json`。如果保留 trace 或下游任务质量下降明显，优先尝试把 `k_bits` 提到 4；K 的误差会参与后续 K 预测，也会影响 V 的预测源。随后再比较提高 `v_bits`、`anchor_bits` 或减小 `group_size`。

配置修改后应重新拟合参数，特别是 `stride`、K/V bit、anchor bit、残差模式、scale dtype 或重建 dtype 变化时。这样拟合看到的闭环误差与部署一致。

## 6. Self-Forcing 完整生成流程

Self-Forcing 的适配、启动命令、trace 格式和视频评测统一放在 [`self-forcing/README.md`](../self-forcing/README.md)。流程分成两个 SH 脚本。

第一次运行负责采集 clean trace 和拟合参数：

```bash
TRACE_RUNNER=/path/to/self-forcing-trace-runner.sh \
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=/data/fit_prompts.txt \
WORK_DIR=runs/self_forcing_prepare \
CONFIG=configs/int2.json \
bash self-forcing/prepare.sh
```

之后只读取已有参数生成视频，可以更换 prompt 并重复运行：

```bash
MODE=tempokv \
MODEL_REPO=/path/to/self-forcing-model-repo \
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=/data/generation_prompts.txt \
PARAM_DIR=runs/self_forcing_prepare/parameters \
OUTPUT_DIR=runs/self_forcing_int2 \
CONFIG=configs/int2.json \
bash self-forcing/generate.sh
```

设置 `MODE=bf16` 可以运行不量化的对照生成；该模式不需要 `CONFIG` 和
`PARAM_DIR`。两种模式都会在 `pipeline_report.json` 中保存总耗时、每个 GPU
的耗时和最高显存占用。

两个入口都会检查 `stride=1560`、30 层、12 heads、head dimension 128 和 pre-RoPE normalized K。生成长度还会检查是否真正触发 TempoKV。

## 7. 接入其他模型

TempoKV 不负责加载模型或执行生成。接入点位于模型自己的 attention cache 边界：

1. 在目标模型上运行有代表性的输入，采集每层量化前的 clean K/V；统一为 `[L,...,T,H,D]` 后离线拟合。
2. 模型生成 K/V 且准备写入 cache 时，把单层张量整理成 `[B,H,S,D]`，调用 `codec.encode(..., layer=真实层号)`，由 cache 管理器保存返回的 `PackedKV`。
3. attention 需要读取该 cache 时调用 `codec.decode`，把返回的 K/V 交回原有 attention 计算。
4. 保证 rotary embedding、head 拆分、token/frame 排列等模型变换发生在与采集 trace 相同的位置；拟合和部署的张量语义必须一致。
5. 用未参与拟合的输入验证输出质量、重建误差、峰值显存和端到端耗时。

如果模型使用增量 cache、paged cache、分片张量或自定义 fused attention，需要调用方另外实现相应的存储生命周期和张量适配。当前 codec 处理传入的完整 `[B,H,S,D]` 张量，不会自动修改第三方模型。

## 8. CUDA/Triton 严格模式与缓存

相关配置含义：

- `use_triton=false`：始终使用 PyTorch 参考解码器，适合 CPU、调试和结果对照。
- `use_triton=true, strict_triton=false`：CUDA 张量会尝试 Triton；布局不支持或执行失败时发出 warning，并回退到 PyTorch。
- `use_triton=true, strict_triton=true`：Triton 布局不支持、未正确安装或执行失败都会直接报错。适合性能验收与生产检查，避免静默回退。

Triton 只在 packed 数据位于 CUDA 时尝试启用。首次 decode 前设置独立编译缓存，尤其是多进程任务：

```python
from tempokv import configure_triton_cache

cache_dir = configure_triton_cache("/fast/local/tempokv-triton")
```

该函数优先保留已有的 `TRITON_CACHE_DIR`；否则在指定根目录下按进程 PID 和 rank 建立目录。未显式传 `rank` 时依次读取 `LOCAL_RANK`、`RANK`，最后使用 0。缓存目录应放在本机可写的快速磁盘，并在每个进程第一次加速 decode 之前调用。

建议先用 `strict_triton=false` 完成功能验证，再用 `strict_triton=true` 跑 CUDA 验收；同时用 `use_triton=false` 的 PyTorch 结果作为重建一致性基准。

用量化重建脚本做严格 CUDA 验收：

```bash
python scripts/generate.py exported_cache.npz reconstructed_cuda \
  --config configs/int2.json \
  --k-params fit_output/k_temporal.npz \
  --v-params fit_output/v_cross_kv.npz \
  --device cuda \
  --use-triton \
  --strict-triton \
  --triton-cache /fast/local/tempokv-triton
```

需要允许回退时，把 `--strict-triton` 换成 `--allow-triton-fallback`。

## 9. 最小检查

```bash
pytest
```

上线前至少确认：JSON 可构造 `CodecConfig`；`group_size` 整除 `D`；参数和配置的 `stride` 一致；层号能在 `layer_ids` 中找到；decode dtype 与 `reconstruction_dtype` 一致；CUDA 性能测试中没有意外回退到 PyTorch。
