# LongCat + TempoKV

该目录保存 LongCat 的 TempoKV 接入代码。最终调用脚本位于
`scripts/LongCat/`，trace 与参数统一位于 `data/longcat/`。

当前 LongCat 运行环境通过 `LONGCAT_REPO` 指定；Self-Forcing 不使用这个参数。

## BF16

```bash
MODE=bf16 \
LONGCAT_REPO=/path/to/LongCat-runtime \
CHECKPOINT_DIR=/models/LongCat-Video \
INIT_VIDEO_PATH=results/longcat/base/0-0.mp4 \
OUTPUT_DIR=results/longcat/bf16 \
bash scripts/LongCat/generate.sh
```

## Trace

```bash
LONGCAT_REPO=/path/to/LongCat-runtime \
CHECKPOINT_DIR=/models/LongCat-Video \
INIT_VIDEO_PATH=results/longcat/base/0-0.mp4 \
TRACE_DIR=data/longcat/trace/raw \
bash scripts/LongCat/collect_trace.sh

python experiments/LongCat/longcat/prepare_trace.py \
  data/longcat/trace/raw \
  data/longcat/trace/clean_trace.npz
```

## 参数拟合

```bash
CONFIG=configs/longcat_int4.json \
bash scripts/LongCat/fit_params.sh
```

默认输出为 `data/longcat/params/kh2-vh2-int4/`。

## TempoKV

```bash
MODE=tempokv \
LONGCAT_REPO=/path/to/LongCat-runtime \
CHECKPOINT_DIR=/models/LongCat-Video \
INIT_VIDEO_PATH=results/longcat/base/0-0.mp4 \
OUTPUT_DIR=results/longcat/tempokv-int4 \
CONFIG=configs/longcat_int4.json \
K_PARAMS=data/longcat/params/kh2-vh2-int4/k_temporal.npz \
V_PARAMS=data/longcat/params/kh2-vh2-int4/v_cross_kv.npz \
bash scripts/LongCat/generate.sh
```

LongCat 参数必须记录实际 cache stride，并使用
`coordinate_system=pre_rope_normalized`。Context Parallel、cache offload、
attention sink 与 `torch.compile` 应分别验证后再组合使用。
