# LongCat 调用入口

该目录只保存用户直接调用的 shell：

```text
scripts/LongCat/
├── README.md
├── _common.sh
├── collect_trace.sh
├── fit_params.sh
├── generate.sh
├── run_bf16.sh
└── run_tempokv.sh
```

Python 接入代码位于 `experiments/LongCat/longcat/`，TempoKV 核心算法位于
`src/tempokv/`，trace 与拟合参数位于 `data/longcat/`。

BF16：

```bash
LONGCAT_REPO=/path/to/LongCat-runtime \
CHECKPOINT_DIR=/models/LongCat-Video \
INIT_VIDEO_PATH=results/longcat/base/0-0.mp4 \
OUTPUT_DIR=results/longcat/bf16 \
bash scripts/LongCat/run_bf16.sh
```

TempoKV：

```bash
LONGCAT_REPO=/path/to/LongCat-runtime \
CHECKPOINT_DIR=/models/LongCat-Video \
INIT_VIDEO_PATH=results/longcat/base/0-0.mp4 \
OUTPUT_DIR=results/longcat/tempokv-int4 \
CONFIG=configs/longcat_int4.json \
K_PARAMS=data/longcat/params/kh2-vh2-int4/k_temporal.npz \
V_PARAMS=data/longcat/params/kh2-vh2-int4/v_cross_kv.npz \
bash scripts/LongCat/run_tempokv.sh
```

完整 LongCat 说明见 `experiments/LongCat/longcat/README.md`。
