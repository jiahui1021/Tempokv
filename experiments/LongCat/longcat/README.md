# LongCat + TempoKV

`experiments/LongCat/` 包含完整的本地 LongCat 推理代码：

```text
experiments/LongCat/
├── inference.py          BF16、trace、TempoKV 共用的生成流程
├── longcat_video/        LongCat 模型、VAE、scheduler 与并行代码
└── longcat/              TempoKV bridge、trace 与参数转换
```

运行时只读取本仓库代码和 `CHECKPOINT_DIR` 指向的模型权重，不需要
`LONGCAT_REPO`、`run_long_t2v.py` 或其他模型仓库。

最终命令位于 `scripts/LongCat/`：

```bash
CHECKPOINT_DIR=/models/LongCat-Video \
bash scripts/LongCat/collect_trace.sh

CONFIG=configs/longcat_int4.json \
bash scripts/LongCat/fit_params.sh

CHECKPOINT_DIR=/models/LongCat-Video \
bash scripts/LongCat/run_bf16.sh

CHECKPOINT_DIR=/models/LongCat-Video \
CONFIG=configs/longcat_int4.json \
K_PARAMS=data/longcat/params/kh2-vh2-int4/k_temporal.npz \
V_PARAMS=data/longcat/params/kh2-vh2-int4/v_cross_kv.npz \
bash scripts/LongCat/run_tempokv.sh
```

trace 合并默认读取每个 event 写入的 `trace_stride`，并要求所有 Context Parallel
rank 使用同一个值。CP1、CP2、CP4 对应的典型值分别是 `1560`、`780`、`390`；如
需处理旧 trace，可通过 `CACHE_STRIDE` 显式校验。

默认流程会生成首段视频，因此 `INIT_VIDEO_PATH` 是可选项。LongCat 参数使用
`coordinate_system=pre_rope_normalized`，并记录实际 cache stride。
