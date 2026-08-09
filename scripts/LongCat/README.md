# LongCat 调用入口

LongCat 模型运行代码和 TempoKV 接入都在本仓库 `experiments/LongCat/`。这些
脚本不读取 `LONGCAT_REPO`，也不调用外部 runner。

## 安装

```bash
pip install -e ".[longcat]"
```

## 采集并合并 clean trace

```bash
CHECKPOINT_DIR=/models/LongCat-Video \
PROMPT="a cinematic mountain road" \
bash scripts/LongCat/collect_trace.sh
```

默认输出：

```text
data/longcat/trace/raw/
data/longcat/trace/output/
data/longcat/trace/clean_trace.npz
```

## 拟合参数

```bash
CONFIG=configs/longcat_int4.json \
bash scripts/LongCat/fit_params.sh
```

## BF16

```bash
CHECKPOINT_DIR=/models/LongCat-Video \
PROMPT="a cinematic mountain road" \
bash scripts/LongCat/run_bf16.sh
```

默认输出到 `results/longcat/bf16/`。

## TempoKV

```bash
CHECKPOINT_DIR=/models/LongCat-Video \
CONFIG=configs/longcat_int4.json \
K_PARAMS=data/longcat/params/kh2-vh2-int4/k_temporal.npz \
V_PARAMS=data/longcat/params/kh2-vh2-int4/v_cross_kv.npz \
PROMPT="a cinematic mountain road" \
bash scripts/LongCat/run_tempokv.sh
```

默认输出到 `results/longcat/tempokv/`。

三种模式都会先从 prompt 生成首段视频，然后在本仓库内执行 video
continuation。若希望从现有视频开始，可额外设置：

```bash
INIT_VIDEO_PATH=/data/input.mp4
```

常用可选变量：`NUM_SEGMENTS`、`NUM_FRAMES`、`NUM_COND_FRAMES`、
`NUM_INFERENCE_STEPS`、`CONTEXT_PARALLEL_SIZE`、`NUM_GPUS`、`SEED`、
`OUTPUT_DIR`、`DRY_RUN`。
