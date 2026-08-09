# LongCat 调用入口

LongCat 模型运行代码和 TempoKV 接入都在本仓库 `experiments/LongCat/`。这些
脚本不读取 `LONGCAT_REPO`，也不调用外部 runner。

## 安装

```bash
pip install -e ".[cuda,longcat]"
```

LongCat 与 Self-Forcing 的 `diffusers`、`transformers`、`av` 版本不同，建议
使用独立的虚拟环境。

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

合并阶段默认从 trace 文件中的 `trace_stride` 读取空间步长，并检查所有 Context
Parallel rank 的 metadata 一致。因此 `CONTEXT_PARALLEL_SIZE=1/2/4` 可以分别使用
`1560/780/390`，不需要手动修改合并命令。只有处理旧 trace 或强制校验时才设置
`CACHE_STRIDE`。

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
