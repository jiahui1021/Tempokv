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
CONFIG=configs/longcat.json \
bash scripts/LongCat/fit_params.sh
```

默认输出：

```text
data/longcat/params/kh2-vh1-int4/k_temporal.npz
data/longcat/params/kh2-vh1-int4/v_cross_kv.npz
```

目录名由 `CONFIG` 推导（与 Self-Forcing、Causal Forcing 一致），换成
`configs/longcat.json` 会写到 `kh2-vh1-int2`，不会覆盖 INT4。
`K_HISTORY_LENGTH` / `V_HISTORY_LENGTH` 会同时影响目录名和拟合参数。
拟合输入用 `TRACE_FILE`，与 `collect_trace.sh` 的输出变量同名。

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
CONFIG=configs/longcat.json \
K_PARAMS=data/longcat/params/kh2-vh1-int4/k_temporal.npz \
V_PARAMS=data/longcat/params/kh2-vh1-int4/v_cross_kv.npz \
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

`FIT_DEVICE` 默认 `auto`：有 CUDA 就用 GPU，没有则回退 CPU 并在 stderr 打出告警——
CPU 拟合同时会把方程降到 float64，比看上去慢得多，所以回退不会静默发生。
`FIT_DEVICE=cuda` 让没有 GPU 时直接报错而不是回退。拟合报告里记录实际使用的设备。

`K_HISTORY_LENGTH` / `V_HISTORY_LENGTH` 可以简写成 `KH` / `VH`，效果相同；
两个名字都设置且取值不同会直接报错。不设置时读 `CONFIG` 里的值（默认 `kh=2`、`vh=1`）。

位宽是命令行 override，不是单独的配置文件：`BITS=2` 同时设 K/V，
`K_BITS` / `V_BITS` 分别设（混合精度用）。`BITS` 与某一侧设成不同值会直接报错。
位宽会进入参数目录名，所以 INT2 与 INT4 的参数互不覆盖。

`V_PREDICTOR=cross_kv|temporal` 切换 V 预测器，也不需要单独的配置文件；它会同时
决定参数目录名（`vh1` 与 `vt1`）和参数文件名（`v_cross_kv.npz` 与 `v_temporal.npz`），
两者不会互相覆盖。注意 temporal V 目前只有 legacy 拟合路径能产出，所以在 native
配置上单独设它会在读 trace 之前就报错并说明原因。

## 历史帧数敏感性实验

扫 K 与 V 的历史帧数，报告重建误差、attention 输出误差和预测项占比：

```bash
MAX_SAMPLES=64 MAX_FRAMES=8 LAYERS=0,15,29 \
bash scripts/LongCat/sweep_history.sh    # 先小规模确认能跑通
bash scripts/LongCat/sweep_history.sh    # 全量
```

K 扫 `k_history_length`（1 到 8），V 在 CrossKV 项之上再扫 `v_history_length`
（0 到 5），并额外拆出 V 预测里 CrossKV 项与 history 项各自的能量占比——两项相关，
所以并排报告而不是把总量切开。

每个 vh 取值还会做一次 **CrossKV 项的留一消融**：把 V 预测里的 CrossKV 项或
history 项从**已拟合好的**模型里去掉，看残差各涨多少。

```text
vh       both  cross only  history only  cross buys  history buys
 0    0.01493     0.01493       0.99618     0.98125       0.00000
 1    0.00489     0.48567       0.12061     0.11573       0.48078
```

`cross buys` = 去掉 CrossKV 项后的残差减去两项都在时的残差，`history buys` 同理。
这是**留一消融**而不是"不带该项重新拟合"，所以说的是该项在这个预测器里贡献了多少，
不是"不用它训出来的模型能做到多好"。两项相关，所以两个残差不构成对总残差的分解。

**每个取值都会重新拟合一次。** 非 shared 参数的历史长度在拟合时就固定了，拿一套
参数改 N 得到的不是那个 N 的模型。位宽、group_size 和 stride 由 `CONFIG` 固定。

报告默认写到 `results/longcat/history_sweep.json`。变量说明见
[scripts/common/sweep_history.sh](../common/sweep_history.sh)。
