# Per-channel temporal vs Dense temporal

只比较 temporal 项，不使用 CrossKV。分别测量 K 的 KH2 和 V 的 VH1；两类模型都仅接收对应张量自己的、已经量化重建的历史帧。

## 定义

每个 layer/head 独立保存参数，head dimension 为 D。

- Per-channel：`prediction_j = sum_p history[p]_j * w[p,j] + bias_j`，每个通道只使用自身历史。
- Dense temporal：`prediction_j = sum_p sum_i history[p]_i * W[p,i,j] + bias_j`，允许同一个 head 内的通道混合；不混合不同 heads 或 layers。

CrossKV 权重与当前 K 均不作为 V 预测器输入。本实验没有拟合、改变或调用 CrossKV。

## 相同控制

- 相同源 trace、calibration/evaluation record split、历史长度、warm-up 与每层/head 参数布局。
- 相同 residual bits、anchor bits、group size、scale dtype、reconstruction dtype。
- 相同 centered least-squares 目标，ridge 均向最近一帧 persistence 收缩。两类模型使用相同的 head-level 方程尺度，避免 diagonal 按通道尺度、dense 按 head 尺度造成额外差异。
- 相同逐输出通道 incoming L1 gain 上限；dense 在所有输入通道和 lag 上求和，per-channel 在 lag 上求和。对角矩阵是 dense 的严格特例。
- 默认两轮拟合。每一轮先用当前模型闭环重建 calibration，再拟合下一轮参数；测试时冻结权重，history 来自各模型自身的量化重建。

默认 ridge=1.0、max_gain=1.0，与现有 temporal 架构研究控制保持一致。这里使用匹配目标重新拟合的 per-channel 对照，不将 production 原参数与另一套目标下的 dense 混算。

## 数据与入口

```bash
python rebuttal/run.py --experiment temporal_matrix --stage pipeline \
  --set trace=/data/self_forcing_trace.npz \
  --output-dir rebuttal/outputs/temporal_channel_dense_sf
```

默认配置 `rebuttal/configs/temporal_matrix.json`。支持 `plan`、`fit`、`replay`、`evaluate`、`pipeline`、`summarize`。

2026-11-07 本机实验沿用真实 Self-Forcing `[30,128,24,12,128]` trace，SHA256 `e8f78e349ff297aceaa04b000de5a52e71b91339a7a09419f35df475148804b6`；每第 5 条记录留出测试，即 103 条拟合 / 25 条测试。原始文件未记录 prompt 身份，不能声明 prompt 级独立。

K 和 V 都采用 4-bit residual、4-bit anchor、group64 与 BF16 重建。V 是 pure temporal-only，与之前 CrossKV+V history 的绝对 V NMSE 不应直接比较。

## 输出和口径

- `temporal_predictors.pt`、`fit.json`：两种 temporal、两类张量的独立参数与拟合记录。
- `reconstruction.json`：总体、逐层、逐帧 SSE，以及 prediction / residual-quantization / reconstruction NMSE。
- `summary.csv`、`summary.md`：比较表。

主指标排除首帧共同 anchor，使用 `sum((reconstruction-target)^2)/sum(target^2)`，按总和汇总，不平均逐层 NMSE。两轮历史始终来自自身的重建，未使用干净历史作为输入。

参数大小按实际 FP32 weights+bias 统计，包含已分配的 warm-up 参数槽，不包含 quantized payload。KH2 的布局保存 order×lag；未使用的上三角槽为零，仍计入实际分配大小。

本实验只测缓存重建，没有生成视频，也没有测视频质量或实际生成速度。

## 验证

```bash
python -m pytest -q rebuttal/tests/test_temporal_matrix.py
```

检查 diagonal-special-case / production replay 一致性、跨通道 temporal 信号、相同 ridge/gain/warm-up 控制、未截断目标下的模型包含关系、因果历史和完全不调用 CrossKV。
