# 实验二：anchor 是否截断 span 内误差传播

该实验与延迟实验分开运行、分开出图。扫描配置为：

- `L_span = 2 / 4 / 8 / 12 / 18` latent frames；
- residual INT2、INT4；
- predictor order、group size、stride 固定；
- 每个 `L_span` 把 calibration trace 重新切成独立 span，再单独拟合；
- `L_span=2` 只拟合实际可访问的一阶历史，未访问的高阶参数补零，运行时参数形状和 predictor order 仍保持不变；
- INT2 主配置为 anchor INT4 + residual INT2；新增的 anchor 对照是 anchor INT2 + residual INT2。

Self-Forcing 的压缩调度已支持 span 跨越 3-frame model block 边界。未设置
`TEMPOKV_SPAN_FRAMES` 时仍使用原来的 24 latent-frame 周期。

## 1. 生成完整执行计划

anchor INT2 不属于 legacy 算术；基础配置若是 `configs/legacy.json`，必须用
`--anchor-control-config` 提供一份 native 配置：

```bash
python scripts/paper_experiments/span/run_study.py \
  --trace data/self-forcing/trace/clean_trace.npz \
  --base-config configs/legacy.json \
  --anchor-control-config configs/native.json \
  --checkpoint /models/self_forcing_dmd.pt \
  --prompts assets/generation_prompts.txt \
  --output results/paper/span-v1
```

这一步只写 `manifest.json` 和每个 variant 的配置，不使用 GPU。计划包含：

- 每个 bit × span 的独立 fit；
- 每个配置的 K/V NMSE-by-position；
- 717 pixel-frame 主扫描和 BF16 reference；
- 完成 717 帧扫描后，按每个 bit width 的 PSNR 自动选择实测最优和最差
  `L_span` 做长程验证；PSNR 同分时依次用较低 LPIPS、较短 span 判定；
- 每个视频相对同 prompt / seed BF16 的 PSNR、LPIPS；
- anchor INT2 / residual INT2 对照。

正式执行增加 `--execute`，并设置 `WAN_MODEL_DIR`。已完成的 step 按 expected artifact
自动跳过，可安全续跑。选择结果写入 `long_selection.json`，长程 run 不再预设为
`L_span=2/18`。

## 2. 汇总与出图

```bash
python scripts/paper_experiments/span/analyze.py \
  --study-root results/paper/span-v1
```

正文产物：

- `tables/compression_quality.csv`
- `figures/compression_quality_int2.{png,pdf,svg}`
- `figures/compression_quality_int4.{png,pdf,svg}`

附录 D.1 产物：

- `tables/nmse_by_position.csv`
- `figures/nmse_by_position_int2.{png,pdf,svg}`
- `figures/nmse_by_position_int4.{png,pdf,svg}`
- `tables/long_horizon.csv`
- `figures/long_horizon.{png,pdf,svg}`

NMSE 的定义是同一位置所有样本的 `sum(error²) / sum(reference²)`，没有对逐样本
NMSE 取平均，也不会在代码中强制“误差必须单调上升”。是否翘尾由真实数据决定。

## 3. 无 GPU 验证

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/tempokv-mpl \
python scripts/paper_experiments/span/analyze.py \
  --study-root /tmp/tempokv-span-synthetic --synthetic --seed 0
```

synthetic 文件均标记为流程验证数据，不能作为论文结果。

## 帧数说明

Self-Forcing 的精确输出满足 `(pixel_frames + 3) % 12 == 0`。1400 不是可执行长度，
长程验证会明确记录：请求值 1400，实际完整工作负载 1401。代码、CSV 和图标题都使用
实际计算长度 1401，不把补算或裁剪后的结果冒充 1400。
