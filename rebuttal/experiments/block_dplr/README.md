# CORAL-style block-DPLR：Self-Forcing CrossKV 重建实验

把 [CORAL-KV Algorithm 1 / §4.2](https://arxiv.org/html/2610.12514v1#S4.SS2) 的矩阵结构迁移到现有同层 `K_hat(t) → V(t)` CrossKV 中，测量真实 trace 的闭环量化重建误差。实验没有更换成 CORAL 的跨层输入，也没有更换其 Hadamard/EDEN 量化器，不代表完整 CORAL-KV 复现。

## 矩阵结构

`prediction = per_head_block(K_hat) + (flatten(K_hat) @ A) @ B`。

- 每层保留 H 个完整 `D×D` head 内矩阵。
- 一对全局 `A:[H*D,r]`、`B:[r,H*D]` 混合不同 heads 的信息。
- 每层单独拟合，不在层之间共享权重。
- 矩阵始终以 block 和 A/B 因子应用；重放时不展开成全局 dense 矩阵。

原先的 CrossKV dense 是每个 head 各一张 `128×128` 矩阵，它在全部 1536 通道上本来就是 block diagonal。因此，加入 CORAL 的全局低秩项会增加其参数大小。CORAL 所比较的 full dense 则允许全部 heads 相互混合；两种 dense 的范围不同。

在 Self-Forcing 30 层 / H=12 / D=128、FP32 参数下：原 per-head dense 为 22.5 MiB；block-DPLR 的 r8/r32/r128 分别为 25.3125/33.75/67.5 MiB。这里仅计算当前 K→V 矩阵，固定的 bias/history 等另列。

## 拟合和比较

- 将 trace 每第 5 条记录留作测试，其余只用于校准；原始文件没有 prompt 身份时，不声明 prompt 级独立。
- 只在 calibration 上重新拟合 K temporal 和原 per-head dense/V history/bias，避免沿用训练来源不明的外部参数。
- 固定上述 K predictor、V history/bias，以及 K4/V4/group64/anchor4/BF16 重建方式。
- 从 calibration baseline 的重建 V 计算同一份历史输入，减去固定 history/bias，形成矩阵拟合目标。
- 按论文做三轮交替 headwise ridge / reduced-rank regression，ridge=1e-3。充分统计量与 Cholesky/SVD 使用 FP64；存储因子用 FP32，与此前矩阵压缩比较一致。
- Reduced-rank 更新求解按输入协方差加权的回归目标，使用精确 SVD；不对最终 dense 权重直接截断。
- 对照为重新拟合的原 per-head dense，以及相同 ridge 的 block-only；测试时所有 history 由各方法自身前面的量化重建帧产生。
- 主指标为排除首帧 anchor 的 `sum((Vhat−V)^2)/sum(V^2)`。保存逐层 SSE，汇总采用总和之比。

默认测试三个全局 rank：8/32/128。输入维度是 H*D，因此它与之前每个 head 单独 SVD 的 rank 不同。

## 入口

```bash
python rebuttal/run.py --experiment block_dplr --stage pipeline \
  --set trace=/data/self_forcing_trace.npz \
  --output-dir rebuttal/outputs/block_dplr_sf
```

默认配置：`rebuttal/configs/block_dplr.json`。分阶段为 `fit → replay`，`evaluate` / `pipeline` 可一次完成；`plan` 不读大 trace；`summarize` 重新排版已有结果。代码不覆盖已有参数文件；恢复需 `--resume`，输入、代码和配置改变应使用新目录。

输出：`fit.json`、`block_dplr_predictors.pt`、独立 K/V baseline 参数、`reconstruction.json`、`summary.csv`、`summary.md`。这些结果只测缓存重建，未测生成视频质量或生成加速。

## 与旧结果的关系

旧矩阵压缩表记录的 trace 为 `[30,104,21,12,128]`，SHA256 为 `c064068033f0b9214fc8316f97da178863f8cf87ec081055b08eb4d45fe77d9a`，原路径当前缺失。

本机现有另一份真实 Self-Forcing trace 为 `[30,128,24,12,128]`，SHA256 为 `e8f78e349ff297aceaa04b000de5a52e71b91339a7a09419f35df475148804b6`。2026-11-07 的测试单独使用这份数据：103 条校准、25 条测试；输出明确记录 `same_trace_as_previous_report=false`。结果仅与本次重新测量的对照比较，不写入旧表。

## 数值检查

```bash
python -m pytest -q rebuttal/tests/test_block_dplr.py
```

覆盖跨 head 因子、因子化公式与参数大小、交替目标下降、独立目标公式、rank-zero 与生产重放逐值一致、完整小样本拟合/重放及文件变动拒绝。小样本测试不作为真实实验结果。
