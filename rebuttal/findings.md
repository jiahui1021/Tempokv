# 代码核对记录

## 已确认

- 仓库根目录已有另一项 VBench 工作的 `task_plan.md`、`findings.md`、`progress.md`；本任务不改写它们。
- 三个生成接入为 Self-Forcing、Causal-Forcing、LongCat。
- `src/tempokv/config.py` 和 `src/tempokv/fitting.py` 已支持 K/V 的 `identity`、`taylor`、`taylor2`，以及 `cross_kv_identity` 等组合。
- `scripts/paper_experiments/predictor_ablation/run.py` 已有上一帧复用对照，但其固定行同时改变 K/V，需要按本次问题拆分轴。
- `scripts/evaluation/evaluate_quality.py` 已提供共享 SSIM/LPIPS 后端和逐帧读取，可复用指标口径。
- `scripts/paper_experiments/long_horizon/` 评估长 trace 重建，不等于固定预算下的真实长视频生成。
- Self-Forcing 的 1401 pixel frames 对应 351 latent frames，长度要求 `(frames + 3) % 12 == 0`。

## 时间、样本与资源

- Self-Forcing `inference.py` 写视频时使用 16 fps。LongCat `inference.py` 默认 15 fps，并通过首段加若干续段生成；不能把三个模型的 1401 帧都写成约 90 秒。
- LongCat 总帧数公式为 `num_frames + num_segments * (num_frames - num_cond_frames)`，`num_segments` 是续段次数。
- 校准资源已有 32 条 calibration、32 条 heldout。应按 prompt 原文核查所用参数的真实训练集，不能只相信文件名或直接使用前 30 条 moviegen prompts。
- Self-Forcing 的 30 layers、12 heads、head_dim=128、1560 tokens/frame，在 351 latent frames 下的 K/V BF16 数据约 93.995 GiB，未含权重与临时空间。完整历史的 BF16 参考不能默认在一张 80 GB GPU 上运行，也不能静默改为短窗口。

## 统计与接入边界

- `PackedTensor.storage_bytes()` 不含 predictor 参数；`logical_storage_bytes()` 另行排除开放 span 的预分配容量。
- `TempoKVRuntimeReport` 区分 current、peak、total encoded；`resident_total_bytes` 只加 packed cache 和 BF16，predictor 参数单独记。
- `StreamingTempoKVEncoder.history_bytes()` 提供开放 span 所保留的 predictor history；实际预算还需计入并去除共享张量重复引用。
- Self-Forcing 默认 `TEMPOKV_BF16_LAG_FRAMES=18`；flush 后的最终 cache 大小不能替代生成中的峰值。
- 当前 `ChunkedKVCache` 不允许部分覆盖 packed span，读也需要完整 span；直接按窗口切片不足以证明真实释放历史或预算约束。
- `evaluate_quality.py` 已流式累加 SSE，再由 pooled MSE 算 PSNR；新增 window 统计应保持这一口径。
- `history_ablation/evaluate_cache.py` 只输出汇总 prediction/reconstruction NMSE；`analyze_predictor.py` 可返回 prediction、residual、reconstruction，但没有独立量化残差失真及事件身份。
- `attention_objective.py` 已有可微闭环计算与残差 detach 约定，不能直接等同于 quantization-aware fitting。
- `trace_bridge.py` 采集 clean pre-RoPE K/V，并替换 VAE；需要同配置、同 prompt/seed 的单独视频生成或新的只读采样 observer，不能把 trace 模式输出当作 BF16 视频。
- TemporalParameters 目前存逐通道系数；CrossKVParameters 存 dense 方阵。新增 dense temporal / low-rank cross 需要实验参数格式与参考解码器，现有 Triton 不直接支持这些格式。

## 本机可复用的外部代码

- `../qvg/raft/` 已有 RAFT 源码，版本、权重路径与权重摘要须在实际实验时锁定。
- `../qvg/Quant-VideoGen/scripts/Self-Forcing/run_qvg.sh` 明确为 QVG/PRQ 生成；`quant_type=s2pp*` 会被拒绝，故不能将相邻目录中的 S2/S2++ 输出标作 QVG。
- `../kv-quant-4-videogen/scripts/paper_experiments/reconstruction/methods.py` 有真实 QVG codec 调用，当前重建接口只返回张量，不返回完整存储占用，需要另加返回 packed-state 统计的适配。
- 多个仓库拥有相同的 `pipeline`、`tempokv_adapter` 等导入名，生成调用应分进程，并显式指定 checkout 与 Python 环境。
