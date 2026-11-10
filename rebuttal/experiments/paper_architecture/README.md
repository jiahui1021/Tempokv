# 论文对齐的 Predictor Architecture Ablation

本实验保留旧结果，另行测量论文结构下的预测器消融。不生成视频，不改变正在运行的其他实验。

论文依据是 `sec/3_method.tex` 的 K/V 预测方程及 single-anchor 定义，以及 `sec/9_appendix.tex` 的 calibration、per-layer/head 参数与 Triton 实现说明。

## 固定对照

- Self-Forcing 全部 30 layers、12 heads、D=128；每个 layer/head 独立拟合，不跨层共享矩阵。
- K 是 KH2 per-channel temporal；V 是当前重建 K 的 dense projection 加 VH1 per-channel history。
- 主设置 K4/V4、anchor4、group64、BF16 scale/reconstruction；原 native 拟合器，两轮重建历史拟合。
- 8 个校准 prompt，每帧 8 个空间位置。使用既有 BF16 轨迹中的前 48 个真实缓存帧，分成两个非重叠 24-frame 校准样本。24-frame 长度沿用采集实现，论文没有明确指定该长度。
- 评估使用另 32 个 prompt 的完整 180-frame 轨迹。**只有首帧 anchor，不每 24 帧重置**。主 NMSE 排除首帧，按误差 SSE/信号 SSE 汇总。
- 校准文件将 BF16 来源激活转存为 FP32，K/V 数组约 1.05 GiB。文件存储 dtype 与模型激活来源精度分别记录。

## 独立消融

Temporal 表改变 K temporal，覆盖原始 per-channel、identity、scalar、matched channel refit、dense、block-diagonal、low-rank，以及权重 FP16/INT8/packed INT4。V 的结构始终保持 dense+VH1；按论文先 K 后 V 的依赖，为新重建 K 重新校准同结构 V。报告 K 与 V 误差、K temporal 参数和全预测器参数，并分别计时 decode K/V。

另有 V-history 子表：固定论文 K 与 dense K→V 矩阵，单独比较 VH1 的 per-channel、同条件 diagonal refit、scalar、dense、block2 FP16、rank1 FP16。该子表保留 CrossKV，不把 V 改成无 CrossKV 的纯 temporal 模型。history 的截距合入既有 V bias，避免冗余参数影响预算。

CrossKV 表固定论文 K 与 V-history 系数，覆盖原始 dense、同目标 dense refit、FP16/INT8/packed INT4 权重、逐矩阵 SVD、同校准集 rank-constrained refit、diagonal、block-diagonal、Shared Tucker。SVD/Tucker 明确标记 post-hoc；只有重拟合项标记 calibration optimization，不能以截断 SVD 代表低秩模型最佳表现。

## 存储预算

不使用空张量填充预算。记录真实参数 payload、量化 scales、bias、V-history 和 warm-up 槽位。改变部分的存储与完整预测器存储分别报告；运行时不持久缓存压缩权重的 FP32 展开。

典型匹配组：

| 对象 | 预算及候选 |
| --- | --- |
| K temporal | 原 per-channel FP32、block2 FP16、每 lag rank1 FP16，均 1.0546875 MiB（含 warm-up、bias） |
| Cross W | 约 5.713 MiB：dense INT8、SVD/refit rank16 FP32、block32、Tucker64 |
| Cross W | 11.25 MiB：dense FP16、SVD/refit rank32 FP32、Tucker90（约 11.212 MiB） |

`budget_groups.json` 列出相同上限且实际存储相差不超过 5% 的配对；未花完的字节保持真实记录。其他点用于完整误差—存储—耗时曲线，不把不同预算直接表述为同预算结果。

## 解码耗时

全部结构在同一 PyTorch reference 后端比较。参数先驻留 GPU；编码、拟合、文件读取不计入；权重反量化、临时投影和输出分配计入。完整 180 帧解码，warm-up 2 次、测量 5 次，对全部 layers 的耗时求和，报告 K、V、K+V。每层测量前后检查独占 GPU，发现其他 GPU 进程即拒绝该轮时间。

额外记录原生产 Triton baseline，作为单独的实现对照；不把不同后端的耗时差归因为预测器结构。

## 运行

```bash
python -m rebuttal.experiments.paper_architecture.runner --config config.json --output results/architecture/paper_aligned --stage plan
python -m rebuttal.experiments.paper_architecture.runner --config config.json --output results/architecture/paper_aligned --stage fit
CUDA_VISIBLE_DEVICES=3 python -m rebuttal.experiments.paper_architecture.runner --config config.json --output results/architecture/paper_aligned --stage measure --device cuda:0
```

配置指定 `codec_config`、`calibration_trace`、`evaluation_trace`。CPU fitting 与 GPU measurement 分开；GPU 不空闲时等待，不抢占正在生成视频的进程。部署副本在 `results/architecture/paper_aligned/code`，通过进程内 import path 加载；不改其他实验已冻结的核心源文件。

结果文件：`data.json`、`paper_baseline.json`、`models/`、`fit.json`、`measurements.json`、`summary.csv`、`budget_groups.json`、`production_backend.json`、`completion.json`。计划或拟合完成不等于三项测量全部完成。
