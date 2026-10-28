# Self-Forcing KV spectral pilot

对原生 BF16 路径运行 E1（clean KV 频谱）、E2（单块删除 Attention Oracle）和 E3（固定 token 预算联合删除）。默认关闭，不需要 TempoKV 拟合参数。所有统计来自真实输入；CPU smoke 的合成数据明确标注，不可作为研究结果。

## 运行

```bash
pip install -e '.[self-forcing,spectral]'
export CHECKPOINT=/models/self_forcing_dmd.pt
export WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B
export PROMPTS=assets/moviegen128_first30.txt

MODE=spectral \
SPECTRAL_CONFIG=scripts/spectral/configs/self_forcing.yaml \
SPECTRAL_OUTPUT_DIR=reports/spectral/self_forcing_pilot \
FRAMES=189 LOCAL_ATTN_SIZE=24 SINK_SIZE=0 \
END_INDEX=2 NUM_GPUS=1 SEED=0 \
OUTPUT_DIR=results/spectral/self_forcing_pilot \
bash scripts/Self-Forcing/generate.sh

python scripts/spectral/analyze_results.py \
  --input reports/spectral/self_forcing_pilot \
  --output reports/spectral/self_forcing_pilot/summary

python -m pytest -q scripts/tests/test_spectral_*.py
```

`FRAMES=189` 为 48 latent frames。`DRY_RUN=1` 和 Python 入口的 `--dry-run` 检查启动命令、配置和输入文件是否存在，但不读取模型权重；运行前仍需提供存在的 checkpoint 和 prompt 文件。`SPECTRAL_CONFIG` 和 `SPECTRAL_OUTPUT_DIR` 都是必需的，后者覆盖配置文件内的 `output_dir`。原有 `bf16` 模式依然无需 `CONFIG` 或 `PARAM_DIR`。`spectral` 的生成路径和缓存与 `bf16` 相同，正常输出视频，额外计算分析数据。

随附 `.yaml` 使用 JSON 语法（合法 YAML），仅依赖基础安装也能读取。普通 YAML 需安装 `PyYAML`；绘图需 `matplotlib`。配置拼写错误和非法形状会直接报错。

## 坐标、阶段与采样

- E1 使用 clean forward 的 Pre-RoPE K 和原始 V，从每次运行的 `grid_sizes=[B,3]` 中每个样本的 `[F,H,W]` 恢复网格。选定 head 仍保留完整 H/W，不使用空间 token 子采样做二维 FFT。
- 每个 clean frame 的选定 head 特征复制到 CPU，进行正交 FFT，处理后释放。只保留上一 clean frame、块级标量和少量固定空间位置的时间轨迹，最多 `max_clean_frames` 帧。阶段由 `note_clean_refresh()` 标记；timestep 不参与判断。`sampling.only_phase` 只过滤 Oracle，绝不阻止 clean 特征采集。
- 空间半径定义为 `sqrt(fftfreq(H)^2 + fftfreq(W)^2)`，单位 cycles/token。E1 扫 `cutoffs`；E2 块分数使用 `score_cutoff` 的高频能量占比。K/V 分别记录。几乎零能量样本返回 NaN 并记录标志。
- 时间 FFT 使用连续 clean 帧的同一批 row-major 空间位置。遇到帧缺口分段，不跨缺口变换；只有一个帧的段保存样本及位置，但不生成频谱。跨帧相似度为同一块全部通道与空间位置的 K/V cosine 均值；没有紧邻上一 clean frame 时为 NaN。
- E2 使用此次有效窗口的 Post-RoPE Q/K 和原始 V，不重施加 RoPE。逐样本、逐 head 采样 Query；K/V 始终保留整个有效窗口。reference、单块删除和联合删除都调用同一个运行时 `attention()`，保持原始 dtype、head、默认缩放和无额外 mask 的语义。
- Query 的 latent frame 是从 0 开始的绝对编号。每个选定 frame 采样至多 `query_tokens_per_head` 个位置，`denoise_calls=[0]` 表示该 chunk 的第一次 denoise forward。空 `latent_frames` / `denoise_calls` 表示不按该字段过滤。调用数仍受 `max_oracle_calls_per_layer` 限制。
- 原生无 sink 连续窗口的位置为 `current_start - (local_start_index - attention_start) + relative_index`；滑窗后仍对应绝对位置。库函数支持显式位置数组，运行时初版仅支持 `SINK_SIZE=0`。不明确的 gap/sink 映射直接拒绝。
- 候选只来自有对应历史 clean 特征、且完整存在于当前窗口的空间块。Current Chunk 和之前 `protect_recent_frames` 个历史帧始终保护。边缘小块保留真实 token 数。候选与 Query 使用本地确定性随机数，不改变生成的随机数状态。相同 prompt/sample/layer/head/frame/call 的候选保持一致，不受 rank 分配和输出目录名影响。

## E2 / E3 的含义

`oracle_error = ||O - O_without_block|| / (||O|| + eps)` 是单块删除影响，不是可相加的联合误差。Attention Mass 以全部 K 做分母，分块计算 FP32 QK，避免构造完整 Attention Matrix。归一化分别保存最大 logit 与中心化的 log-sum-exp，避免极大的公共 logit 偏置吞掉归一化余量。FP32 点积本身已经丢失的差异不在此计算中恢复。

每次调用在同一采样候选集上计算 Spearman、Top-k recall 和保留策略误差。相关性与 recall 使用各分数共同有限的子集，所有比较的有效候选保持一致；表中给出 `finite_candidates,sampled_candidates,excluded_missing_candidates`。Top-k 边界有并列分数时 recall 为 NaN；常量分数的 Spearman 为 NaN。recall 仅针对采样候选，不能称为全局 recall。

E3 在同一次 Q/K/V 上独立重算联合删除结果。预算是采样候选 token 的保留比例；未抽到的历史块也完整保留并参与归一化。输出同时报告候选 token 数、实际保留总 token 数和实际历史保留比例。不同策略在同一个预算上严格保留相同数量 token；排序最后一个块可只保留部分 token，所有策略使用相同边界规则，避免边缘块大小造成预算不等。

| 策略 | 保留优先级 |
|---|---|
| random | 固定 seed 随机排序块，默认重复 3 个 seed |
| uniform | 按历史帧/空间块顺序，从粗到细均匀分布优先级 |
| similarity | 跨帧 cosine 较低的块优先 |
| attention | Attention Mass 较高的块优先 |
| frequency | clean K/V 高频占比均值较高的块优先 |
| frequency_attention | `alpha*z(frequency)+(1-alpha)*z(attention)`，扫配置中的 alpha |
| oracle_greedy（可选） | 单块删除误差排序，仅为诊断基线，不是最优上界 |

缺失分数的块排在有限分数之后；某策略所有分数缺失时整行标记 `not measured`，不补造分数。单独频率、相似度、Attention 与组合分数的比较来自同一候选集，不预设频率有效，也不从相关性推出因果。

## 输出与合并

每次 inference 创建独立路径，不覆盖已有 prompt、sample、rank 或重复运行：

```text
SPECTRAL_OUTPUT_DIR/
  rank-N/inference-<counter>-<unique-id>/sample-N/
    metadata.json
    spectrum/spatial_energy.csv
    spectrum/temporal_energy.npz
    sensitivity/block_scores.csv
    sensitivity/correlation.csv
    compression/error_budget.csv
    compression/window-N.npz          # 仅显式开启 snapshot_calls 后
```

汇总脚本递归读取所有 rank/sample，默认只统计 `status=completed` 的运行，失败和未完成运行的已保存前缀不进入均值或图表。相同 prompt、generation seed、sample、batch 大小及实验身份的完成重试只保留一份，取 metadata 修改时间最新者；时间相同时按路径确定顺序。rank、inference 目录名和输出路径不代表独立采样。`analysis_selection.json` 保存保留路径、每个排除原因、排除计数和实验身份，`summary.md` 同时显示计数。只有失败运行时仍生成 `not measured` 报告。

一个汇总只接受同一实验配置和模型身份。配置中的 `output_dir` 可以不同；其余配置、已记录的 checkpoint/weight 标识、代码版本、数据来源、坐标、实际网格以及运行时 dtype/backend/缩放不同会直接报错，要求分开汇总，不跨配置计算加权均值。不同 generation seed 和 sample 作为独立观测保留。旧报告没有 `generation_seed` 时按 `inference_id` 保留独立观测，不把分析 seed 当成生成 seed；缺少 prompt 内容或权重内容校验值的旧报告只能使用其已有 prompt ID 和 checkpoint 路径，不能识别同一路径下被替换的内容。不要把不同 prompt 文件的相同编号放入同一输入目录，除非 metadata 带有可区分的 prompt 身份。

若目标目录已存在，新汇总写入其 `analysis-<unique-id>/` 子目录。生成 `frequency_by_layer.png`、`frequency_vs_importance.png`、`error_vs_retention.png`、四份汇总 CSV、`analysis_selection.json` 和 `summary.md`。没有数据时图和报告明确显示 `not measured`。

| 文件 | 列或数组 |
|---|---|
| `metadata.json` | git commit、生成 seed、分析 seed、模型路径、完整配置、实际网格/窗口、当前与局部位置、phase、dtype、backend、分析耗时、进程峰值 GPU allocation、缺失/截断计数、完成/失败状态 |
| `spatial_energy.csv` | `prompt_id,sample_id,rank,layer,head,frame,role,cutoff,ratio,total_energy,near_zero_energy,grid_h,grid_w,phase` |
| `temporal_energy.npz` | `heads,cutoffs`；每个 `layer_<L>_<k/v>_segment_<S>` 的 `frames,positions,samples[T,P,heads,D]`；T≥2 时还有 `frequency,energy[T,heads],low_ratio[cutoffs,heads],near_zero` |
| `block_scores.csv` | 公共索引列 + `history_frame,spatial_block_id,absolute_token_range,absolute_token_indices,block_tokens,freq_k_high_ratio,freq_v_high_ratio,temporal_similarity,attention_mass,oracle_error,frame_age,score,query_token_indices,candidate_scope,candidate_count,eligible_candidate_count,reference_norm` |
| `correlation.csv` | 公共索引列 + `score_name,combination_alpha,spearman,topk_recall_sampled,finite_candidates,sampled_candidates,excluded_missing_candidates,topk_fraction,candidate_scope` |
| `error_budget.csv` | 公共索引列 + `policy,retention_ratio,random_seed,combination_alpha,candidate_tokens,target_kept_candidate_tokens,untouched_tokens,kept_candidate_tokens,actual_retained_tokens,partial_block_tokens,relative_attention_output_error,nmse,candidate_scope,budget_granularity,status,total_history_tokens,actual_retained_history_tokens,actual_history_retention,protected_tokens,query_token_indices` |

公共索引列：`prompt_id,sample_id,rank,inference_id,layer,head,call_id,denoise_call,query_frame,phase,grid_h,grid_w,frame_seqlen,effective_k_tokens,current_start,local_start_index,attention_start`。

`absolute_token_range` 是 `[min,max+1)` 包围区间，二维块跨行时并不连续；`absolute_token_indices` 才是精确列表。`score` 为 K/V 高频占比均值。输出中 NaN 明确表示未测量或分数未定义。

## 独立离线重算

默认不存原始历史 KV。需要复查同一窗口时，显式设置 `compression.snapshot_calls: 1`，可保存至多配置指定的 1–4 个单样本/单 head 窗口，并受 `max_snapshot_tokens` 限制。快照保存采样 Query、该 head 完整窗口的 Post-RoPE K/V、候选、分数和保护位置；FP32 文件精确保留 BF16 值。不会保存全部层/帧的 BF16 KV。

```bash
python scripts/spectral/replay_compression.py \
  --input <sample-dir>/compression/window-0.npz \
  --output /tmp/spectral-replay.csv --device cpu
```

CPU 重算使用 PyTorch SDPA，输出记录其 backend，可能与 GPU FlashAttention 有浮点差异。具备完整 Self-Forcing 环境时可使用 `--device cuda --runtime-attention` 调用运行时实现。分数排序来自原始 E2，reference 和联合删除误差在所选 backend 上重新计算。

## 无模型 CPU 检查

```bash
python scripts/spectral/smoke.py --output /tmp/tempokv-spectral-smoke
python scripts/spectral/analyze_results.py \
  --input /tmp/tempokv-spectral-smoke --output /tmp/tempokv-spectral-summary
python -m pytest -q scripts/tests/test_spectral_*.py
```

smoke 使用固定 seed、3×5 空间网格、2 heads 的合成输入。它验证完整输出、绘图和重算流程，不代替真实模型实验。

限制：目前只接入 text-to-video few-step `CausalInferencePipeline`，不支持 initial context/I2V、sink/gap 运行时、量化路径、Pooling/Multi-Resolution 或 Sparse Attention 内核。额外分析会同步 GPU 并增加耗时；打印的是分析段时间及整个进程的峰值分配，不能解释成净分析显存增量。CPU 回归覆盖 hook 关闭/开启和滑窗前后的数值一致性；没有 CUDA 和 checkpoint 的环境不能完成 GPU 端到端验收。

## 实验组标注

本目录为 **A_FREQUENCY_ORACLE**：A_E1_SPECTRUM（频谱）、A_E2_SENSITIVITY（块删除重要性）、A_E3_SELECTION（固定预算选择）。新增大纲的 **B_SPECTRAL_KV** 位于 `scripts/paper_experiments/spectral/`，使用另一套配置、启动入口、结果根目录与分析工具。B_E2 是 chunk 轴统计，B_E3 是摘要评估，不能与本组同编号结果混用。
