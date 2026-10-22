# B_SPECTRAL_KV：独立的谱摘要实验组

本目录实现第二份大纲，属于 **B 组**。原有 **A_FREQUENCY_ORACLE** 保留在 `scripts/spectral/`，配置、启动入口、分析工具和结果目录均独立。重号实验不可合并。

| 原有 A 组 | 新增 B 组 | 含义差异 |
|---|---|---|
| A_E1_SPECTRUM | B_E1_SPECTRUM | B 增加含/去 DC、r95、pre/post-RoPE 和时空联合频谱 |
| A_E2_SENSITIVITY | B_E2_CHUNK_AXIS | A 测历史块删除影响；B 测相邻 chunk 同一步的特征相似度及外推误差 |
| 无 | B_E2B_REFRESH | 最后去噪步与 clean 写入 pass 的差异；另有独立 KVSKIP 变体 |
| A_E3_SELECTION | B_E3_OFF_SHADOW | A 比较固定 token 预算选择；B 比较 DCT/平均池化摘要与删除 |
| 无 | B_E3_ON_ROLLOUT | 明确开启后改变生成结果的在线原型 |
| 无 | B_E4_DISTANCE | 频带跨帧相关性、稳定半径及距离幂律拟合 |
| 无 | B_E5_RESIDUAL | 原始 KV 与已有预测器残差的谱对比，独立离线输入 |

## 实现与验证范围

- 已实现数学基础、流式统计、独立缓存/保留策略、带 token 数量偏置的 Attention、分组 LSE 合并、影子评估和显式在线变体。
- Self-Forcing 与 Causal-Forcing 的原生 BF16 路径均有可选接入。阶段来自 pipeline 的 clean 标记和去噪步计数，绝不从 timestep=0 猜测。Causal-Forcing 的首 chunk 独立去噪日程单独计数。
- 未启用时保持原计算路径；A 组仍使用 `--mode spectral`，B 组使用 `--mode spectral_kv`，不可同时安装。
- 目前以 CPU 数学、合成输入和启动检查验证。真实 GPU 完整生成、长视频漂移、VBench、LPIPS、FID 与吞吐收益均为 **not measured**。
- 在线原型保留原生 BF16 backing，另维护谱缓存；报告包含两者的实际张量占用。摘要 token 减少不代表这个原型已经节省显存，也不构成生产加速结论。
- `group_lse` 是有界分块 Attention 和精确 LSE 合并路径，支持注入返回 LSE 的 backend。当前没有宣称已验证 fused FlashAttention 加速。
- LongCat、LongLive、真实量化组合和外部 Light/Focused Forcing 接入未实现。组合字符串 `+TEMPOKV...` 会报错，避免静默降级。`ExternalSelectorPolicy` 只提供明确的本地函数接口。

## 启动与分组

```bash
pip install -e '.[self-forcing,spectral]'
export CHECKPOINT=/models/self_forcing_dmd.pt
export MODEL_ROOT=/models/Wan2.1-T2V-1.3B
export PROMPTS=assets/moviegen128_first30.txt

EXPERIMENT=B_E1_SPECTRUM RUNTIME=self_forcing \
FRAMES=189 LOCAL_ATTN_SIZE=21 END_INDEX=2 \
bash scripts/paper_experiments/spectral/run.sh

# 另一模型的依赖使用 .[causal-forcing,spectral]，权重也需替换。
EXPERIMENT=B_E2_CHUNK_AXIS RUNTIME=causal_forcing \
FRAMES=189 LOCAL_ATTN_SIZE=21 END_INDEX=2 \
bash scripts/paper_experiments/spectral/run.sh

python scripts/paper_experiments/spectral/analyze.py \
  --input reports/spectral_kv --output reports/spectral_kv/summary
```

`DRY_RUN=1` 检查命令、配置及文件存在性，不加载权重。`FRAMES` 为最终视频帧数，189 对应 48 latent frames；717 对应 180 latent frames。全历史对照使用 `LOCAL_ATTN_SIZE=-1`，滚动窗口对照使用 21。两种设置应写入不同 `SPECTRAL_KV_OUTPUT_DIR`，防止跨窗口混比。沿用现有 `seed + prompt_index*1000003` 的配对规则。

所有结果进入 `reports/spectral_kv/<profile>/<runtime>/<window-and-sink>/rank-N/inference-<唯一编号>/<B_实验编号>/`，每次 inference 使用新目录。每个样本通过 `sample_id` 标注；所有记录含 `suite_id,experiment_id,prompt_id,rank`。每个实验输出 `records.jsonl`，统计量另存无 pickle 的 NPZ；顶层 `metadata.json` 保存配置、方法、网格、时间、显存与是否改变轨迹。原有 A 组分析工具排除 B 组数据，B 组工具也排除 A 组。

随附每个实验独立配置、`READONLY_PILOT.yaml`（多种只读统计一次采集）以及在线 DCT/AVG/DROP/PYRAMID/FULL 变体配置。`SPECTRAL_KV_CONFIG` 可显式指定配置；`SPECTRAL_KV_OUTPUT_DIR` 指定 B 组结果根目录；`OUTPUT_DIR` 指定生成输出。配置使用 JSON-compatible YAML，无 PyYAML 也可读。

## 各实验口径

**B_E1**：来自完整运行时 H/W 网格，不硬编码 30×52。K 的 pre/post-RoPE 分别记录，V 标为 raw。空间频率半径除以 sqrt(0.5) 归一化，r95 默认指去 DC 后的 95% 能量半径，同时给出含 DC 版本和 DC 比例。无非 DC 能量时不补造 r95。时空统计使用同一空间位置，最多保留 `temporal_window` 帧；跨缺口重置。联合 DC 仅指时间和空间同时为零的频点，不删除整个时间 DC 平面。

**B_E2**：在选定 transformer block 输出上，对齐同一去噪步、相邻 chunk，按空间/时间频带比较 cosine、reuse 误差和线性外推误差。只保留两个先前 chunk，并按 `feature_channels` 固定选取每头前若干通道。首块形状不同、chunk 缺口或历史不足时标记未测量。`chunks` 只过滤影子评估，不阻止相邻 chunk 统计。

**B_E2b**：最后去噪步明确由真实日程长度判定。临时保存选定层/头当前块，clean pass 后计算逐频带误差并释放。`KVSKIP_raw` 和 `KVSKIP_lin` 是单独的在线变体，需独立目录和配对长 rollout；跳过后不存在 clean reference，不会生成伪差异谱。

**B_E3-off**：完整窗口 K/V 保持在 softmax 分母中；只采样 Query。将 Post-RoPE K 逆变换到 pre-RoPE，再做 DCT 或平均池化，摘要位置为原网格的分数 cell center，时间位置为原绝对帧号；重新施加运行时 RoPE。BF16 逆变换存在舍入差异，不能解释成纯压缩误差。DCT 与 AVG 在相同目标网格、同一窗口和 Query 上比较；记录实际 token 数。R 是请求的降采样倍数，非整除网格下以实际数量为准。`B1` 加 `log(count)`，`B0` 为消融。当前块、最近保护历史和在窗口内的 sink 保持完整。Top-k 是采样 Query 的 frame attention mass 诊断，不能声称数学最优上界。

**B_E3-on**：配置必须明确 `online=true`。`SPECSUM_*`、`DROP_win*`、`PYR_*` 会改变 Attention 输出并影响后续生成。完整策略直接复用原生 Attention；谱缓存中的 FULL 策略另有数值回归。`keep_originals=true` 保存原件供重新选回或“从原件压缩”诊断，会额外占内存；否则按当前表示迭代压缩，不能重新扩展已丢弃的信息。动态 top-k 在线模式要求保留原件。

**B_E4**：固定空间位置与采样通道，按频带计算不同帧距的 cosine。稳定半径取从低频开始连续达到阈值的频带前缀；缺失频带不能当作稳定。只有两个以上不同帧距的正半径才能拟合 `r*=a*distance^-b`。保留 ring 与标量行数均有上限，不预设 b 大于 1/2。

**B_E5**：只读独立 NPZ，支持真实提供的预测值或复用已有 TempoKV K/V 参数；不运行量化器，不修改拟合逻辑。参数的层、头、维度、网格 stride 与 pre-RoPE 坐标必须匹配，warm-up 明确跳过。

## 在线与 KVSKIP 示例

```bash
EXPERIMENT=B_E3_ON_ROLLOUT bash scripts/paper_experiments/spectral/run.sh
EXPERIMENT=B_E3_ON_AVG bash scripts/paper_experiments/spectral/run.sh
EXPERIMENT=B_E2B_SKIP_RAW bash scripts/paper_experiments/spectral/run.sh

# 仅在独立校准 prompt 上收集，测试 prompt 不参与拟合。
EXPERIMENT=B_E2B_CALIBRATION bash scripts/paper_experiments/spectral/run.sh
python scripts/paper_experiments/spectral/fit_calibration.py \
  --input reports/spectral_kv/b_e2b_calibration/self_forcing/window-21-sink-0 \
  --output reports/spectral_kv/calibration/kvskip_lin.npz
EXPERIMENT=B_E2B_SKIP_LIN bash scripts/paper_experiments/spectral/run.sh
```

`KVSKIP_lin` 逐层逐头学习 K/V 仿射映射，仅保存 FP64 汇总矩阵。样本或秩不足会跳过并报告；缺少任何运行层/头/role 的参数都会拒绝执行。`calibration_file` 相对配置文件所在目录解析。输入和输出文件都不会被覆盖。

## 离线 E5 与漂移检查

```bash
python scripts/paper_experiments/spectral/residual_spectrum.py \
  --input /data/clean_grid_with_predictions.npz --output reports/spectral_kv/e5_offline
# 或使用 --k-params / --v-params 复用实际拟合参数。

python scripts/paper_experiments/spectral/compare_rollouts.py \
  --reference /data/bf16.npz --candidate /data/spectral.npz \
  --output reports/spectral_kv/paired_drift --space latent
```

E5 输入：`k_pre,v` 和可选 `predicted_k,predicted_v` 均为 `[F,H,W,heads,D]`；必需标量 `layer,stride=H*W,coordinate_system` 和整数 `grid=[F,H,W]`。坐标值可为 `pre/pre_rope/pre_rope_normalized`。漂移输入：`values=[T,...]`、相同的标量 `prompt_id,seed`；批量先按 sample 拆分。latent 不计算 PSNR；仅显式 `--space pixel --data-range <范围>` 时计算像素 PSNR。LPIPS/FID/VBench 使用项目既有评测流程，本工具不会用替代分数冒充。

## CPU 检查

```bash
python scripts/paper_experiments/spectral/smoke.py --output /tmp/spectral_kv_smoke
python scripts/paper_experiments/spectral/smoke.py --output /tmp/spectral_kv_online_smoke \
  --online --method SPECSUM_dct_R4_B1
python scripts/paper_experiments/spectral/analyze.py \
  --input /tmp/spectral_kv_smoke --output /tmp/spectral_kv_summary
python -m pytest -q scripts/tests/test_spectral_kv_*.py
```

合成检查结果标记为 synthetic，不能用于证明频率有研究价值。分析按 B 实验编号输出独立 CSV、图和说明；缺失数据为 `not measured`。已有输出目录中的报告会保留，新分析使用唯一子目录。

## 本地验收记录

最终联合检查：353 passed、1 skipped（可选 SciPy 对照未安装）、1 deselected（修改前即失败的空 CONFIG 检查）。其中包含 A 组回归和 B 组数学、统计、缓存、两种模型原生 BF16 接入、sink 滑窗、校准、离线工具和启动检查。GPU 完整生成与真实实验结果仍未测量。

Causal-Forcing 的 21 帧窗口由本入口明确标记为仓库现有的 sliding-window ablation；其启动器只接受 sink=0。Self-Forcing 的 sink 测试使用显式绝对位置映射。默认输出路径及汇总按 runtime/window/sink/profile/是否改变轨迹划分，不跨设置平均。计时中的 cache pass 是 generator 的子区间，不能与总时长重复相加。
