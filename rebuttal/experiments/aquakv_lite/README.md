# aquakv_lite：不是 AQUA-KV

这是早期的简化消融，**不能**当作 AQUA-KV（arXiv 2501.19392）的结果。它复用 TempoKV 的分组标量量化器，只在 trace 上算 NMSE，没有 HIGGS，也没有接进 Self-Forcing 的 cache。正式移植在 `rebuttal/experiments/aquakv/`。

# 实验记录：简化跨层岭回归（对照用）

状态：**代码已实现，仅有合成数据的冒烟测试；尚未在真实模型 trace 上运行，没有任何真实结果。**

## 实现的公式（AQUA-KV, arXiv 2501.19392）

对 trace 中连续的 transformer 层 l，token 为视频 latent 位置（records × frames），特征为该层全部 H 个 head × D 通道拼接成的向量（与论文的逐层线性回归器一致，跨 head 混合）：

- 第 0 层：`K̂_0 = Q(K_0)`，`V̂_0 = Q(V_0)`（无预测器）
- 第 l≥1 层：
  - `K̃_l = [K̂_{l-1}, 1] A_l`，`K̂_l = K̃_l + Q(K_l − K̃_l)`
  - `Ṽ_l = [V̂_{l-1}, K̂_l, 1] B_l`，`V̂_l = Ṽ_l + Q(V_l − Ṽ_l)`
- `A_l`、`B_l` 用岭回归最小二乘（偏置不正则）**逐层顺序拟合**，输入是校准集上刚刚重建出的 `K̂_{l-1}, V̂_{l-1}, K̂_l`（与编码递推一致，不是干净 teacher forcing）。
- 评测阶段在 heldout trace 上冻结参数，不再拟合；`teacher_forcing_diagnostic` 仅作诊断对照。

## 受控对比

- 同一 calibration / heldout trace（`check_trace_split` 防止同源）。
- 同一残差量化器：默认从 TempoKV 的 `codec_config` 读取 `k_bits`、`group_size`、`residual_mode`、`scale_dtype`、`reconstruction_dtype`，调用 `tempokv.quantization.roundtrip`；手动覆盖与 TempoKV 不一致时直接报错。
- 三组：`direct`（不预测，直接量化）、`aquakv`、`tempokv`（给出 `tempokv.k_params/v_params` 时用现有 `cache_metrics.replay`）。
- 指标：K/V 的 prediction NMSE、残差量化误差、重建 NMSE，给出总体、仅预测层、逐层三种口径。
- 存储：payload（码字 + 每组 scale + 非对称零点）与 predictor 参数字节**分开报告**，`total_bits_per_element` 才把参数摊进去。TempoKV 报告参数文件实际字节。

## 与原始 AQUA-KV 的差异（不是“原版视频 AQUA”）

- 原文针对 LLM，用 HIGGS/向量量化等量化器；这里换成 TempoKV 同款分组标量量化，以保证对比只差在预测器。
- 原文可用低秩/更小预测器与更复杂的训练细节；这里是稠密岭回归 + 偏置，参数量为每层 `(F+1)F + (2F+1)F`（F=H·D），可能很大，已如实计入存储。
- 视频模型 KV 通常含 self/cross attention、按块生成；本实现只要求 trace 为 `[L,records,frames,H,D]` 且层号连续，不处理 anchor/span，也不做逐 head 分组预测。
- 只做 trace 重放 NMSE，不生成视频、不评 PSNR/SSIM/LPIPS。

## 运行

```bash
.venv/bin/python rebuttal/run.py --experiment aquakv_lite --stage fit \
  --set calibration_trace=<cal.npz> --set evaluation_trace=<heldout.npz> --output-dir rebuttal/outputs/aquakv_lite/run1
.venv/bin/python rebuttal/run.py --experiment aquakv_lite --stage evaluate \
  --set calibration_trace=<cal.npz> --set evaluation_trace=<heldout.npz> \
  --set tempokv.k_params=<k.pt> --set tempokv.v_params=<v.pt> --output-dir rebuttal/outputs/aquakv_lite/run1
```

默认配置 `rebuttal/configs/aquakv_lite.json`；输出 `fit.json`、`evaluate.json`、`aquakv_predictor.pt`、`tempokv_replay.json`。测试：`rebuttal/tests/test_aquakv_lite.py`（合成冒烟）。正式 AQUA-KV 是 `rebuttal/experiments/aquakv/`。
