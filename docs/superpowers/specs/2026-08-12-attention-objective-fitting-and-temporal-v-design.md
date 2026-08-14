# 设计：attention 目标拟合与 temporal-V 预测器

日期：2026-08-12
状态：已实现，待 GPU fitting / generation 验证

## 1. 背景

TempoKV 的 Self-Forcing legacy 模式经过多轮对齐后，质量仍与参考实现（Quant-VideoGen，下称 QVG）存在差距。逐轮排查的结论是：过去的对齐工作全部集中在 **codec 算术**（量化 scale floor、shared K `[3,3,128]`、IEEE GEMM、RoPE 精度、trace 协议、packing），而两处**结构性差异**从未触及：

1. **拟合目标不同。** TempoKV 只最小化 K/V 重建的最小二乘误差；QVG 的目标里有一半权重是 **attention 输出误差**，并用 Adam 对参数做梯度精调。
2. **V 预测器结构不同。** TempoKV 的 V 是 `Cross(K̂_t)·W[30,12,128,128] + bias + alpha·V̂_{t-1}`（约 5900 万参数）；QVG 落盘的 V 参数是 `alpha[128] + beta[128]`（256 个参数，全层全头共享）。对 QVG 仓库 `results/` 下全部 `.npz` 按形状扫描，**没有任何一个 128×128 的 Cross(K) 矩阵**。

只要拟合目标和预测器结构不同，参数就不可能一致，解码器对得再准也无法收敛到同一质量。

本设计同时修正这两项。

## 2. 目标与非目标

### 目标

- V 预测器增加与参考实现同构的 **temporal** 形式，并作为 legacy 模式的默认。
- 拟合目标函数引入 **attention 输出误差**，K 和 V 都用该目标做梯度精调。
- 采集端记录 attention 目标所需的 `q_post_rope` 与 `k_post_rope`。
- 拟合保持 GPU 上的向量化执行，速度与当前量级相当（当前全量拟合约 18s）。
- 每一步改动可单独验证，出问题时能定位到具体变量。

### 非目标（YAGNI）

- 不实现 QVG 的多候选路由选择（`_evaluate_v_route`）。
- 不删除现有 CrossKV 预测器，它与 temporal 并存，可随时 A/B 切回。
- 不做 per-head / per-layer 的 V 参数。
- 不改动 K 现有的闭环最小二乘初值逻辑（精调在其之上进行）。

## 3. 参考实现的事实依据

所有结论来自 `/Users/hui/Desktop/code/qvg/Quant-VideoGen` 的实际代码与落盘参数，不来自二手描述。

| 事实 | 位置 |
| --- | --- |
| V 预测器就是套在 V 上的 temporal 闭环 rollout，权重形状 `[P,P,D]` 或 `[P,P,H,D]` | `quant_videogen/analysis/attention_objective.py:187` `temporal_closed_loop_rollout` |
| 每帧重建后做 BF16 round-trip；残差 QDQ 在精调时 detach（直通梯度） | 同上，`round_to_bfloat16` / `detach_quantized_residual` |
| 目标函数权重 `(0.5, 0.5)`；两项分别是**当前相位自身张量**的相对 SSE 与 attention 输出的相对 SSE，归一化分母取自基线的全局参考量而非每条 record | `scripts/Self-Forcing/studies/fit_trace_single_pool_gpu.py:48` `BLOCK_OBJECTIVE_WEIGHTS`；第 2570 行的 `loss` 构造 |
| 候选接受条件：复合分数优于历史最优**且**被守护指标不劣于相位起点 | 同文件 `_block_score`（第 2347 行）与 `guardrail_passes`（第 2517 行） |
| attention 项跑真实 SDPA，用记录下来的 `q_post_rope` / `k_post_rope` | 同文件 `_metric_terms`，第 2294–2331 行 |
| 重建 K 的 post-RoPE 由 clean 的 (pre, post) 对反推旋转后施加，不重算 RoPE | 同文件 `_apply_recorded_rope`，第 2245 行 |
| K 与 V 的精调是**交替相位**，每相位只优化一组参数，含早停、保留最优、非回归检查 | 同文件 `refine_k_v_separately`，第 2424 行 |
| V 单独的精调入口，K 在其中冻结 | 同文件 `_refine_temporal_v_attention`，第 628 行 |
| KH3 落盘参数：K `history_weights (3,3,128)` + `history_bias (3,128)`；V `alpha (128,)` + `beta (128,)`，`shared_layers_heads=True` | `results/cross_kv_fast_8_single_gpu_kh3/runtime_params/*.npz` |
| `q_post_rope` 与 k 同形状，即同样的 8 token/帧采样位置 | `quant_videogen/analysis/compiled_trace_bank.py:461` |

## 4. 设计决策

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| CrossKV 去留 | 与 temporal **并存**，legacy 默认 temporal | temporal 直接复用现有 K 的 codec 与 Triton kernel，不需要新 kernel；CrossKV 保留可随时 A/B 切回 |
| 精调范围 | **K 和 V 都精调**，交替相位 | 与参考实现的 `refine_k_v_separately` 一致 |
| guardrail | 保留 | 相位结果不如相位起点就回退，并在报告中记账；同时提供"精调到底有没有用"的直接信号 |
| trace 采集 | `q_post_rope` 与 `k_post_rope` **都存** | 旋转由 (pre, post) 对反推，不依赖任何 RoPE 实现细节；代价是 trace 体积翻倍（约 1.7GB → 3.4GB） |
| 拟合规模控制 | 全量向量化 + 按层分块累加梯度 | 每个 Adam 步只有 24 次帧循环；分块大小不影响结果，拟合可复现 |
| V 共享粒度 | 全层全头共享 `[1,1,128]` | 与参考落盘参数一致；此前 K 从 shared 改成 layered/per-head 反而变差，是同一个教训 |

## 5. 架构

### 5.1 新模块 `src/tempokv/attention_objective.py`

单一职责：提供可微的闭环 rollout 与 attention 目标，不涉及参数存储、不涉及命令行。

| 函数 | 契约 |
| --- | --- |
| `temporal_rollout(target, weight, bias, *, bits, anchor_bits, group_size, detach_residual)` | 输入 `[..., F, H, D]` clean 张量与 `[P,P,D]` 参数，输出同形状重建。锚帧走对称 QDQ，其余帧走闭环预测 + 非对称残差 QDQ，每帧 BF16 round-trip。`detach_residual=True` 时残差不产生梯度 |
| `recover_rotation(pre, post)` | 由 clean 的 pre/post-RoPE 对最小二乘反推每位置的 `(cos, sin)`，返回归一化后的结果 |
| `apply_rotation(x, cos, sin)` | 把旋转施加到任意同形状张量上 |
| `objective_terms(...)` | 返回 6 个 FP64 标量：`k_sse, k_ref, v_sse, v_ref, attn_sse, attn_ref` |
| `nmse(sums)` | 由 6 个分量得到 `k_nmse` / `v_nmse` / `attention_nmse` 三个归一化指标 |
| `phase_score(nmse, baseline, kind, weights)` | 相位复合分数；最终方法为 `0.5/0.5`，消融可显式切换权重 |

### 5.2 改动模块

- `src/tempokv/config.py`：新增 `v_predictor_mode: "cross_kv" | "temporal"`，默认 `cross_kv`。legacy 模式要求 `temporal`。加入 codec signature。
- `src/tempokv/codec.py`：`v_predictor_mode="temporal"` 时，V 用 `predictor_kind="temporal"` 编解码，复用现有路径。副作用：V 解码不再需要重建 K 作为 source。
- `src/tempokv/fitting.py`：`fit_legacy_shared_temporal` 复用于 V（VH1 → `[1,1,128]`）；新增 `refine_attention()` 驱动交替相位、早停与 guardrail。
- `experiments/Self-Forcing/tempokv_adapter/trace_bridge.py`：在与 k/v 相同的采样位置额外记录 `q_post_rope`、`k_post_rope`。
- `experiments/Self-Forcing/tempokv_adapter/prepare_trace.py`：落盘两个新数组，协议串升版。
- `scripts/common/fit_params.py`：提供 steps/lr/patience/token cap，并显式控制 optimizer、objective weights 与 K/V phase；报告写入每相位的接受或回退、最佳步与实际步数。

## 6. 数据流

### Step 1：temporal-V 结构（不需要重采 trace）

temporal-V 只依赖 `v` 本身，现有 12-prompt trace 即可拟合。

```text
clean_trace.npz(k, v)
  → fit_legacy_shared_temporal(k) → k_temporal.npz  [3,3,128] + [3,128]
  → fit_legacy_shared_temporal(v) → v_temporal.npz  [1,1,128] + [1,128]
  → 生成 → PSNR / SSIM / LPIPS
```

**唯一变量：V 预测器结构。**

### Step 2：trace 扩展采集

采集端在每层记录 `q_post_rope` 与 `k_post_rope`，采样位置与 `k_pre_rope` / `v` 完全相同。

```text
keys / values / q_post_rope / k_post_rope   各 [30, 24, 1, 8, 24, 12, 128]
```

### Step 3：attention 目标精调

```text
初值：K、V 均由闭环最小二乘得到（现有代码）
基线：先用初值跑一遍，得到全局参考量 reference_sums = (k_sse, k_ref, v_sse, v_ref, attn_sse, attn_ref)
      以及基线分数 baseline = (k_nmse, v_nmse, attention_nmse)

交替相位循环：
    K 相位 —— Adam 只优化 K 参数，V 冻结
    V 相位 —— Adam 只优化 V 参数，K 冻结

每个梯度步：
    K̂ = temporal_rollout(k, K参数, detach_residual=True)
    V̂ = temporal_rollout(v, V参数, detach_residual=True)
    cos, sin = recover_rotation(k_pre, k_post)
    K̂_post   = apply_rotation(K̂, cos, sin)

    own_sse = ‖K̂−K‖²（K 相位）或 ‖V̂−V‖²（V 相位）
    attn_sse = ‖SDPA(q_post, K̂_post, V̂) − SDPA(q_post, k_post, V)‖²

    loss = 0.5 · own_sse  / reference_sums[own_ref]
         + 0.5 · attn_sse / reference_sums[attn_ref]

    反向、Adam 更新、gain 投影

    候选评分（与 loss 分开，用累计的全局量算）：
        score = 0.5 · own_nmse / baseline[own_nmse]
              + 0.5 · attention_nmse / baseline[attention_nmse]
        接受该候选，当且仅当 score 优于当前最优
        且相位内被守护的每个指标（own_nmse 与 attention_nmse）
        都不劣于相位起点

相位收尾：
    早停（patience 与最小相对改善）
    采用相位内被接受过的最优参数
    一次都没被接受则整相位回退到起点，并在报告中标注
```

**唯一变量：目标函数。**

## 7. 拟合规模与可复现性

- rollout 在层与 event 维度上全并行，每个 Adam 步只有 24 次帧循环。
- 显存通过**按层分块 + 梯度累加**控制；V 参数全层共享，梯度对层求和，因此分块与不分块在数学上等价。
- 不使用随机 minibatch：同一份 trace 两次拟合必须得到完全一致的参数。
- 实现同时支持 legacy/shared `[P,P,D]` 与 native/layer-head `[L,P,P,H,D]` temporal K；V 可为 temporal、CrossKV only 或 `CrossKV + V-history`。后两者保留 aligned reconstructed-K 条件，并在 K phase 中通过 V prediction 路径传播 attention 梯度。
- 同协议消融可选 Adam 或历史单方向 sign-gradient scan、reconstruction/attention 权重以及 K-only、V-only、K+V phase；训练、候选选择和 guardrail 共用同一组非零权重。

## 8. 参数文件与兼容性

- V 参数写入新文件 `v_temporal.npz`；`v_cross_kv.npz` 语义不同，不复用文件名。由 `v_predictor_mode` 决定读取哪个。
- `v_predictor_mode` 进入 codec signature，旧参数文件自动失效——这是期望行为，语义确实变了。
- 参数目录 slug 的 V 段区分结构：`vt1` 表示 temporal、`vc1` 表示 cross_kv，例如 `legacy-kh3-vt1-int4`。

## 9. 错误处理

全部显式失败，不静默降级：

- 要求 attention 精调但 trace 缺少 `q_post_rope` / `k_post_rope` → 报错并提示重新采集。
- 精调过程中出现非有限 loss → 报错。
- 相位回退 → `fit_report.json` 写明 `accepted` 或 `guardrail_fallback`，附 optimizer、objective、phase 和相位起止分数。
- `v_predictor_mode` 与参数文件实际格式不符 → 加载即报错。
- `v_predictor_mode="temporal"` 但参数不是全层全头共享形状 → 报错。

## 10. 测试

| 测试 | 断言 |
| --- | --- |
| 旋转复原 | 构造已知 `(cos, sin)` → `recover_rotation` 反推 → 与真值一致 |
| rollout 梯度语义 | `detach_residual=True` 时残差路径不产生梯度；参数梯度非零且有限 |
| 分块等价 | 按层分块累加梯度 == 不分块（数值一致） |
| 拟合确定性 | 同一 trace 两次拟合，参数 bit-identical |
| guardrail | 构造精调必然变差的场景，断言参数回退且报告标注 `guardrail_fallback` |
| temporal-V 编解码 | VH1 共享 `[1,1,128]` 的编码/解码往返；Triton 路由落在 temporal persistent kernel |
| 目标函数归一 | 重建等于 clean 时 loss 为 0；各分项与手算参考一致 |
| trace 格式 | 4 数组的 shape 与协议校验；缺 q 时报错信息明确 |

## 11. 验收标准

三步各自固定 prompt、seed、checkpoint，只跑 `0_0.mp4`，记录 PSNR / SSIM / LPIPS 与拟合耗时：

- **Step 1** 相对当前 `legacy-kh3-vh1-int4` 基线（PSNR 15.13 dB）给出明确的升降结论。
- **Step 2** trace 校验通过，体积与形状符合预期。
- **Step 3** 相对 Step 1 给出明确的升降结论；`fit_report.json` 中每个相位的接受/回退可读。
- 全流程拟合耗时保持在分钟级以内。

## 12. 风险

- **attention 目标是代理量。** 它在每帧 8 个采样 token 上做 SDPA，不是完整序列的注意力，与最终视频质量之间仍有 gap。缓解：每步都以端到端 PSNR 为最终判据。
- **temporal-V 表达能力远低于 CrossKV**（256 vs 5900 万参数）。若 Step 1 质量下降，需要在此处停下来讨论，而不是继续叠加 Step 3。
- **梯度精调引入超参**（步数、学习率、patience）。缓解：默认值取自参考实现，且 guardrail 保证不会比最小二乘初值更差。
