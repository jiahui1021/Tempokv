# Jensen latency：A100 诊断

日期：2026-09-13。范围：当前仓库 Self-Forcing Jensen 接入；用户确认 GPU 为 A100。本报告为诊断与优化设计，未修改生产代码，未执行 A100 latency benchmark。

## 结论

最强的瓶颈证据指向 **channel augmentation 跨越 FlashAttention-2 的 128 维 specialization 边界**。生产 d=128、g=64，Jensen 本来只有 G=2 项 correction，但当前 Q/K/V 一起扩到 136 维。多出来的 8 个 channels 并不意味着 GPU kernel 只多做 6.25% 的工作。

重写 kernel 有较强的收益依据，目标应为 **保留 Q/K/V 的 128 维 attention，在 softmax 前融合 rank-2 Jensen correction**。只重写 query norm、scale square 或 cache decoder，无法消除 attention specialization 的变化，预计只能处理次要开销。实际排序和加速倍数仍需当前 A100 数据验证。

## 证据与边界

| 证据 | 内容 | 可得结论 / 限制 |
|---|---|---|
| 生产代码 | padded width；K/V direct decode into widened buffer；Q augmentation；FA2 attention；output slice | 当前确实走 widened-head 路径，见 [causal_model.py](/Users/hui/Desktop/code/Tempokv/experiments/Self-Forcing/wan/modules/causal_model.py:154)、[调用路径](/Users/hui/Desktop/code/Tempokv/experiments/Self-Forcing/wan/modules/causal_model.py:584) |
| 仓库历史 benchmark 注释 | A100 128→136 约 +66% | [bench_jensen_alternatives.py](/Users/hui/Desktop/code/Tempokv/tests/bench_jensen_alternatives.py:5)；没有找到对应原始日志，不能当成本次实测 |
| 官方 FA2 v2.8.3 源码 | HEADDIM_SWITCH 对 129–192 选择 kHeadDim=192 | [官方 dispatch](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/static_switch.h)；仓库历史“160 tier”依赖安装版本，不是普适规则 |
| 仓库历史 Flex benchmark 注释 | plain Flex 56.7 ms，Jensen score_mod 76.4 ms，约 +35% | [bench_jensen_score_mod.py](/Users/hui/Desktop/code/Tempokv/tests/bench_jensen_score_mod.py:5)；此前使用 default compile 和 FP32 strided captures，不能代表调优后的最好结果 |
| 论文 | score_mod 融合 correction，不生成 Q×K bias；约 +5% end-to-end | [main.tex](/Users/hui/Desktop/code/arXiv-2605.26266v1/main.tex:1000)、[latency](/Users/hui/Desktop/code/arXiv-2605.26266v1/main.tex:591)；该 latency 比较为 MAGI-1 / L4 / g=32，不是本仓库 A100 对照 |
| 本次 CPU 检查 | 28 tests passed in 1.03s | 验证已有数学/cache invariants；不验证 CUDA BF16 attention 输出和 latency |

Self-Forcing 的安装文档未固定 flash-attn release；pyproject 中 2.7.4.post1 是 LongCat dependency，不能据此推断当前 A100 安装版本。

A100 不适用 FA3→FA2 的 Hopper 损失，也不适用面向 Hopper/Blackwell 的 FlexAttention FA4 backend。

## 当前开销的优先级

### 1. Attention head specialization：最高优先级

当前流程为：原始 Q/K/V 128 → Q 增加两个 norm channels，K 增加两个 Delta² channels，全部补到 136 → 普通 FA2 → 截回 128。

原始 softmax scale 保持 1/sqrt(128)，因此 correction 系数正确。问题在执行规格与布局，而非 correction 符号或归一化系数。

### 2. Scale 准备与全缓存遍历：第二优先级

[cache.py](/Users/hui/Desktop/code/Tempokv/src/tempokv/runtime/cache.py:516) 每次清零输出，逐 span 调 step_sizes，拼接 anchor/residual scales，写入 BF16 strided padding，再平方。可优化为缓存写入时准备 Delta²，或者单次读取 packed scales、完成 multiplier/cast/square/布局写入。

Delta² 只在 span 内容或量化 scale 变化时更新；query group norms 随每次 attention 的 Q 改变，不能跨 denoising step 缓存。新增缓存、替换缓存、淘汰窗口和 BF16 chunks 必须正确处理。

现有 decoder 的 width 仍为 packed D=128，并支持显式 output strides；没有证据表明 Jensen 必然使 decoder 回退到 Torch 或变成 136 维解码。见 [triton_ops.py](/Users/hui/Desktop/code/Tempokv/src/tempokv/triton_ops.py:810)。

### 3. Query augmentation 与 output copy：第三优先级

[query_channels](/Users/hui/Desktop/code/Tempokv/src/tempokv/runtime/jensen.py:46) 由多个 Torch ops 组成，随后分配 widened Q 并写入 slices。输出从 136 截成 128 后 head stride 仍为 136，合并 H×D 通常需要复制。保持 128 维的融合 attention 可以同时去掉这些 widened buffers 和输出重排。

通用 varlen wrapper 每次构造长度信息、RoPE 全缓存处理属于 baseline 也有的开销，不应全部归因于 Jensen。

## 推荐优化路线

| 优先级 | 方案 | 收益依据 | 代价 / 风险 |
|---|---|---|---|
| 1 | 128 维 FlexAttention score_mod，group-major contiguous Delta²/norms，max-autotune | 无需 widening，最小成本验证论文融合路线 | 默认 Flex 自身可能比 FA2 慢；capture gathers、tile/occupancy 和编译次数需实测 |
| 2 | 若调优 Flex 仍明显慢于 FA2：定制 FA2/A100 forward，把 rank-2 correction 放进 attention tile | 保留成熟 128 维 QK/PV 路径，同时消除 widened head | 修改 attention 内核与扩展，维护成本较高；普通 Triton attention 从零重写未必胜过 FA2 |
| 3 | 在 cache 写入时缓存 Delta²，消除每次 span 拼接、重复清零、cast/square | 去掉重复准备与多个 launch，辅助上面两条方案 | 需要准确的缓存更新与淘汰逻辑；单独做不能消除 head specialization 损失 |

官方 PyTorch 建议对复杂 score_mod 使用 max-autotune，因为修改函数会影响 shared-memory 使用和 kernel 配置。见 [官方调优说明](https://pytorch.org/blog/flexattention-for-inference/)。本仓库可用的 layout/autotune 微测入口是 `tests/bench_jensen_score_mod.py --max-autotune`。

### 定制融合 kernel 的核心

保持 Q/K/V/output 的 D=128。对每个 query tile，计算或加载两组 query energies；对每个 key tile，加载两组 Delta²。对 score tile 执行：

```text
score = Q @ K.T / sqrt(128)
score -= (q_energy_0[:, None] * delta_sq_0[None, :]
        + q_energy_1[:, None] * delta_sq_1[None, :]) / (24 * 128)
online_softmax(score) @ V
```

query energies 在 tile 外层复用；Delta² 沿 key 方向连续加载；不要每个 scalar score 重新对 128 channels 求范数。保持原有 attention mask、softmax scale、BF16/current-chunk correction=0 和量化 multiplier。attention 内部无需 materialize 完整 bias。

默认微测 Q=4680、K=280800、H=12 时，单独 FP32 Q×K bias 约 58.75 GiB（由维度直接计算）。当前实现没有生成该矩阵；独立 dense-bias kernel 会倒退，不是推荐方向。

### 收益上限不能等同于实测

若当前 A100 上能复现 attention +66%，并且融合实现完全恢复到原始 FA2 时间，则该 attention 部分相对当前最多减少约 39.8%（1−1/1.66）。真实融合仍有 correction 开销，端到端还含 cache、RoPE、模型投影、VAE 等工作，因此整体节省必然小于这个理想 attention 上限。

这只是基于历史注释的条件推算，不能作为实际 speedup 承诺；论文 L4 的 +5% 也不能用于设定 A100 的目标值。

## 必须保留的方法与数值约束

- 不通过改变 g、跳过层/步、或 cache mean approximation 来宣称等价优化。这些改变 correction 或实验设定。
- INT1 symmetric 使用 Delta=2*scale，INT2/INT4 常规使用 Delta=scale，BF16 keys/current chunk 使用 0。
- 当前生产将 Delta 先转 BF16 再平方；runtime augment reference 则 FP32 平方后转 BF16。融合版本需决定并验证兼容舍入顺序，不能把现有 CPU tests 误当成生产 BF16 等价性证明。
- 默认正交 rotation 后分组 Q energy 必须按实际 quantizer 的分组坐标计算；不能仅凭总范数不变使用未旋转的 group norms。

## 当前可验证范围与未完成项

已完成：代码路径、论文公式/实现、官方 kernel dispatch、历史 benchmark 可信度、CPU math/cache tests。

未完成：当前 A100 的 FA2 release/build、128/136 的真实时间比、Jensen 准备工作占比、调优 Flex 的时间、任何修改后的 before/after latency。原因是本机无 CUDA，当前 SSH config 为空，未提供可用 A100 访问入口。

可信测量应使用原运行的 window、Q/K 长度、位宽、checkpoint、prompt、seed 和 GPU，仅配对 Jensen=0/1。现有 `BENCHMARK=1 PROFILE=1 SKIP_EXISTING=0` 正常生成入口能比较 attention_kernel 与 self_attention 自身开销；`tests/bench_jensen_attention.py` 可隔离 128/136 和明确 fa_version=2。

旧 latency adapter 未传 Jensen 开关，并将 compilation 时间写为 0，不能直接用作本次结果。Flex 的 growing KV lengths 还需检查编译缓存/length buckets，并严格掩掉 bucket padding；warmup/compile 时间与 steady-state 时间分开报告。

## 同日实现补充（用户授权后）

已添加实验性 `TEMPOKV_JENSEN_BACKEND=flex`，默认保持 `augmented`。新路径使用 narrow K/V，单独准备 BF16 squared steps，并在编译后的 FlexAttention score_mod 内计算所有分组修正。sink/recent window 对 K、V、steps 使用相同切片。query coefficient 和 steps 保持旧路径的中间舍入顺序；因 attention 内部运算不同，输出不保证逐位一致。

新增 `scripts/tests/test_jensen_flex.py` 覆盖 G=1/2/4 的修正项、BF16/FP16 舍入、未量化 token=0、默认/非法开关、两个 predictor direction 的真实 cache-read 路由、gapped sink/recent window，以及 CPU full-graph capture。相关检查 41 passed；2 项 CUDA kernel/reference tests 因本机无 CUDA skipped。

新增 `scripts/tools/benchmark_jensen_fused.py`：固定 FA2，对照原始128维、augmented136维和真实 fused 路径；先做小尺寸 FP32 corrected-softmax 对照，再输出环境版本、首调用时间和稳定 CUDA event 时间。此脚本尚未在 A100 执行。本次没有定制 FA2 CUDA 内核，也没有验证任何实际 latency 收益。
