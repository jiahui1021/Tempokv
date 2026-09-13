# Jensen 自定义 Triton kernel

以下首先记录初版实现；文末记录用户报告 60%+ 开销后的修改。

## 实现范围

新增 `src/tempokv/runtime/jensen_triton_kernel.py`，通过
`TEMPOKV_JENSEN_BACKEND=triton` 接入 Self-Forcing 已选择的 sink/recent window。
默认 backend 保持 `augmented`。此实现是独立的 Triton forward kernel，未修改 FA2 扩展。

- Q/K/V/output 保持原始 head dimension。
- 每个 query tile 在 KV 循环前计算分组 norm 和 rounded coefficient。
- 直接按输入 strides 读取 BSHG squared steps，无 group-major 转换。
- 每个 score tile 加入 rank-G correction，使用 online softmax 累积输出。
- 不分配 Q×K bias、widened Q/K/V 或独立 query norm buffer。
- 单次 kernel launch 写出连续 BSHD output，不需要输出转置复制。

公式保持现有生产舍入：

```text
c_g = half(-sum(float(q_g)^2) / (24 sqrt(D)))
t_ig = half(half(Delta_ig)^2)                  # 由既有 cache read 提供
score_i = dot(q, k_i) / sqrt(D) + sum(float(c_g) * float(t_ig)) / sqrt(D)
```

softmax 和累积使用 FP32，PV 的概率转为输入 half dtype 后执行 dot。
与 FA2/Flex 的归约顺序不同，输出不承诺逐位一致。没有更改 group_size、cache
quantization、窗口、INT1 multiplier、当前块的 zero steps 或缓存均值开关。

## 支持和限制

NVIDIA Ampere+，FP16/BF16，D=64/128，g=32/64/128 且整除 D；equal heads、
无 mask/dropout 的缓存推理。输入支持正 stride 切片，输出连续。
明确拒绝 CPU、不支持的形状及需要梯度的调用；无静默 fallback。
采用 64×64 tile、4 warps、2 stages 的保守配置，尚未进行 GPU 调优。
NQ/NK 不按具体值特化，但 dtype、维度、group 和 stride 对齐类别仍可能生成不同编译版本。

此修改处理 attention 本身；cache step extraction、decode、RoPE 的耗时仍存在。
独立 Triton kernel 是否胜过 widened FA2 必须测量，不能由维度缩小直接推断。

## 验证

本机 Torch 2.13.0，无 CUDA/Triton。最终相关测试 54 passed、23 skipped；
CUDA 跳过项包括新 kernel 21 项，以及已有 Flex 2 项。Python 语法检查及测速脚本
`--help` 通过。**语法检查不代表 Triton lowering、GPU 数值正确性或加速验证。**

新 GPU tests 覆盖：D=64/128、G=1/2/4、BF16/FP16、非连续输入、batch/head、
Q/K tile 尾部、缓存长度变化、原地更新 steps、全零 steps、大 logits 稳定性。
CPU tests 执行真实生产路径片段，验证两种 predictor direction、gapped sink/recent
window 以及选择 Triton 时实际调用对应入口。

## A100 复现

```bash
python -m pytest -q scripts/tests/test_jensen_triton.py
PYTHONPATH=src python scripts/tools/benchmark_jensen_fused.py \
  --backends flex triton --query-tokens 4680 --key-tokens 280800 \
  --heads 12 --group-size 64 --warmup 3 --repeats 10
```

尺寸沿用历史微测默认值，实际判断需替换为真实慢例。报告 FA2 plain、augmented、
Flex 和 Triton 的 correctness error、首调用时间、CUDA event 中位数及相对 augmented
速度比。小尺寸 correctness preflight 会先编译另一形状，因此 shape_first_call_s
不等同于全新进程的全部冷启动时间。基准不包含 decode、RoPE、step extraction。

当前没有可用 A100 连接，未执行 GPU 编译/数值测试/latency 对照，也没有 end-to-end
加速结果。不使用历史 benchmark 注释或论文实验结果作为本实现的测量值。

参考：本地论文 `arXiv-2605.26266v1/main.tex` 的 Implementation Note 与 Taylor-corrected
attention 伪代码；[Triton 官方 fused attention 文档](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
用于核对 online-softmax recurrence。

## 60%+ 开销反馈后的修改

用户确认无法提供远程 GPU 连接，由用户在 A100 执行修改后的程序。
60%+ 是用户报告值，本地没有该次运行的命令、日志、输入形状和计时范围；不能判定
具体占比。初版只有固定 tile，且将整个 attention 替换为未经性能验证的 Triton 实现，
没有同 kernel 的无修正对照，无法把 backend 代价与修正代价分开。

### 性能候选

- Triton 默认 `auto`，在 64×64/4 warps/2 stages、128×64/4/2、128×32/4/3 三组配置中
  测量选择；Q/K 按 2 的幂次 bucket 复用选择，不刻意按每个增长长度重新调优。
- 新候选启用编译器 FP fusion；Jensen coefficient 仍先 round 到输入 half dtype。
- `TEMPOKV_JENSEN_TRITON_CONFIG=legacy` 保留上一版固定 tile 和禁用 FP fusion 的配置。
- 调优写入临时 output，不修改 Q/K/V/steps；首次新 bucket 耗时包括编译和调优。
- 不承诺自动选择优于 FA2；三个候选全部来自同一自定义 kernel，仍可能整体落后。

### 测量修复

- `src/tempokv/profiling.py` 原先只包裹普通 `causal_model.attention`；现在还包裹
  `jensen_flex_attention` 与 `jensen_triton_attention`，统一计入 `attention_kernel`，
  用子节点记录实际执行的 backend，避免重复计入同名汇总。
- `read_step_sizes` 单列为 `jensen_steps`；原来这部分只出现在 self_attention self time。
- `generate.log` 打印 Jensen backend 和 Triton config；仅设置 JENSEN=1 仍使用默认 augmented。
- benchmark 增加 plain Flex、plain Triton。Triton plain 使用编译期关闭 correction，
  连范数和 steps 读取都去掉，区别于输入全零 steps 后仍执行 correction。
- `--tune-triton` 对每个固定配置分别测 plain/fused，保留 legacy 对照。除 GPU events 外
  还报告同步 wall time；数值检查失败会退出，不能把错误候选推荐给用户。

`comparisons` 中：

```text
backend_extra_ms = plain_backend_ms - plain_fa2_ms
correction_extra_ms = corrected_backend_ms - plain_backend_ms
total_vs_fa2_pct = 100 * (corrected_backend_ms / plain_fa2_ms - 1)
```

两个毫秒差值可相加；两个百分比的分母不同，不能直接相加。
自动配置的 plain/fused 可能各自选中不同 tile；严格固定 tile 对照看 `triton_64x64`
等配置条目。微测不包含 cache decode、step extraction 和 RoPE，完整生成通过 PROFILE 比较。

```bash
PYTHONPATH=src python scripts/tools/benchmark_jensen_fused.py \
  --backends triton --tune-triton --query-tokens 4680 --key-tokens 280800 \
  --heads 12 --group-size 64 > jensen-benchmark.json
```

上述形状是历史微测默认值，不推定为用户真实慢例；实际使用时与运行窗口对应。
输出包含 `best_measured_fixed_config`，只推荐本次形状中最快的固定 Triton 配置。

本轮本地检查：57 passed、33 CUDA skipped；Python 语法和 benchmark help 通过。
新增测试覆盖 profiler 实际别名、缺失别名报告、无重复计时、backend/correction 占比计算，
以及待 GPU 运行的每个 tile、自动调优和编译期关闭 correction 对照。尚无新 latency 结果。
