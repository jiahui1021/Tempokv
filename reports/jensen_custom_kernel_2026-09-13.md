# Jensen 自定义 Triton kernel

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
