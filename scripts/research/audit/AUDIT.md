# 实施前审计（M0）

HEAD：`25558596e49156a5fabef79150c2cf51240969bb`。独立本地分支：`research/kv-exploration`。原有未提交 A/B 组文件保留，实施前 SHA256 清单见 `baseline.json`。

| 入口 | 实际语义 | 新研究约束 |
|---|---|---|
| causal_model.py / qkv_fn | Q/K 归一化后为 `[B,S,Heads,D]`；V 未旋转 | S1 读取 clean Pre-RoPE K 和 V；网格从 `[B,3]` 读取 |
| native BF16 分支 | 连续张量保存 Post-RoPE K，窗口滚动；sink 可产生不连续绝对位置 | 使用显式映射；reference 直接调用原 attention，不重旋转 |
| TempoKV 分支 | ChunkedKVCache 保存 Pre-RoPE K，读后施加绝对 RoPE | 观察 hook 在真实解码/窗口后；已有 pruning/Jensen 不可混作无偏 BF16 对照 |
| attention.py | FA2/FA3/SDPA，原生入口无额外 causal mask，输入 `[B,L,H,D]` | 同 dtype/scale/head/窗口/kernel；高精度结果另标，不替代生产 reference |
| pipeline denoise loop | 当前块被每一步重写；每块后 note_clean_refresh 明确标记 | 不依据 timestep=0 判断 clean；step/index/chunk 分开记录 |
| final latent | 最后 denoise 的 denoised_pred 写入输出；clean forward 的预测被忽略 | E1-clean 从最终 latent 或最后 denoise 输出计算，不能把 refresh 返回值当成结果 |
| trace/pruning | trace 每帧默认8 token；pruning 原来只在 TempoKV 分支 | 不将稀疏 trace 做二维 FFT；统一研究选择器用于 BF16 与量化对照 |
| evaluate_quality.py | 已有视频配对质量/一致性评估 | 不重写 LPIPS/FID；没有真实视频就标记 not_run |

现有 A/B 组可复用的部分仅包括已验证的 FFT、RoPE、窗口映射、相对误差、分组 Attention 基础。缺口：统一 research.v1 schema、当前/历史保护合同、条件替换/合并、归一化非对称扰动、含开销位预算、逐事件 frozen 干预、每步 RNG 对齐的反事实重放、step budget、2×2交互、因果档案与回访评价。

本机 CUDA 不可用；GPU 生成与真实性能、画质、长期研究结论不作完成声明。`source=tempokv` 仅用于 E7 观察/明确的扩展；主机制分析以 BF16 为准。研究不会修改正式 codec 或拟合逻辑，不将 Predictor 作为新实验的前提。
