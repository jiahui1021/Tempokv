# 六项补充实验的实现方案

日期：2026-11-06。模式：代码规划。所有实验设置为建议值，尚未执行；不包含实验结果或审稿意见回复。

## 1. 统一组织与比较约定

### 1.1 目录

```text
rebuttal/
  README.md
  implementation_plan.md
  code_map.md
  experiment_matrix.json
  run.py                         # 以下为拟新增代码，目前未实现
  requirements.txt
  configs/
  common/                        # manifest、视频指标、cache 指标、trace、统计
  adapters/                      # 三个模型、observer、研究 codec、真实 QVG
  experiments/
    previous_frame.py
    temporal_window.py
    motion/
    architecture/
    quant_calibration.py
    quant_aware_fit.py            # 可选扩展
    memory/
  tests/
  outputs/<run_id>/
```

现有入口复用，不把 `scripts/paper_experiments/` 整体复制过来。`code_map.md` 列出逐文件责任与必要接入改动。输出目录按 run_id 新建；续跑只复用身份匹配且完整的结果，不覆盖用户既有稿件、结果或项目状态。

### 1.2 共用样本、模型和开销

| 项目 | 建议首版 | 理由与限制 |
| --- | --- | --- |
| prompts | 已有 `assets/calibration/heldout_prompts.txt` 的 32 条；先与所用参数的真实 calibration/development prompts 比较原文与摘要 | 当前仓库已有 32/32 切分，避免直接取前 30 条造成拟合与评价重叠 |
| calibration | 每个模型使用自己的 trace 与参数；沿用已冻结的 production calibration protocol | 不跨模型复用 predictor 参数，不用长视频 test trace 重拟合 |
| seeds | 首版 seed 0，配对固定；多 seed 扩展与首版分开记录 | 可复用既有视频；单 seed 结果不宣称跨 seed 稳定性 |
| 生成质量位宽 | 正式实验优先使用论文当前主设置；若其尚未在可读文件中冻结，建议先列 INT4，另列 INT2 压力设置 | `configs/legacy.json` 当前为 K4/V4，不能擅自假定论文主结果全是 INT2 |
| trace 诊断位宽 | K/V INT2 与 INT4 | 离线成本较低，可以观察不同量化强度 |
| Temporal-window | Self-Forcing、Causal-Forcing 各 32 对视频，1401 pixel frames | 合计 128 个方法-视频文件；两组完整生成设置一致时可复用 |
| 架构与量化目标诊断 | 三模型各自独立 held-out trace；端到端首版集中 Self-Forcing | 区分通用压缩规律与高成本生成验证 |
| Motion fidelity | 尽量复用已有的完整配对视频集合，剔除拟合/调参 prompts，最低以 32 对为首版 | 使用有效样本数 N，不把缺失视频或缺失 trace 默认为零 |
| Fixed budget generation | Self-Forcing；BF16 / QVG / TempoKV | 先保证三方法在一个生成器上公平比较 |

LongCat 是建议的第三模型扩展：其默认 15 fps、分段生成，与两个主模型不同。若保留 1401 帧，时长约 93.4 秒，应得到六个完整 15 秒窗口和一个 3.4 秒尾窗；单列尾窗，完整视频不丢帧。若主表必须严格只有六行，则主表只纳入两个 16 fps 模型，LongCat 全部窗口放扩展表。不能更改播放帧率来掩盖时长差异。

LongCat 总帧数必须从实际生成公式计算：`num_frames + num_segments * (num_frames - num_cond_frames)`；其中 `num_segments` 是续段次数。参数组合及第一段/续段标识写入清单，不能把 shell 的 `NUM_FRAMES` 当成整段输出帧数。

### 1.3 统一身份与中间结果

每个 run 保存：model/checkpoint、生成器与 codec commit、config SHA-256、parameter SHA-256、prompt 原文摘要与源编号、seed/sample_id、尺寸、pixel/latent frames、fps/timebase、attention window、sink、span、BF16 lag、anchor/group/bit 设置、后端、运行状态与失败原因。新架构额外记录 rank、共享范围、目标、拟合轮数与参数格式版本。

共用输出以如下粒度记录；实际读写可用分片 CSV/JSONL，不能一次将所有长 trace 放进显存：

| 文件 | 一行含义 | 必须保留的核心字段 |
| --- | --- | --- |
| `manifest.json` | 实验与输入清单 | 上述身份、配对键、预期与实际样本数、状态 |
| `frames.csv` | 一对视频的一帧 | model/prompt/seed/method、frame_index、timestamp、SSE、pixel_count、SSIM、LPIPS |
| `windows.csv` | prompt × seed × window × method | start/end、实际帧数/时长、PSNR/SSIM/LPIPS |
| `cache_events.jsonl` | 同轨迹 clean cache 的一次采样/编码 | prompt/seed、absolute latent frame、layer/head/token 集、span/anchor、误差能量与元素数、metric_scope |
| `motion_pairs.csv` | 一段 BF16 视频的相邻帧对 | timestamp、flow magnitude、cut 标记、global/residual proxy、有效性 |
| `memory_events.csv` | 一次写入/编码/淘汰/读取状态 | step、实际/可见历史、分项 bytes、峰值、淘汰范围 |
| `summary.json`、`summary.csv` | 明确分组后的汇总 | N、缺失原因、均值与区间、协议、原始记录路径 |

`metric_scope` 至少区分 `bf16_trace_replay`、`online_local_reconstruction`、`end_to_end_video_fidelity`。这些量的数值不能混成一个“cache error”。

所有统计按模型、位宽和 attention 设置分别汇总。置信区间以 prompt 为重采样单位；多 seed 时保留同一 prompt 的 seed 组。frame、layer/head 和 window 不能冒充独立视频样本。`run_id` 与配置摘要不匹配的旧输出不参与续跑。

## 2. 上一 frame predictor 对照

对应 B5T8 Q2。新增 `experiments/previous_frame.py`，复用 `build_fixed_temporal_parameters`、已有拟合/生成入口和共用质量、cache 统计。

已有 `identity` 的定义是 `X_t = reconstructed_X_(t-1)`。正式记录要说明“frame”是相同空间 token 位置上的上一 **latent KV frame**；不能将未量化的上一帧或上一 RGB frame 作为 decoder 不可获得的输入。

建议补全的比较行：

| variant | K temporal | V temporal | K→V | 用途 |
| --- | --- | --- | --- | --- |
| `production` | 原有 learned per-channel | 原有 learned history | 原有 dense | 当前方法 |
| `k_identity` | identity，只取上一重建帧 | 保持 learned history 家族与长度 | 同结构、重新拟合 | 单独回答 temporal K 的必要性 |
| `kv_identity_cross` | identity | identity | 同结构、重新拟合 | 对齐现有 `identity_cross_vh2` 的完整上一帧替换 |
| `kv_identity_no_cross` | identity | identity | 移除 | 可复用已有行，用于补充说明 CrossKV 的贡献 |

前三行为首版必选；第四行在已有匹配视频可复用时纳入。`identity` 模式可保留与 production 相同的 history 槽位并将更早权重设为零；同时报告实际只访问一帧，避免把历史长度差异隐藏起来。CrossKV 应按各自重建 K 重新拟合；V history 的结构与拟合规则保持不变，允许其数值随闭环重新拟合。

质量使用相同 BF16 参考和同一组 prompts/seed；报告 K/V prediction NMSE、K/V reconstruction NMSE、PSNR/SSIM/LPIPS、参数大小、拟合时间。生成长度首版沿用已有 717 帧，1401 帧可复用相同变体的长视频结果，不为此增加整套必跑生成。

验收：identity 的每个非 anchor prediction 确实等于上一重建 frame；没有未来信息；原有 production 比较复现；更换 K 后 CrossKV 参数身份改变；所有方法视频完整配对。已有 60-prompt runner 的输出只有在 prompts、seed、长度、位宽、参数和 lag 等身份完全匹配时才可复用。

## 3. Temporal-window Error Analysis

对应 Qfdf W1/Q1、B5T8 W3。新增 `experiments/temporal_window.py`，共用 `common/video_metrics.py`。

### 3.1 生成与窗口

主实验在 Self-Forcing、Causal-Forcing 上分别生成 BF16 和 TempoKV，1401 pixel frames。保持相同 attention context、seed、初始噪声、prompts、生成长度与其他采样参数。每模型自己的 BF16 作为参考，不跨模型配对。

以实际时间定义半开窗口 `[0,15)`、`[15,30)`、…、`[75,end)`；在 16 fps、1401 帧时：

| window | frame indices，左闭右开 | 实际长度 |
| --- | --- | --- |
| 1 | `[0,240)` | 15 s |
| 2 | `[240,480)` | 15 s |
| 3 | `[480,720)` | 15 s |
| 4 | `[720,960)` | 15 s |
| 5 | `[960,1200)` | 15 s |
| 6 | `[1200,1401)` | 12.5625 s |

读取视频 timebase/PTS，确认输出为等帧率；如采用 frame_index/fps 计算时间，记录该假设。拒绝帧数、尺寸、fps 或配对身份不符；不通过截断较长视频来掩盖差异。模型 adapter 分别验证合法长度，Self-Forcing 的 `(frames + 3) % 12 == 0` 对 1401 成立。

### 3.2 指标与图表

一次解码逐帧累加所有窗口指标，默认不采样帧。window PSNR 由窗口全部像素的 SSE/count 得到 MSE 后再转 dB；SSIM/LPIPS 是窗口内 frame 均值。保留 `mean_frame_psnr` 时必须另列名称，不能与 pooled PSNR 混用。零误差 PSNR 记 `null + zero_error=true`，展示为 ∞，不填 0。

先保存每 prompt/window 明细，再按 prompt 等权汇总均值与 95% bootstrap CI；另存 pooled SSE，供复查总体 PSNR。曲线横轴为实际时间，最后窗口标出较短长度。产出每模型三条指标曲线，以及首窗到末窗的配对变化表；该变化本身不证明量化误差之外的成因。

可复用 `frames.csv` 直接生成后续 Motion 的同范围 fidelity，避免重复解码和 LPIPS 推理。

### 3.3 BF16 参考的可行性

Self-Forcing 当前维度下，351 latent frames 的完整 BF16 K/V 约 93.995 GiB，尚未计入模型和临时空间。不能假定单张 80 GB 卡可运行。方案要求显存足够的环境，或经过一致性验证的精确 cache offload/分片路径；数据并行复制模型不会降低单条视频的 cache 大小。

这影响执行环境，不改变实验定义。若无法运行完整参考，该组标为未完成；不能缩短 BF16 attention window 再继续称为纯量化质量对照。Trace 模式跳过 VAE，也不能代替 BF16 视频。

验收：每个完整 pair 有 6 行窗口结果、总计 1401 帧，每帧仅进入一个窗口；聚合 SSE 与全视频统计一致；参考自比为零误差；原有整视频评测口径保持一致。

## 4. Motion Robustness Analysis

对应 Qfdf W2/Q2、cCap Q3。新增 `experiments/motion/`。主流程是 BF16 motion → 固定分组 → 匹配 fidelity/cache error → proxy 与困难片段。

### 4.1 BF16 optical flow 与分组

优先适配本机 `../qvg/raft/`，实现 `FlowBackend`，固定代码版本、checkpoint、推理迭代和预处理。首版不训练新 estimator。所有相邻帧都参与主统计；若做低分辨率 flow，必须将横纵向量分别按宽高缩放回原图坐标，裁去 padding 后再算 magnitude。不能直接比较不同 resize 后的 flow 像素值。

对 BF16 视频计算用户给出的

`m = sum_txy ||F_(t→t+1)(x,y)||_2 / ((T-1) H W)`。

每模型独立计算分位点 1/3 和 2/3，再冻结 Low/Medium/High。相同 prompt/seed 的不同量化方法始终沿用同一个 BF16 分组。阈值和样本成员保存到 `motion_groups.json`，不根据 TempoKV 的坏结果调整组别。

相同尺寸/fps 内以原始 `px/frame` 为主。跨尺寸/fps 如需联合展示，另报除以图像对角线并乘 fps 的 `diagonal/second`，保留原量，不将归一化后的阈值套回原分组。分位边界有 ties 时按冻结的阈值归类并报告实际 N；不为了凑齐人数把相同 motion 人为拆开。

Flow 在 abrupt cut 上可能无物理意义：保留全 pair 的原始 `m` 以对应既定定义，同时另报去除 cut pair 的 `m_continuous`、cut 数量和占比。Camera/object proxy 和逐段相关性只使用有效连续 pair；cut 作为独立困难事件报告。

### 4.2 Fidelity、cache error 与时间对应

建议补全的组表：`model / bits / motion_group / N_video / N_cache / mean_motion / K pred NMSE / V pred NMSE / K recon NMSE / V recon NMSE / PSNR / SSIM / LPIPS`，按 prompt 汇总并给区间。

Cache 主诊断应把同一份 BF16 clean K(pre-RoPE)/V 输入真实 codec，使用闭环重建历史；比较重建和该次输入，排除模型轨迹变化。另可在 TempoKV 生成时记录 `online_local_reconstruction`，目标为它自己的量化前 cache，不能与 BF16 轨迹直接相减后称为量化局部误差。

现有 trace 若没有 prompt/seed/absolute frame 映射，只能作整体诊断，不能拼接到具体运动片段。新增 observer 或 trace sidecar 保存 sample_id、prompt 摘要、seed、绝对 latent 帧、采样 token 坐标、layer/head、clean pass 标记；sampling 固定，不按误差选择位置。主统计在生成的原生 GPU 张量上归约后只写标量，避免整段 cache 拷回本机。

视频时间与 latent 时间必须由模型 adapter 明确映射。Wan 的首帧和随后四倍时域转换有边界特殊性；保存 pixel interval / latent interval 的映射表，不能简单把 frame 编号乘 4 后假定一一对应。LongCat 还须去掉 conditioning 的重复输出，并保留每段全局偏移。无法定位的事件明确缺失，不前后随意补齐。

初版 scatter 一点代表一个 prompt-seed 视频，分别画 motion vs K/V reconstruction NMSE、motion vs LPIPS。报告每模型的 Spearman ρ、N、区间；误差和 motion 的覆盖范围必须一致。window 级关联只作为补充，按 prompt 聚类，不把所有 layers/frames 摊平计算显著性。不将相关关系写成因果结论。

### 4.3 Camera 与 local motion proxy

在均匀采样且通过有效性检查的 flow correspondences 上，用 robust affine estimation 拟合全局位移；homography 可作为单列的替代，不根据结果好坏逐视频切换。输出 inlier ratio、拟合是否失败和使用的点数。

`global_magnitude = mean ||F_global||`；`local_proxy = mean ||F - F_global||`。两者分别和同范围 K/V cache NMSE 作图、计算相关性。残差只能称 local/object-motion proxy，不能声称获得真实 object motion；parallax、非平面场景和遮挡也会进入残差。拟合失败和低 inlier 的视频单列，不强行赋零。

### 4.4 遮挡与突变困难案例

建议补全类别：large local motion、large camera motion、occlusion/disocclusion、abrupt scene change。每类先按 BF16 画面及 flow/cut/前后向不一致等信号筛候选，再人工核对类别；自动信号只是 proxy。每类目标 3 个片段，可重叠但记录类别；找不到真实 scene change 时报告缺失，不将拼接视频伪装成模型自然生成。

输出同一时间范围的 BF16/TempoKV 并排帧、PSNR/LPIPS/SSIM、K/V reconstruction NMSE 与 prediction NMSE、clip 起止时间、选取规则。前后向 flow 不一致不能单独当成真实遮挡标注。若需要新增专门 prompts 生成困难案例，单列为 targeted cohort，不混入原有 motion 分位分组。

验收：已知平移和 resize 的合成图像给出正确坐标单位；flow pair 数等于 T−1；每视频分组唯一；motion/cache/fidelity 的范围匹配；cut 与无效估计可追溯；存在负结果或困难失败时如实保留。

## 5. Predictor Architecture Ablation

对应两个设计问题。两组实验独立，calibration、validation、test prompts 分开。下表是缺失飞书表格的建议补全，不推定为原表。

### 5.1 A：Temporal Predictor

| variant | 形式 | 固定项 |
| --- | --- | --- |
| `identity` | 上一重建帧，权重 1、bias 0 | 同一 anchor、quantizer；复用实验 1 |
| `scalar_linear` | 每个 lag 一个标量，所有通道共享 | history 阶数、层/头共享范围固定 |
| `channel_linear` | 当前逐通道线性 | production 对照 |
| `dense_linear` | 每个 lag 一个 D×D 通道映射 | history 阶数和共享范围不变，只改变通道连接 |

主轴先只更换 K temporal；V temporal 的结构、长度和 K→V dense 结构固定，所有 learned 行按同一规则重新拟合，使 cross 输入匹配新的重建 K。若需要检验 V temporal，单独建第二张表，固定 K；禁止把旧 runner 中同时改变 K/V 的行称为单独改变 K 的结果。非线性 MLP 不作为首版必跑项。

Scalar/channel/dense 的 bias、warm-up order、样本池、可访问历史与正则选择协议一致。生产使用 shared K 时，dense 也采用相同层/头共享范围；不能把 dense per-head 与 channel 全局共享的差别全部归为连接结构。正则可在同一 development split 上各自选择，固定候选网格与搜索开销，test 不参与选择。

新建 `architecture/temporal.py` 和 `fit.py`；scalar 可展开为已有对角参数以验证功能，但参数成本按实际运行的存储分别记录。Dense temporal 放独立研究参数与参考 codec 中，不绕过旧参数的 shape 校验。

### 5.2 B：K→V Projection

K temporal 与 V history 完全固定，只变当前重建 K 到 V 的映射；V history/bias 的重新拟合规则对所有行一致。共享范围沿用 production 的 `cross_kv_weight_mode`，不能同时改成跨 head 的大矩阵。

| variant | 形式 | 说明 |
| --- | --- | --- |
| `diagonal` | D 个逐通道系数 | 成本最低的连接对照 |
| `low_rank_1_8` | `A∈R^(D×r), B∈R^(r×D), r=D/8` | D=128 时 r=16 |
| `low_rank_1_4` | r=D/4 | D=128 时 r=32 |
| `low_rank_1_2` | r=D/2 | D=128 时 r=64 |
| `dense` | 当前 D×D | 当前方法 |

非 128 的 head_dim 由适配器解析，rank 取有效整数并写清实际值。`r=D` 仅作低秩实现与 dense 等价性的正确性检查，不作为“压缩”行。

拟合最小化相同 calibration 数据上的 `||Y - XAB - history_term - bias||²`：可用 dense SVD 作初始化，随后进行 reduced-rank regression、交替最小二乘或受 rank 约束的优化。首版建议有固定迭代上限的交替最小二乘；保存收敛记录、初始化和 development 选择规则。最多三轮外部“运行—修复—重跑”不等于优化器只能更新三步；求解器内部步数在配置中预先固定。SVD-only 若保留，必须单独标注，不将其当作低秩可达最优表现。

### 5.3 共同输出、后端和验收

先在三模型 held-out trace 上输出 prediction/quantization/reconstruction error、实际参数 bytes、fit time、预测/解码成本。再在 Self-Forcing 上对预先确定的主比较行生成相同样本的视频；如先用 development 选代表 rank，必须冻结选择再生成 test，不按 test 结果挑最好的一行。

所有架构首版使用同一个 PyTorch 参考编码/解码路径；先确认它复现 production。Dense 与 low-rank 的端到端质量可以比较；其参考实现耗时不能冒充已优化 Triton 的线上收益。Factorized decoder 实际执行 `(X @ A) @ B`，实际存 A/B，不能预先展开 dense W 后仍报告 factorized 内存。

产出两张独立表：`variant / parameters / actual_parameter_bytes / fit_time / decoder_backend / K,V pred NMSE / K,V recon NMSE / PSNR / SSIM / LPIPS / measured_latency`。未运行的指标记缺失和原因。

验收：dense 的对角特例复现 channel；rank=D 可复现 dense 映射；每个 warm-up order 有合法预测；两组实验互不改变对方的结构；真实序列闭环稳定；端到端路径实际使用新参数且没有静默退回旧 dense 模型。

## 6. Quantization-aware Calibration Analysis

新增 `experiments/quant_calibration.py`。默认范围是用户的实验 1：测量已有 predictor 的量化后失真及其与 prediction error 的关系。

### 6.1 默认测量

每个 K/V、非 anchor 编码单元，以 decoder 可见的重建历史和当前重建 K 构造 `P=f(C)`，记录：

- `R = X - P`；`prediction_mse = sum(R²)/count`。
- `R_hat = D(Q_b(R))`；`residual_quant_mse = sum((R-R_hat)²)/count`。
- `residual_quant_nmse_x = sum((R-R_hat)²)/sum(X²)`，作为跨 predictor 的主归一化量。
- `residual_quant_nmse_r = sum((R-R_hat)²)/sum(R²)`，另列残差归一化量；近零分母标缺失/退化原因。
- `X_hat = cast_storage(P+R_hat)`；`actual_reconstruction_nmse = sum((X-X_hat)²)/sum(X²)`。
- 同时记录 residual dynamic range、量化 step/scale 的统计、元素数与能量分子/分母。不同 quantizer 配置分开汇总。

在无最终 dtype 舍入且 P 保持一致时，`X-(P+R_hat) = R-R_hat`；实际 BF16 存储还包含最后的加法/类型转换影响。因此不能只给 reconstruction NMSE 换个名字就当成精确测到了用户定义的 `L_quant`。

每个 span 的 anchor、warm-up、普通预测位置分别标记，不只剔除整段视频的第 0 帧。实际 R_hat 必须使用真实 quantizer、scale dtype、zero-point 和同样的量化分组；如现有工具未返回中间值，在参考 replay/observer 中直接归约，不由 BF16 最终重建值倒推出 R_hat。

建议补全比较行：direct quantization、identity、当前 learned predictor；已有的不同 calibration size/目标参数可在身份匹配时追加。每种位宽固定结构与 quantizer，比较 prediction MSE 与 residual_quant_mse 是否同向，不跨不同 bit/group/scale 配置简单相关。

输出按模型、K/V、bit、predictor、layer/head/时间位置的明细与汇总，附 prediction error vs quantization distortion scatter 和 Spearman；统计区间仍按 prompt。BF16 参考的 reconstruction error 为其存储误差，不能填入“BF16 residual quantization”假指标。

### 6.2 可选实验 2

只有需要进一步比较目标时再实现 `quant_aware_fit.py`，不作为首版默认依赖。结构不变，从同一个 prediction-MSE 参数初始化，固定拟合数据、优化预算和 validation 规则，评价使用未参与选择的 test prompts。

首个可行方案是量化 assignment 固定的交替更新：用真实 closed-loop rollout 获得量化 codes/scales，固定其重建残差作为本轮局部 surrogate，再更新 predictor；每轮重新执行真实量化目标，保留 calibration/development 上的记录。不直接对硬 round 求导。简单令 fake-quant 的导数恒为 1，会让 `P + Q(X-P)` 中的梯度抵消，不能仅加一个 STE 就宣称实现成功。

若 development 上真实量化后失真下降，再用预先规定的代表设置做 end-to-end generation；如没有改善，同样记录。该扩展提供目标比较证据，不预设必须提出新的训练方法，也不预设原 prediction-MSE 目标一定足够。

验收：FP32 无舍入的误差恒等关系成立；BF16 最终误差单独可测；直接量化、恒等 predictor、raw anchor 与多个 span 的统计范围正确；所有选择不使用 test。默认分析完成即可交付，无需自动启动可选拟合。

## 7. Fixed Memory Budget / Long History

对应 B5T8 W3/Q3。新增 `experiments/memory/`，区分“可存储历史”与“长历史是否有用”。

### 7.1 A：按真实 storage 计算容量

分别报告两个明确口径：

1. `M_storage(H)`：冻结完毕的历史 cache 本身，包括 codes、anchors、scales、zero-points、QVG codebooks/assignments 及实际 tensor metadata；适合回答压缩后的可存储容量。
2. `M_live(H,t)`：生成时实际持有的 KV 状态，包括 packed capacity、BF16 lag/current block、predictor history、未释放依赖和缓存副本；固定预算生成按 `max_t M_live(H,t) <= B_KV` 约束。

静态 predictor 参数单列 `M_params`，解码/attention 临时空间单列 `M_workspace`，另报实测整卡峰值；`B_KV` 不包含模型权重。`M_storage` 的优势不能直接写成整卡可生成优势。

对每个实际 tensor 按 storage 去重，包含 view 所持有的底层容量；同一组共享参数或缓存引用不重复计数。TempoKV 的 `storage_bytes()` 不含 predictor；`logical_storage_bytes()` 不含开放 span 的增长容量；现有 streaming 的 resident 统计已加 history，不能重复添加。结束后的 flush 记录为 post-generation，不能替代生成过程峰值。

BF16 计算依据为每层每 token 的 K+V 两份、实际 head 数/head_dim、batch、element_size；不同模型不硬编码相同维度。QVG 使用真实 PRQ codec，包含每级 centroid、assignment 和最终 residual；不以 `bits/16` 替代计数，也不把 S2/S2++ 当成 QVG。

第一版预算建议取同一 Self-Forcing 配置下 BF16 保留 48、96、192 latent frames 的真实 bytes，转换成 GiB 供展示。预算生成前固定，报告每方法最大的合法 history 长度、对应 seconds、峰值 bytes、unused bytes、是否达到 351 帧上限。容量只核算，不需要生成新视频；不根据哪组结果更好事后挑预算。

`T_max = max{H: max_t M_live(H,t) <= B_KV}` 在合法 block/span 对齐的 H 上枚举验证。带 anchor、增长容量或重锚定时不假定完全线性或严格单调；如要二分，先证明该具体计数模型的单调性。首次对少量 H 用真实 packed states 对照形状计数，QVG GPU codec 不可用时标记缺测，不以模拟结果冒充。

### 7.2 B：两类真实生成对照

| 对照 | 约束 | 可回答的问题 |
| --- | --- | --- |
| Matched Context | 同一 attention window H、sink、生成长度、prompts/seed | 固定 context 下各压缩方法造成的 fidelity 损失 |
| Matched Memory | 相同 B_KV，各方法采用其可行 H；horizon 仍相同 | 更高压缩能否换取更多有效历史与更好长程质量 |

首版采用 Self-Forcing 的 1401 帧、held-out 32 prompts、seed 0、sink=0；BF16/QVG/TempoKV 在同一生成器上运行。Matched Context 使用预先固定的共同 H（建议 96 latent frames，执行前验证每预算是否可容纳；不满足的小预算不列该对照），不强制把相同 H 的内存也写成相同。

除了 BF16 的短窗口和 QVG 的中间窗口，必须增加 **TempoKV-short**：同一 TempoKV、同一预算上限下，分别用 BF16 的短 H 和 TempoKV 的长 H。这个配对直接检验“增加历史”的贡献；跨方法差异同时包含 quantization 和历史长度，不能单独归因于历史更长。短 H 允许少用预算，unused bytes 如实报告。

每个预算可得到 BF16-Hbf、QVG-Hqvg、TempoKV-Htk、TempoKV-Hbf。Matched Context 另列共同 H 的三方法行。所有方法的 history 由计数器决定，不预设一定是 BF16 < QVG < TempoKV；若某配置有较大固定开销或不能容纳最小 block，报告实际限制。

使用相同完整历史 BF16 reference 计算 PSNR/SSIM/LPIPS，评价后半段/末窗的 VBench consistency 等既有协议指标，并保留全片结果。逐像素 fidelity 对自由生成轨迹分离较敏感，需与 consistency/VBench 并列解释，不能仅凭 PSNR 断定历史更有用。BF16 参考不受比较预算 B_KV 限制；无可用完整参考时，仍可完成容量和无需参考的质量指标，但配对 fidelity 明确缺测。

### 7.3 真正的预算约束与 QVG 公平性

现有 `LOCAL_ATTN_SIZE` 能限制可见历史，不能自动证明 TempoKV 的旧压缩数据已释放。当前 cache 要求完整 packed span 读取，单个预测 span 也有对先前重建状态的依赖；直接删中间帧会破坏解码。

`budget_manager.py` 首版使用独立 span、整 span 淘汰与合法窗口边界。建议先用 12 latent frames/span 的预算专用协议，保存实际 anchor 开销；三方法的 Matched Context/Memory 及 TempoKV-short 都固定这一协议。它与原全程单 anchor 主结果分开标记，不能复用不同 span 的质量结果。若保留原全程单 span，则必须实现并核算边界重锚定或所有解码支持状态，不声称 attention mask 一变就释放了内存。

管理器在每个写入、编码、淘汰和 read event 核对 B_KV。淘汰同步处理 K/V、span、开放 encoder/history 与持有引用；absolute frame position/RoPE 不重置。记录 attention 实际可访问的最早/最晚帧及依赖支持范围；不可见但为解码保留的旧数据仍计内存。所有必要的 BF16 lag 与最小 current block 在预算内预留，不能在不记录的情况下改为 lag=0。

QVG 优先在本目录适配真实 `compress/uncompress` 到同一个 Self-Forcing pipeline；相邻仓库的 `run_qvg.sh` 用于实现对照。若直接调用不同 checkout 的生成器，应先验证 BF16 的 seed/noise、attention、RoPE、采样配置与输出等价，否则跨方法结果存在生成器混杂，不能作为正式 matched-memory 证据。QVG 的 quant_type 必须为真实 PRQ/QVG，包含 stages、centroids 与 KMeans 设置。

验收：每次事件均在预算内；attention 真的读取预期历史；长生成中 cache 不随总 horizon 无限增长；span 淘汰后无残余持有；全局位置与 sink=0 语义正确；整卡临时峰值另有记录；长短 TempoKV 的实际 H 确有区别。第一轮小模型/小张量验证通过后，才运行正式长视频。

## 8. 实现次序、输出与检查

| 顺序 | 实现包 | 前置依赖 | 完成标志 |
| --- | --- | --- | --- |
| P0 | 清单、流式视频指标、cache 误差分项、模型 adapter 的 plan 模式 | 现有入口和参数 | 相同输入可配对、可续跑，统计单位一致 |
| P1 | 实验 1、实验 5.1、实验 6A | 现有参数/trace/codec | 无需新增视频即可完成对照重建、量化目标测量和容量表 |
| P1 | 实验 2、Motion fidelity/flow 分组 | 配对 BF16/TempoKV 视频；缺失时生成 | 六窗口明细、运动三组表和 LPIPS scatter |
| P2 | Motion cache 对齐、camera/local proxy、困难片段 | 有身份的 trace/observer | 匹配范围的 cache 关联和可复查片段 |
| P2 | 实验 4A/4B 离线拟合与参考解码 | calibration/development/test split | 两张独立架构表、真实 factorized 参数 |
| P3 | 架构端到端、实验 6B 预算管理与生成 | 参考 codec、真实 QVG、内存/位置检查 | 真实生成、两类对照、TempoKV 长短配对 |
| 可选 | 实验 5.2 quantization-aware fitting | 5.1 结果与固定目标比较设置 | 真实量化后失真比较，不预设改善 |

拟议统一入口，以下命令在实现后才可运行：

```bash
python rebuttal/run.py --experiment previous_frame --stage plan --config rebuttal/configs/previous_frame.json
python rebuttal/run.py --experiment temporal_window --stage evaluate --config rebuttal/configs/temporal_window.json
python rebuttal/run.py --experiment motion --stage evaluate --config rebuttal/configs/motion.json
python rebuttal/run.py --experiment architecture --stage fit --config rebuttal/configs/architecture.json
python rebuttal/run.py --experiment quant_calibration --stage evaluate --config rebuttal/configs/quant_calibration.json
python rebuttal/run.py --experiment fixed_memory --stage plan --config rebuttal/configs/fixed_memory.json
```

`plan` 只写输入检查和命令清单；`fit/replay/generate/evaluate/summarize` 仅开放各实验真正支持的阶段。全局 `--resume` 校验身份；`--dry-run` 不启动模型；缺失依赖为带原因的失败/缺测，不能静默降级换方法。

只增加覆盖实质风险的检查：窗口边界及末窗、真实误差恒等关系与 BF16 舍入、identity 闭环历史、factorized decoder、运动坐标单位、cache span 淘汰与绝对位置、真实 storage 与 runtime 峰值。已有 production 相关回归检查按受影响模块选取；本次规划本身只检查文档与路径，不运行 GPU 实验。

每个结果保留原始记录、N 和缺失原因；表中结果来自实际运行。飞书中未展示的表格内容、论文主位宽、未给出的参数路径均不伪装成已确认事实。发现负结果如实保留，不据此更改用户的研究结论。
