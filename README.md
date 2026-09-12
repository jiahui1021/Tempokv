# TempoKV

TempoKV 用时间预测和 CrossKV 预测压缩视频生成模型的 KV cache。仓库包含三个完整接入：
Self-Forcing、Causal Forcing 和 LongCat。

## 安装

```bash
pip install -e .                       # 基础代码
pip install -e ".[cuda,self-forcing]"  # 按模型选一个推理环境
pip install -e ".[cuda,causal-forcing]"
pip install -e ".[cuda,longcat]"
pip install -e ".[metrics]"            # 需要质量评测时
```

Self-Forcing 与 LongCat 依赖的 `diffusers`、`transformers` 和 `av` 版本不同，建议使用
独立虚拟环境。

## 完整流程

三个模型都按同一顺序运行：**采集 clean trace → 拟合参数 → TempoKV 生成 → BF16 对照**。
每一步是一条脚本，模型相关的路径用环境变量传入：

| 模型 | 脚本目录 | 必需环境变量 |
|---|---|---|
| Self-Forcing | `scripts/Self-Forcing/` | `CHECKPOINT`、`WAN_MODEL_DIR`、`PROMPTS` |
| Causal Forcing | `scripts/Causal-Forcing/` | `CHECKPOINT`、`MODEL_ROOT`、`PROMPTS` |
| LongCat | `scripts/LongCat/` | `CHECKPOINT_DIR`、`PROMPT` |

以 Self-Forcing 为例：

```bash
export CHECKPOINT=/models/self_forcing_dmd.pt
export WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B

# 1. 采集 trace
PROMPTS=assets/fit_prompts.txt bash scripts/Self-Forcing/collect_trace.sh

# 2. 拟合参数
FIT_DEVICE=cuda bash scripts/fit_params.sh self-forcing

# 3. 生成，以及同一组 prompt、seed、帧数的 BF16 对照
PROMPTS=assets/generation_prompts.txt bash scripts/Self-Forcing/run_tempokv.sh
PROMPTS=assets/generation_prompts.txt bash scripts/Self-Forcing/run_bf16.sh
```

换成上表另一行的脚本目录和环境变量即可跑另外两个模型；`scripts/fit_params.sh` 的参数
相应改成 `causal-forcing` 或 `longcat`。Causal Forcing 首次使用可先运行
`bash scripts/Causal-Forcing/download_models.sh`。

## 拟合协议

两个协议都可用，用 `CONFIG` 选：

| 协议 | 配置 | K predictor | CrossKV 权重 | Cross 矩阵乘 |
|---|---|---|---|---|
| legacy（默认） | `configs/legacy.json` | 跨层共享 | layer-shared | ieee |
| native | `configs/native.json` | 每层每头 | layer-head | tf32 |

配置里写 `fit_mode` 就够了，其余结构默认值跟着协议走。两个协议的位宽选项完全一样
（下一节），但参数**不能互换**：协议写进 codec signature，加载时不匹配直接报错，
参数目录也分别以 `legacy-` / `native-` 开头。

```bash
CONFIG=configs/native.json FIT_DEVICE=cuda bash scripts/fit_params.sh self-forcing
```

## 位宽

一个 span 的第一个 unit 是 **anchor**，直接量化；后面每个 unit 只存**预测残差**。
两者的位宽独立设置，拟合和生成必须传入相同的值：

| 变量 | 范围 | 作用 |
|---|---|---|
| `BITS` | 1–8 | 同时设置 K/V 残差位宽 |
| `K_BITS`、`V_BITS` | 1–8 | 分别设置 K/V 残差位宽（混合位宽） |
| `ANCHOR_BITS` | 1–15 | 设置 anchor 量化位宽 |
| `ANCHOR_BITS=16` | — | anchor 不量化，直接按 bf16 存 |

默认是 anchor INT4 + K/V 残差 INT4，legacy 和 native 都一样。

```bash
BITS=2 bash scripts/fit_params.sh self-forcing                    # K/V 都是 INT2
K_BITS=3 V_BITS=5 bash scripts/fit_params.sh self-forcing         # K INT3、V INT5
ANCHOR_BITS=8 BITS=2 bash scripts/fit_params.sh self-forcing      # 宽 anchor、窄残差
ANCHOR_BITS=16 BITS=2 bash scripts/fit_params.sh self-forcing     # 无损 anchor 上界
```

所有位宽都用连续 bit packing，INT1/3/5/7 这些非 2 的幂不浪费 padding。残差停在 8 位，
因为超过 8 位的残差已经比它替换的 bfloat16 更大。

anchor 的每组还要存一个 bf16 scale，所以 N 位 anchor 的实际开销是 `N + 16/group_size` 位/元素。
到 INT16 就是 16.25 位——**比原始 bf16 还大，而且仍然有损**。所以 `ANCHOR_BITS=16` 不是
INT16，而是跳过量化器、直接以 bf16 存 anchor：正好 16.00 位/元素，逐位无损，不存 scale。
中间档位仍然是有损压缩，但 scale 精度让收益在 INT12 附近就基本饱和：标准正态输入下
INT8→INT12 的 RMSE 降约 11 倍，INT12 之后已经很平。

K/V 位宽相等时目录名沿用 `intN`，不相等时写成 `kN-vM`；anchor 不是默认的 INT4 时追加
`-aN`。例如 `K_BITS=3 V_BITS=5 ANCHOR_BITS=12` 得到
`legacy-kh2-vh1-k3-v5-a12-layershared`。

## Jensen 修正（attention score，默认关闭）

量化给 cached key 加的噪声在 score 上是零均值的，但 softmax 里的 `exp` 是凸的，所以
`E[e^δ] ≥ 1`：每个被量化的 cached score 对配分和的贡献都被系统性放大，cached block 会
从**未量化的当前 chunk** 那里抢走 attention mass。`JENSEN=1` 在 softmax 之前减掉这个偏置

```
b_i = (1 / 24d) · Σ_j Δ²_{i,j} · ‖q_j‖²
```

`Δ` 就是 codec 已经存着的 scale，所以**不增加任何存储**，也**不改变任何存储值**——
`K̂`、`V̂`、重建 NMSE 开不开都逐位相同，变的只有 attention 输出。同理它不进 codec
signature，所以开关两边共用同一份拟合参数，不用重新拟合。

代价在计算：修正是一个秩 `G = d/group_size` 的双线性型，折进 Q/K 的额外通道里就能让
普通 attention kernel 顺带算出来，不需要 FlexAttention 或自定义 kernel；head_dim 从 128
补到 136（flash-attn 要 8 的倍数，且 FA2 要求 Q/K/V 同宽）。增加 6.25% 的通道并不
代表只增加 6.25% 的耗时：跨过 128 维边界会切换到更大的 kernel 规格，具体取决于
FlashAttention 版本。

```bash
JENSEN=1 PROMPTS=assets/generation_prompts.txt bash scripts/Self-Forcing/run_tempokv.sh
```

输出目录自动加 `-jensen` 后缀，所以开关两边的视频不会互相覆盖，可以直接对比。

新增实验路径 `TEMPOKV_JENSEN_BACKEND=flex`：Q/K/V 保持 128 维，在编译后的
FlexAttention 内计算同一 Jensen 修正。采用连续的分组修正数据和自动调优，保留原路径
的 BF16 中间舍入顺序，不生成完整 Q×K bias，输出不保证逐位一致。默认仍为 `augmented`；新路径尚未在
A100 验证编译、正确性和速度，不保证更快。需要支持 FlexAttention 的 Torch（建议
沿用运行 Wan 的 Torch >=2.5）及匹配的 Triton。缓存增长可能触发额外编译，首轮耗时与
稳定运行耗时需分别比较。两个 backend 不会自动区分输出目录，比较时指定不同目录：

```bash
JENSEN=1 TEMPOKV_JENSEN_BACKEND=flex OUTPUT_DIR=results/self_forcing/jensen-flex \
  PROMPTS=assets/generation_prompts.txt bash scripts/Self-Forcing/run_tempokv.sh
```

`scripts/tools/benchmark_jensen_fused.py` 比较原始 FA2、136 维修正和新融合路径，先做
小尺寸数值检查，再分别报告首调用和稳定耗时。默认尺寸来自已有长窗口微测，应使用
实际运行的 query/key 长度替换；不包含缓存解码和 RoPE，不能替代完整生成对照。

目前只有 Self-Forcing 接入。默认关闭：在本仓库的 trace 上，生产的 INT4 残差位宽下这个
偏置比论文测量的 INT2 直接量化设定小约两个数量级，端到端收益尚未验证。有希望的是
`K_BITS=2`，以及 anchor 帧——anchor 走直接量化，`Δ` 明显更大，实测偏置是残差帧的 12 倍。

## 其他拟合选项

| 变量 | 作用 |
|---|---|
| `CONFIG` | codec 配置 JSON，同时决定拟合协议（默认 `configs/legacy.json`） |
| `TRACE_FILE` | 指定采集好的 `clean_trace.npz` |
| `OUTPUT_DIR` | 指定参数输出目录 |
| `KH`、`VH` | 设置 K/V history length |
| `k_predictor_mode` | 配置中的 `shared`、`identity`、`taylor`、`taylor2` 或 `direct` |
| `V_PREDICTOR` | `cross_kv`、`temporal`、`identity`、`taylor`、`taylor2`、`cross_kv_identity`、`cross_kv_taylor`、`cross_kv_taylor2`、`innovation` 或 `direct` |
| `CACHE_STRIDE` | LongCat Context Parallel 的 local stride |
| `FIT_DEVICE` | `cuda`、`cpu` 或 `auto` |
| `DRY_RUN=1` | 只显示最终命令，不执行拟合 |

完整选项：`python scripts/common/fit_params.py --help`。模型专用的旧入口
`scripts/<Model>/fit_params.sh` 仍可使用。

参数写到 `data/<模型>/params/<参数方案>/`（Causal Forcing 多一层模式目录），每个目录包含
`k_temporal.npz`、一个 V 参数文件和 `fit_report.json`。上表每一项都写进 codec signature，
所以参数不能跨模型、协议、位宽、stride 或配置复用——加载时签名不匹配会直接报错。

## 查看结果

生成结果写入 `results/<模型>/`，TempoKV 用与参数目录同名的参数方案作子目录，BF16 对照用
`bf16`。至少检查：

- 视频是否完整生成；
- `tempokv_report.json` 中 encode/decode 计数是否非零；
- BF16 与 TempoKV 是否使用相同 prompt、seed 和帧数；
- 峰值显存、完整生成时间、质量指标和压缩比。

```bash
# BF16 与 TempoKV 的视频质量对比
python scripts/evaluation/evaluate_quality.py \
  results/self_forcing/bf16/videos \
  results/self_forcing/legacy-kh2-vh1-int4-layershared/videos \
  --device cuda --output results/self_forcing/evaluation/quality.json

# 完整生成的耗时分解（先用 PROFILE=1 跑一次生成）
python scripts/tools/profile_tempokv_runtime.py \
  results/self_forcing/legacy-kh2-vh1-int4-layershared

# Cross kernel microbenchmark，需要 NVIDIA GPU
python scripts/tools/benchmark_tempokv_kernels.py \
  --route v --sweep production --cross-matmul-precision ieee
```

## Self-Forcing 四卡消融

需要把 K/V history、独立 temporal-V 对照、50 条视频和 cache/视频指标放进同一张表时，用
[`run_history_ablation_4gpu.sh`](scripts/Self-Forcing/run_history_ablation_4gpu.sh)：默认
K 轴 `h_K=0..5` 固定 CrossKV `h_V=1`，V 轴 CrossKV `h_V=0,1,2,4,5` 固定 `h_K=3`，外加
`w/o cross` 的 temporal V `h_V=1`。脚本要求四张卡，每个 variant 一张，配置、参数、视频、
日志和汇总格式见
[`history_ablation/README.md`](scripts/paper_experiments/history_ablation/README.md)。

## 详细说明

- [Self-Forcing](scripts/Self-Forcing/README.md)、[Causal Forcing](scripts/Causal-Forcing/README.md)、[LongCat](scripts/LongCat/README.md)
- [脚本目录](scripts/README.md)
- 论文实验：[K/V history 消融](scripts/paper_experiments/history_ablation/README.md)、[CrossKV sharing](scripts/paper_experiments/cross_kv_sharing/README.md)、[Predictor replacement](scripts/paper_experiments/predictor_ablation/README.md)、[latency](scripts/paper_experiments/latency/README.md)
