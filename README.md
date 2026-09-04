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

## 位宽

一个 span 的第一个 unit 是 **anchor**，直接量化；后面每个 unit 只存**预测残差**。
两者的位宽独立设置，拟合和生成必须传入相同的值：

| 变量 | 范围 | 作用 |
|---|---|---|
| `BITS` | 1–8 | 同时设置 K/V 残差位宽 |
| `K_BITS`、`V_BITS` | 1–8 | 分别设置 K/V 残差位宽（混合位宽） |
| `ANCHOR_BITS` | 1–15 | 设置 anchor 量化位宽 |
| `ANCHOR_BITS=16` | — | anchor 不量化，直接按 bf16 存 |

默认是 anchor INT4 + K/V 残差 INT4（`configs/legacy.json`）。

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

## 其他拟合选项

| 变量 | 作用 |
|---|---|
| `CONFIG` | codec 配置 JSON（默认 `configs/legacy.json`） |
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
所以参数不能跨模型、位宽、stride 或配置复用——加载时签名不匹配会直接报错。旧的 native 或
非 layer-shared 参数需要按当前 legacy 配置重新拟合。

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
