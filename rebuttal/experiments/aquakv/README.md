# 独立 AQUA-KV → Self-Forcing

当前入口实现独立 AquaKV 的完整流程：**采集 clean KV → 校准 AquaKV 自己的预测器 → 独立 heldout 重放 → BF16/AquaKV 视频生成 → 视频评测 → 汇总**。不读取 TempoKV 的参数、配置、拟合器或量化器。

代码验证和真实实验分开记录。当前没有真实 Self-Forcing 视频实验结果；本地检查不能代表生成质量或 GPU 性能。

## 官方来源与保留的算法

使用 [AQUA-KV 官方仓库](https://github.com/goodevening13/aquakv) 的固定版本 `e744183f2f3a4a90381f20a6975456871fd1ae29`。官方 HIGGS、回归求解器和网格均保存在 `upstream/`，附 Apache-2.0 LICENSE；来源及本地调整见 [SOURCE.md](upstream/SOURCE.md)。运行实验时不需要在线下载官方仓库。

- 第 0 层独立使用 HIGGS 4-bit（默认 `d=2,n=256`），也支持显式保留 BF16。
- 其余层用 HIGGS 压缩预测残差，默认 `d=2,n=16`（名义 2-bit）。`n=64/256` 对应名义 3/4-bit。
- K 预测输入是上一层重建 K。V 预测输入按官方顺序拼接：`[当前层重建 K, 上一层重建 V]`。
- 校准调用官方 `fit_linear_regression`，使用相对 Hessian damping `reg_rate`，逐层用已经重建的输入校准后继预测器。AquaKV 原版需要这一步；这是 AquaKV 自己的校准，与 TempoKV 拟合无关。
- 校准使用与实际推理一致的参数精度，默认 BF16；保存实际 BF16 权重，避免只按 BF16 估算参数大小、实际却使用高精度权重。
- 原版的 dense 映射保留跨 head 混合；没有替换成 TempoKV 的时序映射或逐 head 结构。

## Self-Forcing 适配

Self-Forcing 每个块执行多次去噪和一次 clean refresh。当前去噪块保持可覆盖的 BF16 数据，只有 pipeline 明确发出 clean refresh 后才进入历史缓存。历史按 latent frame 划分，逐层压缩超出 BF16 保留范围的部分。

读取缓存时，跨层预测依次使用**重建后的 pre-RoPE K/V**。每层只暂存解码该层所需的历史，最后一层释放；缓存中不保留完整 BF16 重建镜像。注意力前按绝对位置恢复 Wan 的 3D RoPE；有限窗口按 sink 与 recent 的并集选择，过期完整块释放。

默认 `prefix_frames=1,bf16_lag_frames=18`，单位是 **latent frames**。首帧保留 BF16，最近 18 个已定稿帧也保留 BF16，与 TempoKV 的默认保留长度相同。缓存 K 和同次生成的 BF16 参考统一使用 `complex64` RoPE，Q 使用原生精度。`sink_size` 控制注意力窗口。

复现旧实验时，使用旧配置中的 `recent_buffer_frames=3` 与 `rope_precision=native`。旧阈值达到时会压缩所有待压缩历史。若从当前默认配置覆盖，还需设置 `cache.bf16_lag_frames=null`。

评测默认读取同一输出目录中完成并校验过的 BF16 和 AquaKV 视频。需要复用外部视频时，可显式设置 `reference_dir` 或 `candidate_dir`。

`flush_tail=true` 只在生成结束后压缩未满阈值的尾部。报告同时保存生成中的峰值，避免用结束后的大小代替峰值。每条视频必须确实使用过 K/V 压缩和解压，否则生成阶段失败并说明原因。

## 环境

CPU 校准和重放需要 `torch,numpy`。CUDA 生成使用仓库现有 Self-Forcing 环境和本地模型权重。安装 HIGGS 的 CUDA 依赖：

```bash
python -m pip install -r rebuttal/experiments/aquakv/requirements-cuda.txt
```

CPU 使用 Hadamard 参考实现。CUDA 使用 Triton 3.0 及以上版本做融合量化和解码，其他场景的 CUDA 参考路径使用 `fast-hadamard-transform`。视频评测复用通用的 PSNR/SSIM/LPIPS 入口。LPIPS 需要对应依赖和本地权重。

## 一个命令运行全流程

从仓库根目录执行，替换两个模型路径：

```bash
bash rebuttal/run_aquakv.sh \
  --set checkpoint=/models/self_forcing.pt \
  --set model_root=/models/Wan2.1-T2V-1.3B \
  --output-dir rebuttal/outputs/aquakv/sf_higgs2
```

该脚本等价于 `python rebuttal/run.py --experiment aquakv --stage pipeline`。默认配置是 `rebuttal/configs/aquakv.json`，使用现有独立 calibration/heldout prompt 文件。校准采集默认 81 RGB 帧；最终视频默认 1401 RGB 帧、full history。BF16 full-history 参考需要约 94 GiB 的 KV 数据空间，另需模型与临时空间。帧数、prompt 数量和位宽都由配置显式决定。

脚本使用当前环境的 `python`，可通过 `AQUAKV_PYTHON=/path/to/python` 指定。`AQUAKV_STAGE=plan` 只输出计划。执行不会自动下载模型，也不会启动远端任务。

## 分阶段运行

将机器上的路径保存到一个新配置，例如 `rebuttal/outputs/aquakv_local.json`：

```json
{
  "extends": "../configs/aquakv.json",
  "checkpoint": "/models/self_forcing.pt",
  "model_root": "/models/Wan2.1-T2V-1.3B"
}
```

所有阶段使用相同配置和目录：

```bash
python rebuttal/run.py --experiment aquakv --config rebuttal/outputs/aquakv_local.json --stage collect --output-dir rebuttal/outputs/aquakv/sf_higgs2
python rebuttal/run.py --experiment aquakv --config rebuttal/outputs/aquakv_local.json --stage fit --output-dir rebuttal/outputs/aquakv/sf_higgs2
python rebuttal/run.py --experiment aquakv --config rebuttal/outputs/aquakv_local.json --stage replay --output-dir rebuttal/outputs/aquakv/sf_higgs2
python rebuttal/run.py --experiment aquakv --config rebuttal/outputs/aquakv_local.json --stage generate --output-dir rebuttal/outputs/aquakv/sf_higgs2
python rebuttal/run.py --experiment aquakv --config rebuttal/outputs/aquakv_local.json --stage evaluate --output-dir rebuttal/outputs/aquakv/sf_higgs2
python rebuttal/run.py --experiment aquakv --config rebuttal/outputs/aquakv_local.json --stage summarize --output-dir rebuttal/outputs/aquakv/sf_higgs2
```

也可直接提供 `calibration_trace` 和 `evaluation_trace`（单个 NPZ、NPZ 目录或文件列表），跳过采集。格式为匹配的 `[L,records,frames,H,D]` keys/values、从 0 开始的连续完整 `layer_ids`、`coordinate_system=pre_rope_normalized`，以及覆盖所有记录的 prompt hashes。独立评测根据**预测器文件内保存的实际校准来源**检查泄漏，不依赖用户另行声明的 calibration 路径。

`evaluate` 是真实视频评测，`replay` 是 heldout trace 重放。生成仅接受 BF16 校准参数，与 Self-Forcing 的计算精度一致。`validation_records` 可留出校准内部诊断记录，不用于最终 heldout 结果。

恢复使用 `--resume`；配置、来源文件、代码或 HIGGS 网格改变后应使用新输出目录。`fit` 默认写入输出目录，不能指定 `predictor_path`；外部 `predictor_path` 用于选择已经校准好的输入文件，供独立 `replay` / `generate` 使用并记录指纹。`pipeline` 使用已经提供的两套 trace 时会自动跳过采集。

## 输出

| 输出 | 内容 |
| --- | --- |
| `collection/calibration/` / `collection/evaluation/` | 对应 split 的 BF16 视频及 clean KV，分别在 `videos/`、`traces/` 下 |
| `aquakv_predictor.pt`、`fit.json` | AquaKV 独立权重、官方版本、HIGGS 设置、实际校准来源及误差 |
| `replay.json` | heldout AquaKV 与不使用预测器的 HIGGS 对照；逐层误差和真实存储字节 |
| `runs/bf16/videos`、`runs/aquakv/videos` | 相同生成器、prompt、seed、sample、窗口设置的视频与生成摘要 |
| `runs/aquakv/aquakv_activity.rank-*.jsonl` | 每个 prompt 的实际压缩/解压次数、缓存峰值、参数和网格字节 |
| `quality/`、`evaluate.json` | 逐帧、逐窗口、逐视频 PSNR/SSIM/LPIPS 及汇总 |
| `summary.json` | 已完成各阶段的报告 |

存储按实际 bit-packed 网格索引、FP32 scales 和 padding 计算，预测器另列。Self-Forcing 的 1536 channels 在默认 1024 Hadamard 分组下需要 padding，因此**名义 2-bit 不代表实际总存储就是 2-bit**。网格字节在生成报告中另列；文件序列化开销不混入 tensor 存储。计时是实际参考实现的计时，不宣称具备融合解码内核的速度。

## 检查

怀疑生成适配改变了输出时，用同一个 checkpoint、prompt、seed 和 RoPE 精度跑
`generate`，设置 `generation_methods=["bf16","bf16_adapter","aquakv"]`、
`record_latent_hashes=true`。可先限制到一条 prompt 和 93 个 RGB 帧。需要已拟合
的 AquaKV 参数。三路分别走原生 BF16、关闭压缩的 AquaKV 缓存、正常 AquaKV 压缩。
各自输出的 `latent_hashes.rank-*.jsonl` 按生成块记录 VAE 解码前的 latent 哈希。
先比较原生 BF16 和关闭压缩的 AquaKV 路径，再检查开启压缩后从哪一块开始分叉。
使用新输出目录可以排除旧参考视频的生成环境和视频编码差异。

```bash
python -m pytest -q rebuttal/tests/test_aquakv.py
```

测试包含官方 HIGGS 重建等价性、实际索引打包、官方回归输入、独立参数、clean-refresh 覆盖、跨层解码、prefix/buffer、绝对 RoPE、窗口与释放，以及执行 Self-Forcing 实际 inference loop 的小模型检查。CUDA 检查在无 GPU 时跳过。
