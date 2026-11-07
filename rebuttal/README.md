# TempoKV 补充实验框架

本目录已经实现六项补充实验的入口、拟合/重建评测、视频指标和生成适配器。`implementation_plan.md`、`code_map.md`、`experiment_matrix.json` 保留最初的实验设计；它们不是实验结果。当前代码通过本地 CPU 与小型真实视频回归测试，真实模型的 CUDA 长视频生成、RAFT 和正式质量结果仍须在具备数据/权重的实验环境验收。

## 入口与配置

从仓库根目录运行。环境需要项目的模型依赖，以及 `numpy torch pytest scipy scikit-image opencv-python av matplotlib`；计算 LPIPS 另需 `lpips` 及其本地权重。RAFT、QVG 和生成模型的代码/权重路径必须显式配置，框架不负责下载模型。

```bash
python rebuttal/run.py --experiment all --stage plan --output-dir /tmp/tempokv-rebuttal-plan
python -m pytest -q rebuttal/tests
```

`all` 只生成六项实验的计划，不启动模型。`rebuttal/configs/` 中的八个命名配置分别覆盖六项实验、可选 `quant_aware_fit` 和 `collect_trace`。缺少真实 trace、checkpoint、参考视频等输入时，计划列出缺项；执行阶段必须补齐。默认 `configs/legacy.json` 是 K4/V4 的配置示例，**不能未经核对就当作论文最终主设置**。

建议在自己的 JSON 中 `extends` 对应配置，再填入实验机器上的绝对路径。例如：

```json
{
  "extends": "../configs/previous_frame.json",
  "calibration_trace": "/data/tempo/cal.npz",
  "evaluation_trace": "/data/tempo/test.npz",
  "checkpoint": "/models/self_forcing.pt",
  "model_root": "/models/Wan2.1-T2V-1.3B",
  "reference_dir": "/data/bf16/videos"
}
```

`extends` 相对当前配置文件解析，其他已知输入路径相对仓库根目录解析。示例可存为 `rebuttal/outputs/my_previous_frame.json`，该输出目录被 Git 忽略。默认 prompt 文件仅提供独立划分示例，需与实际 trace 和模型生成来源一致。若有独立 identity 文件，可配置 `calibration_identity` / `evaluation_identity`；否则自动读取 trace 同名 `.identity.json`。

同一实验的各阶段使用相同完整配置和输出目录：

```bash
python rebuttal/run.py --experiment previous_frame --config rebuttal/outputs/my_previous_frame.json --stage fit --output-dir rebuttal/outputs/previous_frame
python rebuttal/run.py --experiment previous_frame --config rebuttal/outputs/my_previous_frame.json --stage replay --output-dir rebuttal/outputs/previous_frame
```

`--set dotted.key=value` 可以覆盖配置。`--resume` 仅控制恢复，不改变实验身份；恢复或跨执行阶段都会检查输入内容和代码指纹。纯 plan 可以在输入准备好后进入执行。修改配置、trace/identity 内容或代码后须使用新输出目录，不可给旧结果换标签。

## 六项实验与输出

| 实验 | 阶段 | 输出与控制 |
| --- | --- | --- |
| `previous_frame` | `fit → replay`，可选 `generate → evaluate` | production、learned one-frame K、copy K、copy K/V + cross、无 cross 的 copy K/V；全部保留真实 residual codec，并按变化后的重建 K 重新拟合 |
| `temporal_window` | `generate → evaluate → summarize` | per-prompt/per-window PSNR、SSIM、LPIPS，prompt bootstrap 和末窗减首窗；1401@16fps 是 87.5625 秒，最后窗口 201 帧 |
| `motion` | `evaluate → summarize`，可选共用 `generate` | BF16 非 cut 光流分组、fidelity/cache error、Spearman、camera/local proxy、困难片段；详见 [输入约定](experiments/motion/README.md) |
| `architecture` | 分别以 `axis=temporal` / `axis=cross` 运行 `fit → replay`，可选 `generate → quality` | 两个独立消融；误差、参数字节、独立预热重复 decode 计时及生成视频质量 |
| `quant_calibration` | `fit → replay`（`evaluate` 同 replay） | direct / identity / production，在固定 bit-width 下测 prediction、真实 residual quantization 和最终重建误差；不据此宣称 L2 与量化目标等价 |
| `fixed_memory` | `capacity`，再 `generate → evaluate → summarize` | KV owned state 容量；真实 SF BF16/QVG/TempoKV matched-memory、matched-context、TempoKV 长短历史；详见 [配对要求](experiments/memory/README.md) |

架构的两个 axis 必须用不同输出目录。Cross 的 `dense`、`diagonal`、`low_rank` 使用相同 zero-centered ridge 目标与重拟合流程，生产参数单列为 `production`；低秩实际存储并使用 A/B 因子。`decode_seconds_*` 是相同 PyTorch reference 后端的纯 K→V 解码计时（包含输出分配），不是生产 Triton 内核或端到端生成加速。Research generation 使用有效 span（默认 24 latent frames），每条视频必须验收 K/V encode/decode 均非零；日志在 `codec_activity.rank-*.jsonl`。

`quant_aware_fit` 是可选的真实量化器局部随机搜索对照，使用 calibration 优化、validation 选择，最终 test 必须另留。导出的 `fit_report.json` 保存 calibration/validation 身份与参数指纹；`quant_calibration` 自动读取同目录报告（或显式 `parameter_fit_report`），阻止把选择参数所用的 validation 重新当最终 test。它不代表最佳可达到的量化感知方法。`collect_trace` 支持 Self-Forcing 的 `generate → convert`，保留 prompt/seed、空间采样和 latent→RGB 区间，不把不同 prompt 随意池化。

## 结果使用边界

- 默认 `require_prompt_identity=true`。验证依据是整个 trace 的身份，而不是 `max_records` 截取后的第一批记录；显式 sidecar 与内嵌身份冲突会失败。未记录真实训练来源的外部 predictor 不能仅凭声明一个 calibration 文件就视为来源已核验。
- 目录视频配对读取实际生成摘要；fixed-memory 必须核验 prompt 文本、noise seed、sample，支持 `seed_<seed>/videos` 参考目录。QVG 编码隔离生成器 RNG，避免把不同随机轨迹计为量化损失。
- Motion 的 cache join 匹配方法身份；`cache_variant` 用于显式映射视频标签与 replay 标签。困难片段在共同 RGB/cache 边界计算，无法对齐则明确缺测。quality 复用还需成功完成记录和输出 CSV 指纹；失败重跑不会用新视频身份认证旧分数。旧 quality 缺这些记录时须重算。
- Fixed-memory 仍以 full-history BF16 为参考，指标含上下文变化和压缩的共同影响；纯量化影响需在相同 attention window 下另外配对 BF16。容量表是显式 tensor-shape/schedule 核算，device resident peak 和生成质量须以实际 worker 结果验收。
- SF 是默认主模型；CF 的标准生成适配器已接入。研究架构/在线 observer 与固定预算 worker 首版仅支持 SF，其余模型可做带身份 trace replay；不能用这部分测试宣称三个模型端到端均已验证。
- AQUA-KV 的 novelty 对照、training-free 用语及论文结论需要另外撰写和实证。代码与合成测试通过不等于已经回答这些研究问题。

## 2026-11-07 修复验证

本轮修复覆盖可运行默认配置、恢复身份、完整 trace 划分、随机流隔离、视频与 cache 方法身份、quality 复用指纹、cut 分组、困难案例区间、架构公平拟合及实际量化验收。具体本地测试计数记录在 [progress.md](progress.md)；没有运行真实模型的 CUDA 长视频实验。

## AQUA-KV（Self-Forcing）

`rebuttal/experiments/aquakv/` 是官方 AQUA-KV（commit `e744183f2f3a4a90381f20a6975456871fd1ae29`）的移植：HIGGS/`HiggsQuantizer`、顺序最小二乘、在重建后的上一层上拟合。它接在 Self-Forcing 的 pre-RoPE cache 上。`python rebuttal/run.py --experiment aquakv --stage plan` 只打印比特表和集群命令，不填写 PSNR/SSIM/LPIPS。适配说明和命令在 [experiments/aquakv/README.md](experiments/aquakv/README.md)。

`aquakv_lite` 是早期消融：复用 TempoKV 分组量化，只在 trace 上算 NMSE。它不是 AQUA-KV，不能写进对照表。
