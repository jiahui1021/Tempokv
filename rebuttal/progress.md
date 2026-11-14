# 本次工作记录

## 2026-11-06

- 任务限定为六项补充实验的代码修改规划，所有新增文档集中在 `rebuttal/`。
- 已读取 `ccf-experiment-designer`、`planning-with-files`，未请求新增批准、未调用外部研究服务。
- 已核对现有上一帧 predictor、质量评测和长 trace 重建入口；保留旧任务计划。
- 初步检索包含不存在的 `wrappers/`，仅影响文件列表；改用实际 `experiments/` 接入目录继续核对。
- 已完成六项实验所需代码入口核对，补充实际 fps、样本切分、BF16 内存下限、QVG/PRQ 身份和 packed span 释放限制。
- 补充检索时假设的 Self-Forcing 子包路径及 LongCat sample 路径不存在；已使用 `rg --files` 确认实际目录，未依赖这些路径形成方案。
- 开始整理主方案与逐文件修改表；本次仍只新增规划文件。
- 已写入 `README.md`、`implementation_plan.md`、`code_map.md` 和 `experiment_matrix.json`，完整覆盖六项实验。
- 方案明确 2 个主模型的分窗口实验、第三模型的独立协议、两项架构消融、默认量化目标测量，以及固定预算下 TempoKV 长短历史自对照。
- 核对到 streaming resident 统计已经包含 predictor history；方案要求避免重复相加。
- 已完成一次最终检查：六项实验完整；实验清单 JSON、文档链接、23 个已有代码路径、32/32 样本切分、窗口算术和 BF16 KV bytes 全部通过。
- `git status` 显示本次只有 `rebuttal/` 七个新增规划文件，已有跟踪代码没有差异；没有运行模型、GPU 实验或现有测试套件。
- 本次代码规划任务完成；方案中的新脚本仍为拟新增内容，不表述为已实现。

## 2026-11-06：开始代码实现

- 用户明确要求开始全面修改，统一入口集中在 `rebuttal/`。
- 三组互不重叠的模块分配独立文件责任：视频/motion、架构、固定预算；共用入口和 cache/trace 诊断由主任务负责。
- 默认 `python3` 未安装 numpy/torch/pytest，正在查找已有可用 Python 环境；不据此跳过代码实现。

## 2026-11-07：审查后修复

- 基于远端 `0d73cc931014a8391873c92cfba5f8656ee133dc` 修复；前述“仅规划”记录是历史状态，当前目录已有实验实现。
- 补齐默认 JSON 配置与实际运行文档；新增 learned-one-frame K 对照，保留 residual codec。
- 修复 resume/跨阶段输入与代码指纹、完整 trace 身份检查及 sidecar 冲突；采样不再隐藏 calibration/evaluation 重叠。
- 架构生成采用有效 span，并验收真实 K/V encode/decode；cross 消融统一目标与拟合，生产基线另列；新增纯 decode 计时。
- 修复 QVG RNG 隔离、fixed-memory 多 seed 的实际摘要/文本配对与生成完成验收。
- 修复 motion CSV 身份、跨方法 cache join、quality 来源核验、cut 分组和困难案例共同时间边界。
- 验证仅为本地 CPU、合成 trace 和小型真实视频测试；CUDA 模型生成、RAFT 与科学结论仍待正式运行。
- 最终验证：`PYTHONPATH=src python -m pytest -q rebuttal/tests` → **87 passed, 1 skipped**（11.90 秒；唯一跳过项需要真实 CUDA）。包括真实 packed codec、拟合/重建、量化目标选择来源、小型 PyAV 视频及失败恢复测试。
- 六项默认 `--stage plan` 通过；48 个 Python 模块可解析，所有默认 JSON 可解析，`git diff --check` 通过。
- 补充质量完成记录与输出 CSV 指纹：失败重跑不能认证旧分数；量化感知参数导出保留 calibration/validation 身份，最终测试检查两个来源。

## 2026-11-07：独立 AQUA-KV 全流程替换早期对照

- AquaKV 不读取 TempoKV 配置、拟合器、量化器或预测参数。
- 固定官方版本 e744183f2f3a4a90381f20a6975456871fd1ae29，包含 HIGGS、网格、回归求解器和 Apache-2.0 LICENSE；保留官方首层独立量化、重建输入与 V 输入顺序。
- 新增独立 BF16 clean trace 采集、AquaKV 校准、heldout replay、Self-Forcing 压缩缓存生成、视频质量评测和汇总；入口 `rebuttal/run_aquakv.sh` / `rebuttal/run.py --experiment aquakv --stage pipeline`。
- Self-Forcing 主模型仅增加可选缓存调用；默认路径不启用 AquaKV。测试执行原生 inference loop 的小模型采集与 clean-refresh/reset 行为，并检查不安装 TempoKV 也可独立校准/重放。
- 最终第三轮验证：`.venv/bin/python -m pytest -q rebuttal/tests tests/test_self_forcing.py tests/test_self_forcing_bf16_cache_window.py tests/test_prepare_self_forcing_trace.py` → **153 passed, 2 skipped**（27.78 秒）。跳过项需要 CUDA。
- 17 个 Python 文件通过解析，独立默认 JSON 和 6 个官方网格通过检查；脚本语法、统一 plan 入口与 `git diff --check` 通过。
- 没有执行真实 Self-Forcing checkpoint/CUDA 视频实验，没有产生真实质量或性能结果。未推送远端。

## 2026-11-07：CORAL-style block-DPLR 矩阵真实重建实验

- 按用户要求，将 CORAL 的 head 内完整矩阵加全局跨 head 低秩项用于当前同层 CrossKV。入口 `rebuttal/run.py --experiment block_dplr --stage pipeline`，默认配置 `rebuttal/configs/block_dplr.json`。
- 旧表原 trace 缺失；使用现有 Self-Forcing `[30,128,24,12,128]` trace，SHA256 `e8f78e349ff297aceaa04b000de5a52e71b91339a7a09419f35df475148804b6`，本轮 103 条拟合 / 25 条测试。明确与旧表分开，重新拟合全部 baseline，未混用旧结果。
- K4/V4、anchor4、group64、BF16 重建；固定 K temporal、V history/bias。三轮 headwise ridge / exact reduced-rank 更新；矩阵参数 FP32。
- 本轮 dense V NMSE 0.002428772；DPLR r8/r32/r128 为 0.002221015 / 0.001898709 / 0.001341994，分别降低 8.55% / 21.82% / 44.75%。矩阵大小由 22.5 MiB 增至 25.3125 / 33.75 / 67.5 MiB。
- 新增相同 ridge 的 block-only 对照 NMSE 0.002429898。校验了生产重放等价性、全部 SSE 比率、矩阵字节数和 120 组拟合目标轨迹。
- 结果、CSV、参数与汇总位于 `rebuttal/outputs/block_dplr_self_forcing_20261107/`；这是实际 trace 重建结果，尚无视频生成质量或生成速度测量，也不代表完整 CORAL 跨层编解码器复现。
- 最终验证：160 passed, 2 CUDA skips（39.42 秒）；模块/配置可解析，`git diff --check` 通过。没有推送远端。

## 2026-11-07：Per-channel temporal vs Dense temporal 真实重建比较

- 用户明确只比较 temporal 项。本实验 K KH2、V VH1 各自仅使用自身的量化重建历史，CrossKV 全部关闭，未调用 CrossKV 拟合或重建。
- 入口 `rebuttal/run.py --experiment temporal_matrix --stage pipeline`，默认配置 `rebuttal/configs/temporal_matrix.json`。
- 沿用真实 Self-Forcing `[30,128,24,12,128]` trace，103 条校准 / 25 条测试；K4/V4、anchor4、group64、BF16 重建。两类参数均按层/head 独立，两轮拟合，head-level ridge=1.0，incoming gain cap=1.0。
- K temporal：Per-channel NMSE 0.001124080，Dense 0.001719166，增加 52.94%；实际 FP32 temporal weights+bias 为 1.0546875 / 90.3515625 MiB。
- V temporal-only：Per-channel NMSE 0.004669325，Dense 0.004968584，增加 6.41%；参数为 0.3515625 / 22.67578125 MiB。
- Dense 的 K/V 输出分别有 99.9837% / 100% 达到当前增益上限，因此结果限于当前约束下的实现；不据此评价无约束 Dense 的最佳能力。
- 结果、预测参数、逐层/逐帧误差与增益诊断保存在 `rebuttal/outputs/temporal_channel_dense_self_forcing_20261107/`。只有缓存重建结果，无视频质量或生成速度结果。
- 最终验证：166 passed, 2 CUDA skips（32.54 秒）。对角特例与生产重放一致、目标/增益控制、因果历史、无 CrossKV 调用、全部 SSE/字节数/比例和来源划分均通过核对。

## 2026-11-07：AQUA-KV 移植（不覆盖 main 的 rebuttal 修复）

- 本地 `rebuttal/` 压缩包基于 `0d73cc9`，比 `origin/main`（`ace2b8e`）旧。没有整包覆盖。main 上已有文件保持原样，只在 `run.py`、`experiment_matrix.json`、`README.md`、`progress.md` 上追加 AQUA-KV 入口；Self-Forcing 只增加 `aquakv` 模式和 `aquakv_bridge.py`。
- 压缩包里的简化版（TempoKV 分组量化 + trace NMSE）改名为 `aquakv_lite`，不作为 AQUA-KV 结果。
- 正式移植在 `experiments/aquakv/`：vendored HIGGS 与 `fit_linear_regression`（commit `e744183f2f3a4a90381f20a6975456871fd1ae29`），顺序校准，Self-Forcing pre-RoPE cache。`plan` / `generate` / 无视频的 `evaluate` 不产生质量数字。
- 本机无 GPU，没有跑 Self-Forcing 生成，也没有填写 PSNR/SSIM/LPIPS。
- `python3 -m pytest -q rebuttal/tests`：108 passed，1 skipped。唯一跳过项仍是 CUDA RNG。`ace2b8e` 原有用例在装上 av、opencv、matplotlib 后为 87 passed、1 skipped；新增 11 个 AQUA-KV 测试和 10 个 `aquakv_lite` 测试。原有用例全部通过。
