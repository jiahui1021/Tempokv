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
