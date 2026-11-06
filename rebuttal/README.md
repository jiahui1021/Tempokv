# TempoKV 补充实验代码规划

本目录对应用户要求的 `tempokv/rebuttal`，位于仓库根目录，与 `src/tempokv/` 并列。

**当前交付是修改方案，尚未实现新实验、启动生成或产生实验结果。** 现有代码保持不变。

| 文件 | 内容 |
| --- | --- |
| [implementation_plan.md](implementation_plan.md) | 六项实验的比较设置、指标、文件划分、依赖和验收标准 |
| [code_map.md](code_map.md) | 已核对的现有代码、新增模块，以及少量可能需要修改的公共接入 |
| [experiment_matrix.json](experiment_matrix.json) | 可供后续实现读取的实验设计清单；不是已经可运行的配置 |
| [findings.md](findings.md) | 本机代码核对依据 |
| [task_plan.md](task_plan.md)、[progress.md](progress.md) | 本次规划的范围与检查记录 |

建议采用以下安排：

1. **共用数据与统计代码**：一次配对生成 BF16/TempoKV，复用于分窗口、运动分组、困难片段；同一份 held-out trace 复用于架构、量化目标和重建统计。
2. **Temporal-window 主实验测两个模型**：Self-Forcing、Causal-Forcing，各 32 条独立评测 prompts，1401 帧、seed 0。LongCat 作为第三模型扩展，单列其分段生成与时长设置。两个主模型共享 Wan 系列结构，结果不能写成不同 backbone 的普遍验证。
3. **上一帧复用直接复用已有实现**：已有 `identity` / `cross_kv_identity`，新增工作集中在统一比较设置和输出。
4. **Temporal 架构与 K→V 架构独立比较**；低秩拟合使用同一 calibration set，SVD 只作初始化或另列参考。
5. **量化目标先做误差测量**，默认不增加新的拟合算法；现有拟合是否足够应由实测支持。
6. **固定预算先做真实占用核算，再做真实视频生成**。第一版生成比较放在 Self-Forcing；BF16、QVG、TempoKV 使用相同生成器、prompts、seed 与生成长度。

需要在实现中固定的口径：

- 1401 帧在 16 fps 下约 87.56 秒，前五个窗口各 15 秒，第六个窗口保留实际剩余时长；不能六等分后都标成 15 秒。
- Self-Forcing 的完整历史 BF16 参考仅 K/V 数据就约 94 GiB，还需要权重与临时空间；短窗口 BF16 不能替代完整历史参考。
- 生成时保留的 BF16 尾部、predictor history、量化辅助数据和预分配容量都影响可用历史长度；结束后 flush 的大小不足以回答固定预算问题。
- 所有缺失飞书表格中的比较行都在主方案中标为“建议补全”，不推定为用户原表内容。
