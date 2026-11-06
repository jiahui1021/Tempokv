# 逐文件修改清单

状态：规划。下面“新增”路径尚未实现；现有路径均在本机核对。所有实验专用代码、配置、测试和输出都集中在 `rebuttal/`。

## 已有代码与复用方式

| 现有文件 | 已有能力 | 本次复用边界 |
| --- | --- | --- |
| `src/tempokv/config.py` | `identity/taylor/taylor2`、CrossKV 组合、K/V 历史、参数共享方式 | 复用生产配置；实验性新架构用独立配置，不向旧配置塞未知字段 |
| `src/tempokv/fitting.py` | `build_fixed_temporal_parameters`、`fit_predictors`、闭环拟合 | 复用现有拟合；低秩、scalar/dense temporal 的新求解放在本目录 |
| `src/tempokv/parameters.py` | `TemporalParameters` 与 dense `CrossKVParameters` | 现有参数原样加载；新增架构另存带版本号的参数，不冒充旧格式 |
| `src/tempokv/codec.py` | 真实量化、参考预测/解码、`PackedTensor.storage_bytes()` / `logical_storage_bytes()` | 作为重建正确性与存储核算基准；不把低秩展开成 dense 后声称省内存 |
| `src/tempokv/quantization.py` | 实际 residual quantization / dequantization | 各项实验共用；不另写一个近似量化器替代正式测量 |
| `src/tempokv/attention_objective.py` | 可微闭环、residual detach 约定 | 仅为可选量化目标扩展提供计算工具，不将其现有 attention 目标标为量化目标 |
| `src/tempokv/runtime/cache.py` | frame cache、绝对位置、逐 cache decoder、完整 span 读取 | 复用位置与读取语义；固定预算需要真正释放 span 及其所有持有者 |
| `src/tempokv/runtime/streaming.py` | 增量编码、BF16 lag、`event_hook`、history 占用 | 复用事件与峰值记录；其 `_observe_resident` 已计入 history，不再次相加 |
| `src/tempokv/runtime/metrics.py` | current / peak / cumulative bytes、参数大小 | 复用指标，明确三者区别；新增分项快照在本目录输出 |
| `scripts/evaluation/evaluate_quality.py` | 流式配对、RGB PSNR、SSIM、LPIPS | 复用 `QualityBackend` 及其预处理；新增逐帧/分窗口记录，不复制 LPIPS 实现 |
| `scripts/evaluation/evaluate_vbench.py` | 项目既有 VBench 入口 | 用于真实长视频生成后的质量评价，记录协议与版本 |
| `scripts/paper_experiments/predictor_ablation/run.py` | 上一帧复用、Taylor 与 learned 对照、拟合和生成 | 选择已有变体或调用其底层入口；旧 runner 默认同时改 K/V，不能原样当作单轴架构实验 |
| `scripts/paper_experiments/history_ablation/evaluate_cache.py` | prediction / reconstruction NMSE | 复用定义；扩展为逐事件、逐 frame 的误差记录 |
| `scripts/tools/analyze_predictor.py` | 返回 prediction、residual、reconstruction | 用作首版诊断和交叉检查；正式记录应与 codec 的实际计算核对 |
| `scripts/paper_experiments/long_horizon/methods.py` | 长 trace 闭环重建与逐帧曲线 | 用于重建分析；不能用于声称已完成固定预算生成 |
| `experiments/Self-Forcing/tempokv_adapter/{bridge,generate,trace_bridge}.py` | Self-Forcing 生成、编码、采样接入 | 第一批端到端实验主接入；trace 模式替换 VAE，视频需要单独配对生成 |
| `experiments/Causal-Forcing/causal_forcing/tempokv_adapter/{bridge,generate}.py` | Causal-Forcing 生成和压缩 | 第二模型；短窗口已有 benchmark 专用限制，按其公开实验模式使用 |
| `experiments/LongCat/longcat/{bridge,generate}.py`、`experiments/LongCat/inference.py` | LongCat 分段生成、cache 接入 | 第三模型的 trace 诊断与可选生成；单列 fps、分段、conditioning 设置 |

## 新增文件及责任

| 新增路径 | 责任 | 支持实验 |
| --- | --- | --- |
| `rebuttal/run.py` | 选择 experiment/stage，输出命令清单、续跑和汇总 | 全部 |
| `rebuttal/configs/*.json` | 冻结每项实验的具体比较设置；包含主方案确定后的真实路径 | 全部 |
| `rebuttal/common/manifest.py` | prompt/seed/model/checkpoint/config/参数/视频/trace 身份、摘要、状态 | 全部 |
| `rebuttal/common/video_metrics.py` | 一次解码产生逐帧 SSE/SSIM/LPIPS、timestamp 和 window 索引 | 1、2、3、4、6 |
| `rebuttal/common/cache_metrics.py` | prediction、residual quantization、实际 reconstruction 误差与分项能量 | 1、3、4、5 |
| `rebuttal/common/trace.py` | 读取已有 trace，保存 prompt、绝对帧、token 位置等身份；分片处理 | 1、3、4、5 |
| `rebuttal/common/statistics.py` | 按 prompt 配对汇总、bootstrap CI、Spearman、样本数和缺失原因 | 2、3、4、5、6 |
| `rebuttal/adapters/{self_forcing,causal_forcing,longcat}.py` | 独立进程生成、参数映射、能力与长度检查 | 生成与 trace |
| `rebuttal/adapters/observer.py` | 在 clean cache 写入/编码处只读记录；GPU 上归约，输出有限统计 | 3、5、6 |
| `rebuttal/adapters/experimental_codec.py` | scalar/dense temporal、factorized cross 的统一 PyTorch 参考编码/解码 | 4 |
| `rebuttal/adapters/qvg.py` | 本机真实 QVG/PRQ codec、packed state 计数、相同 Self-Forcing 生成器接入 | 6 |
| `rebuttal/experiments/previous_frame.py` | 生产、K-only identity、K/V identity 等明确的比较轴 | 1 |
| `rebuttal/experiments/temporal_window.py` | 时间窗口、prompt-window 明细、汇总与曲线 | 2 |
| `rebuttal/experiments/motion/{flow,groups,camera,cases,report}.py` | BF16 flow、分组、camera/local proxy、困难片段、关联分析 | 3 |
| `rebuttal/experiments/architecture/{temporal,cross_low_rank,fit,report}.py` | 两项独立架构实验的拟合、重建、生成比较 | 4 |
| `rebuttal/experiments/quant_calibration.py` | 默认只测现有参数，区分量化失真与最终存储失真 | 5 |
| `rebuttal/experiments/quant_aware_fit.py` | 可选扩展；与默认分析入口分开 | 5 扩展 |
| `rebuttal/experiments/memory/{accounting,capacity,budget_manager,generation,report}.py` | 实际 bytes、可行历史长度、真正释放历史、双重对照和结果 | 6 |
| `rebuttal/tests/test_*.py` | 时间/误差/预算/位置等必要正确性检查 | 各项关键风险 |
| `rebuttal/outputs/<run_id>/` | 输入清单、配置、逐条记录、统计、图表、日志和失败原因 | 全部 |

各目录补齐必要的 `__init__.py`。专用可选依赖写入 `rebuttal/requirements.txt`；模型现有环境继续独立使用，不混装三个生成模型的依赖。

## 可能需要修改的公共代码

优先从独立 worker 调用已有接口。只有已有接口不足以表达正确行为时，修改下表的接入；实验逻辑仍留在 `rebuttal/`。默认配置继续走原有计算。

| 现有文件 | 拟议最小改动 | 触发条件 |
| --- | --- | --- |
| `experiments/Self-Forcing/tempokv_adapter/bridge.py` | 可选、显式传入 `codec_factory` / observer；继续复用原生成器 | 架构变体需要真实生成，现有构造只能加载旧参数 |
| `experiments/Causal-Forcing/causal_forcing/tempokv_adapter/bridge.py` | 同类可选只读 observer 接入 | 第二模型需要同轨迹 cache 误差，现有事件不足 |
| `src/tempokv/runtime/streaming.py` | 在编码事件暴露身份、误差归约所需只读输入/结果；支持预算管理器清除被淘汰的开放状态 | motion/量化诊断或真正释放历史无法只靠现有 `event_hook` 完成 |
| `src/tempokv/runtime/cache.py` | 明确释放完整独立 span 的接口，维护 `global_frames` 与 K/V 配对 | 固定预算生成需要释放旧 span；禁止仅改变 attention mask |

现有 Triton 的权重布局和 `PackedTensor` 并不支持所有新架构。首版研究比较全部采用同一个参考后端；在没有真实适配与测量前，不写成生产加速已支持。若共享 hook 最终不需要，则不修改对应公共文件。

## 本机相邻代码依赖

- RAFT：`../qvg/raft/`。以显式路径、commit、checkpoint SHA-256 加载，不复制进仓库。
- QVG/PRQ：`../qvg/Quant-VideoGen/scripts/Self-Forcing/run_qvg.sh`、`quant_videogen/real/prq.py`、`compress.py` / `uncompress.py`。固定 `quant_type`、bits、stages、centroids、block size、KMeans iterations。
- QVG 重建参考：`../kv-quant-4-videogen/scripts/paper_experiments/reconstruction/methods.py`。该接口只返回重建张量，不能直接获得完整 memory accounting。
- 各 checkout 的导入名可能相同，必须使用独立 worker 进程；不得让一次 import 顺序隐式决定加载哪个生成器或 codec。
