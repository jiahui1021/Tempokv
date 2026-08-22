# 实验一：固定长度开销与长度扩展曲线

这套代码严格分开回答两个问题：

1. `fixed_717.csv`：717 帧时，各方法的绝对耗时和拆解是多少。
2. `horizon.csv`、`horizon_absolute_log.svg`、`horizon_overhead_share.svg`：耗时和 cache encode/decode 占比如何随生成长度变化。两张图独立输出，不使用双轴，也不把两个问题混成一张图。

## 运行协议

- 方法固定为 BF16、RTN INT2/INT4、QVG INT2/INT4、TempoKV INT2/INT4。
- 长度固定为 180、360、540、717、1050、1400 帧，每个方法和长度恰好 5 次。
- 每次正式测量前紧邻运行一轮完整 warmup，warmup 报告会校验后丢弃。
- 首次编译单独运行、校验并写入 `compilation.csv`，不进入 warmup 或 E2E。
- E2E 必须包含 VAE decode，表中另列 cache encode、cache decode、VAE decode 和其余 denoising 时间。
- 每个计时段必须在前后调用 `torch.cuda.synchronize()`。正式模式拒绝只用异步 CUDA event 的分段报告。
- 所有方法必须报告同一组冻结字段：Self-Forcing、832×480、prompt SHA-256、seed、A100 80GB、GPU UUID、CUDA、attention backend 和 attention lifecycle。GPU UUID 用来确认七种方法确实运行在同一张卡上。
- paired BF16 overhead 按同长度、同 repeat 逐对相减，再报告中位数与 p10–p90。不会用“两个汇总值相减”代替配对统计。

## 准备 manifest

复制 `manifest.example.json`，填写 prompt SHA-256、CUDA、attention 信息，以及七个方法的 adapter 命令。命令是 argv 数组，可用以下占位符：

```text
{method} {phase} {frames} {repeat} {seed}
{report_path} {output_dir} {session_id} {session_scope}
```

同样的信息会注入环境变量：

```text
PAPER_EXPERIMENT_METHOD
PAPER_EXPERIMENT_PHASE       compilation | warmup | measure
PAPER_EXPERIMENT_FRAMES
PAPER_EXPERIMENT_REPEAT
PAPER_EXPERIMENT_SEED
PAPER_EXPERIMENT_REPORT
PAPER_EXPERIMENT_OUTPUT
PAPER_EXPERIMENT_SESSION_ID
PAPER_EXPERIMENT_SESSION_SCOPE
PAPER_EXPERIMENT_RUNTIME_INSTANCE_ID
PAPER_EXPERIMENT_FROZEN_JSON
```

adapter 必须保持模型加载和正式计时的边界，并把报告写到
`PAPER_EXPERIMENT_REPORT`。当前编排器会启动多个 adapter 进程，因此 manifest 只能如实填写
`coordinator_session_multiple_processes`；填写 `single_process` 会直接拒绝。若论文协议要求七种方法在同一 Python 进程内常驻，需要各 runtime 另行提供常驻 worker，不能由本编排器谎报。
warmup 与紧随其后的 measure 会收到同一个 `PAPER_EXPERIMENT_RUNTIME_INSTANCE_ID`，严格输入必须原样报告为
`runtime_instance_id`。adapter 只有在两次调用确实连接同一个已加载的常驻 runtime 时才能回报该标识；若两条命令各自重新加载模型，不能把编排器提供的标识当作同一实例的证明。

## 报告 adapter

正式 runtime 先写一个严格 timing JSON：

```json
{
  "effective_frames": 717,
  "runtime_instance_id": "persistent-worker-uuid",
  "observed_frozen": {
    "model": "Self-Forcing",
    "width": 832,
    "height": 480,
    "prompt_sha256": "...",
    "seed": 0,
    "gpu_name": "NVIDIA A100-SXM4-80GB",
    "gpu_uuid": "GPU-...",
    "cuda_version": "...",
    "attention_backend": "...",
    "attention_lifecycle": "..."
  },
  "timings_s": {
    "e2e": 52.0,
    "cache_encode": 1.0,
    "cache_decode": 2.0,
    "vae_decode": 7.0,
    "denoising_other": 42.0
  },
  "synchronization": {
    "api": "torch.cuda.synchronize",
    "before_each_segment": true,
    "after_each_segment": true
  },
  "exclusions": {"model_loading": true, "compilation": true}
}
```

转换为统一报告：

```bash
python scripts/paper_experiments/latency/report_adapter.py \
  --format strict --input /path/to/runtime_timing.json
```

首次编译输入只包含：

```json
{
  "one_time_compilation_s": 23.4,
  "observed_frozen": {
    "model": "Self-Forcing",
    "width": 832,
    "height": 480,
    "prompt_sha256": "...",
    "seed": 0,
    "gpu_name": "NVIDIA A100-SXM4-80GB",
    "gpu_uuid": "GPU-...",
    "cuda_version": "...",
    "attention_backend": "...",
    "attention_lifecycle": "..."
  }
}
```

```bash
python scripts/paper_experiments/latency/report_adapter.py \
  --format compilation --input /path/to/compilation.json
```

现有 Self-Forcing `pipeline_report.json` 和 `profile/rank-0.json` 可以用于流程联调：

```bash
python scripts/paper_experiments/latency/report_adapter.py \
  --format self-forcing-profile \
  --pipeline-report /path/to/pipeline_report.json \
  --profile /path/to/profile/rank-0.json
```

这个兼容入口会如实标记分段为异步 CUDA events。它只能搭配 `run.py --pilot`，正式模式会拒绝。

## 干跑与 CPU 自检

干跑只生成完整执行计划，不执行 GPU 命令：

```bash
python scripts/paper_experiments/latency/run.py \
  --manifest scripts/paper_experiments/latency/manifest.example.json \
  --dry-run
```

synthetic 模式会生成 210 个带水印的假测量，用来验证聚合、CSV 和绘图。不能作为论文结果：

```bash
python scripts/paper_experiments/latency/run.py \
  --manifest scripts/paper_experiments/latency/manifest.example.json \
  --synthetic
```

## Self-Forcing 帧数限制

当前仓库的 Self-Forcing runtime 只接受满足 `(FRAMES + 3) % 12 == 0` 的最终视频帧数。指定长度中只有 717 能直接运行；180、360、540、1050、1400 不能静默改成邻近值，否则横轴和实测长度不一致。正式 run report 要求 `effective_frames == requested_frames`，因此会拒绝裁剪或取整后的冒充结果。完成这五个点需要 runtime 先支持精确长度，或在论文协议中公开改用一组合法长度。
