# Causal Forcing 运行时

本目录是 TempoKV 仓库内自带的 Causal Forcing 推理运行时。上游版本与本地改动
见 [UPSTREAM.md](UPSTREAM.md)。不需要 clone 外部 `thu-ml/Causal-Forcing`，
不需要设置外部 `PYTHONPATH`。

```text
experiments/Causal-Forcing/
├── inference.py                     BF16 / trace / TempoKV 共用的生成入口
├── configs/                         上游 Causal Forcing 配置
└── causal_forcing/
    ├── wan/ utils/ pipeline/ demo_utils/   vendored 推理运行时
    ├── kv_cache.py                  指向 tempokv.runtime.cache 的再导出
    └── tempokv_adapter/             TempoKV 桥接、trace 采集、入口驱动
```

公开命令位于 [scripts/Causal-Forcing/](../../scripts/Causal-Forcing/)。

## KV cache 语义

Causal Forcing 每个 block 的流程是：

```text
4 次 noisy denoising forward     → ephemeral，只服务当前 block
1 次 clean/context refresh       → finalize，成为长期历史
```

cache 只保存 **pre-RoPE normalized K** 与 **raw V**。RoPE 在读出之后按每个
span 的绝对帧位置重新施加，V 从重建出的 pre-RoPE K 解码。三条不变式：

1. 写进 cache 的 K 是 `norm_k(k_proj(x))`，不是 `causal_rope_apply(...)` 之后的 K。
2. V 的 decode source 必须是解码后的 K，闭环在 RoPE 之前完成。
3. 哪一次 forward 算 clean 由 pipeline 显式声明，不在 attention 里根据 timestep 猜。

`finalize` 后的帧先以 BF16 累积，攒满一个 span 再整体编码，因此一个 span 只有
一个 anchor，span 内 `frame 0 → 1 → 2 → 3 → ...` 的预测链不会被 block 边界打断。
最近不足一个 span 的帧保持 BF16。

## 尚未实现

rolling eviction、sink 保留和 long video 还没有接入。cache 已经保存绝对帧位置
和独立的 predictor span，加这些功能不需要推倒重写；在那之前，
`sink_size > 0` 与局部注意力窗口同时使用会直接报错，而不是给出与上游不同的结果。
