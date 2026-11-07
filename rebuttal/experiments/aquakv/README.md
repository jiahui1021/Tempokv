# AQUA-KV on Self-Forcing

对照论文是 [Cache Me If You Must: Adaptive Key-Value Quantization for Large Language Models](https://arxiv.org/abs/2501.19392)。代码快照是 `official_repo/`，commit `e744183f2f3a4a90381f20a6975456871fd1ae29`（2025-11-05，`goodevening13/aquakv` 的 `grid_generator`）。`aquakv_lite` 是另一套分组标量量化消融，不是这个实验。

本目录在没有 GPU 的环境里只做算法移植和 CPU 测试。`plan` 不填写 PSNR / SSIM / LPIPS。

## 官方算法实际在做什么

量化器是 `HiggsQuantizer`（Hadamard + EDEN 格点），不是 Quanto。主路径用的格点是 `EDEN{d}-{n}`。README 默认 `EDENN_D=2`、`EDENN_N=16`（名义 2 bit）、`HADAMARD_GROUPSIZE=1024`、`RECENT_BUFFER_SIZE=128`、`PREFIX_SIZE=4`。第 0 层写死 `edenn_d=2, edenn_n=256`（名义 4 bit），除非 `--not_quantize_first_layer` 把它留成精确值。`QUIPSHARP` 的码本索引超过 uint8，官方量化器写不进去，这里不移植。

HIGGS 对每个 Hadamard 组用 float32 L2 范数做 scale，做未归一化的 Walsh-Hadamard（`H @ H = n I`），再按 `edenn_d` 找最近码字。反量化走码本、乘回 scale、转 fp16、做逆变换（`scale=1/groupsize`），最后裁掉通道填充。`quantize_dequantize` 再 cast 回输入 dtype。

预测器是带偏置的稠密 `nn.Linear`。Key：`K_l ~ Linear(K̂_{l-1})`。Value：`V_l ~ Linear(concat(K̂_l, V̂_{l-1}))`。权重布局是 `[out, in]`，即 `Y ~ X @ W.T + b`。拟合函数是 `fit_linear_regression`：`percdamp` 默认 `1e-3`，加在**全部**对角线上，包括全 1 的 bias 列；`pinverse`；累加 dtype float32。权重存成激活 dtype。`valid_nsamples` 默认 32/256，用固定 `randperm` 种子切开。验证样本不进正规方程，但仍重建并传到下一层。同一层用同一种子。

校准按层顺序进行，输入是上一层的重建，不是干净激活。`train_predictors.py` 不看 `PREFIX_SIZE` 和 `RECENT_BUFFER_SIZE`。

推理 cache（`cache_utils.py`）把前 `prefix_size` 个 token 留在普通 `DynamicCache`。其余 token 先留在 fp 缓冲里，缓冲达到 `recent_buffer_size` 再整段量化。Llama 交给 cache 的是 RoPE 之后的 key，cache 会先把 RoPE 撤销，`combine_heads` 把 `[B,H,S,D]` 合成 `[B,S,H*D]`，量化，读出时再转回去并重新加上 RoPE。刚写入的那一步返回的是原始 K/V；之后的读才是反量化。第 0 层在官方评测里被反量化成 fp 再存着（源码注释写这是为了少调试），码本本身仍然算进比特预算。

官方 `HiggsQuantizer.quantize` 把 `idx` 分配在 CPU 上，CUDA 写入会失败。`official_repo/aquakv/quantizers.py` 只改了这一处：`device=device`。CPU 上的码字与上游一致。`fast_hadamard_transform` 是 CUDA 扩展，这里用 PyTorch 的 Sylvester FWHT 顶上同一个符号。

## 移植了什么

- `HiggsQuantizer` 与 `fit_linear_regression` 直接调用 vendored 函数，不重写数学。
- 校准是同一套正规方程。视频放不进一次 `torch.stack`，所以按官方 `chunk_size` 路径逐个视频累加 `CXX`、`CXY`。验证视频不进入累加。
- 推理接在 Self-Forcing 已经存在的 pre-RoPE cache 上：干净 block 定稿后（`_incremental_tempokv` 触发的 `quantize_kv_cache`）才量化；正在去噪的 block、前 `prefix_tokens`、已定稿序列末尾 `recent_buffer_tokens` 保持 bf16。
- 比较表把 KV 码字、scale、前缀、最近缓冲、第 0 层和预测器字节都除以 K/V 元素个数。TempoKV 一侧是 legacy 非对称分组量化（bf16 scale + uint8 zero point）加上预测器。预测器文件存在时用文件大小，否则用 legacy shared-K + layer-shared CrossKV 的解析字节，并在表里标明来源。

默认比较行是 `EDEN2-64`（名义 3 bit）加 packed 索引。通道 1536 按官方 group 1024 会填到 2048，所以每个元素的信息比特大约是 4，再加 scale、第 0 层 `EDEN2-256`、前缀、缓冲和预测器。按 351×1560 token、30 层这条形状算，AQUA-KV 是 4.154 bit/元素（预测器 410,695,680 字节），legacy k4/v4 是 4.375 bit/元素（解析预测器 400,896 字节），差距约 −0.221。这是存储核算，不是测出来的质量。差距写在 `plan.json` 的 `bit_gap_aquakv_minus_tempokv`，不改码本去凑平。`uint8` 是官方物理布局（`n<256` 时一个码字占一整字节），用 `--set index_packing=uint8` 打开。README 默认的 `EDEN2-16` 作为未对齐比特的参考行，不参与主比较。

第 0 层按码本存储。官方评测把第 0 层留成 fp 的驻留量写在 `official_eval_fp_resident_total_bytes`，不拿它当压缩比。

## 不得不改的地方

- Token 是一个 latent 位置，通道是 `num_heads * head_dim = 1536`，head-major，对应官方 `combine_heads`。预测器跨层、逐位置，不跨时间。
- Self-Forcing 的 cache 已经是 qk RMSNorm 之后、RoPE 之前的 K，以及原始 V，形状 `[B,S,H,D]`。RoPE 在 `causal_model.py` 读出之后原地施加。这里不再撤销 RoPE，否则会转两次。Wan 的 qk RMSNorm 在 Llama 里没有对应物；缓存的就是 attention 在 RoPE 之前真正使用的张量。
- 因果块：当前去噪 block 保持精确（对应官方“写入当步返回原值”）。定稿后，前缀之外、最近 128 个 token 之外的位置写成码。前缀和缓冲可以切开一帧，存储按 token 而不是按整帧。序列末尾的缓冲不在结束时再压掉。`sink_size` 必须是 0，cache 不滚动。
- Cross-attention cache 不量化。
- 校准样本是一条 `(prompt, seed)` 视频，不是一个空间 token。这样验证切分不会把同一条视频的 token 拆进训练集。32 条校准里 `valid_nsamples=4`，比例仍是官方的 1/8；官方绝对数 32 会把 32 条视频全部划成验证、没有训练。
- 现有 `trace_bridge` 会抽稀 token 并把整段留在内存里。采集是单独的 `AQUAKV_COLLECT`：pre-RoPE `ChunkedKVCache`，每个干净 block 追加到 bf16 的 int16 bitcast memmap，并 `--skip_video_save`。身份是 prompt 文件里该行的 sha256，与 `prompt_records` 相同。
- 索引默认 packed，使驻留字节等于信息比特。反量化先 unpack，再调用官方 `dequantize`。
- RoPE 精度 `complex64`，与 TempoKV bridge 和 BF16 参考脚本一致。

## 公平性上保持不动的东西

评测 prompt、seed、帧数、权重、`scripts/evaluation/evaluate_quality.py`（相对 BF16 视频）、以及“KV 驻留 + 预测器”这条内存口径。质量字段在视频目录出现之前保持 null。

## 集群命令

先跑一条 prompt 计时，再按条数放大。不要把下面的磁盘和显存估计当成已测量的墙钟时间。

`python rebuttal/run.py --experiment aquakv --stage plan` 会把可复制的命令写进 `plan.json` 的 `commands`。需要的环境变量是 `CHECKPOINT`、`WAN_MODEL_DIR`，TempoKV 那一臂还要 `TEMPOKV_K_PARAMS` 和 `TEMPOKV_V_PARAMS`（legacy k4/v4 的 `k_temporal.npz` 与 CrossKV 文件）。

预期计算量：

- 校准 32 条 BF16 全历史，加上 held-out 32 条 × seed 0 × {bf16, tempokv, aquakv}，一共 128 次 351 latent 帧 rollout。
- 每 3 个 latent 帧一个 block，每 block 4 次去噪加 1 次干净刷新。
- BF16 全历史 KV 约 94 GiB（`experiment_matrix.json` 的 `full_bf16_kv_gib_estimate`），加上权重和 workspace，需要 H200 这一档（141 GiB）。本 bridge 不做 cache offload。
- 采集磁盘大约是每条视频一份 bf16 K/V memmap。32 条约 3 TiB。`plan.json` 里的 `collection_disk_bytes` 是按形状算出来的，不是 `du`。
- 校准 scratch 只留上一层的重建，32 条 K+V 大约 100 GiB，写完下一层就删。预测器 bf16 大约 0.38 GiB。
- 量化后的平均比特见 `plan.json` 的 `comparison`。主行是 packed `EDEN2-64`。

评测：

```bash
python scripts/evaluation/evaluate_quality.py <bf16/videos> <candidate/videos> --output quality.json --no-fid
```

`rebuttal/run.py --experiment aquakv --stage evaluate` 在三个视频目录都存在且含 mp4 时才调用它。否则 `quality` 为 null，`status` 为 `not_run`。
