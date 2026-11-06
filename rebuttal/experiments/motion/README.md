# Motion 与分窗口输入约定

统一入口为 `python rebuttal/run.py`，本目录通过 `run(config, stage, output_dir)` 调用。

配对输入可以直接写 `pairs`，也可以用 `pairs_manifest` 指向 `{"pairs": [...]}`；两个目录的相对视频文件名完全一致时，可设置 `reference_dir` / `candidate_dir`。推荐显式输入：

```json
{
  "pairs": [
    {
      "model": "self_forcing",
      "prompt_id": "38",
      "seed": 0,
      "sample_id": 0,
      "method": "tempokv_int4",
      "comparison_id": "full_history",
      "reference": "/absolute/bf16/38.mp4",
      "candidate": "/absolute/tempokv/38.mp4",
      "cache_events": "/absolute/cache_events.jsonl"
    }
  ],
  "metrics": ["psnr", "ssim", "lpips"],
  "video_decoder": "pyav",
  "window_seconds": 15,
  "expected_frames": 1401,
  "flow": {
    "backend": "raft",
    "raft_root": "/absolute/qvg/raft",
    "checkpoint": "/absolute/raft-things.pth",
    "device": "cuda",
    "iterations": 20,
    "resize_width": 416,
    "forward_backward": true
  },
  "camera": {"grid_step": 16, "ransac_threshold": 3, "min_inlier_ratio": 0.3},
  "cuts": {"mean_absolute_rgb_threshold": 60},
  "difficult_cases": {"clips_per_category": 3, "clip_seconds": 3, "export_clips": true}
}
```

路径示例需要替换为实际本机数据。RAFT 仅加载显式本地 checkpoint，不自动下载；`flow.backend=farneback` 可显式使用 OpenCV 传统光流，结果注明算法。不同 flow 后端不合并统计。

默认 PyAV 读取并核对实际 PTS。`video_decoder=opencv` 是显式的等帧率假设，结果记录 `frame_index/fps` 时间口径。默认三项指标缺少依赖会失败；`metrics=["psnr"]` 仅供明确选择的检查或分阶段计算，不伪装成已计算 LPIPS。

生成身份由配对清单声明。可另提供 `reference_metadata` 与 `candidate_metadata` sidecar 路径；同时提供时，默认核对 model、prompt_id、seed、frames、width、height、attention_window、sink_size；可通过 `matched_metadata_fields` 冻结实际字段名称。缺少 sidecar 的结果保留 `identity_validation=declared_in_pair_manifest`。

目录自动配对优先读取各目录 `summary.json`（或其父目录），从 `rank_summaries[].videos[]` 的 `prompt_index`、`sample_index`、`noise_seed` 获取真实身份并要求两边一致。也可设置 `video_index_manifest`，内容为 `{"videos": [{"relative_path": "0_0.mp4", "prompt_id": 0, "sample_id": 0, "seed": 0}]}`。显式共享索引仍标为用户声明；匹配生成摘要只代表身份索引匹配，不代表全部生成设置已核实。无索引时保留完整 stem 作为 prompt_id 并标记未验证，不猜测 `0_0` 的含义。

Cache 记录必须具备以下信息，缺失时 fidelity 和 motion 仍完成，cache 关联明确记缺测：

- `model`、`prompt_id`、`seed`、可选 `sample_id`，且匹配视频身份。
- `metric_scope`，默认只关联 `bf16_trace_replay`。
- `role` 为 K/V；`error_sse`、`signal_sse`、`prediction_sse`、`is_anchor`。
- 明确的半开 RGB frame 范围 `pixel_start_frame` / `pixel_end_frame`。兼容同义字段 `pixel_start` / `pixel_end`；不从 `absolute_latent_frame` 猜测。

完整视频的 cache correlation 要求 K/V 各自覆盖完整质量计算范围。Camera/local proxy 关联把连续光流聚合到同一个显式 cache RGB 区间；切镜头或估计失败的区间不混入。多个 layer/head 的 NMSE 通过误差与能量的分子/分母汇总。

Cache join 同时核对事件中已有的 `method`、`variant`、`comparison_id`。默认与 pair 的对应身份匹配；variant 默认采用 pair 的 `variant` 或 `method`。如果视频方法标签是 `tempokv_int4`、replay 事件标签是 `production`，须在 pair 中显式设置 `cache_variant: "production"`。`cache_method`、`cache_comparison_id` 可声明其他命名映射。事件同一身份中混有带标签和不带标签的配置会拒绝关联。旧 online observer 文件没有实验标签时，仍由 pair 的独立 `cache_events` 文件声明配置，输出标记 `declared_single_configuration_file`；这类文件不可拼接不同配置。

`case_candidates.json` 仅为基于 BF16 的自动候选，不能当作已核实遮挡或 scene change。人工标注可使用 `difficult_cases.annotations` 指向 `{"cases": [{"case_id": "...", "manual_verified": true, "category": "...", "notes": "..."}]}`；类别与起止时间可修正。`difficult_cases.json` 区分 verified 与 candidate-only。输出 MP4/PNG 供展示，重新编码片段不参与定量 fidelity。

困难案例若具备完整 K/V 时间映射，会向外扩展到最近的共同显式 cache 边界，并在同一个实际 RGB 区间重新汇总 fidelity 和导出片段。输出同时保留 `requested_*` 与实际起止时间；不按像素长度拆分或估计 cache SSE。缺少完整映射时保留原片段并明确记录 cache 缺测。

可选 `quality_output_dir` 复用已生成的 `frames.csv` 与 `videos.csv`。必须同时存在成功评测后生成的 `quality_completion.json`（绑定输入声明和输出 CSV 指纹）及本版本生成的 `video_pairs.json`，包含原视频路径、字节数、SHA-256 和指标/decoder 合同；换视频、缺视频、换指标配置或缺旧指纹均需重算质量，不能仅凭相同 prompt ID 复用。

主 motion 分数及分位数组仅使用非 cut 帧对；跨 cut 不估计光流。全部帧对均为 cut 的视频标为 `unclassified`，仍保留其质量和困难案例信息。可选 `motion_groups` 导入已冻结阈值，其 `motion_basis` 必须为 `non_cut_adjacent_frames`；旧含 cut 阈值须重算。缺失模型的阈值会依据该模型 BF16 单独计算；相同 magnitude 的 ties 不为凑齐组数而拆分。
