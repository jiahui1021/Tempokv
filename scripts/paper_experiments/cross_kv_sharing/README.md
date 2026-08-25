# CrossKV sharing experiment

This formal Self-Forcing study compares four CrossKV projection layouts while
keeping the legacy shared-K setup unchanged:

- `layer_head`: independent matrix for every layer and head
- `layer_shared`: one matrix per head, shared across layers
- `head_shared`: one matrix per layer, shared across heads
- `shared`: one matrix shared across all layers and heads

The generated configs differ only in `cross_kv_weight_mode`. The formal
generation protocol creates one BF16 reference and four TempoKV videos with the
same prompt, seed, frame count, checkpoint, and WAN model directory.

`configs/legacy.json` selects `layer_shared` explicitly. `run_study.py` retains
the exhaustive sharing study. The final latency command is intentionally much
smaller: it fits `layer_shared_tf32`, generates exactly one candidate video,
and compares it with an existing fresh `layer_shared + IEEE` pipeline report.

Recommended final command:

```bash
CALIBRATION_TRACE=/traces/calibration.npz BASELINE_REPORT=/runs/layer_shared/pipeline_report.json CHECKPOINT=/models/self_forcing_dmd.pt WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B PROMPTS=assets/moviegen128.txt OUTPUT_DIR=results/paper/layer_shared_tf32_one_video bash scripts/paper_experiments/cross_kv_sharing/run_one_video.sh
```

The candidate run always enables benchmark and runtime profiling. It rejects a
baseline that is not a fresh one-video benchmark or that uses a different
checkpoint, prompt, frame count, seed, prompt index, GPU count, model config,
attention window, or span. The output `latency_comparison.json` reports raw
end-to-end and `decode_v` times, both speedups, Triton route counts, and the
final `latency_reduced` decision.

Running `run_study.py` directly without `--execute` remains a model-free plan
mode that writes the resolved configs, manifest, and planned comparison files.

`layer_shared` retains one matrix per head during every layer decode, so it does
not reduce CrossKV FLOPs. The TF32 row tests the lowest-risk compute-side
latency option already supported by the Triton kernel. Treat a microbenchmark
win as provisional until both `decode_v_ms` and `generation_seconds` improve
without an unacceptable quality regression.

Before the full video run, the V kernel can be screened with the same production
shape and effective launch-count candidates:

```bash
python scripts/tools/benchmark_tempokv_kernels.py --route v --units 24 --sweep both --persistent-units 2 4 8 --block-s 8 16 32 --block-d 16 32 64 --num-warps 4 8 --num-stages 2 3 4 --cross-matmul-precision ieee --json results/crosskv-ieee.json
python scripts/tools/benchmark_tempokv_kernels.py --route v --units 24 --sweep both --persistent-units 2 4 8 --block-s 8 16 32 --block-d 16 32 64 --num-warps 4 8 --num-stages 2 3 4 --cross-matmul-precision tf32 --json results/crosskv-tf32.json
```
