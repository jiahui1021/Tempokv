# Paired cache recovery

The question is how much departure from the BF16 reference remains reversible
after low-bit history has already affected later clean latents and cache writes.
Each prompt/seed has one native reference and one shared low-bit warm prefix.
After the fork, the branches use the same initial noise and the runtime verifies
the same RNG state before and after every denoise and clean-write call.

| Branch | History at the fork | Subsequent writes |
|---|---|---|
| `continue` | Same low-bit warm state | Keep rounding newly aged chunks |
| `stop` | Keep existing rounded BF16 values | BF16 |
| `undo_rounding` | Restore the exact pre-rounding values saved in the warm run | BF16 |
| `reencode_own` | Replay the warm run's own clean latents in order at native absolute positions | BF16 |
| `sham` | Replay the same clean latents, then discard that cache and use the stopped state | BF16 |
| `reencode_reference` | Replay clean latents from the paired BF16 reference | BF16 |

`undo_rounding` and `reencode_reference` are diagnostic controls with
unavailable information at deployment. Undoing rounding preserves the
history-dependent hidden state that existed when each entry was first written.
Re-encoding uses only one clean write per chunk and does not re-denoise the
prefix. `sham` must match `stop` bitwise, and reference re-encoding must
match native continuation bitwise. The 16-bit setting tests the same branch
paths with identity rounding.

The native attention kernel reads the full history. Key rounding is applied in
post-RoPE coordinates and values use their native coordinates. Current and
recent chunks are protected during rounding. A span is rounded once when it
first ages out of protection. Every branch has the same live token positions
and the same future query–key count. Re-encoding adds separate prefix work.
The native BF16 backing and CPU diagnostic snapshots remain allocated. These
experiments report neither packed-cache savings nor perceptual video quality.

The causal fork with a shared low-bit prefix, exact undo-rounding, and sham
controls addresses reversibility. LongLive re-caches on prompt switching,
ID-Forcing changes cache-write provenance and conditioning for extrapolation,
and the 33-method benchmark varies when low-bit quantization is applied.

- [LongLive, §3.1](https://arxiv.org/html/2509.22622v2)
- [ID-Forcing, §3.1–3.2](https://arxiv.org/html/2610.03120v1)
- [33-method benchmark, Appendix A](https://arxiv.org/html/2603.27469v1)

## Run

```bash
python scripts/research/cache_recovery.py --bits 4 --dry-run

CUDA_VISIBLE_DEVICES=0 python scripts/research/cache_recovery.py \
  --checkpoint "$SELF_FORCING_CHECKPOINT" --model-root "$WAN_MODEL_DIR" \
  --prompts "$PROMPTS" --prompt-start 2 --prompt-count 2 --seed 0 \
  --bits 4 --fork-chunk 6 --future-chunks 8 --protect-chunks 2 \
  --output-dir reports/research/cache-recovery-pilot-4bit

python -m pytest scripts/tests/test_research_cache_recovery.py -q
```

The CLI requires the repository's native Self-Forcing inference environment.
The launch manifest and one immutable JSON document per prompt/seed use
`research.cache_recovery.v1`. A document stores the before-fork clean-latent
errors, branch-level costs, K and V fork errors for each absolute prefix
chunk, and an ordered vector of future clean-latent relative L2 values.
Vector index 0 is the first chunk after the fork. The two branch hashes of
the prefix distinguish the observable warm history from the input that was
used to construct the oracle cache. Compare branches within the same prompt,
seed, bit width and fork location. Compute prompt-level statistics before
aggregating repeated chunks.
