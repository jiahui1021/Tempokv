# Fixed-memory video identity and references

QVG PRQ runs inside an isolated PyTorch RNG context. Its KMeans initialization
does not advance the video's denoising-noise stream. CUDA integration still
requires validation on the execution GPU; CPU tests do not measure video quality.

Successful generation verifies every video's native `summary.json` record
against the job's prompt file, actual per-prompt noise seed, and sample index.
It writes `videos/generation_identity.json`, retaining prompt text and the
summary digest. Evaluation and resume reject changed or missing identities.
The identity uses actual `noise_seed`, which is `base_seed + prompt_index *
1_000_003` in the bundled Self-Forcing runner, rather than the CLI base seed.

For one seed, `bf16_reference_dir` may point directly to reference videos or to
their parent containing `videos/`. For multiple seeds, use:

```text
references/
  seed_0/videos/{*.mp4,summary.json,generation_identity.json}
  seed_1/videos/{*.mp4,summary.json,generation_identity.json}
```

`seed_<seed>` selects a candidate reference cohort; its directory name is never
accepted as identity evidence. Every pair must match the generation summary's
prompt index, actual noise seed, sample index, and prompt text. Filenames may
differ. A wrong seed or reordered/different prompt cohort is rejected.

Older references that have native `summary.json` but no identity snapshot must
also specify `bf16_reference_prompts`, the original reference generation prompt
file. It is not inferred from the candidate's prompts. A manually supplied file
declares the prompt provenance; the native summary verifies indices and seeds.
References with no native summary cannot be automatically paired by filename.

The default protocol still uses a declared complete-history BF16 reference for
all jobs. This measures fidelity to that reference, including history-window
effects; it is not labelled pure quantization error or ground-truth quality.
The reference's complete-history setting remains a protocol requirement, not
something proved by the native summary's identity records.
