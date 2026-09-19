# Codec configs

One JSON per codec setting. Pick one with `CONFIG=configs/<name>.json`; the
bit widths can also be overridden per run with `BITS` / `K_BITS` / `V_BITS` /
`ANCHOR_BITS`, so a config only has to pin what those variables cannot.

Fitted parameters are stamped with the config's signature and refuse to load
against a different one, so a config change means refitting.

## Production

| config | fit | K/V | anchor | K hist | notes |
|---|---|---|---|---|---|
| `legacy.json` | legacy | 4/4 | INT4 | 2 | the default; shared K predictor, layer-shared CrossKV |
| `native.json` | native | 4/4 | INT4 | 2 | per-layer/head K predictor, layer-head CrossKV, TF32 cross matmul |

## Structure ablations

Each changes the fitted parameter layout, which no override can do.

| config | fit | K/V | anchor | notes |
|---|---|---|---|---|
| `legacy-cross-lh.json` | legacy | 4/4 | INT4 | layer-head CrossKV — the CrossKV sharing study |
| `legacy-k-lh.json` | legacy | 4/4 | INT4 | per-layer/head K predictor — the K sharing study |
| `legacy-k-lh-cross-lh.json` | legacy | 4/4 | INT4 | both layer-head; differs from `native.json` only in fitting math and IEEE matmul |

Bit-width and V-predictor ablations need no file of their own: run `legacy.json`
with `K_BITS` / `V_BITS` / `ANCHOR_BITS` (16 keeps the anchor in bf16),
`V_PREDICTOR=temporal`, or `K_HISTORY_LENGTH`.
