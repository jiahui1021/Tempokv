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

## Ablations

| config | fit | K/V | anchor | notes |
|---|---|---|---|---|
| `legacy-cross-lh.json` | legacy | 4/4 | INT4 | layer-head CrossKV under the legacy family — the CrossKV sharing study |
| `legacy-cross-lh-a5.json` | legacy | 5/3 | INT5 | same, wider anchor |
| `legacy-cross-lh-a6.json` | legacy | 6/4 | INT6 | same, wider still |
| `legacy-k5v3-anchor-max.json` | legacy | 5/3 | INT5 | anchor widened to match K, layer-shared CrossKV |
| `legacy-k5v4-anchor-max.json` | legacy | 5/4 | INT5 | same |
| `legacy-k6v4-anchor-max.json` | legacy | 6/4 | INT6 | same |
| `legacy-k3v2-a16.json` | legacy | 3/2 | **bf16** | unquantized anchor, narrow residuals |
| `legacy-k4v2-a16.json` | legacy | 4/2 | **bf16** | same, wider K residual |
| `legacy-vtemporal.json` | legacy | 4/2 | INT4 | temporal V predictor instead of CrossKV |
| `legacy_kh3_vt1_int4.json` | legacy | 4/4 | INT4 | K history 3 with a temporal V predictor |
