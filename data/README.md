# Model data

Reusable trace and fitted parameters are grouped by model:

```text
data/
└── <model>/
    ├── trace/
    │   ├── raw/
    │   ├── output/
    │   └── clean_trace.npz
    └── params/
        └── <parameter-profile>/
            ├── k_temporal.npz
            └── v_cross_kv.npz
```

Generated videos and evaluation reports belong in `results/`, not here.
