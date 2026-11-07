# AQUA-KV source

Source: https://github.com/goodevening13/aquakv

Pinned commit: `e744183f2f3a4a90381f20a6975456871fd1ae29`.

`quantizers.py`, `edenn.py`, `linear_utils.py`, `grids/*.pt` are from this commit, licensed under Apache-2.0 (included in `LICENSE`). No model weights or calibration data are included.

Local changes: Hadamard imports use `hadamard.py`, which calls the original fast CUDA transform on CUDA and a PyTorch reference transform on CPU. Quanto is imported lazily, so the HIGGS workflow does not require unused Quanto dependencies. Progress display uses an optional tqdm wrapper. The HIGGS formulas, grids, scale precision, FP16 inverse transform, and the regression implementation are otherwise unchanged. CUDA numerical parity is tested when `fast_hadamard_transform` is available.

The video adapter keeps pre-RoPE normalized Wan K, uses `[current reconstructed K, previous reconstructed V]` for V prediction, and fits its own predictors with upstream `fit_linear_regression`. It never loads TempoKV predictors, configuration, quantization or fitting code.
