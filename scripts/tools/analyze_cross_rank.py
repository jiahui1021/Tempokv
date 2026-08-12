#!/usr/bin/env python3
"""Decide whether a low-rank all-head Cross map can replace the per-head one.

The shipped Cross predictor is block diagonal: one dense ``[D, D]`` map per
head, which cannot mix heads at all.  A low-rank all-head map factors the whole
``[H*D, H*D]`` projection as ``[H*D, r] @ [r, H*D]``, costing ``2r/D`` of the
per-head arithmetic.  The two are different hypothesis classes, so the swap is
only safe if the data says so.

Two questions are answered separately:

1. How much of the fitted per-head map survives a rank-r truncation?  Because a
   block-diagonal matrix's singular values are exactly its blocks' singular
   values pooled together, this needs no large SVD.
2. What does a rank-r all-head map fitted directly on the trace actually cost
   in V reconstruction error?  This is the decisive number: a directly fitted
   map may find cross-head structure the per-head map could never represent, so
   it can beat its own truncation.

Question 1 needs only the fitted parameters; question 2 needs the trace too.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.parameters import CrossKVParameters  # noqa: E402


def _energy_ranks(values: torch.Tensor, fractions=(0.9, 0.99, 0.999)) -> dict[float, int]:
    ordered = torch.sort(values.flatten(), descending=True).values
    energy = ordered.double().square()
    cumulative = energy.cumsum(0) / energy.sum().clamp_min(1e-30)
    return {
        fraction: int(torch.searchsorted(cumulative, fraction).item()) + 1
        for fraction in fractions
    }


def _spectrum(parameters: CrossKVParameters) -> None:
    weight = parameters.weight
    if weight.ndim != 4:
        raise ValueError("expected layered CrossKV weights [L, H, D, D]")
    layers, heads, dim, _ = weight.shape
    print(f"fitted per-head Cross map: {layers} layers x {heads} heads x {dim}x{dim}")
    print(f"{'layer':>6} {'rank@90%':>9} {'rank@99%':>9} {'rank@99.9%':>11} {'2r/D at 99%':>12}")
    print("-" * 52)
    totals = []
    for layer in range(layers):
        # Block diagonal: the full map's singular values are the blocks' pooled.
        values = torch.linalg.svdvals(weight[layer].double())
        ranks = _energy_ranks(values)
        totals.append(ranks)
        print(
            f"{layer:>6} {ranks[0.9]:>9} {ranks[0.99]:>9} {ranks[0.999]:>11} "
            f"{2 * ranks[0.99] / dim:>11.2f}x"
        )
    median99 = int(np.median([item[0.99] for item in totals]))
    print(
        f"\nmedian rank keeping 99% of the fitted map's energy: {median99} "
        f"(of {heads * dim}); a rank-{median99} all-head map costs "
        f"{2 * median99 / dim:.2f}x the per-head arithmetic"
    )
    if 2 * median99 / dim >= 1.0:
        print(
            "-> truncation buys nothing here: the fitted per-head maps are not "
            "low rank, so only a directly fitted all-head map could win"
        )


def _reduced_rank_errors(
    keys: torch.Tensor, values: torch.Tensor, ranks: tuple[int, ...], ridge: float
) -> None:
    """Fit an all-head map on the trace and score truncations of it."""
    layers, heads, dim = keys.shape[0], keys.shape[-2], keys.shape[-1]
    width = heads * dim
    print(f"\nall-head regression on the trace: width {width}, ridge {ridge}")
    print(f"{'layer':>6} {'per-head':>10} " + " ".join(f"{'r=' + str(r):>10}" for r in ranks))
    print("-" * (18 + 11 * len(ranks)))
    for layer in range(layers):
        x = keys[layer].reshape(-1, heads, dim).double()
        y = values[layer].reshape(-1, heads, dim).double()
        samples = x.shape[0]
        flat_x = x.reshape(samples, width)
        flat_y = y.reshape(samples, width)
        reference = flat_y.square().sum()

        # Per-head baseline: one independent ridge solve per head.
        per_head_error = 0.0
        for head in range(heads):
            gram = x[:, head].T @ x[:, head]
            rhs = x[:, head].T @ y[:, head]
            penalty = ridge * torch.diagonal(gram).mean().clamp_min(1e-12)
            solution = torch.linalg.solve(
                gram + penalty * torch.eye(dim, dtype=torch.float64), rhs
            )
            per_head_error += (x[:, head] @ solution - y[:, head]).square().sum()

        gram = flat_x.T @ flat_x
        rhs = flat_x.T @ flat_y
        penalty = ridge * torch.diagonal(gram).mean().clamp_min(1e-12)
        dense = torch.linalg.solve(
            gram + penalty * torch.eye(width, dtype=torch.float64), rhs
        )
        # Reduced-rank regression: truncate in the whitened space so the
        # truncation minimizes prediction error, not map error.
        root = torch.linalg.cholesky(
            gram + penalty * torch.eye(width, dtype=torch.float64)
        )
        left, singular, right = torch.linalg.svd(root.T @ dense, full_matrices=False)
        cells = []
        for rank in ranks:
            truncated = torch.linalg.solve_triangular(
                root.T,
                left[:, :rank] @ torch.diag(singular[:rank]) @ right[:rank],
                upper=True,
            )
            error = (flat_x @ truncated - flat_y).square().sum()
            cells.append(f"{float(error / reference):>10.5f}")
        print(
            f"{layer:>6} {float(per_head_error / reference):>10.5f} " + " ".join(cells)
        )
    print(
        "\nColumns are V NMSE on the fitting data.  A rank whose column beats "
        "the per-head column is strictly better: cheaper arithmetic, fewer "
        "parameters, and lower error."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v-params", type=Path, help="fitted v_cross_kv.npz")
    parser.add_argument("--trace", type=Path, help="clean trace .npz")
    parser.add_argument("--keys-name", default="keys")
    parser.add_argument("--values-name", default="values")
    parser.add_argument("--ranks", type=int, nargs="*", default=(16, 32, 64, 128))
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--layers", type=int, default=4, help="layers to regress")
    args = parser.parse_args(argv)
    if args.v_params is None and args.trace is None:
        parser.error("pass --v-params, --trace, or both")

    if args.v_params is not None:
        _spectrum(CrossKVParameters.load(args.v_params))

    if args.trace is not None:
        with np.load(args.trace, allow_pickle=False) as archive:
            keys = torch.from_numpy(np.array(archive[args.keys_name], copy=True))
            values = torch.from_numpy(np.array(archive[args.values_name], copy=True))
        layers = min(args.layers, int(keys.shape[0]))
        _reduced_rank_errors(
            keys[:layers], values[:layers], tuple(args.ranks), args.ridge
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
