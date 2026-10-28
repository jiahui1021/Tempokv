"""Suite A regressions for review R8/R9: normalized mass and sample selection."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tempokv.spectral.config import SpectralConfig
from tempokv.spectral.probe import BUDGET, CORRELATION, SCORES, SPATIAL
from tempokv.spectral.sensitivity import Block, attention_mass


ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("spectral_review_analyzer", ROOT / "scripts/spectral/analyze_results.py")
analyzer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyzer)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_size", [1, 2, 8])
def test_large_logits_preserve_candidate_ranking_and_total_mass(dtype, chunk_size):
    # This was the review reproduction: the old absolute FP32 logZ produced
    # [1.0, 0.5] instead of [1/7, 0.5], and full-window mass was 4 rather than 1.
    q = torch.tensor([10000., -10000.], dtype=dtype).reshape(1, 2, 1, 1)
    k = torch.tensor([10000.] * 7 + [9900.], dtype=dtype).reshape(1, 8, 1, 1)
    blocks = [Block(0, 0, (0, 1)), Block(0, 1, (7,)), Block(0, 2, tuple(range(8)))]
    observed = attention_mass(q, k, blocks, chunk_size=chunk_size)
    probs = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()).softmax(-1)
    expected = [float(probs[..., list(block.indices)].sum(-1).mean()) for block in blocks]
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)
    assert observed[0] < observed[1]
    assert observed[-1] == pytest.approx(1, abs=1e-7)


@pytest.mark.parametrize("offset", [1e8, -1e8])
@pytest.mark.parametrize("chunk_size", [1, 3, 16])
def test_mass_is_invariant_to_large_common_logit_offset(offset, chunk_size):
    q = torch.ones(2, 3, 2, 1)
    k = torch.zeros(2, 8, 2, 1)
    blocks = [Block(0, 0, (0, 1)), Block(0, 1, (2, 3, 4)), Block(0, 2, tuple(range(8)))]
    before = attention_mass(q, k, blocks, chunk_size=chunk_size)
    after = attention_mass(q, k + offset, blocks, chunk_size=chunk_size)
    np.testing.assert_allclose(before, [.25, .375, 1], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-7)


def _report(root, name, values=(1., 0.), *, status="completed", generation_seed=42,
            sample_id=0, model="checkpoint-A.pt", config=None, runtime=None, mtime=1):
    path = root / name / f"sample-{sample_id}"
    path.mkdir(parents=True)
    for folder in ("spectrum", "sensitivity", "compression"):
        (path / folder).mkdir()
    payload = {
        "suite_id": "A_FREQUENCY_ORACLE", "status": status, "prompt_id": 7,
        "sample_id": sample_id, "rank": 0, "inference_id": name,
        "source": "native_bf16", "model": model,
        "coordinates": {"spectrum": "clean pre-RoPE K, raw V", "oracle": "runtime post-RoPE Q/K, raw V"},
        "config": SpectralConfig.from_dict(config or {}).to_dict(),
        "effective_grids": {"0": [2, 2, 1, 2]}, "effective_windows": runtime or [],
    }
    if generation_seed is not None:
        payload["generation_seed"] = generation_seed
    meta = path / "metadata.json"
    meta.write_text(json.dumps(payload))
    os.utime(meta, ns=(mtime, mtime))
    for relative, columns in (("spectrum/spatial_energy.csv", SPATIAL),
                              ("sensitivity/block_scores.csv", SCORES),
                              ("sensitivity/correlation.csv", CORRELATION),
                              ("compression/error_budget.csv", BUDGET)):
        with (path / relative).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            if relative == "spectrum/spatial_energy.csv":
                writer.writerows({"layer": 0, "head": 0, "role": role, "cutoff": .25, "ratio": value,
                                  "frame": frame, "phase": "clean"}
                                 for frame, value in enumerate(values) for role in ("k", "v"))
    np.savez_compressed(path / "spectrum/temporal_energy.npz", heads=np.asarray([0]), cutoffs=np.asarray([.25]))
    return path


def _read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_failed_prefix_and_completed_retry_do_not_change_summary_mean(tmp_path):
    root = tmp_path / "reports"
    _report(root, "failed-prefix", values=(1.,), status="failed", mtime=3)
    _report(root, "first-completed", mtime=1)
    newest = _report(root, "successful-retry", mtime=2)
    out = analyzer.analyze(root, tmp_path / "summary")
    rows = _read(out / "frequency_summary.csv")
    assert {int(row["rows"]) for row in rows} == {2}
    assert {float(row["ratio_mean"]) for row in rows} == {.5}
    selection = json.loads((out / "analysis_selection.json").read_text())
    assert selection["discovered_samples"] == 3
    assert selection["retained_samples"] == 1
    assert selection["excluded_failed_samples"] == 1
    assert selection["excluded_retry_samples"] == 1
    assert selection["retained_metadata"] == [str(newest / "metadata.json")]
    assert "Excluded failed samples: 1" in (out / "summary.md").read_text()


def test_only_failed_samples_create_unmeasured_summary(tmp_path):
    _report(tmp_path / "reports", "failed", status="failed")
    out = analyzer.analyze(tmp_path / "reports", tmp_path / "summary")
    assert not _read(out / "frequency_summary.csv")
    assert not _read(out / "budget_summary.csv")
    selection = json.loads((out / "analysis_selection.json").read_text())
    assert selection["retained_samples"] == 0 and selection["excluded_failed_samples"] == 1
    assert "not measured" in (out / "summary.md").read_text()


def test_distinct_generation_seeds_and_sample_identities_survive(tmp_path):
    _report(tmp_path, "seed-42", generation_seed=42)
    _report(tmp_path, "seed-43", generation_seed=43)
    _report(tmp_path, "sample-1", generation_seed=42, sample_id=1)
    roots, _, selection = analyzer._select_samples(tmp_path)
    assert len(roots) == 3 and selection["excluded_retry_samples"] == 0


def test_output_paths_and_ranks_do_not_create_independent_retry_samples(tmp_path):
    _report(tmp_path, "rank-0/attempt", config={"output_dir": "original-output"}, mtime=1)
    latest = _report(tmp_path, "rank-1/retry", config={"output_dir": "new-output"}, mtime=2)
    meta_path = latest / "metadata.json"
    payload = json.loads(meta_path.read_text())
    payload["rank"] = 1
    meta_path.write_text(json.dumps(payload))
    roots, _, selection = analyzer._select_samples(tmp_path)
    assert roots == [latest]
    assert selection["excluded_retry_samples"] == 1


@pytest.mark.parametrize("difference", ["config", "weights", "backend"])
def test_different_experiment_or_weight_identity_is_not_silently_averaged(tmp_path, difference):
    backend = [{"dtype": "torch.bfloat16", "attention_backend": "wan.modules.attention"}]
    _report(tmp_path / "reports", "first", runtime=backend)
    kwargs = {"runtime": backend}
    if difference == "config":
        kwargs["config"] = {"sampling": {"query_tokens_per_head": 16}}
    elif difference == "weights":
        kwargs["model"] = "checkpoint-B.pt"
    else:
        kwargs["runtime"] = [{"dtype": "torch.bfloat16", "attention_backend": "other.backend"}]
    _report(tmp_path / "reports", "different", **kwargs)
    with pytest.raises(ValueError, match="different experiment configurations or model/runtime identities"):
        analyzer.analyze(tmp_path / "reports", tmp_path / "summary")
    assert not (tmp_path / "summary").exists()


def test_legacy_missing_generation_seed_does_not_guess_analysis_seed(tmp_path):
    _report(tmp_path, "legacy-a", generation_seed=None)
    _report(tmp_path, "legacy-b", generation_seed=None)
    roots, _, selection = analyzer._select_samples(tmp_path)
    assert len(roots) == 2
    assert selection["unknown_generation_seed_samples"] == 2
    assert selection["excluded_retry_samples"] == 0


def test_different_batch_size_is_a_distinct_sampling_identity(tmp_path):
    _report(tmp_path, "batch-one", sample_id=0)
    _report(tmp_path, "batch-two", sample_id=0)
    _report(tmp_path, "batch-two", sample_id=1)
    roots, _, selection = analyzer._select_samples(tmp_path)
    assert len(roots) == 3 and selection["excluded_retry_samples"] == 0
