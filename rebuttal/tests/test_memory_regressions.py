from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rebuttal.adapters.qvg import QVGBackend
from rebuttal.experiments.memory.evaluation import generation_pairs
from rebuttal.experiments.memory.identity import file_digest, save_job_identities


def backend(tmp_path):
    (tmp_path / "quant_videogen/real").mkdir(parents=True)
    (tmp_path / "quant_videogen/real/prq.py").touch()
    return QVGBackend(tmp_path, block_size=2, k_centroids=2, v_centroids=2)


@pytest.mark.parametrize("raises", [False, True])
def test_qvg_keeps_generator_rng_after_stochastic_backend(monkeypatch, tmp_path, raises):
    codec = backend(tmp_path)
    original_fork = torch.random.fork_rng
    requested_devices = []

    def cpu_fork(*, devices):
        requested_devices.append(devices)
        # Exercise real CPU save/restore without requiring CUDA in this test.
        return original_fork(devices=[])

    monkeypatch.setattr(torch.random, "fork_rng", cpu_fork)

    def stochastic_quantize(value, **kwargs):
        torch.randint(0, 8, (2, 2))  # Same global RNG operation as QVG KMeans.
        if raises:
            raise RuntimeError("codec error after consuming random numbers")
        return dict(centroids_list=[], cluster_ids_list=[],
                    residual_quant=value, scales=torch.ones(1))

    codec._module = SimpleNamespace(triton_prq_quantize_tensor=stochastic_quantize)
    value = SimpleNamespace(is_cuda=True, device=torch.device("cuda:1"),
                            shape=(1, 2, 8, 4), dtype=torch.bfloat16,
                            contiguous=lambda: torch.ones(1, 2, 8, 4))
    torch.manual_seed(17)
    expected = torch.randn(6)
    torch.manual_seed(17)
    if raises:
        with pytest.raises(RuntimeError, match="codec error"):
            codec.encode_tensor(value, role="k")
    else:
        codec.encode_tensor(value, role="k")
    assert torch.equal(torch.randn(6), expected)
    assert requested_devices == [[torch.device("cuda:1")]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RNG integration requires a CUDA device")
def test_qvg_restores_actual_cuda_rng(tmp_path):
    codec = backend(tmp_path)
    value = torch.ones((1, 2, 8, 4), device="cuda", dtype=torch.bfloat16)

    def stochastic_quantize(x, **kwargs):
        torch.randint(0, 8, (2, 2), device=x.device)
        return dict(centroids_list=[], cluster_ids_list=[], residual_quant=x, scales=x[:1])

    codec._module = SimpleNamespace(triton_prq_quantize_tensor=stochastic_quantize)
    before = torch.cuda.get_rng_state(value.device).clone()
    codec.encode_tensor(value, role="k")
    assert torch.equal(torch.cuda.get_rng_state(value.device), before)


def cohort(directory, seed, *, sample=0, filenames=None):
    directory.mkdir(parents=True)
    records = []
    for i in range(2):
        path = directory / (filenames[i] if filenames else f"{i}_0.mp4")
        path.write_bytes(b"identity-test-only; decoding is outside this test")
        records.append({"prompt_index": i, "sample_index": sample,
                        "noise_seed": seed + i * 1_000_003, "path": str(path)})
    (directory / "summary.json").write_text(json.dumps({"rank_summaries": [{"videos": records}]}))


def setup_plan(tmp_path, seeds=(0, 1)):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a running dog\na moving camera\n")
    jobs = []
    for seed in seeds:
        folder = tmp_path / f"candidate_{seed}"
        cohort(folder / "videos", seed)
        jobs.append({"folder": str(folder), "name": f"seed_{seed}", "method": "tempokv",
                     "seed": seed, "expected_video_count": 2, "history_frames": 96,
                     "budget_bytes": 1000, "command": ["python", "--data_path", str(prompts)],
                     "input_digests": {"prompts": file_digest(prompts)}})
    (tmp_path / "generation_plan.json").write_text(json.dumps({"jobs": jobs}))
    return prompts, jobs


def test_multi_seed_pairs_use_recorded_noise_seed_and_prompt_identity(tmp_path):
    prompts, _ = setup_plan(tmp_path)
    root = tmp_path / "references"
    for seed in (0, 1):
        # Reference filenames deliberately differ: the summary is authoritative.
        cohort(root / f"seed_{seed}" / "videos", seed, filenames=["dog.mp4", "camera.mp4"])
    pairs = generation_pairs({"bf16_reference_dir": str(root),
                              "bf16_reference_prompts": str(prompts)}, tmp_path)
    assert len(pairs) == 4
    assert {p["seed"] for p in pairs} == {0, 1, 1_000_003, 1_000_004}
    for pair in pairs:
        assert f"seed_{pair['generation_seed']}" in pair["reference"]
        assert pair["identity_validation"] == "generation_indices_and_prompt_text_matched"


@pytest.mark.parametrize("reference_seed,sample", [(1, 0), (0, 1)])
def test_wrong_reference_seed_or_sample_is_rejected(tmp_path, reference_seed, sample):
    prompts, _ = setup_plan(tmp_path, (0,))
    root = tmp_path / "references"
    cohort(root, reference_seed, sample=sample)
    with pytest.raises(ValueError, match="matching prompt/noise_seed/sample"):
        generation_pairs({"bf16_reference_dir": str(root),
                          "bf16_reference_prompts": str(prompts)}, tmp_path)


def test_matching_indices_cannot_hide_different_prompts(tmp_path):
    _, _ = setup_plan(tmp_path, (0,))
    root = tmp_path / "references"
    cohort(root, 0)
    reference_prompts = tmp_path / "other_prompts.txt"
    reference_prompts.write_text("a sleeping cat\na moving camera\n")
    with pytest.raises(ValueError, match="prompt text differ"):
        generation_pairs({"bf16_reference_dir": str(root),
                          "bf16_reference_prompts": str(reference_prompts)}, tmp_path)


def test_reference_cannot_inherit_candidate_identity_without_summary(tmp_path):
    prompts, _ = setup_plan(tmp_path, (0,))
    root = tmp_path / "references"
    root.mkdir()
    (root / "0_0.mp4").write_bytes(b"video")
    with pytest.raises(ValueError, match="generation summary is required"):
        generation_pairs({"bf16_reference_dir": str(root),
                          "bf16_reference_prompts": str(prompts)}, tmp_path)


def test_reference_snapshot_preserves_prompt_text_and_checks_summary(tmp_path):
    _, jobs = setup_plan(tmp_path, (0,))
    job = jobs[0]
    save_job_identities(job)
    reference = Path(job["folder"]) / "videos"
    pairs = generation_pairs({"bf16_reference_dir": str(reference)}, tmp_path)
    assert len(pairs) == 2
    summary = reference / "summary.json"
    summary.write_text(summary.read_text() + "\n")
    with pytest.raises(ValueError, match="summary changed"):
        generation_pairs({"bf16_reference_dir": str(reference)}, tmp_path)


def test_candidate_summary_must_match_job_seed(tmp_path):
    prompts, jobs = setup_plan(tmp_path, (0,))
    root = tmp_path / "references"
    cohort(root, 0)
    summary = Path(jobs[0]["folder"]) / "videos/summary.json"
    payload = json.loads(summary.read_text())
    payload["rank_summaries"][0]["videos"][0]["noise_seed"] = 999
    summary.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="summary seed/sample does not match job"):
        generation_pairs({"bf16_reference_dir": str(root),
                          "bf16_reference_prompts": str(prompts)}, tmp_path)


@pytest.mark.parametrize("has_summary", [False, True])
def test_generation_requires_actual_identity_before_complete_or_resume(monkeypatch, tmp_path, has_summary):
    from rebuttal.experiments.memory.generation import execute_jobs

    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a running dog\na moving camera\n")
    checkpoint, runtime = tmp_path / "checkpoint.pt", tmp_path / "runtime.yaml"
    checkpoint.touch()
    runtime.touch()
    folder = tmp_path / "output"
    job = {"folder": str(folder), "name": "seed_0", "seed": 0,
           "config_path": str(folder / "worker_config.json"), "expected_video_count": 2,
           "identity": "fixture-job", "input_digests": {"prompts": file_digest(prompts)},
           "worker_config": {"model_root": str(tmp_path)},
           "command": ["python", "--data_path", str(prompts),
                       "--checkpoint_path", str(checkpoint), "--config_path", str(runtime)]}
    calls = []

    def successful_process(*args, **kwargs):
        calls.append(args)
        cohort(folder / "videos", 0)
        if not has_summary:
            (folder / "videos/summary.json").unlink()
        (folder / "memory_report.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("rebuttal.experiments.memory.generation.subprocess.run", successful_process)
    plan = {"jobs": [job], "missing_configuration": []}
    if not has_summary:
        with pytest.raises(RuntimeError, match="generation failed"):
            execute_jobs(plan)
        saved = json.loads((folder / "job.json").read_text())
        assert saved["status"] == "failed"
        assert "invalid_generation_identity" in saved["failure_reason"]
    else:
        execute_jobs(plan)
        assert (folder / "videos/generation_identity.json").is_file()
        assert job["status"] == "complete"
        execute_jobs(plan, resume=True)
        assert job["status"] == "reused"
        assert len(calls) == 1
