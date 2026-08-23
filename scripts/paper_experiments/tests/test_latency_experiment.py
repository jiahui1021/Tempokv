from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
RUN_PATH = SCRIPTS_ROOT / "paper_experiments" / "latency" / "run.py"
ADAPTER_PATH = SCRIPTS_ROOT / "paper_experiments" / "latency" / "report_adapter.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("paper_latency_run", RUN_PATH)
adapter = _load("paper_latency_adapter", ADAPTER_PATH)


def _manifest(tmp_path: Path) -> dict:
    template = json.loads(
        (SCRIPTS_ROOT / "paper_experiments" / "latency" / "manifest.example.json").read_text(
            encoding="utf-8"
        )
    )
    template["output_dir"] = str(tmp_path / "output")
    return template


def test_synthetic_builds_separate_fixed_and_horizon_artifacts(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    output = runner.execute(manifest, dry_run=False, synthetic=True, pilot=False)

    measurements = json.loads((output / "measurements.json").read_text(encoding="utf-8"))
    assert len(measurements) == 7 * 6 * 5
    with (output / "fixed_717.csv").open(newline="", encoding="utf-8") as handle:
        fixed = list(csv.DictReader(handle))
    with (output / "horizon.csv").open(newline="", encoding="utf-8") as handle:
        horizon = list(csv.DictReader(handle))
    assert len(fixed) == 7
    assert len(horizon) == 7 * 6
    assert all(row["effective_frames"] == "717" for row in fixed)
    assert "cache_encode_s_median" in fixed[0]
    assert "cache_decode_s_p90" in fixed[0]
    assert "vae_decode_s_p10" in fixed[0]
    assert "paired_bf16_overhead_pct_median" in fixed[0]

    absolute = (output / "horizon_absolute_log.svg").read_text(encoding="utf-8")
    share = (output / "horizon_overhead_share.svg").read_text(encoding="utf-8")
    assert "Absolute end-to-end latency" in absolute
    assert "Cache encode + decode share" not in absolute
    assert "Cache encode + decode share" in share
    assert "Absolute end-to-end latency" not in share
    assert "SYNTHETIC — NOT PAPER RESULTS" in absolute
    warmup = json.loads(
        (output / "raw" / "TempoKV_INT2" / "frames-717" / "repeat-0" / "warmup.json").read_text(
            encoding="utf-8"
        )
    )
    measure = json.loads(
        (output / "raw" / "TempoKV_INT2" / "frames-717" / "repeat-0" / "measure.json").read_text(
            encoding="utf-8"
        )
    )
    assert warmup["runtime_instance_id"] == measure["runtime_instance_id"]


def test_each_measurement_is_immediately_preceded_by_full_warmup(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    output = runner.execute(manifest, dry_run=True, synthetic=False, pilot=False)
    plan = json.loads((output / "execution_plan.json").read_text(encoding="utf-8"))
    non_compilation = [item for item in plan if item["phase"] != "compilation"]
    assert len(non_compilation) == 7 * 6 * 5
    assert all(item["phase"] == "pair" for item in non_compilation)
    assert all(item["warmup_report"].endswith("warmup.json") for item in non_compilation)
    assert all(item["measure_report"].endswith("measure.json") for item in non_compilation)


def _valid_report(manifest: dict) -> dict:
    return {
        "schema_version": runner.RUN_SCHEMA,
        "method": "TempoKV_INT2",
        "phase": "measure",
        "repeat": 0,
        "requested_frames": 717,
        "effective_frames": 717,
        "session_id": "session",
        "session_scope": runner.SUPPORTED_SESSION_SCOPE,
        "runtime_instance_id": "loaded-worker-1",
        "frozen": manifest["frozen"],
        "timings_s": {
            "e2e": 10.0,
            "cache_encode": 1.0,
            "cache_decode": 2.0,
            "vae_decode": 2.0,
            "denoising_other": 5.0,
        },
        "synchronization": {
            "api": "torch.cuda.synchronize",
            "before_each_segment": True,
            "after_each_segment": True,
        },
        "exclusions": {"model_loading": True, "compilation": True},
    }


def test_formal_validation_rejects_async_segments_and_changed_frozen_fields(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    report = _valid_report(manifest)
    report["synchronization"] = {
        "api": "cuda_events_async",
        "before_each_segment": False,
        "after_each_segment": False,
    }
    with pytest.raises(runner.ContractError, match="before and after every timing segment"):
        runner.validate_run_report(
            report,
            expected_method="TempoKV_INT2",
            expected_phase="measure",
            expected_repeat=0,
            expected_frames=717,
            expected_session_id="session",
            expected_session_scope=runner.SUPPORTED_SESSION_SCOPE,
            expected_frozen=manifest["frozen"],
        )

    report = _valid_report(manifest)
    report["frozen"] = dict(report["frozen"], seed=123)
    with pytest.raises(runner.ContractError, match="frozen-field mismatch"):
        runner.validate_run_report(
            report,
            expected_method="TempoKV_INT2",
            expected_phase="measure",
            expected_repeat=0,
            expected_frames=717,
            expected_session_id="session",
            expected_session_scope=runner.SUPPORTED_SESSION_SCOPE,
            expected_frozen=manifest["frozen"],
        )


def test_manifest_refuses_false_single_process_claim(tmp_path) -> None:
    payload = _manifest(tmp_path)
    payload["session_scope"] = "single_process"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runner.ContractError, match="cannot claim single_process"):
        runner._manifest(path)


def test_measurement_must_reuse_warmup_runtime_instance(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    report = _valid_report(manifest)
    with pytest.raises(runner.ContractError, match="same loaded runtime instance"):
        runner.validate_run_report(
            report,
            expected_method="TempoKV_INT2",
            expected_phase="measure",
            expected_repeat=0,
            expected_frames=717,
            expected_session_id="session",
            expected_session_scope=runner.SUPPORTED_SESSION_SCOPE,
            expected_frozen=manifest["frozen"],
            expected_runtime_instance_id="loaded-worker-from-warmup",
        )


def test_strict_adapter_rejects_a_different_runtime_instance(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    context = {
        "method": "TempoKV_INT2",
        "phase": "measure",
        "frames": 717,
        "repeat": 0,
        "session_id": "session",
        "session_scope": runner.SUPPORTED_SESSION_SCOPE,
        "runtime_instance_id": "assigned-pair-id",
        "frozen": manifest["frozen"],
    }
    raw = {
        "effective_frames": 717,
        "runtime_instance_id": "different-process-id",
        "observed_frozen": manifest["frozen"],
        "timings_s": _valid_report(manifest)["timings_s"],
        "synchronization": _valid_report(manifest)["synchronization"],
        "exclusions": _valid_report(manifest)["exclusions"],
    }
    with pytest.raises(runner.ContractError, match="assigned by the coordinator"):
        adapter.strict_report(raw, context)


def test_paired_overhead_uses_matching_repeat(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    output = runner.execute(manifest, dry_run=False, synthetic=True, pilot=False)
    with (output / "fixed_717.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    assert float(rows["BF16"]["paired_bf16_overhead_s_median"]) == pytest.approx(0.0)
    # TempoKV synthetic samples use the same repeat skew as BF16.  This checks
    # that the result is formed from paired repeats, not two unrelated samples.
    records = json.loads((output / "measurements.json").read_text(encoding="utf-8"))
    paired = []
    for repeat in range(5):
        current = next(
            item for item in records
            if item["method"] == "TempoKV_INT2"
            and item["effective_frames"] == 717
            and item["repeat"] == repeat
        )
        baseline = next(
            item for item in records
            if item["method"] == "BF16"
            and item["effective_frames"] == 717
            and item["repeat"] == repeat
        )
        paired.append(current["timings_s"]["e2e"] - baseline["timings_s"]["e2e"])
    assert float(rows["TempoKV_INT2"]["paired_bf16_overhead_s_median"]) == pytest.approx(
        sorted(paired)[2]
    )
