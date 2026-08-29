from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


CALIBRATION_DIR = Path(__file__).parents[1] / "paper_experiments" / "calibration"
if str(CALIBRATION_DIR) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_DIR))


def _module(name: str):
    path = CALIBRATION_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"calibration_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _module("protocol")
build_groups = _module("build_groups")
run_minimal = _module("run_minimal")
plot_figure = _module("plot_figure")
run_shift = _module("run_shift")


def _pool_archive(path: Path, *, layers=2, rows=8, frames=4, heads=2, dim=128, stride=16):
    rng = np.random.default_rng(0)
    shape = (layers, rows, frames, heads, dim)
    np.savez(
        path,
        keys=rng.normal(size=shape).astype(np.float32),
        values=rng.normal(size=shape).astype(np.float32),
        layer_ids=np.arange(layers, dtype=np.int32),
        trace_stride=np.asarray(stride, dtype=np.int32),
        coordinate_system=np.asarray("pre_rope_normalized"),
        cross_source_kind=np.asarray("normalized"),
    )
    return path


def _groups_file(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# The codec is fixed, and both the fitter and the evaluator must see it
# --------------------------------------------------------------------------


def test_the_study_codec_is_int2_with_a_fixed_anchor_schedule() -> None:
    assert protocol.CODEC["k_bits"] == 2
    assert protocol.CODEC["v_bits"] == 2
    assert protocol.CODEC["anchor_bits"] == 4
    assert protocol.CODEC["group_size"] == 64
    assert protocol.CODEC["residual_mode"] == "asymmetric"


def test_stride_is_the_only_model_dependent_codec_field() -> None:
    """Everything else must be byte-identical across models and modules."""
    assert "stride" not in protocol.CODEC
    first = protocol.codec_config(1560)
    second = protocol.codec_config(1024)
    assert first["stride"] == 1560 and second["stride"] == 1024
    assert {k: v for k, v in first.items() if k != "stride"} == {
        k: v for k, v in second.items() if k != "stride"
    }


def test_codec_config_rejects_a_nonpositive_stride() -> None:
    with pytest.raises(ValueError):
        protocol.codec_config(0)


# --------------------------------------------------------------------------
# Prompt groups
# --------------------------------------------------------------------------


def test_prompt_groups_reject_members_outside_the_pool(tmp_path: Path) -> None:
    path = _groups_file(
        tmp_path / "groups.json",
        {"prompt_count": 4, "groups": {"calibration": [0, 1, 9]}},
    )
    with pytest.raises(ValueError, match="outside the pool"):
        protocol.PromptGroups.load(path)


def test_prompt_groups_reject_a_repeated_prompt(tmp_path: Path) -> None:
    path = _groups_file(
        tmp_path / "groups.json",
        {"prompt_count": 4, "groups": {"calibration": [0, 0]}},
    )
    with pytest.raises(ValueError, match="repeats a prompt"):
        protocol.PromptGroups.load(path)


def test_calibration_and_evaluation_prompts_must_be_disjoint(tmp_path: Path) -> None:
    """A shared prompt would make every held-out number partly in-sample."""
    path = _groups_file(
        tmp_path / "groups.json",
        {
            "prompt_count": 4,
            "groups": {"calibration": [0, 1, 2], "heldout": [2, 3]},
        },
    )
    groups = protocol.PromptGroups.load(path)
    with pytest.raises(ValueError, match="must be disjoint"):
        groups.require_disjoint("calibration", "heldout")


def test_unknown_group_names_say_what_is_available(tmp_path: Path) -> None:
    path = _groups_file(
        tmp_path / "groups.json", {"prompt_count": 2, "groups": {"calibration": [0]}}
    )
    groups = protocol.PromptGroups.load(path)
    with pytest.raises(ValueError, match="calibration"):
        groups.group("heldout")


# --------------------------------------------------------------------------
# Pool slicing
# --------------------------------------------------------------------------


def test_a_prompt_owns_a_contiguous_block_of_records(tmp_path: Path) -> None:
    pool = protocol.PoolTrace.load(
        _pool_archive(tmp_path / "pool.npz", rows=8), prompt_count=4
    )
    assert pool.rows_per_prompt == 2
    assert list(pool.rows_for([0])) == [0, 1]
    assert list(pool.rows_for([3])) == [6, 7]
    assert list(pool.rows_for([0, 2])) == [0, 1, 4, 5]


def test_a_pool_that_does_not_divide_by_the_prompt_count_is_refused(
    tmp_path: Path,
) -> None:
    """The trace and the prompt groups would describe different runs."""
    with pytest.raises(ValueError, match="do not divide evenly"):
        protocol.PoolTrace.load(_pool_archive(tmp_path / "pool.npz", rows=7), prompt_count=4)


def test_the_legacy_event_layout_collapses_to_the_same_record_order(
    tmp_path: Path,
) -> None:
    """[L,E,B,P,T,H,D] is event-major, so flattening preserves prompt blocks.

    The loader selects this path on tensor rank rather than on the archive's
    layout name, so the fixture does not restate the protocol string that the
    repository keeps confined to its trace emitters.
    """
    rng = np.random.default_rng(0)
    keys = rng.normal(size=(2, 4, 1, 2, 3, 2, 128)).astype(np.float32)
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        keys=keys,
        values=keys.copy(),
        layer_ids=np.arange(2, dtype=np.int32),
        stride=np.asarray(1560, dtype=np.int32),
        coordinate_system=np.asarray("pre_rope_normalized"),
    )
    pool = protocol.PoolTrace.load(path, prompt_count=2)
    assert pool.keys.shape == (2, 8, 3, 2, 128)
    assert pool.rows_per_prompt == 4
    np.testing.assert_allclose(pool.keys[:, :4], keys[:, :2].reshape(2, 4, 3, 2, 128))


def test_the_archive_layout_name_is_carried_into_provenance(tmp_path: Path) -> None:
    path = tmp_path / "pool.npz"
    _pool_archive(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    np.savez(path, **arrays, trace_layout=np.asarray("some_named_layout_v1"))
    assert protocol.PoolTrace.load(path, prompt_count=4).source_layout == (
        "some_named_layout_v1"
    )
    _pool_archive(path)
    assert protocol.PoolTrace.load(path, prompt_count=4).source_layout == "standard_lsthd"


def test_a_slice_is_written_without_the_pinned_event_layout(tmp_path: Path) -> None:
    """The 12-prompt event protocol cannot express a variable-size set."""
    pool = protocol.PoolTrace.load(
        _pool_archive(tmp_path / "pool.npz", rows=8, stride=16), prompt_count=4
    )
    written = pool.write_slice(tmp_path / "slice.npz", [1, 2])
    with np.load(written, allow_pickle=False) as archive:
        assert "trace_layout" not in archive
        assert archive["keys"].shape[1] == 4
        assert int(np.asarray(archive["trace_stride"]).item()) == 16
        assert str(np.asarray(archive["trace_role"]).item()) == "calibration"
        assert list(np.asarray(archive["pool_prompts"])) == [1, 2]


def test_an_empty_prompt_slice_is_refused(tmp_path: Path) -> None:
    pool = protocol.PoolTrace.load(_pool_archive(tmp_path / "pool.npz"), prompt_count=4)
    with pytest.raises(ValueError):
        pool.rows_for([])


# --------------------------------------------------------------------------
# Draws and statistics
# --------------------------------------------------------------------------


def test_draws_are_deterministic_for_a_seed() -> None:
    population = list(range(16))
    first = protocol.draw_subsets(population, 4, 3, seed=7)
    second = protocol.draw_subsets(population, 4, 3, seed=7)
    assert first == second
    assert protocol.draw_subsets(population, 4, 3, seed=8) != first


def test_drawing_the_whole_pool_collapses_to_one_sample() -> None:
    """Repeats of an exhaustive draw are the same set, not independent runs."""
    draws = protocol.draw_subsets(list(range(8)), 8, 5, seed=0)
    assert len(draws) == 1
    assert draws[0] == tuple(range(8))


def test_a_size_larger_than_the_pool_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot draw"):
        protocol.draw_subsets(list(range(4)), 8, 1, seed=0)


def test_summarize_reports_a_zero_spread_for_a_single_sample() -> None:
    assert protocol.summarize([2.0]) == {
        "mean": 2.0,
        "std": 0.0,
        "min": 2.0,
        "max": 2.0,
        "samples": 1,
    }


# --------------------------------------------------------------------------
# Distribution shift accounting
# --------------------------------------------------------------------------


def test_error_increase_compares_within_one_evaluation_set() -> None:
    """Held-out sets differ in difficulty, so only same-column ratios mean anything."""
    cells = [
        {"calibration": "action", "evaluation": "action",
         "k_nmse": {"mean": 0.10}, "v_nmse": {"mean": 0.20}},
        {"calibration": "action", "evaluation": "scene",
         "k_nmse": {"mean": 0.55}, "v_nmse": {"mean": 0.80}},
        {"calibration": "scene", "evaluation": "action",
         "k_nmse": {"mean": 0.11}, "v_nmse": {"mean": 0.24}},
        {"calibration": "scene", "evaluation": "scene",
         "k_nmse": {"mean": 0.50}, "v_nmse": {"mean": 0.80}},
    ]
    rows = run_shift._error_increase(cells, "action", "scene")
    by_evaluation = {row["evaluation"]: row for row in rows}
    assert by_evaluation["action"]["k_nmse_increase"] == pytest.approx(0.10)
    assert by_evaluation["scene"]["k_nmse_increase"] == pytest.approx(0.10)
    assert by_evaluation["scene"]["v_nmse_increase"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Cross-model normalization
# --------------------------------------------------------------------------


def test_each_model_is_normalized_by_its_own_reference_size() -> None:
    """Absolute NMSE is not comparable across models; the ratio is."""
    run_models = _module("run_models")
    points = [
        {"calibration_size": 2, "k_nmse": {"mean": 0.4}, "v_nmse": {"mean": 0.8}},
        {"calibration_size": 8, "k_nmse": {"mean": 0.2}, "v_nmse": {"mean": 0.4}},
    ]
    rows = run_models._normalize(points, 8)
    assert rows[0]["k_nmse_ratio"] == pytest.approx(2.0)
    assert rows[1]["k_nmse_ratio"] == pytest.approx(1.0)
    assert rows[1]["v_nmse_ratio"] == pytest.approx(1.0)


def test_a_model_spec_must_name_a_pool_and_a_groups_file() -> None:
    run_models = _module("run_models")
    name, pool, groups = run_models._parse_model("longcat=/a/pool.npz:/b/groups.json")
    assert (name, str(pool), str(groups)) == (
        "longcat",
        "/a/pool.npz",
        "/b/groups.json",
    )
    with pytest.raises(ValueError):
        run_models._parse_model("longcat=/a/pool.npz")


# --------------------------------------------------------------------------
# Prompt pool construction
# --------------------------------------------------------------------------


def test_the_pool_is_balanced_and_the_groups_are_disjoint(tmp_path: Path) -> None:
    prompts = [
        f"A man walks through a room number {index}." if index % 2 == 0
        else f"An empty valley at dawn number {index}."
        for index in range(40)
    ]
    indices, groups, manifest = build_groups.build(
        prompts, calibration_size=8, heldout_size=8
    )
    assert len(indices) == 16
    assert len(set(indices)) == 16
    assert not set(groups["calibration"]) & set(groups["heldout"])
    for name in ("action", "scene"):
        assert not set(groups[f"{name}_calibration"]) & set(groups[f"{name}_heldout"])
        assert len(groups[f"{name}_calibration"]) == 4
        assert len(groups[f"{name}_heldout"]) == 4
    assert [item["distribution"] for item in manifest[:4]] == [
        "action",
        "scene",
        "action",
        "scene",
    ]


def test_a_prompt_file_without_both_distributions_is_refused() -> None:
    with pytest.raises(ValueError, match="scene prompts"):
        build_groups.build(
            [f"A man runs number {index}." for index in range(20)],
            calibration_size=4,
            heldout_size=4,
        )


def test_classification_uses_the_documented_subject_rule() -> None:
    assert build_groups.classify("A woman walks down a Tokyo street.") == "action"
    assert build_groups.classify("Drone view of waves crashing on cliffs.") == "scene"


# --------------------------------------------------------------------------
# The minimal run is a real run
# --------------------------------------------------------------------------


def test_the_minimal_run_covers_three_distinct_kv_geometries() -> None:
    """Module 3 is only evidence if the models do not share a shape."""
    shapes = {
        (spec["layers"], spec["heads"], spec["dim"], spec["stride"])
        for spec in run_minimal.MODELS.values()
    }
    assert len(shapes) == len(run_minimal.MODELS) == 3


def test_the_minimal_groups_are_disjoint_and_balanced(tmp_path: Path) -> None:
    groups = protocol.PromptGroups.load(run_minimal.write_groups(tmp_path / "g.json"))
    groups.require_disjoint("calibration", "heldout")
    groups.require_disjoint("action_calibration", "action_heldout")
    groups.require_disjoint("scene_calibration", "scene_heldout")
    assert len(groups.group("action_calibration")) == len(
        groups.group("scene_calibration")
    )


def test_the_minimal_pool_slices_cleanly_by_prompt(tmp_path: Path) -> None:
    spec = run_minimal.MODELS["longcat"]
    path = run_minimal.synthesize_pool(spec, tmp_path / "pool.npz", seed=0)
    pool = protocol.PoolTrace.load(path, run_minimal.POOL_PROMPTS)
    assert pool.rows_per_prompt == spec["records_per_prompt"]
    assert pool.stride == spec["stride"]


# --------------------------------------------------------------------------
# Figure inputs
# --------------------------------------------------------------------------


def test_panel_rows_carry_the_spread_not_just_the_mean() -> None:
    report = {
        "points": [
            {
                "calibration_size": 4,
                "k_nmse": {"mean": 0.1, "std": 0.01, "samples": 3},
                "v_nmse": {"mean": 0.2, "std": 0.02, "samples": 3},
            }
        ]
    }
    row = plot_figure.size_rows(report)[0]
    assert row["k_nmse_std"] == 0.01
    assert row["v_nmse_samples"] == 3
