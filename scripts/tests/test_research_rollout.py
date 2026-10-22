"""CPU replay checks with recurrent, branch-owned clean refreshes."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from tempokv.research.rollout import (
    FrozenEvent, ReplayCase, ReplayMemoryError, ReplayRNGError, ReplayTargetError,
    RolloutOutput, TraceRecorder, capture_rng_state, paired_replay, restore_rng_state, rng_fingerprint,
)


def seed_all():
    torch.manual_seed(17); np.random.seed(19); random.seed(23)


def recurrent_callback(observed=None, *, hit_mode="once", mutate_reference=False):
    observed = {} if observed is None else observed
    def run(case, recorder):
        # Every run creates and refreshes its own cache from its own denoised
        # state, never substituting a reference clean latent or reference cache.
        cache = {"value": torch.ones(1, 2), "refreshes": []}
        latents = []
        frame0 = 0
        for chunk, frames in enumerate((1, 2, 1, 3)):
            for step in range(2):
                recorder.record_rng("before_step", (chunk, step))
                draw = torch.randn_like(cache["value"]) + float(np.random.normal()) + random.random()
                denoised = .7 * cache["value"] + .1 * draw
                if case is not None:
                    for event in case.events:
                        if event.chunk == chunk and event.step == step:
                            denoised = denoised + case.strength
                            if hit_mode != "missing": recorder.record_injection(event)
                            if hit_mode == "twice": recorder.record_injection(event)
                recorder.record_output(chunk, step, denoised)
                cache["value"] = denoised
                recorder.record_rng("after_step", (chunk, step))
            recorder.record_rng("before_clean_refresh", chunk)
            cache["value"] = .9 * cache["value"] + .01 * torch.randn_like(cache["value"])
            cache["refreshes"].append(float(cache["value"].sum()))
            recorder.record_rng("after_clean_refresh", chunk)
            recorder.record_chunk(chunk, frame0, frames)
            latents.append(cache["value"][:, None].expand(-1, frames, -1).clone())
            frame0 += frames
        output = RolloutOutput(torch.cat(latents, dim=1), {"refreshes": cache["refreshes"]})
        if case is None:
            observed["reference_tensor"] = output.latents
            observed["reference_refreshes"] = cache["refreshes"].copy()
            observed["reference_end"] = rng_fingerprint()
        else:
            observed[case.case_id] = cache["refreshes"].copy()
            if mutate_reference:
                observed["reference_tensor"].add_(100)
        return output
    return run


def test_null_identity_one_shot_effect_variable_horizons_and_branch_owned_refresh():
    seed_all()
    observed, branches = {}, []
    event = FrozenEvent(2, 3, 1, 0, (0, 1))
    cases = [ReplayCase("null", ()), ReplayCase("effect", (event,), strength=.4)]
    def consume(case, trace, output, rows):
        branches.append((case.case_id, set(trace.outputs), len(rows)))
        assert output.latents.shape == (1, 7, 2)
    result = paired_replay(recurrent_callback(observed), cases, target_chunk=1,
                           horizons=(0, 1, 2, 4), record_steps={(1, 0), (3, 1)}, on_branch=consume)
    assert result.reference_latents.device.type == "cpu"
    assert set(result.reference_trace.outputs) == {(1, 0), (3, 1)}
    assert len(result.reference_trace.outputs) == 2
    assert all(keys == {(1, 0), (3, 1)} for _, keys, _ in branches)
    assert result.reference_trace.chunks == {0: (0, 1), 1: (1, 2), 2: (3, 1), 3: (4, 3)}
    null = [row for row in result.rows if row["case_id"] == "null" and row["status"] == "measured"]
    assert all((row["model_output_relative_error"] or row["clean_latent_relative_error"] or 0) == 0 for row in null)
    effect = [row for row in result.rows if row["case_id"] == "effect"]
    current = next(row for row in effect if row["metric_scope"] == "D_clean_current")
    assert current["clean_latent_relative_error"] > 0 and (current["frame0"], current["frames"]) == (1, 2)
    later = [row for row in effect if row["metric_scope"] == "D_clean_horizon" and row["status"] == "measured"]
    assert {row["frames"] for row in later} == {1, 3}
    assert all(row["clean_latent_relative_error"] > 0 for row in later)
    missing = next(row for row in effect if row["horizon_chunks"] == 4)
    assert missing["status"] == "not_measured" and missing["clean_latent_relative_error"] is None
    assert observed["null"] == observed["reference_refreshes"]
    assert observed["effect"][0] == observed["reference_refreshes"][0]
    assert observed["effect"][1:] != observed["reference_refreshes"][1:]
    assert result.branch_summaries[1]["hits"][0]["hits"] == 1
    assert all(item["paired_rng"] for item in result.branch_summaries)
    assert rng_fingerprint() == observed["reference_end"]
    assert not torch.equal(result.reference_latents, result.last_output.latents)


def test_cases_factory_runs_after_reference_and_copy_protects_original_latents():
    seed_all()
    observed = {}
    factory_calls = []
    def factory(reference):
        factory_calls.append(tuple(reference.chunks))
        assert reference.outputs and observed["reference_end"] == rng_fingerprint()
        # Selection may use RNG; branches still start from the original entry.
        random.random(); np.random.random(); torch.rand(1)
        return [ReplayCase("null", ())]
    result = paired_replay(recurrent_callback(observed, mutate_reference=True), factory,
                           record_steps={(1, 1)}, horizons=(0,), target_chunk=1)
    assert factory_calls == [(0, 1, 2, 3)]
    assert not torch.equal(result.reference_latents, observed["reference_tensor"])
    torch.testing.assert_close(result.reference_latents, result.last_output.latents, rtol=0, atol=0)
    assert rng_fingerprint() == observed["reference_end"]


def test_torch_python_numpy_states_and_next_draws_match_reference_only_run():
    seed_all()
    entry = capture_rng_state()
    callback = recurrent_callback()
    callback(None, TraceRecorder(record_steps=set()))
    expected = (torch.rand(3), random.random(), np.random.random(3))
    restore_rng_state(entry)
    paired_replay(callback, [ReplayCase("a", ()), ReplayCase("b", ())], record_steps=set())
    torch.testing.assert_close(torch.rand(3), expected[0], rtol=0, atol=0)
    assert random.random() == expected[1]
    np.testing.assert_array_equal(np.random.random(3), expected[2])
    snapshot = capture_rng_state(); before = rng_fingerprint()
    torch.rand(1); random.random(); np.random.random()
    assert rng_fingerprint() != before
    restore_rng_state(snapshot)
    assert rng_fingerprint() == before


@pytest.mark.parametrize("rng", ["torch", "python", "numpy", "after_final_call"])
def test_rng_divergence_is_detected_and_reference_end_restored(rng):
    seed_all()
    observed = {}
    callback = recurrent_callback(observed)
    def divergent(case, recorder):
        if case is not None and rng != "after_final_call":
            {"torch": lambda: torch.rand(1), "python": random.random, "numpy": np.random.random}[rng]()
        output = callback(case, recorder)
        if case is not None and rng == "after_final_call": torch.rand(1)
        return output
    with pytest.raises(ReplayRNGError, match="fingerprints or final RNG"):
        paired_replay(divergent, [ReplayCase("bad", ())], record_steps=set())
    assert rng_fingerprint() == observed["reference_end"]


@pytest.mark.parametrize("mode", ["missing", "twice"])
def test_frozen_event_must_be_hit_exactly_once(mode):
    seed_all()
    observed = {}
    event = FrozenEvent(0, 0, 1, 0, (0,))
    with pytest.raises(ReplayTargetError, match="exactly once"):
        paired_replay(recurrent_callback(observed, hit_mode=mode), [ReplayCase("bad-hit", (event,), strength=.1)])
    assert rng_fingerprint() == observed["reference_end"]


def test_recorder_clones_selected_steps_only_and_rejects_live_tensor_rng_keys():
    seed_all()
    recorder = TraceRecorder(max_capture_mb=.001, record_steps={(2, 1)})
    tensor = torch.ones(2, 3)
    assert not recorder.record_output(0, 0, torch.zeros(100000))
    assert recorder.record_output(2, 1, tensor)
    tensor.zero_()
    assert bool((recorder.outputs[2, 1] == 1).all())
    assert recorder.capture_bytes == 24
    with pytest.raises(ValueError, match="twice"):
        recorder.record_output(2, 1, tensor)
    with pytest.raises(TypeError, match="live tensors"):
        recorder.record_rng("step", tensor)
    with pytest.raises(ValueError, match="overlap"):
        recorder.record_chunk(0, 0, 3); recorder.record_chunk(1, 2, 2)


def test_shared_memory_cap_counts_reference_trace_reference_latents_and_active_branch():
    seed_all()
    def callback(case, recorder):
        recorder.record_rng("call", 0)
        recorder.record_chunk(0, 0, 1)
        recorder.record_output(0, 0, torch.ones(1, 40))
        return RolloutOutput(torch.ones(1, 1, 40))
    # Reference peak is 480 bytes, but a branch additionally retains its step
    # snapshot and trajectory, exceeding this 550-byte shared cap.
    with pytest.raises(ReplayMemoryError):
        paired_replay(callback, [ReplayCase("null", ())], max_capture_mb=550 / 1024 ** 2, horizons=(0,))
    result = paired_replay(callback, [ReplayCase("null", ())], max_capture_mb=.01, horizons=(0,))
    assert all(row["status"] == "measured" for row in result.rows)
    assert result.reference_trace.capture_bytes == 160


def test_callback_and_hook_errors_restore_reference_end_rng_and_no_cases_work():
    seed_all()
    observed = {}
    callback = recurrent_callback(observed)
    result = paired_replay(callback, [], record_steps=set())
    torch.testing.assert_close(result.reference_latents, result.last_output.latents)
    assert result.rows == [] and result.branch_summaries == []
    def broken(case, trace):
        if case is not None:
            torch.rand(1); random.random(); np.random.random()
            raise RuntimeError("callback failure")
        return callback(case, trace)
    with pytest.raises(RuntimeError, match="callback failure"):
        paired_replay(broken, [ReplayCase("bad", ())], record_steps=set())
    assert rng_fingerprint() == observed["reference_end"]
    def hook(*args):
        torch.rand(1)
        raise RuntimeError("hook failure")
    with pytest.raises(RuntimeError, match="hook failure"):
        paired_replay(callback, [ReplayCase("null", ())], on_branch=hook, record_steps=set())
    assert rng_fingerprint() == observed["reference_end"]
