import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("prepare_trace.py")
SPEC = importlib.util.spec_from_file_location("causal_prepare_trace", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_event(
    root: Path,
    event_index: int,
    *,
    attention: bool = True,
    partial_attention: bool = False,
    q_shape: tuple[int, ...] | None = None,
) -> None:
    event = root / "rank_000" / f"event_{event_index:06d}_frames_0_2"
    event.mkdir(parents=True)
    shape = (1, 2, 2, 2, 4)
    for layer in range(2):
        arrays = {
            "k_pre_rope": np.full(shape, layer + 1, dtype=np.float32),
            "v": np.full(shape, layer + 2, dtype=np.float32),
            "layer_index": np.asarray(layer, dtype=np.int32),
            "num_layers": np.asarray(2, dtype=np.int32),
            "num_heads": np.asarray(2, dtype=np.int32),
            "head_dim": np.asarray(4, dtype=np.int32),
            "token_indices": np.asarray([0, 2], dtype=np.int32),
            "head_indices": np.arange(2, dtype=np.int32),
            "global_start_frame": np.asarray(0, dtype=np.int64),
            "global_end_frame": np.asarray(2, dtype=np.int64),
            "trace_stride": np.asarray(4, dtype=np.int32),
            "coordinate_system": np.asarray("pre_rope_normalized"),
            "cross_source_kind": np.asarray("normalized"),
            "generation_mode": np.asarray("chunkwise"),
            "num_frame_per_block": np.asarray(1, dtype=np.int32),
        }
        if attention:
            arrays["q_post_rope"] = np.full(
                q_shape or shape, layer + 3, dtype=np.float32
            )
            if not partial_attention:
                arrays["k_post_rope"] = np.full(
                    shape, layer + 4, dtype=np.float32
                )
        np.savez(event / f"layer_{layer:02d}.npz", **arrays)


def test_convert_trace_preserves_rotated_attention_arrays(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "trace.npz"
    _write_event(raw, 0)

    MODULE.convert_trace_root(raw, output)

    with np.load(output, allow_pickle=False) as trace:
        assert trace["keys"].shape == (2, 2, 2, 2, 4)
        assert trace["q_post_rope"].shape == trace["keys"].shape
        assert trace["k_post_rope"].shape == trace["keys"].shape


def test_convert_trace_keeps_legacy_kv_only_compatibility(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "trace.npz"
    _write_event(raw, 0, attention=False)

    MODULE.convert_trace_root(raw, output)

    with np.load(output, allow_pickle=False) as trace:
        assert "q_post_rope" not in trace.files
        assert "k_post_rope" not in trace.files








def test_legacy_event_preserves_rotated_attention_layout(tmp_path: Path) -> None:
    event = tmp_path / "event_000000_tokens_0_1560"
    event.mkdir()
    shape = (1, 1, 1, 12, 128)
    for layer in range(30):
        np.savez(
            event / f"layer_{layer:02d}.npz",
            k_pre_rope=np.full(shape, layer + 1, dtype=np.float32),
            v=np.full(shape, layer + 2, dtype=np.float32),
            q_post_rope=np.full(shape, layer + 3, dtype=np.float32),
            k_post_rope=np.full(shape, layer + 4, dtype=np.float32),
            layer_index=np.asarray(layer, dtype=np.int32),
            token_indices=np.asarray([0], dtype=np.int32),
            head_indices=np.arange(12, dtype=np.int32),
            tokens_start=np.asarray(0, dtype=np.int64),
            tokens_end=np.asarray(1560, dtype=np.int64),
            frame_seq_length=np.asarray(1560, dtype=np.int32),
        )

    arrays, _ = MODULE._event(event, legacy_protocol=True)

    assert arrays["keys"].shape == (30, 1, 1, 1, 12, 128)
    for name in ("values", "q_post_rope", "k_post_rope"):
        assert arrays[name].shape == arrays["keys"].shape
