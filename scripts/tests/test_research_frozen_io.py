"""Bounded one-event frozen archives, using CPU tensors only."""
from __future__ import annotations

from dataclasses import replace
import io
import json
import zipfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tempokv.research.attention_oracle import reference
from tempokv.research.context import FrozenAttention, ProbeContext
from tempokv.research.frozen_io import load_frozen, save_frozen
from tempokv.spectral_kv.model_spec import ModelSpec
from tempokv.spectral_kv.rope3d import rope_apply


def native(q, k, v):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).contiguous()


def example(dtype=torch.bfloat16, *, theta=False):
    generator = torch.Generator().manual_seed(53)
    positions = torch.tensor([frame * 6 + token for frame in (0, 3, 4, 5) for token in range(6)])
    context = ProbeContext("fixture", 0, 0, 7, 3, 4, "denoise", 1, 750, 24, 2, 2, 3,
                           torch.tensor([24, 27, 31, 35]), positions, 1, 1)
    q = torch.randn(1, 4, 2, 12, generator=generator).to(dtype)
    pre, value = [torch.randn(1, 24, 2, 12, generator=generator).to(dtype) for _ in range(2)]
    omega = torch.arange(1, 7, dtype=torch.float32) / 20
    phase = torch.arange(64, dtype=torch.float32)[:, None] * omega
    freqs = None if theta else torch.polar(torch.ones_like(phase), phase)
    coords = torch.stack((positions // 6, (positions % 6) // 3, positions % 3), dim=1)
    spec = ModelSpec(8, 2, 12, (2, 3), 2, (4, 4, 4))
    key = rope_apply(pre, coords[:, 0], coords[:, 1], coords[:, 2], spec,
                     freqs=freqs, precision="float64" if theta else "float32")
    valid = torch.zeros(24, dtype=torch.bool)
    valid[6:12] = True
    sparse = pre.clone(); sparse[:, ~valid] = 0
    window = FrozenAttention(context, q, key, value, native, sparse, None, valid)
    metadata = {"rope_precision": "float64" if theta else "float32", "prompt": "fixture only"}
    if theta: metadata["rope_default_theta"] = 10000.
    return window, freqs, metadata, coords


def rewrite(source, destination, change):
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    change(arrays)
    np.savez_compressed(destination, **arrays)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_roundtrip_exact_native_dtype_sparse_pre_mask_sink_gaps_and_oracle(tmp_path, dtype):
    window, freqs, metadata, coords = example(dtype)
    original = {name: getattr(window, name).clone() for name in ("q_post", "k_post", "v", "k_pre", "pre_valid_mask")}
    path = save_frozen(tmp_path / "event.npz", window, metadata=metadata, freqs=freqs)
    loaded, info = load_frozen(path)
    assert info["schema_version"] == "research.frozen.v1"
    assert info["attention_backend"] == "torch_cpu_sdpa" and info["runtime_kernel"].endswith("native")
    assert info["runtime_grid"] == [2, 2, 3] and info["source"] == "bf16"
    assert info["metadata"] == metadata and info["rotation_available"]
    assert loaded.context.identity == window.context.identity
    for name, expected in original.items():
        torch.testing.assert_close(getattr(loaded, name), expected, rtol=0, atol=0)
        torch.testing.assert_close(getattr(window, name), expected, rtol=0, atol=0)
    torch.testing.assert_close(loaded.context.key_abs_positions, window.context.key_abs_positions)
    torch.testing.assert_close(loaded.context.protected_mask, window.context.protected_mask)
    torch.testing.assert_close(reference(loaded), reference(window), rtol=0, atol=0)
    valid = loaded.pre_valid_mask
    rotated = loaded.rotate_pre(loaded.k_pre[:, valid], coords[valid])
    torch.testing.assert_close(rotated, loaded.k_post[:, valid], rtol=0, atol=0)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["q_post"].dtype == np.float32
        assert archive["key_abs_positions"].dtype == np.int64
        assert archive["freqs"].dtype == np.complex64
        assert archive["q_post"].shape[1] == 4 and archive["k_post"].shape[1] == 24


def test_explicit_theta_rotation_and_missing_rotation_is_not_invented(tmp_path):
    window, _, metadata, coords = example(theta=True)
    explicit = save_frozen(tmp_path / "theta.npz", window, metadata=metadata)
    loaded, info = load_frozen(explicit, attention_fn=native)
    assert info["attention_backend"] == "injected_runtime_callback"
    assert info["rope"]["source"] == "default_theta"
    valid = loaded.pre_valid_mask
    torch.testing.assert_close(loaded.rotate_pre(loaded.k_pre[:, valid], coords[valid]), loaded.k_post[:, valid], rtol=0, atol=0)
    plain = save_frozen(tmp_path / "no-rotation.npz", window, metadata={"rope_theta": 10000.})
    no_rotation, info = load_frozen(plain)
    assert no_rotation.rotate_pre is None and not info["rotation_available"]
    torch.testing.assert_close(no_rotation.k_pre, window.k_pre)
    with pytest.raises(ValueError, match="injected runtime"):
        load_frozen(explicit, device="cuda")


@pytest.mark.parametrize("complex_dtype,precision", [(torch.complex64, "float32"), (torch.complex128, "float64")])
def test_native_precision_uses_captured_frequency_dtype_and_never_guesses_without_it(tmp_path, complex_dtype, precision):
    window, _, metadata, _ = example()
    real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64
    phase = torch.arange(64, dtype=real_dtype)[:, None] * (torch.arange(1, 7, dtype=real_dtype) / 20)
    freqs = torch.polar(torch.ones_like(phase), phase)
    metadata["rope_precision"] = "native"
    path = save_frozen(tmp_path / f"native-{precision}.npz", window, metadata=metadata, freqs=freqs)
    loaded, info = load_frozen(path)
    assert loaded.rotate_pre is not None and info["rope"]["precision"] == precision
    absent = save_frozen(tmp_path / "missing-frequencies.npz", window, metadata={"rope_precision": "native"})
    assert load_frozen(absent)[0].rotate_pre is None
    with pytest.raises(ValueError, match="native RoPE precision"):
        save_frozen(tmp_path / "ambiguous-theta.npz", window,
                    metadata={"rope_precision": "native", "rope_default_theta": 10000.})


def test_exclusive_atomic_no_overwrite_and_serialized_byte_budget(tmp_path):
    window, freqs, metadata, _ = example()
    path = save_frozen(tmp_path / "one.npz", window, metadata=metadata, freqs=freqs)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_frozen(path, window, metadata=metadata, freqs=freqs)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
    # Native BF16 is smaller than its exact FP32 archive representation.
    native_bytes = sum(getattr(window, name).numel() * 2 for name in ("q_post", "k_post", "v", "k_pre"))
    with pytest.raises(ValueError, match="serialization|uncompressed"):
        save_frozen(tmp_path / "too-small.npz", window, max_capture_mb=(native_bytes + 100) / 1024 ** 2,
                    metadata=metadata, freqs=freqs)
    assert not (tmp_path / "too-small.npz").exists()
    dangling = tmp_path / "existing-link.npz"
    uncreated = tmp_path / "uncreated-target.npz"
    dangling.symlink_to(uncreated)
    with pytest.raises(FileExistsError):
        save_frozen(dangling, window, metadata=metadata, freqs=freqs)
    assert dangling.is_symlink() and not uncreated.exists()


def test_uncompressed_zip_limit_is_checked_before_numpy_load(tmp_path, monkeypatch):
    window, freqs, metadata, _ = example()
    path = save_frozen(tmp_path / "large.npz", window, metadata=metadata, freqs=freqs)
    def forbidden(*args, **kwargs): raise AssertionError("NPZ arrays allocated before size check")
    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match="uncompressed"):
        load_frozen(path, max_capture_mb=.001)


def test_declared_huge_shape_and_object_payload_rejected_before_loading(tmp_path, monkeypatch):
    window, freqs, metadata, _ = example()
    path = save_frozen(tmp_path / "valid.npz", window, metadata=metadata, freqs=freqs)
    malicious = tmp_path / "huge-shape.npz"
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(buffer, {"descr": "<f4", "fortran_order": False, "shape": (1, 10 ** 9, 2, 12)})
    buffer.write(b"\0\0\0\0")
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(malicious, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name in source.namelist():
            target.writestr(name, buffer.getvalue() if name == "q_post.npy" else source.read(name))
    def forbidden(*args, **kwargs): raise AssertionError("malicious shape reached np.load")
    with monkeypatch.context() as patch:
        patch.setattr(np, "load", forbidden)
        with pytest.raises(ValueError, match="declared array shape"):
            load_frozen(malicious, max_capture_mb=1)
    objects = tmp_path / "object.npz"
    rewrite(path, objects, lambda arrays: arrays.update(q_post=np.empty((1, 4, 2, 12), dtype=object)))
    with pytest.raises(ValueError, match="object"):
        load_frozen(objects)


@pytest.mark.parametrize("corruption", ["missing_coords", "bad_coords", "missing_grid", "padding", "protection", "bf16_values"])
def test_corrupt_coordinates_grid_padding_protection_and_bf16_are_rejected(tmp_path, corruption):
    window, freqs, metadata, _ = example()
    path = save_frozen(tmp_path / "valid.npz", window, metadata=metadata, freqs=freqs)
    target = tmp_path / f"{corruption}.npz"
    def change(arrays):
        if corruption == "missing_coords": arrays.pop("key_coordinates")
        elif corruption == "bad_coords": arrays["key_coordinates"][0, 1] += 1
        elif corruption == "missing_grid":
            header = json.loads(arrays["metadata_json"].item()); header.pop("runtime_grid")
            arrays["metadata_json"] = np.array(json.dumps(header))
        elif corruption == "padding": arrays["valid_key_mask"][0] = False
        elif corruption == "protection": arrays["protected_mask"][0] = False
        else: arrays["q_post"][0, 0, 0, 0] = .1234567
    rewrite(path, target, change)
    with pytest.raises(ValueError): load_frozen(target)


def test_partial_kv_unknown_shapes_nonfinite_and_model_dump_metadata_are_rejected(tmp_path):
    window, freqs, metadata, _ = example()
    # Arbitrary token sampling loses a complete dense frame.
    short_context = replace(window.context, key_abs_positions=window.context.key_abs_positions[:-1])
    partial = replace(window, context=short_context, k_post=window.k_post[:, :-1], v=window.v[:, :-1],
                      k_pre=window.k_pre[:, :-1], pre_valid_mask=window.pre_valid_mask[:-1])
    with pytest.raises(ValueError, match="partial"):
        save_frozen(tmp_path / "partial.npz", partial)
    # Even whole-frame removal is rejected when it removes a current frame.
    ids = list(range(18))
    context = replace(window.context, key_abs_positions=window.context.key_abs_positions[ids])
    missing_current = replace(window, context=context, k_post=window.k_post[:, ids], v=window.v[:, ids],
                              k_pre=window.k_pre[:, ids], pre_valid_mask=window.pre_valid_mask[ids])
    with pytest.raises(ValueError, match="current-chunk"):
        save_frozen(tmp_path / "partial-current.npz", missing_current)
    bad = replace(window, q_post=window.q_post.clone())
    bad.q_post[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        save_frozen(tmp_path / "nan.npz", bad)
    multi = replace(window, q_post=window.q_post.expand(2, -1, -1, -1), k_post=window.k_post.expand(2, -1, -1, -1),
                    v=window.v.expand(2, -1, -1, -1), k_pre=window.k_pre.expand(2, -1, -1, -1))
    with pytest.raises(ValueError, match="one nonempty sample"):
        save_frozen(tmp_path / "batch.npz", multi)
    with pytest.raises(TypeError, match="history"):
        save_frozen(tmp_path / "history.npz", window, metadata={"all_cache_history": window.k_post})
    with pytest.raises(ValueError, match="partial"):
        save_frozen(tmp_path / "declared-partial.npz", window, metadata={"expected_key_token_count": 100})
    with pytest.raises(ValueError, match="runtime grid"):
        save_frozen(tmp_path / "bad-grid.npz", window, metadata={"runtime_grid": [2, 3, 2]})
    unknown_shape = replace(window, q_post=window.q_post.clone())
    unknown_shape.q_post.resize_(unknown_shape.q_post.numel())
    with pytest.raises(ValueError, match="unknown Q/K/V shape"):
        save_frozen(tmp_path / "unknown-shape.npz", unknown_shape)
