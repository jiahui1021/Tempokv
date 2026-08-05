from __future__ import annotations

from tempokv.runtime import configure_triton_cache


def test_triton_cache_is_process_scoped(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "3")
    path = configure_triton_cache(tmp_path)
    assert path.is_dir()
    assert path.name.endswith("rank-3")


def test_existing_triton_cache_is_respected(tmp_path, monkeypatch) -> None:
    existing = tmp_path / "shared"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(existing))
    assert configure_triton_cache(tmp_path / "ignored") == existing.resolve()
