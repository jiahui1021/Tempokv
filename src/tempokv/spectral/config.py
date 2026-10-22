"""Validated, codec-independent settings for the BF16 spectral pilot."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sampling:
    layers: tuple[int, ...] = (0, 15, 29)
    heads: tuple[int, ...] = (0, 6)
    latent_frames: tuple[int, ...] = (12, 24, 36)
    denoise_calls: tuple[int, ...] = (0,)
    query_tokens_per_head: int = 32
    max_candidate_blocks: int = 24
    only_phase: str = "denoise"
    max_oracle_calls_per_layer: int = 12


@dataclass(frozen=True)
class Frequency:
    spatial_block_size: tuple[int, int] = (4, 4)
    cutoffs: tuple[float, ...] = (0.10, 0.25, 0.40)
    score_cutoff: float = 0.25
    temporal_positions_per_frame: int = 32
    max_clean_frames: int = 192


@dataclass(frozen=True)
class Oracle:
    protect_recent_frames: int = 6
    sink_frames: int = 0
    key_chunk_size: int = 1024
    topk_fraction: float = 0.25


@dataclass(frozen=True)
class Compression:
    retention_ratios: tuple[float, ...] = (0.25, 0.50, 0.75)
    policies: tuple[str, ...] = (
        "random", "uniform", "similarity", "attention", "frequency", "frequency_attention"
    )
    random_seeds: tuple[int, ...] = (0, 1, 2)
    combination_alphas: tuple[float, ...] = (0.25, 0.50, 0.75)
    # Optional offline replay of a bounded number of single-head windows.
    snapshot_calls: int = 0
    max_snapshot_tokens: int = 100_000


@dataclass(frozen=True)
class SpectralConfig:
    source: str = "bf16"
    experiments: tuple[str, ...] = ("spectrum", "sensitivity", "compression")
    seed: int = 42
    sampling: Sampling = field(default_factory=Sampling)
    frequency: Frequency = field(default_factory=Frequency)
    oracle: Oracle = field(default_factory=Oracle)
    compression: Compression = field(default_factory=Compression)
    output_dir: str = "reports/spectral/self_forcing_pilot"
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.source != "bf16":
            raise ValueError("spectral analysis currently requires source=bf16")
        if not self.experiments or set(self.experiments) - {"spectrum", "sensitivity", "compression"}:
            raise ValueError("experiments must contain spectrum, sensitivity and/or compression")
        s, f, o, c = self.sampling, self.frequency, self.oracle, self.compression
        for name in ("layers", "heads", "latent_frames", "denoise_calls"):
            values = getattr(s, name)
            if (name in {"layers", "heads"} and not values) or any(
                type(x) is not int or x < 0 for x in values
            ) or len(values) != len(set(values)):
                raise ValueError(f"sampling.{name} must contain unique nonnegative integers")
        if s.only_phase != "denoise":
            raise ValueError("sampling.only_phase is an Oracle filter and must be denoise")
        for name, value in (
            ("query_tokens_per_head", s.query_tokens_per_head),
            ("max_candidate_blocks", s.max_candidate_blocks),
            ("max_oracle_calls_per_layer", s.max_oracle_calls_per_layer),
            ("temporal_positions_per_frame", f.temporal_positions_per_frame),
            ("max_clean_frames", f.max_clean_frames),
            ("key_chunk_size", o.key_chunk_size),
            ("max_snapshot_tokens", c.max_snapshot_tokens),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if len(f.spatial_block_size) != 2 or any(type(x) is not int or x < 1 for x in f.spatial_block_size):
            raise ValueError("spatial_block_size must have two positive dimensions")
        for name, values in (("cutoffs", f.cutoffs), ("retention_ratios", c.retention_ratios),
                             ("combination_alphas", c.combination_alphas)):
            if not values or any(not math.isfinite(x) for x in values):
                raise ValueError(f"{name} must contain finite values")
        if any(not 0 <= x <= math.sqrt(0.5) for x in (*f.cutoffs, f.score_cutoff)):
            raise ValueError("frequency cutoffs use radial cycles/token in [0, sqrt(0.5)]")
        if any(not 0 <= x <= 1 for x in (*c.retention_ratios, *c.combination_alphas)):
            raise ValueError("retention_ratios and combination_alphas must be in [0,1]")
        if not 0 < o.topk_fraction <= 1:
            raise ValueError("topk_fraction must be in (0,1]")
        if type(o.protect_recent_frames) is not int or o.protect_recent_frames < 0:
            raise ValueError("protect_recent_frames must be nonnegative")
        if o.sink_frames != 0:
            raise ValueError("spectral runtime requires sink_frames=0; gaps need explicit positions")
        if not math.isfinite(self.eps) or self.eps <= 0 or type(self.seed) is not int:
            raise ValueError("eps must be finite and positive, seed must be an integer")
        allowed = {"random", "uniform", "similarity", "attention", "frequency", "frequency_attention", "oracle_greedy"}
        if not c.policies or set(c.policies) - allowed or len(set(c.policies)) != len(c.policies):
            raise ValueError("unsupported or duplicate compression policy")
        if not c.random_seeds or any(type(x) is not int or x < 0 for x in c.random_seeds):
            raise ValueError("random_seeds must contain nonnegative integers")
        if type(c.snapshot_calls) is not int or not 0 <= c.snapshot_calls <= 4:
            raise ValueError("snapshot_calls must be in [0,4] (single-head windows only)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SpectralConfig:
        payload = dict(payload)
        for name, kind in (("sampling", Sampling), ("frequency", Frequency),
                           ("oracle", Oracle), ("compression", Compression)):
            section = dict(payload.pop(name, {}))
            defaults = kind()
            for key, value in section.items():
                if isinstance(getattr(defaults, key, None), tuple):
                    section[key] = tuple(value)
            payload[name] = kind(**section)
        if "experiments" in payload:
            payload["experiments"] = tuple(payload["experiments"])
        return cls(**payload)


def load_config(path: str | Path) -> SpectralConfig:
    source = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(source)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("YAML needs PyYAML: pip install -e '.[spectral]'; JSON is also supported") from exc
        payload = yaml.safe_load(source)
    if not isinstance(payload, dict):
        raise ValueError("spectral config must be a mapping")
    return SpectralConfig.from_dict(payload)
