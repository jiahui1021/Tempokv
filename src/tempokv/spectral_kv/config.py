"""Suite B configuration is intentionally incompatible with the suite A schema."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .method_spec import parse_method

SUITE_ID = "B_SPECTRAL_KV"
EXPERIMENTS = {
    "B_E1_SPECTRUM", "B_E2_CHUNK_AXIS", "B_E2B_REFRESH", "B_E3_OFF_SHADOW",
    "B_E3_ON_ROLLOUT", "B_E4_DISTANCE", "B_E5_RESIDUAL",
}


@dataclass(frozen=True)
class SpectralKVConfig:
    suite_id: str = SUITE_ID
    profile: str = "readonly"
    experiments: tuple[str, ...] = ("B_E1_SPECTRUM", "B_E2_CHUNK_AXIS", "B_E2B_REFRESH", "B_E3_OFF_SHADOW", "B_E4_DISTANCE")
    seed: int = 42
    layers: tuple[int, ...] = (0, 15, 29)
    heads: tuple[int, ...] = (0, 6)
    chunks: tuple[int, ...] = (4, 8, 12)
    denoise_steps: tuple[int, ...] = (0, -1)
    query_tokens: int = 32
    max_analysis_calls_per_layer: int = 12
    max_clean_frames: int = 192
    max_chunks: int = 64
    spatial_bins: int = 8
    temporal_bins: int = 3
    temporal_window: int = 8
    max_lag: int = 24
    correlation_threshold: float = 0.9
    feature_channels: int = 16
    keep_recent_frames: int = 6
    sink_frames: int = 1
    shadow_methods: tuple[str, ...] = (
        "DROP_win6", "SPECSUM_dct_R4_B0", "SPECSUM_dct_R4_B1", "SPECSUM_avg_R4_B1",
        "SPECSUM_dct_R16_B1", "SPECSUM_avg_R16_B1", "SPECSUM_dct_R4_B1_top8",
    )
    method: str = "BF16"
    online: bool = False
    keep_originals: bool = False
    attention_backend: str = "reference"
    calibration_file: str | None = None
    collect_calibration: bool = False
    calibration_samples_per_chunk: int = 64
    calibration_ridge: float = 1e-4
    rope_precision: str = "runtime"
    max_stat_rows: int = 100000
    eps: float = 1e-12

    def __post_init__(self):
        import math
        if self.suite_id != SUITE_ID or not self.experiments or set(self.experiments) - EXPERIMENTS:
            raise ValueError("suite B requires explicit B_* experiment IDs and suite_id=B_SPECTRAL_KV")
        for key in ("layers", "heads", "chunks"):
            values = getattr(self, key)
            if len(set(values)) != len(values) or any(type(v) is not int or v < 0 for v in values):
                raise ValueError(f"{key} must contain unique nonnegative integers (empty means all)")
        if any(type(v) is not int or v < -1 for v in self.denoise_steps):
            raise ValueError("denoise_steps uses zero-based indices or -1 for the actual last step")
        for key in ("query_tokens", "max_analysis_calls_per_layer", "max_clean_frames", "spatial_bins", "temporal_bins",
                    "temporal_window", "max_lag", "feature_channels", "calibration_samples_per_chunk", "max_stat_rows", "max_chunks"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        for key in ("keep_recent_frames", "sink_frames"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if not 0 <= self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be in [0,1]")
        if not math.isfinite(self.eps) or self.eps <= 0 or not math.isfinite(self.calibration_ridge) or self.calibration_ridge <= 0:
            raise ValueError("eps and calibration_ridge must be finite and positive")
        if self.attention_backend not in {"reference", "group_lse"} or self.rope_precision not in {"runtime", "float32", "float64"}:
            raise ValueError("invalid attention_backend or rope_precision")
        if not self.profile or "/" in self.profile or "\\" in self.profile or self.profile in {".", ".."}:
            raise ValueError("profile must be a single nonempty directory name")
        method = parse_method(self.method)
        for name in self.shadow_methods:
            if parse_method(name).family == "kvskip":
                raise ValueError("KVSKIP needs an online paired rollout, not a shadow attention case")
        if method.family != "full" and not self.online:
            raise ValueError("non-BF16 runtime methods require online=true and a separate experimental profile")
        if self.online and not set(self.experiments).intersection({"B_E3_ON_ROLLOUT", "B_E2B_REFRESH"}):
            raise ValueError("online methods must be labeled B_E3_ON_ROLLOUT or B_E2B_REFRESH")
        if method.skip == "lin" and not self.calibration_file:
            raise ValueError("KVSKIP_lin requires a separately fitted calibration_file")
        if self.collect_calibration and self.online:
            raise ValueError("calibration must be collected on a read-only BF16 trajectory")
        if self.temporal_window < 2:
            raise ValueError("temporal_window must be at least two frames")
        if method.topk is not None and self.online and not self.keep_originals:
            raise ValueError("online top-k selection needs keep_originals=true to restore previously summarized frames")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload)
        for key in ("experiments", "layers", "heads", "chunks", "denoise_steps", "shadow_methods"):
            if key in payload:
                payload[key] = tuple(payload[key])
        return cls(**payload)


def load_config(path):
    source = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(source)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("ordinary YAML requires PyYAML; JSON-compatible YAML needs no extra parser") from exc
        payload = yaml.safe_load(source)
    if not isinstance(payload, dict):
        raise ValueError("suite B configuration must be a mapping")
    if payload.get("calibration_file"):
        calibration = Path(payload["calibration_file"]).expanduser()
        payload["calibration_file"] = str((Path(path).resolve().parent / calibration).resolve())
    return SpectralKVConfig.from_dict(payload)


def runtime_environment():
    import os
    missing = [key for key in ("SPECTRAL_KV_CONFIG", "SPECTRAL_KV_OUTPUT_DIR") if not os.environ.get(key)]
    if missing:
        raise ValueError(f"suite B requires {', '.join(missing)}")
    paths = {key: str(Path(os.environ[key]).expanduser().resolve()) for key in ("SPECTRAL_KV_CONFIG", "SPECTRAL_KV_OUTPUT_DIR")}
    config = load_config(paths["SPECTRAL_KV_CONFIG"])
    if config.calibration_file and not Path(config.calibration_file).is_file():
        raise ValueError(f"KVSKIP_lin calibration file does not exist: {config.calibration_file}")
    return paths
