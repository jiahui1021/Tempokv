"""Independent research.v1 configuration; legacy A/B experiment IDs are invalid."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path

EXPERIMENTS = ("S1", "S2", "S3", "E1", "E2", "E3", "E4", "E5", "E6", "E7")
MILESTONES = {"M0": (), "M1": ("S1", "S2", "E1", "E3", "E4"), "M2": ("E1", "E2", "E6"), "M3": ("S3", "E7"), "M4": ("E5",)}

@dataclass(frozen=True)
class Selection:
    layers: tuple[int, ...] = (0, 7, 15, 22, 29)
    heads: tuple[int, ...] = (0, 3, 6, 9)
    chunk_indices: tuple[int, ...] = (4, 8, 16)
    denoise_step_indices: tuple[int, ...] = (0, 1, 2, 3)

@dataclass(frozen=True)
class Sampling:
    queries_per_head: int = 64
    max_history_blocks_per_event: int = 64
    spatial_block_hw: tuple[int, int] = (4, 4)
    protect_recent_latent_frames: int = 6
    protect_current_chunk: bool = True
    sink_policy: str = "preserve"
    max_capture_mb: float = 256
    clean_blocks_per_stream: int = 256
    max_clean_frames: int = 192
    max_pairs: int = 8
    max_pair_frame_distance: int = 24

@dataclass(frozen=True)
class Spectral:
    cutoffs: tuple[float, ...] = (.10, .25, .40)
    score_cutoff: float = .25
    temporal_windows: tuple[int, ...] = (8, 16, 32)
    temporal_positions: int = 32

@dataclass(frozen=True)
class Oracle:
    query_batch_size: int = 64
    block_batch_size: int = 4
    reference_dtype: str = "bf16"
    key_chunk_size: int = 1024

@dataclass(frozen=True)
class Intervention:
    kind: str = "drop"
    role: str = "kv"
    strengths: tuple[float, ...] = (.01, .03, .10)
    horizon_chunks: tuple[int, ...] = (1, 2, 4, 8)
    calibration: str = "relative_kv_rmse"
    target_layer: int = 0
    target_head: int = 0
    target_chunk: int = 4
    target_step: int = 0
    target_block_rank: int = 0
    paired_rollout: bool = False
    max_branches: int = 4
    e6_step_policies: tuple[tuple[float, ...], ...] = ()
    e5_scope: str = "selected"

@dataclass(frozen=True)
class Compression:
    retention: tuple[float, ...] = (.25, .50, .75)
    policies: tuple[str, ...] = ("random", "uniform", "similarity", "attention", "frequency", "frequency_qk", "oracle")
    bits: tuple[int, ...] = (2, 3, 4, 6, 8)
    mixed_bits: tuple[tuple[int, int], ...] = ((4, 2), (2, 4), (6, 4), (4, 6))
    group_size: int = 64
    allocation_max_states: int = 4096

@dataclass(frozen=True)
class Output:
    root: str = "reports/research/self_forcing"
    save_dense_kv: bool = False
    save_videos: bool = False

@dataclass(frozen=True)
class ResearchConfig:
    schema_version: str = "research.v1"
    backend: str = "self_forcing"
    source: str = "bf16"
    experiments: tuple[str, ...] = ("S1", "S2", "E1", "E3", "E4")
    seed: int = 0
    frames: int = 237
    local_attn_size: int = 24
    sink_size: int = 0
    max_events: int = 24
    max_gpu_memory_gb: float | None = None
    selection: Selection = field(default_factory=Selection)
    sampling: Sampling = field(default_factory=Sampling)
    spectral: Spectral = field(default_factory=Spectral)
    oracle: Oracle = field(default_factory=Oracle)
    intervention: Intervention = field(default_factory=Intervention)
    compression: Compression = field(default_factory=Compression)
    output: Output = field(default_factory=Output)
    eps: float = 1e-12

    def __post_init__(self):
        if self.schema_version != "research.v1" or self.backend != "self_forcing":
            raise ValueError("research.v1 currently supports backend=self_forcing")
        if not self.experiments or set(self.experiments) - set(EXPERIMENTS) or len(set(self.experiments)) != len(self.experiments):
            raise ValueError("use fixed S1-S3/E1-E7 experiment IDs, without duplicates")
        if self.source not in {"bf16", "tempokv"} or (self.source == "tempokv" and set(self.experiments) != {"E7"}):
            raise ValueError("source=tempokv is restricted to explicitly labeled E7; core reference is bf16")
        if type(self.seed) is not int or self.seed < 0 or self.frames < 1 or (self.frames + 3) % 12:
            raise ValueError("seed must be nonnegative; video frames must satisfy (FRAMES+3)%12=0")
        if self.local_attn_size == 0 or self.local_attn_size < -1 or self.sink_size < 0:
            raise ValueError("invalid attention window or sink")
        if self.local_attn_size > 0 and self.sink_size >= self.local_attn_size:
            raise ValueError("sink must be smaller than the actual window")
        for name in ("layers", "heads", "chunk_indices", "denoise_step_indices"):
            x = getattr(self.selection, name)
            if not x or len(set(x)) != len(x) or any(type(v) is not int or v < 0 for v in x):
                raise ValueError(f"selection.{name} needs unique nonnegative integers")
        if max(self.selection.chunk_indices) >= self.latent_frames:  # Runtime chunk size is validated again at installation.
            raise ValueError("selected chunk cannot exist within the latent-frame horizon")
        for value in (self.max_events, self.sampling.queries_per_head, self.sampling.max_history_blocks_per_event,
                      self.sampling.clean_blocks_per_stream, self.sampling.max_clean_frames, self.sampling.max_pairs,
                      self.oracle.query_batch_size, self.oracle.block_batch_size, self.oracle.key_chunk_size,
                      self.compression.group_size, self.compression.allocation_max_states, self.intervention.max_branches):
            if type(value) is not int or value < 1: raise ValueError("event, sampling and batching limits must be positive integers")
        if len(self.sampling.spatial_block_hw) != 2 or any(type(v) is not int or v < 1 for v in self.sampling.spatial_block_hw):
            raise ValueError("spatial_block_hw must contain two positive dimensions")
        if not self.sampling.protect_current_chunk or self.sampling.protect_recent_latent_frames < 0 or self.sampling.sink_policy != "preserve":
            raise ValueError("current/recent/sink protection cannot be disabled in core experiments")
        if not math.isfinite(self.sampling.max_capture_mb) or self.sampling.max_capture_mb <= 0:
            raise ValueError("max_capture_mb must be finite and positive")
        if self.max_gpu_memory_gb is not None and (not math.isfinite(self.max_gpu_memory_gb) or self.max_gpu_memory_gb <= 0):
            raise ValueError("max_gpu_memory_gb must be finite and positive")
        if any(not math.isfinite(x) or not 0 <= x <= math.sqrt(.5) for x in (*self.spectral.cutoffs, self.spectral.score_cutoff)):
            raise ValueError("spatial cutoffs use radial cycles/token in [0,sqrt(.5)]")
        if not self.spectral.temporal_windows or any(type(x) is not int or x < 2 for x in self.spectral.temporal_windows):
            raise ValueError("temporal windows need at least two contiguous frames")
        if not self.spectral.cutoffs or type(self.spectral.temporal_positions) is not int or self.spectral.temporal_positions < 1:
            raise ValueError("spectral cutoffs and positive temporal_positions are required")
        if type(self.sampling.max_pair_frame_distance) is not int or self.sampling.max_pair_frame_distance < 0:
            raise ValueError("max_pair_frame_distance must be nonnegative")
        if self.intervention.kind not in {"none", "drop", "gaussian", "low_rank", "temporal_noise", "quantization_like", "zero", "sparse"}:
            raise ValueError("unsupported intervention")
        for name in ("target_layer","target_head","target_chunk","target_step","target_block_rank"):
            if type(getattr(self.intervention,name)) is not int or getattr(self.intervention,name)<0:
                raise ValueError("intervention target indices must be nonnegative integers")
        if self.intervention.paired_rollout and (self.intervention.target_layer not in self.selection.layers or self.intervention.target_head not in self.selection.heads):
            raise ValueError("paired target layer/head must be included in clean capture selection")
        if self.intervention.role not in {"k", "v", "kv"} or self.intervention.calibration not in {"relative_kv_rmse", "local_d_att"}:
            raise ValueError("invalid intervention role or calibration protocol")
        if self.intervention.e5_scope not in {"selected","all"}:
            raise ValueError("E5 scope must be selected or all runtime layers/heads")
        if any(not math.isfinite(x) or x < 0 for x in self.intervention.strengths): raise ValueError("strengths must be nonnegative")
        if any(type(x) is not int or x < 0 for x in self.intervention.horizon_chunks): raise ValueError("horizons must be nonnegative")
        if any(not 0 <= x <= 1 for x in self.compression.retention): raise ValueError("retention must be in [0,1]")
        allowed = {"random", "uniform", "similarity", "attention", "frequency", "frequency_qk", "oracle"}
        if set(self.compression.policies) - allowed: raise ValueError("unsupported selection policy")
        valid_bits=set(range(2,9))|{16}
        if not self.compression.bits or any(type(x) is not int or x not in valid_bits for x in self.compression.bits):
            raise ValueError("bits must be 2..8 or native 16")
        if any(len(pair)!=2 or any(type(x) is not int or x not in valid_bits for x in pair) for pair in self.compression.mixed_bits):
            raise ValueError("mixed_bits must contain K/V pairs of 2..8 or native 16")
        if any(not 0 <= r <= 1 for policy in self.intervention.e6_step_policies for r in policy): raise ValueError("invalid step retention")
        if self.oracle.reference_dtype not in {"bf16", "native"}: raise ValueError("reference uses native production dtype")
        if not math.isfinite(self.eps) or self.eps <= 0: raise ValueError("eps must be positive")

    @property
    def latent_frames(self): return (self.frames + 3) // 4

    def to_dict(self): return asdict(self)

    @property
    def signature(self):
        payload = self.to_dict(); payload["output"].pop("root")
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key, kind in (("selection", Selection), ("sampling", Sampling), ("spectral", Spectral),
                          ("oracle", Oracle), ("intervention", Intervention), ("compression", Compression), ("output", Output)):
            values = dict(data.get(key, {})); defaults = kind()
            for name, value in values.items():
                if isinstance(getattr(defaults, name, None), tuple):
                    values[name] = tuple(tuple(x) if isinstance(x, list) else x for x in value)
            data[key] = kind(**values)
        if "experiments" in data: data["experiments"] = tuple(data["experiments"])
        return cls(**data)

def load_config(path):
    text = Path(path).read_text(encoding="utf-8")
    try: payload = json.loads(text)
    except json.JSONDecodeError:
        try: import yaml
        except ImportError as exc: raise ValueError("install PyYAML or use JSON-compatible YAML") from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict): raise ValueError("research config must be a mapping")
    return ResearchConfig.from_dict(payload)
