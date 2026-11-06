"""Enumerate independent-span capacity using the actual packed tensor shapes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tempokv.config import CodecConfig, load_config


@dataclass(frozen=True)
class ModelShape:
    layers: int = 30
    heads: int = 12
    head_dim: int = 128
    tokens_per_frame: int = 1560
    batch_size: int = 1

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in self.__dict__.values()):
            raise ValueError("model_shape dimensions must be positive integers")

    @property
    def bf16_frame_bytes(self):
        return self.layers * self.batch_size * self.tokens_per_frame * self.heads * self.head_dim * 2 * 2


def codec_config(config: dict) -> CodecConfig:
    path = config.get("codec_config")
    overrides = config.get("codec_overrides", {})
    return load_config(path, overrides) if path else CodecConfig(**overrides)


def span_bytes(method: str, frames: int, shape: ModelShape,
               codec: CodecConfig, qvg: dict | None = None) -> int:
    if frames < 1:
        return 0
    tokens = shape.tokens_per_frame * frames
    bh = shape.batch_size * shape.heads
    d = shape.head_dim
    if method == "bf16":
        return shape.bf16_frame_bytes * frames
    if method == "tempokv":
        if d % codec.group_size:
            raise ValueError("head_dim must be divisible by TempoKV group_size")
        g = d // codec.group_size
        scale_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}[codec.scale_dtype]
        anchor_tokens = shape.tokens_per_frame
        if codec.raw_anchor:
            anchor = anchor_tokens * d * {"float32": 4, "float16": 2, "bfloat16": 2}[codec.reconstruction_dtype]
        else:
            anchor = anchor_tokens * ((d * codec.anchor_bits + 7) // 8 + g * scale_bytes)
        payload = 0
        for bits in (codec.k_bits, codec.v_bits):
            residual_tokens = tokens - anchor_tokens
            codes = residual_tokens * ((d * bits + 7) // 8)
            scale = residual_tokens * g * scale_bytes
            zero = residual_tokens * g if codec.residual_mode == "asymmetric" else 0
            payload += anchor + codes + scale + zero
        return shape.layers * bh * payload
    if method == "qvg":
        c = {"bits": 4, "stages": 2, "k_centroids": 256, "v_centroids": 256,
             "block_size": 64, **(qvg or {})}
        if c["bits"] not in (2, 4) or d % c["block_size"]:
            raise ValueError("QVG supports INT2/INT4 and head_dim divisible by block_size")
        if not 1 <= c["stages"] or any(not 1 <= c[k] <= 256 for k in ("k_centroids", "v_centroids")):
            raise ValueError("QVG packed stages/centroid configuration is invalid")
        payload = 0
        for key in ("k_centroids", "v_centroids"):
            if tokens < c[key]:
                raise ValueError("QVG span tokens must cover the configured centroids")
            # functions.triton_prq_quantize_tensor: BF16 centroids, uint8 IDs,
            # PACK_OUTPUT_INT8=True, BF16 scales; no clipping side tensors.
            payload += (c["stages"] * (c[key] * d * 2 + tokens) +
                        tokens * ((d * c["bits"] + 7) // 8) +
                        tokens * (d // c["block_size"]) * 2)
        return shape.layers * bh * payload
    raise ValueError(f"unknown method {method!r}")


def live_profile(method: str, history: int, horizon: int, shape: ModelShape,
                 codec: CodecConfig, *, span_frames: int = 12,
                 lag_frames: int = 18, block_frames: int = 3, qvg: dict | None = None) -> list[dict]:
    """Before-write high water for the worker's batch independent-span schedule."""
    if min(history, horizon, span_frames, block_frames) < 1 or lag_frames < 0:
        raise ValueError("invalid history/horizon/span/block/lag")
    if span_frames % block_frames:
        raise ValueError("span_frames must be divisible by block_frames")
    if method != "bf16" and history < span_frames + lag_frames + block_frames:
        raise ValueError("history is too short for this span/lag schedule")
    full = span_bytes(method, span_frames, shape, codec, qvg)
    rows = []
    for end in range(block_frames, horizon + block_frames, block_frames):
        end = min(end, horizon)
        start = max(0, end - history)
        encoded = 0 if method == "bf16" else max(0, end - block_frames - lag_frames) // span_frames * span_frames
        first_span = start // span_frames
        packed_count = max(0, encoded // span_frames - first_span) if encoded > start else 0
        packed = packed_count * full
        dense_frames = max(0, end - max(start, encoded))
        dense = dense_frames * shape.bf16_frame_bytes
        rows.append({"end_frame": end, "visible_frames": min(history, end),
                     "decode_support_start": first_span * span_frames if packed_count else start,
                     "packed_bytes": packed, "dense_bytes": dense,
                     "kv_owned_bytes": packed + dense,
                     "measurement": "shape_accounting_not_device_measurement"})
    return rows


def capacity_report(config: dict) -> dict:
    shape = ModelShape(**config.get("model_shape", {}))
    codec = codec_config(config)
    block = int(config.get("block_frames", 3))
    span = int(config.get("span_frames", 12))
    lag = int(config.get("bf16_lag_frames", 18))
    horizon = int(config.get("horizon_latent_frames", (int(config.get("frames", 1401)) + 3) // 4))
    if horizon < 1:
        raise ValueError("horizon_latent_frames must be positive")
    histories = config.get("histories")
    if histories is None:
        histories = sorted(set(range(span, horizon + 1, span)) | {horizon})
    histories = sorted(set(int(h) for h in histories if 0 < int(h) <= horizon))
    if not histories:
        raise ValueError("no legal candidate histories")
    for h in histories:
        if h != horizon and h % span:
            raise ValueError("history candidates must align to independent span boundaries")
    budgets = config.get("budgets_bytes") or [shape.bf16_frame_bytes * int(h)
                                              for h in config.get("budget_basis_latent_frames", [48, 96, 192])]
    if any(int(b) < 1 for b in budgets):
        raise ValueError("budgets_bytes must be positive")
    methods = ["qvg" if m in ("qvg_prq_real_codec", "qvg_prq") else m
               for m in config.get("methods", ["bf16", "qvg", "tempokv"])]
    profiles, frozen_profiles, unsupported = {}, {}, []
    for method in methods:
        for h in histories:
            try:
                frozen = h // span * span_bytes(method, span, shape, codec, config.get("qvg")) + h % span * shape.bf16_frame_bytes
                frozen_profiles[(method, h)] = frozen
                events = live_profile(method, h, horizon, shape, codec, span_frames=span,
                                      lag_frames=lag, block_frames=block, qvg=config.get("qvg"))
                # Frozen storage uses full independent spans and a trailing dense
                # incomplete span; it is deliberately separate from live peak.
                profiles[(method, h)] = {"live_peak_bytes": max(e["kv_owned_bytes"] for e in events),
                                         "frozen_storage_bytes": frozen, "events": events}
            except ValueError as error:
                unsupported.append({"method": method, "history_frames": h, "reason": str(error)})
    rows = []
    fps = float(config.get("fps", 16))
    if fps <= 0:
        raise ValueError("fps must be positive")
    for budget in budgets:
        budget = int(budget)
        for method in methods:
            candidates = [(h, r) for (m, h), r in profiles.items() if m == method and r["live_peak_bytes"] <= budget]
            best = max(candidates, key=lambda item: item[0]) if candidates else None
            frozen_candidates = [(h, n) for (m, h), n in frozen_profiles.items() if m == method and n <= budget]
            frozen_best = max(frozen_candidates, key=lambda item: item[0]) if frozen_candidates else None
            rows.append({"method": method, "budget_bytes": budget, "budget_gib": budget / 2**30,
                         "history_frames": best[0] if best else None,
                         "history_seconds": max(0, 4 * best[0] - 3) / fps if best else None,
                         "live_peak_bytes": best[1]["live_peak_bytes"] if best else None,
                         "frozen_storage_bytes": best[1]["frozen_storage_bytes"] if best else None,
                         "unused_bytes": budget - best[1]["live_peak_bytes"] if best else None,
                         "frozen_capacity_history_frames": frozen_best[0] if frozen_best else None,
                         "frozen_capacity_bytes": frozen_best[1] if frozen_best else None,
                         "reached_horizon": bool(best and best[0] == horizon),
                         "status": "estimated_feasible" if best else "infeasible",
                         "measurement": "exact_tensor_shape_schedule_not_device_measurement"})
    return {"schema_version": "tempokv.rebuttal.memory.capacity.v1", "status": "capacity_accounted",
            "model_shape": shape.__dict__, "bf16_frame_bytes": shape.bf16_frame_bytes,
            "span_frames": span, "bf16_lag_frames": lag, "block_frames": block,
            "horizon_latent_frames": horizon, "rows": rows, "unsupported": unsupported,
            "profile_rows": [{"method": m, "history_frames": h,
                              "live_peak_bytes": r["live_peak_bytes"],
                              "frozen_storage_bytes": r["frozen_storage_bytes"]}
                             for (m, h), r in profiles.items()],
            "budget_scope": "KV_owned_state_only; static_parameters_workspace_model_weights_separate",
            "schedule": "batch_independent_spans; prefix_release; no_incremental_predictor_history",
            "measurement_note": "shape-derived capacity; measured worker checks every state change"}
