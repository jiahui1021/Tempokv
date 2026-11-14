"""Install AQUA-KV into the Self-Forcing pre-RoPE KV cache.

Collection (``AQUAKV_COLLECT=1``) keeps the BF16 chunked cache and writes each
clean block to a memmap. Generation replaces those slots with ``AquaRuntime``,
which quantizes a block only after the clean refresh. Cross-attention is left
exact. RoPE stays in the attention module: this cache stores post-RMSNorm,
pre-RoPE keys, which is the tensor Wan attention consumes.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


def _prompt_sha(prompt_file: Path, index: int) -> str:
    lines = [line.strip() for line in prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if index < 0 or index >= len(lines):
        raise ValueError(f"prompt index {index} is outside {prompt_file}")
    return hashlib.sha256(lines[index].encode()).hexdigest()


class _Collector:
    def __init__(self, root: Path, prompt_file: Path, *, frames: int, frame_seq: int, layers: int,
                 heads: int, dim: int):
        self.root = root
        self.prompt_file = prompt_file
        self.frames = frames
        self.frame_seq = frame_seq
        self.layers = layers
        self.heads = heads
        self.dim = dim
        self.tokens = frames * frame_seq
        self.prompt_index: int | None = None
        self.seed: int | None = None
        self._maps: dict[str, np.memmap] | None = None
        self._folder: Path | None = None
        self._meta: dict[str, Any] = {}

    def set_context(self, prompt_index: int, seed: int) -> None:
        self.prompt_index = int(prompt_index)
        self.seed = int(seed)
        self._maps = None
        self._folder = None

    def _open(self, batch: int) -> dict[str, np.memmap]:
        if self.prompt_index is None or self.seed is None:
            raise RuntimeError("AQUA-KV collection saw a clean block before set_research_context")
        if batch != 1:
            raise ValueError("AQUA-KV collection stores one video at a time; set samples=1")
        rank = os.environ.get("RANK", "0")
        folder = self.root / f"rank-{rank}" / f"prompt-{self.prompt_index:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        shape = (self.layers, self.tokens, self.heads, self.dim)
        maps = {}
        for role in ("k", "v"):
            path = folder / f"{role}.i16"
            maps[role] = np.memmap(path, dtype=np.int16, mode="w+", shape=shape)
        meta = {
            "format": "aquakv_self_forcing_kv_v1",
            "prompt_index": self.prompt_index,
            "prompt_sha256": _prompt_sha(self.prompt_file, self.prompt_index),
            "seed": self.seed,
            "layers": self.layers,
            "frames": self.frames,
            "frame_seq_length": self.frame_seq,
            "num_heads": self.heads,
            "head_dim": self.dim,
            "batch": 1,
            "dtype": "bfloat16",
            "storage": "int16_bitcast",
            "layout": "[layer, token, head, dim]",
            "complete": False,
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._meta = meta
        self._folder = folder
        self._maps = maps
        return maps

    @torch.no_grad()
    def write_block(self, cache: list[dict[str, Any]], start: int, end: int) -> None:
        if end <= start:
            return
        if end > self.tokens:
            raise ValueError("clean block extends past AQUAKV_NUM_FRAMES")
        sample = cache[0]["k"].read(start, end)
        maps = self._maps or self._open(int(sample.shape[0]))
        for layer, entry in enumerate(cache):
            for role in ("k", "v"):
                tensor = entry[role].read(start, end)
                if tensor.shape[0] != 1:
                    raise ValueError("AQUA-KV collection stores one video at a time")
                flat = tensor[0].detach().to(dtype=torch.bfloat16).contiguous().view(torch.int16).cpu().numpy()
                maps[role][layer, start:end] = flat
        maps["k"].flush()
        maps["v"].flush()
        if end == self.tokens and self._folder is not None:
            self._meta["complete"] = True
            (self._folder / "meta.json").write_text(
                json.dumps(self._meta, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )


def install(runtime_root: str | Path) -> type:
    runtime_root = Path(runtime_root).resolve()
    project_root = runtime_root.parents[1]
    for path in (project_root, project_root / "src", runtime_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from rebuttal.experiments.aquakv.cache import AquaRuntime
    from rebuttal.experiments.aquakv.method import AquaSpec, bank_from_state

    spec_path = os.environ.get("AQUAKV_SPEC")
    if not spec_path:
        raise ValueError("AQUAKV_SPEC must point at the experiment JSON")
    payload = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    spec = AquaSpec.from_dict(payload)
    collecting = os.environ.get("AQUAKV_COLLECT") == "1"
    frames = int(os.environ.get("AQUAKV_NUM_FRAMES", "0"))
    if frames < 1:
        raise ValueError("AQUAKV_NUM_FRAMES must be the latent rollout length")
    bank = None
    if not collecting:
        predictor_path = os.environ.get("AQUAKV_PREDICTORS")
        if not predictor_path:
            raise ValueError("AQUA-KV generation needs AQUAKV_PREDICTORS")
        bank = bank_from_state(torch.load(predictor_path, map_location="cpu"))
        if bank.spec.to_dict() != spec.to_dict():
            raise ValueError("AQUAKV_PREDICTORS was fitted with a different spec than AQUAKV_SPEC")
    elif not os.environ.get("AQUAKV_TRACE_DIR") or not os.environ.get("AQUAKV_PROMPT_FILE"):
        raise ValueError("AQUA-KV collection needs AQUAKV_TRACE_DIR and AQUAKV_PROMPT_FILE")

    pipeline_module = importlib.import_module("pipeline.causal_inference")
    base = pipeline_module.CausalInferencePipeline

    class AquaKVCausalInferencePipeline(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._uses_pre_rope_cache = True
            # The base pipeline calls quantize_kv_cache after every clean block
            # only when this flag is set. AQUA uses that hook and no other one.
            self._incremental_tempokv = True
            self._aquakv_spec = spec
            self._aquakv_bank = bank
            self._aquakv_runtime: AquaRuntime | None = None
            self._aquakv_collector = None
            if collecting:
                self._aquakv_collector = _Collector(
                    Path(os.environ["AQUAKV_TRACE_DIR"]),
                    Path(os.environ["AQUAKV_PROMPT_FILE"]),
                    frames=frames,
                    frame_seq=self.frame_seq_length,
                    layers=self.num_transformer_blocks,
                    heads=self.num_heads,
                    dim=self.head_dim,
                )
            if self.num_transformer_blocks != spec.num_layers or self.num_heads != spec.num_heads or self.head_dim != spec.head_dim:
                raise ValueError(
                    f"Wan shape {(self.num_transformer_blocks, self.num_heads, self.head_dim)} "
                    f"does not match the AQUA spec {(spec.num_layers, spec.num_heads, spec.head_dim)}"
                )
            if self.frame_seq_length != 1560:
                raise ValueError("AQUA-KV Self-Forcing bridge expects frame_seq_length 1560")
            for block in self.generator.model.blocks:
                block.self_attn.max_attention_size = frames * self.frame_seq_length

        def set_research_context(self, *, prompt_index: int, seed: int, **metadata: Any) -> None:
            del metadata
            if self._aquakv_collector is not None:
                self._aquakv_collector.set_context(prompt_index, seed)

        def _initialize_kv_cache(self, *args: Any, **kwargs: Any) -> Any:
            result = super()._initialize_kv_cache(*args, **kwargs)
            if collecting:
                return result
            batch = int(kwargs.get("batch_size", args[0] if args else 1))
            dtype = kwargs.get("dtype", args[1] if len(args) > 1 else torch.bfloat16)
            device = kwargs.get("device", args[2] if len(args) > 2 else torch.device("cuda"))
            target = int(kwargs.get("target_frames") or frames)
            runtime = AquaRuntime(
                spec, bank, batch_size=batch, frame_seq_length=self.frame_seq_length,
                max_frames=target, dtype=dtype, device=device,
            )
            self._aquakv_runtime = runtime
            for layer in range(self.num_transformer_blocks):
                self.kv_cache1[layer]["k"] = runtime.slot(layer, "k")
                self.kv_cache1[layer]["v"] = runtime.slot(layer, "v")
            return result

        def quantize_kv_cache(self, tokens_to_quantize_start: int, tokens_to_quantize_end: int,
                              max_tokens_to_quantize: int) -> None:
            if tokens_to_quantize_end > max_tokens_to_quantize:
                raise ValueError("quantization range exceeds the available cache")
            if self._aquakv_collector is not None:
                self._aquakv_collector.write_block(
                    self.kv_cache1, tokens_to_quantize_start, tokens_to_quantize_end,
                )
                return
            if self._aquakv_runtime is None:
                raise RuntimeError("AQUA-KV cache was not initialized")
            with torch.no_grad():
                stats = self._aquakv_runtime.compress_finalized(int(tokens_to_quantize_end))
            self.summary["quantization_events"].append({
                "tokens_start": int(tokens_to_quantize_start),
                "tokens_end": int(tokens_to_quantize_end),
                "quant_end": stats["quant_end"],
                "quantized_tokens": stats["quantized_tokens"],
                "codec": "aquakv",
            })

        def _resident_kv_bytes(self) -> tuple[int, int]:
            if self._aquakv_runtime is None:
                return super()._resident_kv_bytes()
            info = self._aquakv_runtime.resident_bytes()
            tokens = int(self.kv_cache1[0]["local_end_index"])
            adapter = Path(__file__).resolve().parent
            if str(adapter) not in sys.path:
                sys.path.insert(0, str(adapter))
            from efficiency_record import bf16_equivalent_bytes
            equivalent = bf16_equivalent_bytes(
                batch=self._aquakv_runtime.batch, tokens=tokens,
                heads=self.num_heads, head_dim=self.head_dim,
            ) * len(self.kv_cache1)
            return info["kv_bytes"] + info["predictor_bytes"], equivalent

    pipeline_module.CausalInferencePipeline = AquaKVCausalInferencePipeline
    pipeline_package = importlib.import_module("pipeline")
    pipeline_package.CausalInferencePipeline = AquaKVCausalInferencePipeline
    return AquaKVCausalInferencePipeline
