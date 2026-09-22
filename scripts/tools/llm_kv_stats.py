"""Capture KV-cache activations of a small LLM (Qwen3.6-27B) and reuse the
video-side statistics to compare the two cache families.

Usage (single GPU): python llm_kv_stats.py OUT_JSON [MODEL_PATH]

Qwen3.6-27B is hybrid: only the full_attention layers keep a K/V cache
(linear_attention layers do not), and those cached layers are what we measure.

Long-context arm: concatenated local markdown corpus in 8k-token prefills.
Short-prompt arm: moviegen prompts (~60 tokens each), like text "frames".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kv_activation_stats import (aggregate, frame_stats, head_vector_dissim,
                                 profile_dissim, token_stats)

W = Path("/mnt/cpfs/epic-user/hujiahui-20260612")
MODEL = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/mnt/cpfs/public_data/public_model/Qwen/Qwen3.6-27B")
CHUNK = 8192
N_CHUNKS = 3


def build_corpus(tok) -> list[torch.Tensor]:
    texts = []
    for p in sorted((W / "Tempokv").rglob("*.md"))[:40]:
        try:
            texts.append(p.read_text())
        except OSError:
            continue
        if sum(len(t) for t in texts) > CHUNK * N_CHUNKS * 6:
            break
    blob = "\n\n".join(texts)
    ids = tok(blob, add_special_tokens=False).input_ids
    return [torch.tensor(ids[i * CHUNK : (i + 1) * CHUNK]) for i in range(N_CHUNKS)
            if (i + 1) * CHUNK <= len(ids)]


def cached_layers(past):
    """(index, keys, values) for layers that actually hold a K/V cache."""
    out = []
    for li, layer in enumerate(past):
        kt = getattr(layer, "keys", None)
        vt = getattr(layer, "values", None)
        if kt is None or vt is None:
            continue
        out.append((li, kt, vt))
    return out


def run(model, chunks_ids, short_ids):
    cfg = getattr(model.config, "text_config", None) or model.config
    kvh, hd = cfg.num_key_value_heads, cfg.head_dim
    out = {"model": str(MODEL), "full_attention_layers_only": True,
           "n_layers_total": cfg.num_hidden_layers, "arms": {}}

    def stats_from_cache(cache):
        past = cache.layers if hasattr(cache, "layers") else cache
        got = cached_layers(past)
        L = len(got)
        per = {"k": [], "v": []}
        sc = {"k": np.zeros((L, kvh, hd)), "v": np.zeros((L, kvh, hd))}
        hm = {"k": np.zeros((L, kvh, hd)), "v": np.zeros((L, kvh, hd))}
        for li, (l, key, val) in enumerate(got):
            for role, t in (("k", key), ("v", val)):
                for h in range(kvh):
                    x = t[0, h].to(torch.float32).cpu().numpy().astype(np.float64)
                    s = token_stats(x)
                    s["_n"] = x.shape[0]
                    s.update(frame_stats(x, 0, 1))  # LLM has no spatial axis
                    per[role].append(s)
                    sc[role][li, h] = x.std(axis=0)
                    hm[role][li, h] = x.mean(axis=0)
        return per, sc, hm, [g[0] for g in got]

    with torch.no_grad():
        for tag, ids_list in (("long_context", chunks_ids), ("short_prompt", short_ids)):
            per_k, per_v = [], []
            sc_k = sc_v = hm_k = hm_v = None
            layer_idx = None
            for ids in ids_list:
                inp = ids[None, :].to(model.device)
                cache = model(input_ids=inp, use_cache=True).past_key_values
                per, sc, hm, layer_idx = stats_from_cache(cache)
                per_k += per["k"]
                per_v += per["v"]
                if sc_k is None:
                    sc_k = np.zeros_like(sc["k"])
                    sc_v = np.zeros_like(sc["v"])
                    hm_k = np.zeros_like(hm["k"])
                    hm_v = np.zeros_like(hm["v"])
                sc_k += sc["k"] / len(ids_list)
                sc_v += sc["v"] / len(ids_list)
                hm_k += hm["k"] / len(ids_list)
                hm_v += hm["v"] / len(ids_list)
                del cache
                torch.cuda.empty_cache()
            het = {
                "k": {
                    "layer_profile_dissim": profile_dissim(sc_k.reshape(sc_k.shape[0], -1)),
                    "head_profile_dissim": profile_dissim(
                        np.concatenate([sc_k[:, h] for h in range(kvh)], axis=0)),
                    "head_vector_dissim": float(np.mean(
                        [head_vector_dissim(hm_k[l]) for l in range(hm_k.shape[0])])),
                },
                "v": {
                    "layer_profile_dissim": profile_dissim(sc_v.reshape(sc_v.shape[0], -1)),
                    "head_profile_dissim": profile_dissim(
                        np.concatenate([sc_v[:, h] for h in range(kvh)], axis=0)),
                    "head_vector_dissim": float(np.mean(
                        [head_vector_dissim(hm_v[l]) for l in range(hm_v.shape[0])])),
                },
            }
            out["arms"][tag] = {
                "k": {"layers": aggregate(per_k)}, "v": {"layers": aggregate(per_v)},
                "heterogeneity": het,
                "shape": dict(L=len(layer_idx), H=kvh, D=hd,
                              n_chunks=len(ids_list),
                              tokens=int(sum(len(i) for i in ids_list))),
                "full_attention_layer_ids": layer_idx,
            }
            print(f"[llm] {tag} done", flush=True)
    return out


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    chunks = build_corpus(tok)
    prompts = (W / "kv-quant-result/shared/prompts/moviegen_128.txt").read_text().splitlines()[:16]
    short = [torch.tensor(tok(p, add_special_tokens=False).input_ids) for p in prompts]
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=False)
    except (ValueError, KeyError):
        from transformers import AutoModelForConditionalGeneration
        model = AutoModelForConditionalGeneration.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    res = run(model, chunks, short)
    Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    print("wrote", sys.argv[1], flush=True)


if __name__ == "__main__":
    main()
