"""Print the parameter directory name for a codec config and its overrides.

Usage: codec_slug.py CONFIG K_HIST V_HIST K_BITS V_BITS STRIDE V_PREDICTOR
       FIT_HISTORY_SOURCE ANCHOR_BITS   (empty strings mean "use the config")

Standard library only, so shell entry points do not import torch.
"""

import json
import sys

DEFAULT_K_BITS = 2
DEFAULT_V_BITS = 2
DEFAULT_ANCHOR_BITS = 4
DEFAULT_STRIDE = 1560
DEFAULT_V_HISTORY_LENGTH = 0
DEFAULT_V_PREDICTOR_MODE = "cross_kv"

K_PREFIX = {"identity": "ki", "taylor": "kty", "taylor2": "kty2"}
V_PREFIX = {
    "cross_kv": "vh",
    "temporal": "vt",
    "innovation": "vi",
    "direct": "vd",
    "identity": "vid",
    "taylor": "vty",
    "taylor2": "vty2",
    "cross_kv_identity": "cvid",
    "cross_kv_taylor": "cvty",
    "cross_kv_taylor2": "cvty2",
}
# The CrossKV layouts each family deploys, as in
# tempokv.integration.DEPLOYABLE_CROSS_KV_WEIGHT_MODES; the sharing-ablation
# modes must not land in a production directory.
DEPLOYABLE_CROSS = {
    "legacy": ("layer_shared", "layer_head"),
    "native": ("layer_head",),
}


def slug(
    config_path: str,
    k_history: str = "",
    v_history: str = "",
    k_bits: str = "",
    v_bits: str = "",
    stride: str = "",
    v_predictor: str = "",
    fit_history_source: str = "reconstructed",
    anchor_bits: str = "",
) -> str:
    config = json.load(open(config_path, encoding="utf-8"))
    config = config.get("codec", config)

    def pick(override: str, key: str, default: int) -> int:
        if override:
            return int(override)
        value = config.get(key)
        return default if value is None else int(value)

    k_bits_value = pick(k_bits, "k_bits", DEFAULT_K_BITS)
    v_bits_value = pick(v_bits, "v_bits", DEFAULT_V_BITS)
    stride_value = pick(stride, "stride", DEFAULT_STRIDE)
    anchor_bits_value = pick(anchor_bits, "anchor_bits", DEFAULT_ANCHOR_BITS)
    k_history_value = pick(k_history, "k_history_length", 3)
    v_history_value = pick(v_history, "v_history_length", DEFAULT_V_HISTORY_LENGTH)

    # The two fitting families are not interchangeable, so the family leads.
    fit_mode = config.get("fit_mode", "legacy")
    if fit_mode not in DEPLOYABLE_CROSS:
        raise SystemExit(f"fit_mode must be legacy or native, got {fit_mode!r}")
    default_k_predictor = "shared" if fit_mode == "legacy" else "layer_head"
    k_predictor = config.get("k_predictor_mode", default_k_predictor)
    k_prefix = K_PREFIX.get(k_predictor)
    k_label = f"{k_prefix or 'kh'}{k_history_value}"
    # A learned K predictor at its family's own granularity stays unlabelled,
    # which keeps every existing directory name; legacy per-layer/head K (the K
    # sharing study) must not overwrite the pooled parameters.
    k_layout_label = (
        "-k" + k_predictor.replace("_", "")
        if k_prefix is None and k_predictor not in {default_k_predictor, "direct"}
        else ""
    )

    v_predictor = v_predictor or config.get("v_predictor_mode", DEFAULT_V_PREDICTOR_MODE)
    if v_predictor not in V_PREFIX:
        raise SystemExit(
            f"V_PREDICTOR must be one of {', '.join(V_PREFIX)}, got {v_predictor!r}"
        )
    v_label = f"{V_PREFIX[v_predictor]}{v_history_value}"

    deployable = DEPLOYABLE_CROSS[fit_mode]
    cross_weight_mode = config.get("cross_kv_weight_mode", deployable[0])
    # Only CrossKV V reads a per-layer matrix; every other legacy V predictor
    # runs with the layer-shared layout whatever the config says.
    if fit_mode == "legacy" and v_predictor not in {"cross_kv", "direct"}:
        cross_weight_mode = "layer_shared"
    if cross_weight_mode not in deployable:
        raise SystemExit(
            f"{fit_mode} CrossKV parameters must use cross_kv_weight_mode in "
            f"{list(deployable)}, got {cross_weight_mode}"
        )

    precision_label = (
        f"int{k_bits_value}"
        if k_bits_value == v_bits_value
        else f"k{k_bits_value}-v{v_bits_value}"
    )
    # Defaults stay unlabelled so directories fitted before each knob existed
    # keep their names.
    anchor_label = (
        "" if anchor_bits_value == DEFAULT_ANCHOR_BITS else f"-a{anchor_bits_value}"
    )
    stride_label = "" if stride_value == DEFAULT_STRIDE else f"-s{stride_value}"
    if fit_history_source not in {"reconstructed", "clean"}:
        raise SystemExit(
            "FIT_HISTORY_SOURCE must be reconstructed or clean, got "
            f"{fit_history_source!r}"
        )
    fit_label = "-fitclean" if fit_history_source == "clean" else ""
    use_higgs_residual = config.get("residual_quantizer") == "higgs"
    use_higgs_anchor = config.get("higgs_anchor", False) and anchor_bits_value != 16
    quantizer_label = ""
    if use_higgs_residual or use_higgs_anchor:
        group = pick("", "higgs_group_size", 512)
        quantizer_label = f"-higgs-g{group}" if use_higgs_residual else f"-anchorhiggs-g{group}"
        if use_higgs_residual and use_higgs_anchor:
            quantizer_label += "-anchorhiggs"
    return (
        f"{fit_mode}-{k_label}-{v_label}-{precision_label}{anchor_label}"
        f"{stride_label}{k_layout_label}-{cross_weight_mode.replace('_', '')}"
        f"{fit_label}{quantizer_label}"
    )


if __name__ == "__main__":
    print(slug(*sys.argv[1:]))
