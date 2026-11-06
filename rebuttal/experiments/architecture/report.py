"""Flat, inspectable architecture comparison table."""
import csv
from pathlib import Path


def write_table(path: Path, rows):
    fields = ["axis", "variant", "rank", "parameter_bytes", "storage_bytes", "fit_seconds", "replay_seconds",
              "decoder_backend", "decode_seconds_mean", "decode_seconds_median", "decode_warmup", "decode_repeats",
              "decode_timing_scope", "decode_device", "k_prediction_nmse", "v_prediction_nmse", "k_quantization_nmse",
              "v_quantization_nmse", "k_reconstruction_nmse", "v_reconstruction_nmse", "parameters"]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            output = {key: row.get(key) for key in fields}
            for role in ("k", "v"):
                for metric in ("prediction", "quantization", "reconstruction"):
                    output[f"{role}_{metric}_nmse"] = row[role][f"{metric}_nmse"]
            writer.writerow(output)
