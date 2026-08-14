from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "common" / "fit_params.py"
SPEC = importlib.util.spec_from_file_location("tempokv_fit_params", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FIT_PARAMS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIT_PARAMS)


def test_cuda_peak_memory_reset_initializes_requested_device() -> None:
    device = torch.device("cuda:1")

    with (
        patch.object(FIT_PARAMS.torch, "empty") as empty,
        patch.object(FIT_PARAMS.torch.cuda, "reset_peak_memory_stats") as reset,
    ):
        FIT_PARAMS._reset_cuda_peak_memory_stats(device)

    empty.assert_called_once_with(0, device=device)
    reset.assert_called_once_with(device)
