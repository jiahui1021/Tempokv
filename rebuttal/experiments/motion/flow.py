"""Explicit optical-flow implementations with original-image coordinate units."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("motion analysis requires opencv-python") from exc
    return cv2


def restore_flow_units(flow: Any, height: int, width: int) -> Any:
    """Unpad first, resize vectors, then rescale x and y independently."""
    import numpy as np
    cv2 = _cv2()
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError("flow must have shape [H,W,2]")
    old_h, old_w = flow.shape[:2]
    value = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    value[..., 0] *= width / old_w
    value[..., 1] *= height / old_h
    return value


class FarnebackBackend:
    """Explicit classical baseline; never selected as a silent RAFT fallback."""
    def __init__(self, config: dict[str, Any]) -> None:
        self.config, self.cv2 = config, _cv2()
        self.metadata = {"backend": "opencv_farneback", "opencv_version": self.cv2.__version__, "parameters": config}

    def __call__(self, first: Any, second: Any) -> Any:
        cv2 = self.cv2
        h, w = first.shape[:2]
        target = self.config.get("resize_width")
        if target and int(target) < w:
            size = (int(target), max(8, round(h * int(target) / w)))
            first, second = cv2.resize(first, size), cv2.resize(second, size)
        a, b = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, float(self.config.get("pyr_scale", 0.5)), int(self.config.get("levels", 3)),
                                         int(self.config.get("winsize", 15)), int(self.config.get("iterations", 3)),
                                         int(self.config.get("poly_n", 5)), float(self.config.get("poly_sigma", 1.2)), 0)
        return restore_flow_units(flow, h, w)


class _Args(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class RAFTBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("RAFT requires torch") from exc
        self.torch, self.cv2, self.config = torch, _cv2(), config
        checkpoint = Path(config.get("checkpoint", ""))
        root = Path(config.get("raft_root", Path(__file__).resolve().parents[4] / "qvg" / "raft")).resolve()
        core = root / "core"
        if not checkpoint.is_file():
            raise FileNotFoundError("RAFT requires an explicit local checkpoint; no weights are downloaded")
        if not (core / "raft.py").is_file():
            raise FileNotFoundError(f"RAFT checkout is missing core/raft.py: {root}")
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("requested RAFT CUDA device is unavailable")
        self.device = torch.device(device)
        # Upstream uses absolute imports (utils/update/etc). Temporarily isolate
        # its names so unrelated model runtimes cannot choose its dependencies.
        names = {"raft", "update", "extractor", "corr", "utils"}
        saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] in names}
        for key in saved:
            del sys.modules[key]
        sys.path.insert(0, str(core))
        try:
            raft = importlib.import_module("raft")
            utilities = importlib.import_module("utils.utils")
            args = _Args(small=bool(config.get("small", False)), mixed_precision=bool(config.get("mixed_precision", False)) and self.device.type == "cuda",
                         alternate_corr=bool(config.get("alternate_corr", False)), dropout=0)
            self.model, self.padder = raft.RAFT(args), utilities.InputPadder
        except ImportError as exc:
            raise RuntimeError(f"cannot import configured RAFT checkout and dependencies: {exc}") from exc
        finally:
            sys.path.remove(str(core))
            for key in list(sys.modules):
                if key.split(".")[0] in names:
                    del sys.modules[key]
            sys.modules.update(saved)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        self.metadata = {"backend": "raft", "root": str(root), "commit": commit.stdout.strip() if commit.returncode == 0 else None,
                         "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest.hexdigest(), "device": str(device), "parameters": config}

    def __call__(self, first: Any, second: Any) -> Any:
        cv2, torch = self.cv2, self.torch
        h, w = first.shape[:2]
        target = self.config.get("resize_width")
        if target and int(target) < w:
            size = (int(target), max(8, round(h * int(target) / w)))
            first, second = cv2.resize(first, size), cv2.resize(second, size)
        def tensor(frame):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(self.device)
        a, b = tensor(first), tensor(second)
        padder = self.padder(a.shape)
        a, b = padder.pad(a, b)
        with torch.inference_mode():
            _, flow = self.model(a, b, iters=int(self.config.get("iterations", 20)), test_mode=True)
        flow = padder.unpad(flow)[0].permute(1, 2, 0).cpu().numpy()
        return restore_flow_units(flow, h, w)


def make_flow_backend(config: dict[str, Any]) -> Any:
    name = config.get("backend", "raft")
    if name == "raft":
        return RAFTBackend(config)
    if name == "farneback":
        return FarnebackBackend(config)
    raise ValueError("flow.backend must be raft or farneback; no automatic fallback")
