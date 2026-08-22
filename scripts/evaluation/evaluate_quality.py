#!/usr/bin/env python3
"""Compute paired frame metrics and frame FID for matching video directories."""

from __future__ import annotations

import argparse
from itertools import zip_longest
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np


DEFAULT_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


class QualityBackend:
    """Lazy SSIM and LPIPS implementation shared across all video pairs."""

    def __init__(self, device: str = "auto", network: str = "alex") -> None:
        try:
            import torch
            import lpips
            from skimage.metrics import structural_similarity
        except ImportError as error:
            raise RuntimeError(
                "quality evaluation requires torch, lpips, and scikit-image; "
                "install TempoKV with the 'metrics' extra"
            ) from error
        selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if selected == "auto":
            selected = "cpu"
        if str(selected).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {selected!r} is unavailable")
        self.torch = torch
        self.device = torch.device(selected)
        self.structural_similarity = structural_similarity
        self.model = lpips.LPIPS(net=network).to(self.device).eval()

    @staticmethod
    def _rgb(frame: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(frame[..., ::-1])

    def ssim(self, reference: np.ndarray, candidate: np.ndarray) -> float:
        return float(
            self.structural_similarity(
                self._rgb(reference),
                self._rgb(candidate),
                channel_axis=2,
                data_range=255,
            )
        )

    def lpips(self, reference: np.ndarray, candidate: np.ndarray) -> float:
        torch = self.torch

        def tensor(frame: np.ndarray):
            value = torch.from_numpy(self._rgb(frame)).to(
                device=self.device, dtype=torch.float32
            )
            return value.permute(2, 0, 1).unsqueeze(0).div(127.5).sub(1.0)

        with torch.inference_mode():
            return float(self.model(tensor(reference), tensor(candidate)).item())


class FidBackend:
    """Stream paired RGB frames into a directory-level Inception FID."""

    def __init__(self, device: str = "auto", batch_size: int = 32) -> None:
        if batch_size <= 0:
            raise ValueError("--fid-batch-size must be a positive integer")
        try:
            import torch
            from torchmetrics.image.fid import FrechetInceptionDistance
        except ImportError as error:
            raise RuntimeError(
                "FID evaluation requires torchmetrics and torch-fidelity; "
                "install TempoKV with the 'metrics' extra"
            ) from error
        selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if selected == "auto":
            selected = "cpu"
        if str(selected).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {selected!r} is unavailable")
        self.torch = torch
        self.device = torch.device(selected)
        self.batch_size = batch_size
        self.metric = FrechetInceptionDistance(
            feature=2048, normalize=False
        ).to(self.device)
        self.metric.set_dtype(torch.float64)
        self.reference_batch: list[Any] = []
        self.candidate_batch: list[Any] = []
        self.batch_shape: tuple[int, ...] | None = None
        self.frames = 0

    @staticmethod
    def _tensor(torch: Any, frame: np.ndarray) -> Any:
        rgb = np.ascontiguousarray(frame[..., ::-1])
        return torch.from_numpy(rgb).permute(2, 0, 1)

    def add(self, reference: np.ndarray, candidate: np.ndarray) -> None:
        shape = tuple(reference.shape)
        if self.reference_batch and shape != self.batch_shape:
            self.flush()
        self.batch_shape = shape
        self.reference_batch.append(self._tensor(self.torch, reference))
        self.candidate_batch.append(self._tensor(self.torch, candidate))
        self.frames += 1
        if len(self.reference_batch) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.reference_batch:
            return
        reference = self.torch.stack(self.reference_batch).to(self.device)
        candidate = self.torch.stack(self.candidate_batch).to(self.device)
        with self.torch.inference_mode():
            self.metric.update(reference, real=True)
            self.metric.update(candidate, real=False)
        self.reference_batch.clear()
        self.candidate_batch.clear()
        self.batch_shape = None

    def compute(self) -> float:
        self.flush()
        if self.frames < 2:
            raise ValueError("FID requires at least two matched frames")
        with self.torch.inference_mode():
            return float(self.metric.compute().item())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare videos with the same relative filename and report per-video "
            "and pooled RGB PSNR, mean SSIM, mean LPIPS, and frame FID."
        )
    )
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated video extensions (default: common video formats)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Evaluate matched videos even when candidate files are missing",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Compare at most this many frames from each video",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Evaluate every Nth decoded frame (default: 1, all frames)",
    )
    parser.add_argument(
        "--fid-batch-size",
        type=int,
        default=32,
        help="Frames per Inception batch (default: 32)",
    )
    parser.add_argument(
        "--no-fid",
        action="store_true",
        help="skip directory-level FID when only paired PSNR/SSIM/LPIPS are needed",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--lpips-network", choices=("alex", "vgg", "squeeze"), default="alex"
    )
    return parser


def _extensions(text: str) -> tuple[str, ...]:
    values = []
    for value in text.split(","):
        value = value.strip().lower()
        if not value:
            continue
        values.append(value if value.startswith(".") else f".{value}")
    if not values:
        raise ValueError("--extensions must contain at least one extension")
    if len(values) != len(set(values)):
        raise ValueError("--extensions contains duplicate values")
    return tuple(values)


def _video_paths(directory: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"video directory does not exist: {directory}")
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def _iter_frames(path: Path) -> Iterator[np.ndarray]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "PSNR video decoding requires OpenCV; install it with "
            "`pip install opencv-python`."
        ) from error

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open video: {path}")
    produced = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            produced = True
            yield np.asarray(frame)
    finally:
        capture.release()
    if not produced:
        raise ValueError(f"video contains no decodable frames: {path}")


def _psnr(mse: float) -> float | None:
    return None if mse == 0 else 10.0 * math.log10((255.0**2) / mse)


def compare_video(
    reference: Path,
    candidate: Path,
    max_frames: int | None = None,
    *,
    backend: QualityBackend | None = None,
    fid_backend: FidBackend | None = None,
    frame_stride: int = 1,
) -> dict[str, Any]:
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be a positive integer")
    if frame_stride <= 0:
        raise ValueError("--frame-stride must be a positive integer")
    sentinel = object()
    squared_error = 0.0
    values = 0
    frames = 0
    ssim_total = 0.0
    lpips_total = 0.0
    backend = backend or QualityBackend()
    pairs = zip_longest(_iter_frames(reference), _iter_frames(candidate), fillvalue=sentinel)
    decoded_frames = 0
    for reference_frame, candidate_frame in pairs:
        if max_frames is not None and frames >= max_frames:
            break
        if reference_frame is sentinel or candidate_frame is sentinel:
            raise ValueError(
                f"frame count differs: {reference} and {candidate} "
                f"(after {decoded_frames} decoded frame pairs)"
            )
        if reference_frame.shape != candidate_frame.shape:
            raise ValueError(
                f"frame {decoded_frames} shape differs: {reference_frame.shape} vs "
                f"{candidate_frame.shape} ({reference.name})"
            )
        frame_index = decoded_frames
        decoded_frames += 1
        if frame_index % frame_stride != 0:
            continue
        difference = reference_frame.astype(np.float64) - candidate_frame.astype(np.float64)
        squared_error += float(np.square(difference).sum())
        ssim_total += backend.ssim(reference_frame, candidate_frame)
        lpips_total += backend.lpips(reference_frame, candidate_frame)
        if fid_backend is not None:
            fid_backend.add(reference_frame, candidate_frame)
        values += int(difference.size)
        frames += 1
    if frames == 0:
        raise ValueError(f"no frames were compared: {reference} and {candidate}")
    mse = squared_error / values
    return {
        "frames": frames,
        "values": values,
        "squared_error": squared_error,
        "mse": mse,
        "psnr_db": _psnr(mse),
        "ssim": ssim_total / frames,
        "lpips": lpips_total / frames,
    }


def evaluate(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    allow_missing: bool = False,
    max_frames: int | None = None,
    frame_stride: int = 1,
    fid_batch_size: int = 32,
    compute_fid: bool = True,
    device: str = "auto",
    lpips_network: str = "alex",
) -> dict[str, Any]:
    reference_dir = reference_dir.expanduser().resolve()
    candidate_dir = candidate_dir.expanduser().resolve()
    references = _video_paths(reference_dir, extensions)
    _video_paths(candidate_dir, extensions)
    if not references:
        raise ValueError(f"no videos with extensions {extensions} found in {reference_dir}")

    missing = [
        path.relative_to(reference_dir).as_posix()
        for path in references
        if not (candidate_dir / path.relative_to(reference_dir)).is_file()
    ]
    if missing and not allow_missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"candidate directory is missing {len(missing)} matching video(s): {preview}"
        )

    results = []
    total_error = 0.0
    total_values = 0
    total_frames = 0
    total_ssim = 0.0
    total_lpips = 0.0
    backend = QualityBackend(device=device, network=lpips_network)
    fid_backend = (
        FidBackend(device=device, batch_size=fid_batch_size)
        if compute_fid
        else None
    )
    for reference in references:
        relative = reference.relative_to(reference_dir)
        candidate = candidate_dir / relative
        if not candidate.is_file():
            continue
        metrics = compare_video(
            reference,
            candidate,
            max_frames=max_frames,
            backend=backend,
            fid_backend=fid_backend,
            frame_stride=frame_stride,
        )
        total_error += metrics.pop("squared_error")
        total_values += metrics.pop("values")
        total_frames += metrics["frames"]
        total_ssim += metrics["ssim"] * metrics["frames"]
        total_lpips += metrics["lpips"] * metrics["frames"]
        results.append({"video": relative.as_posix(), **metrics})
    if not results:
        raise ValueError("no matching video pairs were available to evaluate")

    pooled_mse = total_error / total_values
    fid = fid_backend.compute() if fid_backend is not None else None
    return {
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "videos": len(results),
        "frames": total_frames,
        "missing": missing,
        "mse": pooled_mse,
        "psnr_db": _psnr(pooled_mse),
        "ssim": total_ssim / total_frames,
        "lpips": total_lpips / total_frames,
        "lpips_network": lpips_network,
        "fid": fid,
        "fid_feature": 2048 if compute_fid else None,
        "fid_reference": "bf16" if compute_fid else None,
        "fid_frames": fid_backend.frames if fid_backend is not None else 0,
        "fid_batch_size": fid_batch_size if compute_fid else None,
        "frame_stride": frame_stride,
        "max_frames_per_video": max_frames,
        "device": device,
        "per_video": results,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    report = evaluate(
        args.reference_dir,
        args.candidate_dir,
        extensions=_extensions(args.extensions),
        allow_missing=args.allow_missing,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        fid_batch_size=args.fid_batch_size,
        compute_fid=not args.no_fid,
        device=args.device,
        lpips_network=args.lpips_network,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return report


if __name__ == "__main__":
    main()
