"""Small decord-compatible reader for macOS arm64.

The VBench code only needs VideoReader, bridge.set_bridge, cpu, len,
get_avg_fps, and get_batch. OpenCV supplies the decoding backend here because
the official decord wheel has no macOS arm64 build.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


class _Bridge:
    def __init__(self) -> None:
        self.name = "native"

    def set_bridge(self, name: str) -> None:
        if name not in {"native", "torch", "numpy"}:
            raise ValueError(f"unsupported bridge: {name}")
        self.name = name


bridge = _Bridge()


class _Batch:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def asnumpy(self) -> np.ndarray:
        return self._array

    def permute(self, *dims):
        import torch

        return torch.from_numpy(self._array).permute(*dims)

    @property
    def shape(self):
        return self._array.shape


class _Context:
    def __init__(self, device_id: int) -> None:
        self.device_id = device_id


def cpu(device_id: int = 0) -> _Context:
    return _Context(device_id)


class VideoReader:
    def __init__(self, source, width=None, height=None, num_threads=1, ctx=None):
        self.source = source
        self.width = width
        self.height = height
        self._path = self._materialize_source(source)
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {source}")
        self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 8.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if count <= 0:
            raise RuntimeError(f"video has no frames: {source}")
        self._count = count

    @staticmethod
    def _materialize_source(source):
        if isinstance(source, (str, Path)):
            return str(source)
        if isinstance(source, io.BytesIO):
            import tempfile

            handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            handle.write(source.getbuffer())
            handle.close()
            return handle.name
        raise TypeError(f"unsupported video source: {type(source)!r}")

    def __len__(self):
        return self._count

    def get_avg_fps(self):
        return self._fps

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            return self.get_batch(range(start, stop, step))
        return self.get_batch([int(index)])._array[0]

    def get_batch(self, indices: Iterable[int]):
        requested = [int(i) for i in indices]
        if not requested:
            empty = np.empty((0, 0, 0, 3), dtype=np.uint8)
            return _Batch(empty)
        if min(requested) < 0 or max(requested) >= self._count:
            raise IndexError("frame index out of range")
        wanted = set(requested)
        frames = {}
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {self._path}")
        frame_index = 0
        while frame_index <= max(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index in wanted:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.width and self.height:
                    frame = cv2.resize(frame, (int(self.width), int(self.height)))
                frames[frame_index] = frame
            frame_index += 1
        cap.release()
        if len(frames) != len(wanted):
            missing = sorted(wanted - frames.keys())[:5]
            raise RuntimeError(f"failed to decode frame(s): {missing}")
        array = np.stack([frames[i] for i in requested], axis=0).astype(np.uint8)
        if bridge.name == "torch":
            import torch

            return torch.from_numpy(array)
        return _Batch(array)

