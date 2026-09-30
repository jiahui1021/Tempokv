#!/usr/bin/env python3
"""Resume VBench-Long clip preparation without rerunning completed videos."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re

import cv2

from vbench2_beta_long.utils import split_video_into_clips


def expected_clips(video: Path, duration: int = 2) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 8)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"invalid video metadata: {video}")
    segment = fps * duration
    return math.ceil(frames / segment)


def resume(stage: Path) -> tuple[int, int]:
    videos = sorted(stage.glob("*.mp4"))
    if len(videos) != 128:
        raise RuntimeError(f"{stage} contains {len(videos)} staged videos; expected 128")
    output = stage / "split_clip"
    output.mkdir(parents=True, exist_ok=True)
    done = 0
    rebuilt = 0
    for index, video in enumerate(videos, start=1):
        folder = output / video.stem
        have = len(list(folder.glob("*.mp4"))) if folder.is_dir() else 0
        need = expected_clips(video)
        if have == need:
            done += 1
            continue
        rebuilt += 1
        print(f"[{index}/128] {video.name}: {have}/{need}; resume", flush=True)
        split_video_into_clips(str(video), str(output), duration=2, fps=8)
    return done, rebuilt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_roots", nargs="+", type=Path)
    args = parser.parse_args()
    for root in args.run_roots:
        stage = root.expanduser().resolve() / "_vbench_long_input"
        done, rebuilt = resume(stage)
        print(f"COMPLETE {root.name}: already_complete={done} rebuilt={rebuilt}", flush=True)


if __name__ == "__main__":
    main()

