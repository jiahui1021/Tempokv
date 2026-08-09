#!/usr/bin/env python3
"""Repository-local LongCat long-video inference runtime."""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torchvision.io import read_video, write_video
from transformers import AutoTokenizer, UMT5EncoderModel

from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.modules.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline


DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, paintings, "
    "still picture, worst quality, low quality, JPEG artifacts, deformed, "
    "disfigured, misshapen limbs, fused fingers, messy background"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bundled LongCat runtime")
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--init_video_path", type=Path)
    parser.add_argument("--workload", choices=("480p_long_gen",), default="480p_long_gen")
    parser.add_argument("--prompt_source", choices=("direct_prompt",), default="direct_prompt")
    parser.add_argument("--prompt", default="a cinematic video")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--prompt_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_segments", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=93)
    parser.add_argument("--num_cond_frames", type=int, default=53)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--context_parallel_size", type=int, default=1)
    parser.add_argument("--attn_sink_frames", type=int, default=0)
    parser.add_argument("--quant_type", choices=("none", "tempokv"), default="none")
    parser.add_argument("--enable_compile", action="store_true")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--offload_kv_cache", action="store_true")
    cache.add_argument("--no_offload_kv_cache", action="store_false", dest="offload_kv_cache")
    parser.set_defaults(offload_kv_cache=False)
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = _parser().parse_args(argv)
    if args.context_parallel_size <= 0:
        raise ValueError("context_parallel_size must be positive")
    if args.num_segments <= 0:
        raise ValueError("num_segments must be positive")
    if args.num_frames <= 0 or (args.num_frames - 1) % 4:
        raise ValueError("num_frames must be positive and satisfy (value - 1) % 4 == 0")
    if (
        args.num_cond_frames <= 0
        or args.num_cond_frames > args.num_frames
        or (args.num_cond_frames - 1) % 4
    ):
        raise ValueError(
            "num_cond_frames must be in [1, num_frames] and satisfy "
            "(value - 1) % 4 == 0"
        )
    if args.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("height and width must be positive")
    if args.fps <= 0 or not 0 <= args.crf <= 51:
        raise ValueError("fps must be positive and crf must be in [0, 51]")
    return args


def init_distributed_environment(args: argparse.Namespace) -> tuple[int, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("LongCat inference requires an NVIDIA CUDA GPU")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size % args.context_parallel_size:
        raise ValueError("WORLD_SIZE must be divisible by context_parallel_size")
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise RuntimeError("no CUDA devices are visible")
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank % device_count)))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        try:
            dist.init_process_group(
                backend="nccl",
                device_id=torch.device("cuda", local_rank),
                timeout=datetime.timedelta(hours=24),
            )
        except TypeError:
            dist.init_process_group(
                backend="nccl",
                timeout=datetime.timedelta(hours=24),
            )
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    init_context_parallel(
        context_parallel_size=args.context_parallel_size,
        global_rank=global_rank,
        world_size=world_size,
    )
    return global_rank, local_rank, world_size


def get_model_and_pipe(args: argparse.Namespace) -> tuple[torch.nn.Module, LongCatVideoPipeline]:
    checkpoint = args.checkpoint_dir.expanduser().resolve()
    cp_split_hw = context_parallel_util.get_optimal_split(
        context_parallel_util.get_cp_size()
    )
    common = {"torch_dtype": torch.bfloat16}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        checkpoint, subfolder="text_encoder", **common
    )
    vae = AutoencoderKLWan.from_pretrained(checkpoint, subfolder="vae", **common)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint, subfolder="scheduler", **common
    )
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint,
        subfolder="dit",
        cp_split_hw=cp_split_hw,
        **common,
    )
    if args.enable_compile:
        dit = torch.compile(dit)
    pipeline = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipeline.to(torch.cuda.current_device())
    return dit, pipeline


def _to_pil_frames(video: np.ndarray) -> list[Image.Image]:
    values = np.asarray(video)
    if values.ndim != 4 or values.shape[-1] not in (3, 4):
        raise ValueError(f"expected video [F,H,W,C], got {values.shape}")
    if np.issubdtype(values.dtype, np.floating):
        values = np.clip(values * 255.0, 0, 255)
    values = values.astype(np.uint8, copy=False)
    return [Image.fromarray(frame[..., :3]) for frame in values]


def _load_video(path: Path, *, minimum_frames: int) -> list[Image.Image]:
    resolved = path.expanduser().resolve()
    frames, _, _ = read_video(str(resolved), output_format="THWC", pts_unit="sec")
    if int(frames.shape[0]) < minimum_frames:
        raise ValueError(
            f"init video contains {frames.shape[0]} frames; at least {minimum_frames} are required"
        )
    return [Image.fromarray(frame.numpy()) for frame in frames[-minimum_frames:]]


def _write_video(path: Path, frames: list[Image.Image], *, fps: int, crf: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(np.stack([np.asarray(frame) for frame in frames])).to(
        torch.uint8
    )
    write_video(
        str(path),
        tensor,
        fps=fps,
        video_codec="libx264",
        options={"crf": str(crf)},
    )


def _clear_cuda() -> None:
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


@torch.no_grad()
def generate(args: argparse.Namespace) -> None:
    global_rank, local_rank, _ = init_distributed_environment(args)
    _, pipeline = get_model_and_pipe(args)
    generator = torch.Generator(device=local_rank).manual_seed(args.seed)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.init_video_path is None:
        initial = pipeline.generate_t2v(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )[0]
        current_video = _to_pil_frames(initial)
        all_frames = list(current_video)
        if global_rank == 0:
            _write_video(
                output_dir / "segment_000.mp4",
                all_frames,
                fps=args.fps,
                crf=args.crf,
            )
        del initial
        _clear_cuda()
    else:
        current_video = _load_video(
            args.init_video_path,
            minimum_frames=args.num_frames,
        )
        all_frames = list(current_video)

    target_size = current_video[0].size
    for segment_index in range(1, args.num_segments + 1):
        output = pipeline.generate_vc(
            video=current_video,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            resolution="480p",
            num_frames=args.num_frames,
            num_cond_frames=args.num_cond_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            use_kv_cache=True,
            offload_kv_cache=args.offload_kv_cache,
            enhance_hf=True,
        )[0]
        new_video = [
            frame.resize(target_size, Image.Resampling.BICUBIC)
            for frame in _to_pil_frames(output)
        ]
        all_frames.extend(new_video[args.num_cond_frames :])
        current_video = new_video
        if global_rank == 0:
            _write_video(
                output_dir / f"segment_{segment_index:03d}.mp4",
                all_frames,
                fps=args.fps,
                crf=args.crf,
            )
        del output
        _clear_cuda()

    if global_rank == 0:
        _write_video(
            output_dir / "longcat.mp4",
            all_frames,
            fps=args.fps,
            crf=args.crf,
        )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        generate(_parse_args(argv))
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"LongCat inference error: {error}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
