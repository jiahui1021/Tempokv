import argparse
import json
import imageio.v3 as iio
import torch
import os
import time
from omegaconf import OmegaConf
from torchvision import transforms
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler, Subset

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--start_index", type=int, default=0)
parser.add_argument("--end_index", type=int, default=-1)
parser.add_argument("--local_attn_size", type=int, default=-1)
parser.add_argument("--sink_size", type=int, default=0)
parser.add_argument("--skip_existing", action="store_true")
parser.add_argument("--skip_video_save", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--save_latents", action="store_true")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    try:
        dist.init_process_group(backend='nccl', device_id=device)
    except TypeError:
        dist.init_process_group(backend='nccl')
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    set_seed(args.seed + global_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    global_rank = 0
    world_size = 1
    set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
runtime_root = os.path.dirname(os.path.abspath(__file__))
default_config = OmegaConf.load(
    os.path.join(runtime_root, "configs", "default_config.yaml")
)
config = OmegaConf.merge(default_config, config)
config.model_kwargs.local_attn_size = args.local_attn_size
config.model_kwargs.sink_size = args.sink_size

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
else:
    pipeline.text_encoder.to(device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
full_num_prompts = len(dataset)
end_index = full_num_prompts if args.end_index < 0 else min(
    args.end_index, full_num_prompts
)
if args.start_index < 0 or args.start_index > end_index:
    raise ValueError("invalid prompt range")
selected_indices = list(range(args.start_index, end_index))
pre_skipped_summaries = []
pre_skipped_batches = []
if args.skip_existing and args.save_with_index:
    pending_indices = []
    for idx in selected_indices:
        expected = [
            os.path.join(args.output_folder, f"{idx}_{seed_idx}.mp4")
            for seed_idx in range(args.num_samples)
        ]
        if expected and all(
            os.path.isfile(path) and os.path.getsize(path) > 0
            for path in expected
        ):
            for seed_idx, output_path in enumerate(expected):
                pre_skipped_summaries.append(
                    {
                        "prompt_index": idx,
                        "sample_index": seed_idx,
                        "noise_seed": int(args.seed) + idx * 1_000_003,
                        "path": output_path,
                        "skipped": True,
                        "generation_seconds": None,
                        "total_seconds": None,
                    }
                )
            pre_skipped_batches.append(
                {
                    "prompt_index": int(idx),
                    "requested_samples": int(args.num_samples),
                    "generated_samples": 0,
                    "skipped_samples": int(args.num_samples),
                    "generation_seconds": None,
                    "video_save_seconds": None,
                    "total_seconds": None,
                    "timing_scope": "prompt_batch",
                }
            )
        else:
            pending_indices.append(idx)
    selected_indices = pending_indices
dataset = Subset(dataset, selected_indices)
num_prompts = len(dataset)
if dist.is_initialized():
    prompts_per_rank = num_prompts // world_size
    remainder = num_prompts % world_size
    # LOCAL_RANK selects the device on each node.  Prompt ownership must use
    # the global rank, otherwise every node repeats the prompts owned by its
    # local rank.
    rank_start = global_rank * prompts_per_rank + min(global_rank, remainder)
    rank_end = rank_start + prompts_per_rank + (
        1 if global_rank < remainder else 0
    )
    dataset = Subset(dataset, range(rank_start, rank_end))
print(
    f"Rank {global_rank}/{world_size} (local {local_rank}): "
    f"total={full_num_prompts}, "
    f"selected={num_prompts}, assigned={len(dataset)}",
    flush=True,
)

sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if global_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    try:
        dist.barrier(device_ids=[local_rank])
    except TypeError:
        dist.barrier()

# Generation reports exclude checkpoint and model loading, matching the
# reference Self-Forcing measurement boundary.
torch.cuda.reset_peak_memory_stats(device)
generation_run_start = time.perf_counter()
video_summaries = list(pre_skipped_summaries) if global_rank == 0 else []
batch_summaries = list(pre_skipped_batches) if global_rank == 0 else []
horizon_probe = os.environ.get("SELF_FORCING_HORIZON_PROBE") == "1"
horizon_probe_result = None


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


def _output_path(prompt: str, idx: int, seed_idx: int) -> str:
    if args.save_with_index:
        return os.path.join(args.output_folder, f"{idx}_{seed_idx}.mp4")
    return os.path.join(args.output_folder, f"{prompt[:100]}-{seed_idx}.mp4")


def _is_valid_video(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _write_video_checked(path: str, frames: torch.Tensor, fps: int) -> None:
    stem, suffix = os.path.splitext(path)
    temporary = f"{stem}.tmp-{os.getpid()}{suffix or '.mp4'}"
    try:
        iio.imwrite(
            temporary,
            frames.to(dtype=torch.uint8).cpu().numpy(),
            fps=fps,
            codec="libx264",
        )
        if not _is_valid_video(temporary):
            raise RuntimeError(f"video encoder produced an empty file: {temporary}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_horizon_probe(path: str, result: dict) -> None:
    temporary = f"{path}.tmp-{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_latents_checked(path: str, latents: torch.Tensor, metadata: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    payload = {
        "latents": latents.detach().to(device="cpu", dtype=torch.bfloat16),
        **metadata,
    }
    try:
        torch.save(payload, temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        if tuple(loaded["latents"].shape) != tuple(latents.shape):
            raise RuntimeError(f"latent shape mismatch after save: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


for i, batch_data in enumerate(dataloader):
    idx = batch_data['idx'].item()
    video_total_start = time.perf_counter()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    if args.skip_existing and args.save_with_index:
        expected_outputs = [
            _output_path("", idx, seed_idx)
            for seed_idx in range(args.num_samples)
        ]
        if all(_is_valid_video(path) for path in expected_outputs):
            print(f"Skipping existing prompt index {idx}")
            for seed_idx, output_path in enumerate(expected_outputs):
                video_summaries.append(
                    {
                        "prompt_index": int(idx),
                        "sample_index": seed_idx,
                        "noise_seed": int(args.seed) + int(idx) * 1_000_003,
                        "path": output_path,
                        "skipped": True,
                        "generation_seconds": None,
                        "total_seconds": None,
                    }
                )
            batch_summaries.append(
                {
                    "prompt_index": int(idx),
                    "requested_samples": int(args.num_samples),
                    "generated_samples": 0,
                    "skipped_samples": int(args.num_samples),
                    "generation_seconds": None,
                    "video_save_seconds": None,
                    "total_seconds": None,
                    "timing_scope": "prompt_batch",
                }
            )
            continue

    prompt_seed = int(args.seed) + int(idx) * 1_000_003
    set_seed(prompt_seed)

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    if args.i2v:
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples
        initial_latent = None

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    # A benchmark interval must not inherit queued work from prompt loading or
    # noise construction.  The matching synchronize after ``pipeline.inference``
    # closes the same device interval.
    torch.cuda.synchronize(device)
    generation_start = time.perf_counter()
    try:
        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent,
            low_memory=low_memory,
            decode_output=not args.skip_video_save,
        )
    except torch.cuda.OutOfMemoryError:
        if not horizon_probe:
            raise
        completed_frames = int(
            getattr(pipeline, "_horizon_completed_frames", 0)
        )
        feasible_frames = (
            0
            if completed_frames < 9
            else completed_frames - ((completed_frames + 3) % 12)
        )
        horizon_probe_result = getattr(pipeline, "horizon_result", None) or {
            "max_feasible_frames": feasible_frames,
            "oom_reached": True,
            "peak_allocated_bytes_at_max": int(torch.cuda.max_memory_allocated(device)),
            "kv_resident_bytes_at_max": int(
                getattr(pipeline.efficiency_sampler, "peak_resident_bytes", 0)
            ),
            "local_attn_size": int(pipeline.generator.model.local_attn_size),
        }
        break
    if horizon_probe:
        horizon_probe_result = getattr(pipeline, "horizon_result", None)
        if horizon_probe_result is None:
            raise RuntimeError("horizon probe completed without a result")
        break
    torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - generation_start
    if args.skip_video_save:
        pipeline.vae.model.clear_cache()
        del video, latents, sampled_noise, initial_latent
        continue
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < full_num_prompts:
        batch_video_start = len(video_summaries)
        save_batch_start = time.perf_counter()
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            output_path = _output_path(prompt, idx, seed_idx)
            latent_path = None
            if args.save_latents:
                latent_path = os.path.join(
                    args.output_folder, "latents", f"{idx}_{seed_idx}.pt"
                )
                _write_latents_checked(
                    latent_path,
                    latents[seed_idx],
                    {
                        "prompt_index": int(idx),
                        "sample_index": seed_idx,
                        "noise_seed": prompt_seed,
                    },
                )
            save_start = time.perf_counter()
            save_seconds = None
            _write_video_checked(output_path, video[seed_idx], fps=16)
            save_seconds = time.perf_counter() - save_start
            video_summaries.append(
                {
                    "prompt_index": int(idx),
                    "sample_index": seed_idx,
                    "noise_seed": prompt_seed,
                    "path": output_path,
                    "latent_path": latent_path,
                    "skipped": False,
                    "generation_seconds": generation_seconds,
                    "save_seconds": save_seconds,
                    # Filled with the prompt-level value after all samples
                    # have been written.  A per-sample cumulative save time is
                    # not a meaningful latency metric for NUM_SAMPLES > 1.
                    "total_seconds": None,
                }
            )
        batch_video_save_seconds = time.perf_counter() - save_batch_start
        batch_total_seconds = time.perf_counter() - video_total_start
        generation_seconds_per_sample = generation_seconds / args.num_samples
        throughput_samples_per_second = (
            args.num_samples / batch_total_seconds
            if batch_total_seconds > 0
            else None
        )
        for entry in video_summaries[batch_video_start:]:
            entry["batch_generation_seconds"] = generation_seconds
            entry["batch_video_save_seconds"] = batch_video_save_seconds
            entry["batch_total_seconds"] = batch_total_seconds
            entry["generation_seconds_per_sample"] = generation_seconds_per_sample
            entry["throughput_samples_per_second"] = throughput_samples_per_second
            entry["total_seconds"] = batch_total_seconds
        batch_summaries.append(
            {
                "prompt_index": int(idx),
                "requested_samples": int(args.num_samples),
                "generated_samples": int(args.num_samples),
                "skipped_samples": 0,
                "generation_seconds": generation_seconds,
                "video_save_seconds": batch_video_save_seconds,
                "total_seconds": batch_total_seconds,
                "generation_seconds_per_sample": generation_seconds_per_sample,
                "throughput_samples_per_second": throughput_samples_per_second,
                "timing_scope": "prompt_batch",
                "kv_cache_memory": getattr(pipeline, "summary", {}).get(
                    "last_kv_cache_memory"
                ),
                "quantization_events": list(
                    getattr(pipeline, "summary", {}).get(
                        "quantization_events", []
                    )
                ),
                # The full per-block observation list, not just a peak: the
                # runner recomputes the peak after this trip through JSON, and
                # the sample count stays auditable.
                "efficiency_samples": [
                    list(sample)
                    for sample in getattr(
                        getattr(pipeline, "efficiency_sampler", None),
                        "samples",
                        [],
                    )
                ],
            }
        )
        if int(idx) == args.start_index and torch.cuda.is_available():
            # Prompt 0 absorbs Triton compilation and the allocator's first
            # growth.  Everything reported comes from the prompts after it.
            torch.cuda.reset_peak_memory_stats()
    del video, latents, current_video, all_video, sampled_noise, initial_latent

generation_wall_seconds = time.perf_counter() - generation_run_start
if horizon_probe and global_rank == 0 and horizon_probe_result is not None:
    _write_horizon_probe(
        os.path.join(args.output_folder, "horizon_probe.json"),
        horizon_probe_result,
    )
rank_summary = {
    "rank": int(global_rank),
    "local_rank": int(local_rank),
    "world_size": int(world_size),
    "generation_wall_seconds": generation_wall_seconds,
    "batches": batch_summaries,
    "videos": video_summaries,
}
if dist.is_initialized():
    gathered = [None] * world_size if global_rank == 0 else None
    dist.gather_object(rank_summary, gathered, dst=0)
    rank_summaries = gathered if global_rank == 0 else None
else:
    rank_summaries = [rank_summary]
if global_rank == 0:
    summary_path = os.path.join(args.output_folder, "summary.json")
    temporary_summary = f"{summary_path}.tmp-{os.getpid()}"
    with open(temporary_summary, "w", encoding="utf-8") as handle:
        json.dump(
            {"rank_summaries": rank_summaries},
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    os.replace(temporary_summary, summary_path)
