from typing import List, Optional
from types import SimpleNamespace
import os
import torch
from tqdm.auto import tqdm

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from kv_cache import ChunkedKVCache

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = len(self.generator.model.blocks)
        if self.num_transformer_blocks < 1:
            raise ValueError("Self-Forcing model contains no transformer blocks")
        first_attention = self.generator.model.blocks[0].self_attn
        self.num_heads = int(first_attention.num_heads)
        self.head_dim = int(first_attention.head_dim)
        self.frame_seq_length = int(getattr(args, "frame_seq_length", 1560))

        self.kv_cache1 = None
        self.args = args
        self.quant_config = getattr(
            args,
            "quant_config",
            SimpleNamespace(quant_type="none", kv_cache_cpu_offload=False),
        )
        self.summary = {
            "quantization_events": [],
            "last_kv_cache_memory": None,
        }
        if getattr(self.quant_config, "kv_cache_cpu_offload", False):
            raise ValueError(
                "bundled Self-Forcing does not support KV cache CPU offload"
            )
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        decode_output: bool = True,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        if self.local_attn_size == -1:
            target_tokens = num_output_frames * self.frame_seq_length
            for block in self.generator.model.blocks:
                block.self_attn.max_attention_size = target_tokens
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                target_frames=num_output_frames,
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                layer = self.kv_cache1[block_index]
                if layer["k"].max_num_chunks < num_output_frames:
                    raise ValueError(
                        "existing Self-Forcing cache is smaller than this request"
                    )
                layer["k"].clear()
                layer["v"].clear()
                layer["global_end_index"] = 0
                layer["local_end_index"] = 0
        self.summary["quantization_events"] = []
        self.summary["last_kv_cache_memory"] = None

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        quantization_factor = 8
        hide_progress = (
            torch.distributed.is_initialized()
            and torch.distributed.get_rank() != 0
        )
        with tqdm(
            total=len(all_num_frames),
            disable=hide_progress,
            dynamic_ncols=True,
        ) as progress_bar:
            for chunk_index, current_num_frames in enumerate(all_num_frames):
                if profile:
                    block_start.record()

                if (
                    chunk_index >= quantization_factor
                    and chunk_index % quantization_factor == 0
                ):
                    quantized_frames_start = sum(
                        all_num_frames[: chunk_index - quantization_factor]
                    )
                    quantized_frames_end = sum(all_num_frames[:chunk_index])
                    self.quantize_kv_cache(
                        quantized_frames_start * self.frame_seq_length,
                        quantized_frames_end * self.frame_seq_length,
                        int(self.kv_cache1[0]["local_end_index"]),
                    )

                noisy_input = noise[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

                # Step 3.1: Spatial denoising loop
                for index, current_timestep in enumerate(self.denoising_step_list):
                    # set current timestep
                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * current_timestep

                    if index < len(self.denoising_step_list) - 1:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        # for getting real output
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                # Step 3.2: record the model's output
                output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

                # Step 3.3: rerun with timestep zero to update KV cache using clean context
                context_timestep = torch.ones_like(timestep) * self.args.context_noise
                self.note_clean_refresh()
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

                if profile:
                    block_end.record()
                    torch.cuda.synchronize()
                    block_time = block_start.elapsed_time(block_end)
                    block_times.append(block_time)

                # Step 3.4: update the start and end frame indices
                current_start_frame += current_num_frames
                progress_bar.update(1)

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output.  Trace collection throws the pixels away,
        # so it skips the VAE entirely instead of paying for a full decode.
        if decode_output:
            video = self.vae.decode_to_pixel(output, use_cache=False)
            video.mul_(0.5).add_(0.5).clamp_(0, 1)
        else:
            if profile:
                raise ValueError("profile=True requires decode_output=True")
            video = None

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def note_clean_refresh(self) -> None:
        """Announce that the next generator call writes clean, keepable K/V.

        Trace collection used to recognise the clean pass by comparing the
        timestep against ``context_noise``, which quietly mis-selects noisy
        passes whenever the denoising schedule changes. The pipeline knows
        which pass is which, so it says so.
        """

    def quantize_kv_cache(
        self,
        tokens_to_quantize_start: int,
        tokens_to_quantize_end: int,
        max_tokens_to_quantize: int,
    ) -> None:
        """BF16/trace hook. TempoKV replaces this method in ``bridge.py``."""
        del tokens_to_quantize_start, tokens_to_quantize_end, max_tokens_to_quantize

    def _initialize_kv_cache(
        self, batch_size, dtype, device, target_frames: Optional[int] = None
    ):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        # Keep absolute frame positions even with local attention. The read
        # window is local, while storage remains frame-addressable so RoPE and
        # TempoKV spans never change coordinates.
        max_num_chunks = int(target_frames or 21)
        rope_precision = os.environ.get(
            "SELF_FORCING_CACHE_ROPE_PRECISION", "native"
        )
        if rope_precision not in {"native", "complex64"}:
            raise ValueError(
                "SELF_FORCING_CACHE_ROPE_PRECISION must be native or complex64"
            )

        for _ in range(self.num_transformer_blocks):
            key_cache = ChunkedKVCache(
                batch_size,
                self.frame_seq_length,
                self.num_heads,
                self.head_dim,
                max_num_chunks,
                dtype,
                device,
                layout="BSHD",
                cache_role="K",
            )
            key_cache.rope_precision = rope_precision
            kv_cache1.append({
                "k": key_cache,
                "v": ChunkedKVCache(
                    batch_size,
                    self.frame_seq_length,
                    self.num_heads,
                    self.head_dim,
                    max_num_chunks,
                    dtype,
                    device,
                    layout="BSHD",
                    cache_role="V",
                ),
                "global_end_index": 0,
                "local_end_index": 0,
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros(
                    [batch_size, 512, self.num_heads, self.head_dim],
                    dtype=dtype,
                    device=device,
                ),
                "v": torch.zeros(
                    [batch_size, 512, self.num_heads, self.head_dim],
                    dtype=dtype,
                    device=device,
                ),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
