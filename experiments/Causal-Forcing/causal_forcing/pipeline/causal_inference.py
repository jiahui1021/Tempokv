from typing import List, Optional
import os
import time
import torch

from causal_forcing.utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from causal_forcing.kv_cache import ChunkedKVCache

from causal_forcing.demo_utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
import tqdm


def _set_cache_index(cache: dict, name: str, value: int) -> None:
    current = cache[name]
    if isinstance(current, torch.Tensor):
        current.fill_(int(value))
    else:
        cache[name] = int(value)


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
            model_root=getattr(args, "model_root", None),
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = (
            WanTextEncoder(model_root=getattr(args, "model_root", None))
            if text_encoder is None else text_encoder
        )
        self.vae = (
            WanVAEWrapper(model_root=getattr(args, "model_root", None))
            if vae is None else vae
        )

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # Optional: separate denoising schedule for the first chunk (block 0).
        # If the config does not provide `denoising_step_list_first_chunk`, the
        # first chunk uses the same schedule as the rest (backwards compatible).
        if hasattr(args, "denoising_step_list_first_chunk") and args.denoising_step_list_first_chunk is not None:
            self.denoising_step_list_first_chunk = torch.tensor(
                args.denoising_step_list_first_chunk, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list_first_chunk = timesteps[1000 - self.denoising_step_list_first_chunk]
        else:
            self.denoising_step_list_first_chunk = None

        self.num_transformer_blocks = len(self.generator.model.blocks)
        if self.num_transformer_blocks < 1:
            raise ValueError("Causal Forcing model contains no transformer blocks")
        first_attention = self.generator.model.blocks[0].self_attn
        self.num_heads = int(first_attention.num_heads)
        self.head_dim = int(first_attention.head_dim)
        self.frame_seq_length = int(getattr(args, "frame_seq_length", 1560))

        self.kv_cache1 = None
        # The native BF16 path uses dense post-RoPE tensors.  TempoKV and
        # trace collection set this flag before the first inference so they
        # can keep their frame-addressable pre-RoPE ChunkedKVCache.
        self._uses_pre_rope_cache = False
        self.args = args
        # The TempoKV bridge replaces `finalize_kv_cache` and reads `summary`;
        # the BF16 and trace paths leave both inert.
        self.summary = {
            "finalize_events": [],
            "last_kv_cache_memory": None,
            "terminal_cache_event": None,
        }
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        # Timing state; populated only when `report_timing=True` is passed to
        # `inference()`. Kept as attributes so callers can read them afterward.
        self.last_generation_time = None
        self.first_chunk_time = None
        self.diffusion_generation_s = None
        self.vae_decode_s = None

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
        report_timing: bool = False,
        retain_final_cache: bool = False,
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
            retain_final_cache (bool): Keep and finalize the last generated
                block's clean K/V. Enable this only when a caller will continue
                generation from this cache or explicitly needs terminal cache
                state. Trace collection enables it automatically.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        # Do not leak timing values when a pipeline instance is reused with a
        # different timing setting for the next prompt.
        self.last_generation_time = None
        self.first_chunk_time = None
        self.diffusion_generation_s = None
        self.vae_decode_s = None
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            # default here
            # self.independent_first_frame: False
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        # Keep this marker for the legacy timing field. The comparable timer
        # starts later, after text encoding and KV cache initialization.
        if report_timing:
            torch.cuda.synchronize()
            self._gen_start_time = time.perf_counter()

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
                if self._uses_pre_rope_cache:
                    if layer["k"].max_num_chunks < num_output_frames:
                        raise ValueError(
                            "existing Causal Forcing cache is smaller than this request"
                        )
                    layer["k"].clear()
                    layer["v"].clear()
                else:
                    # Dense BF16 tensors are reused across prompts.  Clearing
                    # the index state is sufficient because every attention
                    # read is bounded by local_end_index; zeroing the whole
                    # multi-gigabyte cache would add avoidable benchmark cost.
                    required_tokens = num_output_frames * self.frame_seq_length
                    if layer["k"].shape[1] < required_tokens:
                        raise ValueError(
                            "existing Causal Forcing cache is smaller than this request"
                        )
                _set_cache_index(layer, "global_end_index", 0)
                _set_cache_index(layer, "local_end_index", 0)
        self.summary["finalize_events"] = []
        self.summary["last_kv_cache_memory"] = None
        self.summary["terminal_cache_event"] = None
        self.reset_kv_cache_state()

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.note_clean_refresh()
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                self.finalize_kv_cache(current_start_frame, 1)
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.note_clean_refresh()
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                self.finalize_kv_cache(current_start_frame, self.num_frame_per_block)
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Start after text encoding, cache initialization, and any context
        # refresh, immediately before the autoregressive temporal loop. This
        # is the shared diffusion-only benchmark interval.
        if report_timing:
            torch.cuda.synchronize()
            diffusion_started = time.perf_counter()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for block_index, current_num_frames in enumerate(tqdm.tqdm(all_num_frames)):
            if (
                os.environ.get("CAUSAL_FORCING_BENCHMARK") == "1"
                and block_index in {1, 8, 16, 32, 59}
            ):
                current_end_frame = current_start_frame + current_num_frames
                attention_start_frame = (
                    0
                    if self.local_attn_size == -1
                    else max(0, current_end_frame - int(self.local_attn_size))
                )
                print(
                    "[AttentionCheck] "
                    f"block={block_index} "
                    f"current_end_frame={current_end_frame} "
                    f"attention_start_frame={attention_start_frame} "
                    f"attention_length_frames={current_end_frame - attention_start_frame}",
                    flush=True,
                )
            # Optional: time the first block (TTFC). Excludes the KV-cache
            # refresh pass that follows the main denoising.
            if report_timing and block_index == 0:
                torch.cuda.synchronize()
                _first_block_start = time.time()

            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Select denoising schedule: block 0 may use a dedicated schedule
            # when provided by the config; otherwise all blocks share the same list.
            current_denoising_list = (
                self.denoising_step_list_first_chunk
                if block_index == 0 and self.denoising_step_list_first_chunk is not None
                else self.denoising_step_list
            )

            # Step 3.1: Spatial denoising loop. Every pass here rewrites the
            # current block's cache chunks with a different noisy K/V, so the
            # block stays ephemeral BF16 until the clean refresh below.
            self.note_ephemeral_block(len(current_denoising_list))
            for index, current_timestep in enumerate(current_denoising_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(current_denoising_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = current_denoising_list[index + 1]
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

            # Record first-chunk latency (denoising only, before KV cache refresh).
            if report_timing and block_index == 0:
                torch.cuda.synchronize()
                self.first_chunk_time = time.time() - _first_block_start
                print(f"First chunk time: {self.first_chunk_time:.2f}s")

            is_terminal_block = block_index == len(all_num_frames) - 1
            if is_terminal_block and not retain_final_cache:
                # No later attention call can read this block. A clean refresh
                # would therefore run the whole generator once more only to
                # create cache state that is immediately discarded; TempoKV's
                # terminal force-flush would also encode a span that is never
                # decoded. Release both caches before VAE decoding instead.
                self.discard_terminal_kv_cache(
                    current_start_frame, current_num_frames
                )
            else:
                # Step 3.3: rerun with timestep zero to update KV cache using
                # clean context whenever future generation or trace collection
                # needs the block.
                context_timestep = (
                    torch.ones_like(timestep) * self.args.context_noise
                )

                self.note_clean_refresh()
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

                # Step 3.3b: the cache now holds this block's clean state. This
                # is the only point at which a block becomes long-term history,
                # and it happens exactly once per retained block.
                self.finalize_kv_cache(current_start_frame, current_num_frames)

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()
        if retain_final_cache or not all_num_frames:
            if all_num_frames:
                self.summary["terminal_cache_event"] = {
                    "action": "retained",
                    "start_frame": int(
                        current_start_frame - all_num_frames[-1]
                    ),
                    "num_frames": int(all_num_frames[-1]),
                }
            self.flush_kv_cache()

        # Record diffusion time after the final clean refresh/cache flush and
        # before VAE decoding. This is the primary benchmark metric.
        if report_timing:
            torch.cuda.synchronize()
            self.diffusion_generation_s = time.perf_counter() - diffusion_started
            self.last_generation_time = self.diffusion_generation_s

        # Step 4: Decode the output
        if report_timing:
            torch.cuda.synchronize()
            vae_started = time.perf_counter()
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if report_timing:
            torch.cuda.synchronize()
            self.vae_decode_s = time.perf_counter() - vae_started

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

    # -- cache write-mode state machine ---------------------------------
    #
    # The pipeline, not the attention module, decides what a cache write means.
    # Nothing downstream inspects a timestep to guess whether a pass was clean.

    def reset_kv_cache_state(self) -> None:
        """Announce that the whole cache was cleared. TempoKV overrides this."""

    def note_ephemeral_block(self, num_writes: int) -> None:
        """Announce noisy denoising writes that must not become history."""
        del num_writes

    def note_clean_refresh(self) -> None:
        """Announce that the next forward writes clean, keepable K/V.

        Trace collection uses this to sample only the clean pass. Without it a
        collector would have to guess from the timestep, or sample every pass
        and discard four fifths of the transfers.
        """

    def finalize_kv_cache(self, start_frame: int, num_frames: int) -> None:
        """Announce that ``num_frames`` of clean state are now historical.

        BF16 and trace runs only record the event; the TempoKV bridge replaces
        this method with the streaming span encoder.
        """
        self.summary["finalize_events"].append(
            {"start_frame": int(start_frame), "num_frames": int(num_frames)}
        )

    def flush_kv_cache(self) -> None:
        """Announce the end of a rollout, so a partial span can still pack."""

    def discard_terminal_kv_cache(
        self, start_frame: int, num_frames: int
    ) -> None:
        """Release terminal K/V that no future attention call can consume."""
        self.summary["terminal_cache_event"] = {
            "action": "discarded",
            "start_frame": int(start_frame),
            "num_frames": int(num_frames),
        }
        # Dropping the owning lists releases dense and packed self-attention
        # cache tensors as well as the text cross-attention cache before VAE.
        self.kv_cache1 = None
        self.crossattn_cache = None

    def _initialize_kv_cache(
        self, batch_size, dtype, device, target_frames: Optional[int] = None
    ):
        """
        Initialize a Per-GPU KV cache for the Wan model.

        BF16 deliberately keeps the native Causal Forcing representation:
        post-RoPE keys and raw values in preallocated contiguous tensors.  The
        frame-addressable ChunkedKVCache is reserved for TempoKV and trace
        collection, whose read-time RoPE and span metadata are part of their
        method definitions.
        """
        kv_cache1 = []
        if not self._uses_pre_rope_cache:
            if self.local_attn_size != -1:
                max_num_chunks = int(self.local_attn_size)
            else:
                max_num_chunks = int(target_frames or 21)
            if max_num_chunks < 1:
                raise ValueError("Causal Forcing KV cache capacity must be positive")
            max_tokens = max_num_chunks * self.frame_seq_length

            def attach_compat_methods(tensor: torch.Tensor) -> torch.Tensor:
                """Keep the old test/runtime write API over a dense tensor.

                The model only uses tensor slicing.  A few lightweight pipeline
                harnesses write/read the cache directly, so expose the same
                narrow methods without turning the BF16 cache into a chunked
                object or changing its storage layout.
                """
                tensor.max_num_chunks = max_num_chunks
                tensor.kv_cache_size = max_tokens

                def write(
                    start_index: int,
                    end_index: int,
                    data: torch.Tensor,
                    *,
                    global_start_frame: int | None = None,
                ) -> None:
                    del global_start_frame
                    if end_index - start_index != data.shape[1]:
                        raise ValueError("dense cache write shape does not match range")
                    tensor[:, start_index:end_index].copy_(data)

                def read(
                    start_index: int,
                    end_index: int,
                    source_tensor: torch.Tensor | None = None,
                ) -> torch.Tensor:
                    del source_tensor
                    return tensor[:, start_index:end_index]

                def clear() -> None:
                    tensor.zero_()

                tensor.write = write
                tensor.read = read
                tensor.clear = clear
                return tensor

            for _ in range(self.num_transformer_blocks):
                key = attach_compat_methods(torch.zeros(
                    (batch_size, max_tokens, self.num_heads, self.head_dim),
                    dtype=dtype,
                    device=device,
                ))
                value = attach_compat_methods(torch.zeros_like(key))
                kv_cache1.append({
                    "k": key,
                    "v": value,
                    "cache_backend": "bf16",
                    # Keep these as Python integers: converting a CUDA scalar
                    # with `.item()` in every attention call would synchronize
                    # the GPU and distort the native BF16 benchmark.
                    "global_end_index": 0,
                    "local_end_index": 0,
                })
            self.kv_cache1 = kv_cache1
            return

        # Storage stays frame-addressable for the whole rollout even when
        # attention only reads a local window, so absolute frame positions —
        # and therefore RoPE and TempoKV spans — never shift under eviction.
        max_num_chunks = int(target_frames or 21)

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": ChunkedKVCache(
                    batch_size,
                    self.frame_seq_length,
                    self.num_heads,
                    self.head_dim,
                    max_num_chunks,
                    dtype,
                    device,
                    layout="BSHD",
                    cache_role="K",
                ),
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
            # The text cross-attention cache stays BF16: it is small, fixed
            # at 512 tokens, and does not grow along video time, so it carries
            # no temporal redundancy for TempoKV to predict.
            crossattn_cache.append({
                "k": torch.zeros(
                    [batch_size, 512, self.num_heads, self.head_dim],
                    dtype=dtype, device=device,
                ),
                "v": torch.zeros(
                    [batch_size, 512, self.num_heads, self.head_dim],
                    dtype=dtype, device=device,
                ),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
