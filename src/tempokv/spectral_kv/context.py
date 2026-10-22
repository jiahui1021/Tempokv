"""Suite B call context, supplied by pipeline events rather than timestep values."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RunContext:
    chunk_idx: int = 0
    step_idx: int = 0
    is_clean_pass: bool = False
    frame_idx0: int = 0
    is_last_denoise: bool = False

    @property
    def phase(self):
        return "clean" if self.is_clean_pass else "denoise"
