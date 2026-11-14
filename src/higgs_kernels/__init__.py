"""CUDA HIGGS primitives shared by independent codec implementations."""


def for_device(device, group_size):
    if device.type != "cuda":
        return None
    try:
        from . import triton_ops
    except ImportError:
        return None
    return triton_ops if triton_ops.supports(group_size) else None
