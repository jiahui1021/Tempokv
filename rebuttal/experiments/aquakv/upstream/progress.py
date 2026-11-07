"""Progress display is optional for standalone CPU calibration."""
try:
    from tqdm import trange
except ImportError:
    def trange(*args, **kwargs):
        return range(*args)
