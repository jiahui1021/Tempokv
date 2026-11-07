"""CORAL-style head-block DPLR structure on same-layer CrossKV (trace only)."""


def run(config, stage, output_dir):
    from .experiment import run as execute
    return execute(config, stage, output_dir)
