"""Pure temporal per-channel versus full within-head matrix comparison."""


def run(config, stage, output_dir):
    from .experiment import run as execute
    return execute(config, stage, output_dir)
