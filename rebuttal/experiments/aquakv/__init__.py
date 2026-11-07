"""Independent AQUA-KV on Self-Forcing: collect/fit/replay/generate/evaluate."""


def run(config, stage, output_dir):
    from .workflow import run as execute
    return execute(config, stage, output_dir)
