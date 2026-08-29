#!/usr/bin/env python3
"""Propose the calibration pool and its prompt groups from a prompt file.

The study needs one pool of prompts collected in a known order, partitioned
into calibration and held-out sets, and split again into two deliberately
dissimilar distributions for the shift experiment.

The A/B split is proposed here by an explicit, auditable keyword rule: a prompt
counts as *action* when it names a person or animal subject, and as *scene*
otherwise.  The rule is a starting point, not the experiment.  The manifest it
writes is plain JSON with prompt text echoed alongside each decision, and it is
meant to be read and corrected by hand before any GPU time is spent -- a
mislabelled prompt would weaken exactly the claim module 2 makes.

Outputs
    pool_prompts.txt    the prompts to collect, in pool order
    prompt_groups.json  named groups, addressed by pool position
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


# Subject nouns that make a prompt a "human / action" prompt.  Kept explicit
# and short so a reader can judge the rule rather than reverse-engineer it.
ACTION_TERMS = (
    "man", "woman", "men", "women", "person", "people", "boy", "girl",
    "child", "children", "couple", "crowd", "chef", "artist", "runner",
    "dancer", "musician", "player", "worker", "rider", "driver", "soldier",
    "astronaut", "cowboy", "pirate", "monk", "he ", "she ", "his ", "her ",
    "dog", "cat", "bird", "horse", "monkey", "panda", "rabbit", "elephant",
    "wolf", "bear", "fox", "tiger", "lion", "sloth", "otter", "penguin",
    "mammoth", "dinosaur", "robot", "android",
)


def classify(prompt: str) -> str:
    """Return ``action`` when the prompt names an acting subject."""
    text = " " + re.sub(r"\s+", " ", prompt.lower()) + " "
    for term in ACTION_TERMS:
        needle = term if term.endswith(" ") else f"{term} "
        if f" {needle}" in text or f" {term}s " in text or f" {term}, " in text:
            return "action"
    return "scene"


def read_prompts(path: Path) -> list[str]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"{path} contains no prompts")
    return lines


def build(
    prompts: list[str],
    *,
    calibration_size: int,
    heldout_size: int,
) -> tuple[list[int], dict[str, list[int]], list[dict[str, object]]]:
    """Choose the pool and its groups, alternating the two distributions.

    The pool interleaves action and scene prompts so that the size sweep's
    calibration and held-out sets are both distribution-balanced; module 1 then
    measures size alone, and module 2 measures distribution alone.
    """
    labelled = [(index, prompt, classify(prompt)) for index, prompt in enumerate(prompts)]
    buckets = {
        name: [item for item in labelled if item[2] == name]
        for name in ("action", "scene")
    }
    per_distribution = (calibration_size + heldout_size) // 2
    for name, members in buckets.items():
        if len(members) < per_distribution:
            raise ValueError(
                f"the prompt file yields only {len(members)} {name} prompts, but "
                f"the requested pool needs {per_distribution}; supply more "
                "prompts or lower --calibration-size/--heldout-size"
            )
    chosen = {name: buckets[name][:per_distribution] for name in buckets}

    # Interleave so pool position parity carries the distribution label.
    pool: list[tuple[int, str, str]] = []
    for offset in range(per_distribution):
        pool.append(chosen["action"][offset])
        pool.append(chosen["scene"][offset])

    total = len(pool)
    if total != calibration_size + heldout_size:
        raise ValueError("pool size does not match the requested split")

    positions = {"action": [], "scene": []}
    for position, (_, _, label) in enumerate(pool):
        positions[label].append(position)

    calibration = list(range(calibration_size))
    heldout = list(range(calibration_size, total))
    groups: dict[str, list[int]] = {
        "calibration": calibration,
        "heldout": heldout,
    }
    for name in ("action", "scene"):
        member_set = set(positions[name])
        groups[f"{name}_calibration"] = [p for p in calibration if p in member_set]
        groups[f"{name}_heldout"] = [p for p in heldout if p in member_set]

    manifest_prompts = [
        {
            "pool_position": position,
            "source_index": source_index,
            "distribution": label,
            "prompt": prompt,
        }
        for position, (source_index, prompt, label) in enumerate(pool)
    ]
    return [item[0] for item in pool], groups, manifest_prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-size", type=int, default=32)
    parser.add_argument("--heldout-size", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.calibration_size <= 0 or args.heldout_size <= 0:
        raise SystemExit("calibration and held-out sizes must be positive")
    if (args.calibration_size + args.heldout_size) % 2:
        raise SystemExit("the pool size must be even so the two distributions match")
    prompts = read_prompts(args.prompts)
    indices, groups, manifest_prompts = build(
        prompts,
        calibration_size=args.calibration_size,
        heldout_size=args.heldout_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool_file = args.output_dir / "pool_prompts.txt"
    pool_file.write_text(
        "\n".join(prompts[index] for index in indices) + "\n", encoding="utf-8"
    )
    manifest = {
        "prompts_file": str(pool_file),
        "source_prompts_file": str(args.prompts),
        "prompt_count": len(indices),
        "prompt_indices": indices,
        "groups": groups,
        "prompts": manifest_prompts,
    }
    manifest_file = args.output_dir / "prompt_groups.json"
    manifest_file.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"pool prompts   {pool_file}")
    print(f"prompt groups  {manifest_file}")
    for name, members in sorted(groups.items()):
        print(f"  {name:<22} {len(members):>3} prompts")
    print("\nReview the action/scene labels in prompt_groups.json before collecting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
