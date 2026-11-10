"""Continue strict48GB generation with five VBench cases on GPU lanes 1--3."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from rebuttal.common.manifest import read_json, write_json

DEFAULT_OUT = Path("/root/workspace/Tempokv/results/fixed_memory/strict48gb_k5v3_20261109_2330")
DEPS = Path("/root/workspace/Tempokv/results/fixed_memory/dependencies")
COMPAT = Path("/root/workspace/Tempokv/results/fixed_memory/resumed_pilot3_20261109_2237/compat_peft010")
LIMIT = 48_000_000_000


def optional_json(path):
    return read_json(path) if path.is_file() else {}


def source_failed(out):
    status = optional_json(out / "status.json")
    if status.get("status") == "failed" or status.get("failures"):
        raise RuntimeError(f"Source strict study failed: {out / 'status.json'}")
    if status.get("status") == "running" and status.get("pid"):
        try:
            os.kill(int(status["pid"]), 0)
        except ProcessLookupError as error:
            raise RuntimeError("Source strict study process exited without completion") from error
    summary = optional_json(out / "device_memory_summary.json")
    if summary.get("violations"):
        raise RuntimeError("Source strict study recorded a whole-device memory violation")
    return status


def create_once(path, content):
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"Existing content differs; refusing to overwrite: {path}")
        return
    with path.open("x") as handle:
        handle.write(content)


def prepare_wrapper(out):
    import imageio_ffmpeg
    binary = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    bindir = out / "bin"
    bindir.mkdir(exist_ok=True)
    link = bindir / "ffmpeg"
    if link.exists() or link.is_symlink():
        if link.resolve() != binary:
            raise RuntimeError(f"Existing ffmpeg differs: {link}")
    else:
        link.symlink_to(binary)
    wrapper = out / "strict_vbench_python.sh"
    paths = [COMPAT, DEPS / "vbench", DEPS / "vbench_python", ROOT, ROOT / "src"]
    for path in paths:
        if not path.is_dir():
            raise FileNotFoundError(path)
    prefix = ":".join(map(str, paths))
    content = ("#!/bin/sh\nset -eu\n"
               f"export PYTHONPATH={shlex.quote(prefix)}${{PYTHONPATH:+:$PYTHONPATH}}\n"
               f"export PATH={shlex.quote(str(bindir))}:$PATH\n"
               f"export VBENCH_CACHE_DIR={shlex.quote(str(DEPS / 'vbench_cache'))}\n"
               "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1\n"
               f"exec {shlex.quote(sys.executable)} {shlex.quote(str(ROOT / 'rebuttal/experiments/memory/strict_python.py'))} \"$@\"\n")
    create_once(wrapper, content)
    wrapper.chmod(0o755)
    return wrapper


def check_jobs(plan):
    jobs = plan["jobs"]
    if len(jobs) != 5 or len({str(Path(j["folder"]).resolve()) for j in jobs}) != 5:
        raise ValueError("Expected exactly five independent generated conditions")
    if len({j["name"] for j in jobs}) != 5:
        raise ValueError("Duplicate condition names")
    for job in jobs:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", job["name"]):
            raise ValueError("Unsafe condition name")
        if not Path(job["folder"]).is_absolute() or job.get("expected_video_count") != 3:
            raise ValueError("Expected three real videos with absolute source paths")
    return jobs


def condition(out, name):
    """CPU controller; evaluate_vbench starts the strict CUDA Python children."""
    from rebuttal.experiments.memory.evaluation import evaluate_vbench
    from rebuttal.experiments.memory.identity import job_video_identities
    plan = read_json(out / "generation_plan.json")
    job = next(j for j in check_jobs(plan) if j["name"] == name)
    records, _ = job_video_identities(job)
    if len(records) != 3 or not all(Path(record["path"]).is_file() for record in records):
        raise RuntimeError("Incomplete generation identity/video cohort")
    case = out / "vbench_cases" / name
    case.mkdir(parents=True, exist_ok=True)
    selected = {"status": "generation_complete", "jobs": [job]}
    create_once(case / "generation_plan.json", json.dumps(selected, indent=2) + "\n")
    config = read_json(out / "study_config.json")
    settings = {"prompts": config["prompts"], "_resume": True,
                "vbench": {"python": str(out / "strict_vbench_python.sh"),
                           "device": "cuda", "local_checkpoints": True}}
    result = evaluate_vbench(settings, case)
    if len(result["rows"]) != 2 or {r["scope"] for r in result["rows"]} != {"full", "late"}:
        raise RuntimeError("VBench did not complete both scopes")
    for row in result["rows"]:
        path = case / "vbench" / name / row["scope"] / "evaluation_config.json"
        evaluation = read_json(path)
        if evaluation.get("status") not in ("complete", "reused"):
            raise RuntimeError(f"Incomplete VBench scope: {path}")
        if len(evaluation.get("dimensions", [])) != 8:
            raise RuntimeError(f"Expected existing default eight dimensions: {path}")
    write_json(case / "completion.json", result)


def controller(out, poll_seconds):
    from rebuttal.experiments.memory.hard_memory_limit import gpu_rows, MIB
    from rebuttal.experiments.memory.strict_study import environment
    out.mkdir(parents=True, exist_ok=True)
    owner = (out / "vbench_followup.lock").open("a+")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = {}
    active_lock = threading.Lock()
    stop = threading.Event()
    peaks, violations, audit_errors = {}, [], []
    states = {}

    def kill_own_groups():
        with active_lock:
            processes = list(active.values())
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def audit():
        try:
            with (out / "vbench_device_memory_audit.jsonl").open("a") as handle:
                while not stop.is_set():
                    rows = gpu_rows()
                    handle.write(json.dumps({"time": time.time(), "devices": rows}) + "\n")
                    handle.flush()
                    for row in rows:
                        used = row["memory_used_mib"] * MIB
                        peaks[row["uuid"]] = max(peaks.get(row["uuid"], 0), used)
                        if used >= LIMIT:
                            violations.append({"time": time.time(), **row})
                    if violations:
                        stop.set()
                        kill_own_groups()
                    stop.wait(.25)
        except BaseException:
            audit_errors.append(traceback.format_exc())
            stop.set()
            kill_own_groups()

    def check():
        if stop.is_set():
            raise RuntimeError(f"VBench memory audit failed: {violations or audit_errors}")
        return source_failed(out)

    def status(phase, **extra):
        write_json(out / "vbench_status.json", {"status": "running", "phase": phase,
                   "pid": os.getpid(), "cases": list(states.values()), **extra})

    thread = threading.Thread(target=audit, daemon=True)
    thread.start()
    try:
        status("waiting_for_generation")
        while True:
            check()
            plan = optional_json(out / "generation_plan.json")
            if plan.get("status") == "generation_complete":
                break
            stop.wait(poll_seconds)
        jobs = check_jobs(plan)
        prepare_wrapper(out)
        pending = queue.Queue()
        for job in jobs:
            states[job["name"]] = {"condition": job["name"], "status": "pending"}
            pending.put(job)
        status("vbench_full_and_late", real_conditions=5, logical_comparisons=plan.get("logical_comparisons"))

        def lane(gpu):
            while not stop.is_set():
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    return
                name = job["name"]
                case = out / "vbench_cases" / name
                case.mkdir(parents=True, exist_ok=True)
                error = None
                for attempt in range(1, 4):
                    try:
                        check()
                        states[name] = {"condition": name, "status": "running", "gpu": gpu, "attempt": attempt}
                        command = [sys.executable, str(Path(__file__).resolve()), "--output", str(out), "--condition", name]
                        child_environment = environment(out, gpu)
                        child_environment["PATH"] = str(out / "bin") + os.pathsep + child_environment.get("PATH", "")
                        with (case / f"controller_attempt{attempt}.log").open("a") as handle:
                            with active_lock:
                                if stop.is_set():
                                    raise RuntimeError("Audit stopped before client launch")
                                process = subprocess.Popen(command, cwd=ROOT, env=child_environment,
                                            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                                active[gpu] = process
                            code = process.wait()
                            with active_lock:
                                active.pop(gpu, None)
                        archive = case / "attempt_logs" / str(attempt)
                        for log in (case / "vbench").glob("*/*/run.log"):
                            destination = archive / log.relative_to(case / "vbench")
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(log, destination)
                        if code:
                            raise RuntimeError(f"Condition child exited {code}; see {case / f'controller_attempt{attempt}.log'}")
                        check()
                        result = read_json(case / "completion.json")
                        if result.get("status") != "complete":
                            raise RuntimeError("Condition completion record missing")
                        error = None
                        break
                    except Exception:
                        error = traceback.format_exc()
                        write_json(case / f"failure_attempt{attempt}.json", {"error": error, "gpu": gpu})
                        if stop.is_set():
                            break
                        stop.wait(5)
                states[name] = {"condition": name, "status": "failed" if error else "complete", "gpu": gpu,
                                "attempts": attempt, "case_directory": str(case), **({"error": error} if error else {})}
                write_json(case / "status.json", states[name])

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(lane, gpu) for gpu in (1, 2, 3)]
            try:
                while not all(future.done() for future in futures):
                    check()
                    status("vbench_full_and_late")
                    stop.wait(5)
            except BaseException:
                stop.set()
                kill_own_groups()
                raise
            for future in futures:
                future.result()
        check()
        if any(state["status"] != "complete" for state in states.values()):
            raise RuntimeError("One or more VBench cases failed after at most three attempts")
        quality_file = out / "matched_context_quality" / "quality_completion.json"
        status("waiting_for_matched_context_quality")
        while True:
            source = check()
            if quality_file.is_file() and source.get("status") == "generation_and_quality_complete_vbench_pending":
                break
            if source.get("status") == "generation_and_quality_complete_vbench_pending" and not quality_file.is_file():
                raise RuntimeError("Source finished without quality completion evidence")
            stop.wait(poll_seconds)
        quality = read_json(quality_file)
        from rebuttal.experiments.temporal_window import quality_output_fingerprints
        if quality.get("status") != "complete" or quality.get("outputs") != quality_output_fingerprints(quality_file.parent):
            raise RuntimeError("Quality completion evidence is incomplete or changed")
        source_summary = read_json(out / "device_memory_summary.json")
        if source_summary.get("violations") or violations or audit_errors:
            raise RuntimeError("Memory violations prevent experiment completion")
        result = {"status": "complete", "real_conditions": 5, "unique_video_count": 15,
                  "logical_comparisons": plan.get("logical_comparisons"), "cases": list(states.values()),
                  "quality_completion": str(quality_file), "quality_status": quality["status"],
                  "device_peak_bytes": peaks, "whole_device_limit_bytes": LIMIT,
                  "cuda_limit_mib": 40000, "torch_limit_mib": 39000,
                  "enforcement": "verified MPS CUDA limit and one experiment client per GPU; NVML audit"}
        write_json(out / "vbench_status.json", result)
        write_json(out / "experiment6_complete.json", result)
    except BaseException:
        stop.set()
        kill_own_groups()
        write_json(out / "vbench_status.json", {"status": "failed", "pid": os.getpid(),
                    "cases": list(states.values()), "traceback": traceback.format_exc(),
                    "violations": violations, "audit_errors": audit_errors})
        raise
    finally:
        stop.set()
        thread.join(timeout=3)
        write_json(out / "vbench_device_memory_summary.json", {"device_peak_bytes": peaks,
                    "violations": violations, "audit_errors": audit_errors,
                    "whole_device_limit_bytes": LIMIT, "audit_interval_seconds": .25})
        owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--condition")
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.condition:
        condition(args.output.resolve(), args.condition)
    else:
        controller(args.output.resolve(), args.poll_seconds)


if __name__ == "__main__":
    main()
