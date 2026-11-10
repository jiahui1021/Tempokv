"""Run only the unblocked strict48GB BF16 and QVG INT4 branches."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from rebuttal.common.manifest import read_json, write_json, file_sha256

LIMIT = 48_000_000_000


def aggregate(out, cfg, conditions):
    from rebuttal.experiments.memory.identity import save_job_identities
    jobs = []
    for condition in conditions:
        folder = out / "generation" / condition["condition"]
        videos = folder / "videos"
        videos.mkdir(parents=True, exist_ok=True)
        summaries, reports = [], []
        for pid in range(3):
            source = out / "samples" / condition["condition"] / f"prompt_{pid}"
            assert read_json(source / "completion.json")["status"] == "complete"
            destination = videos / f"{pid}_0.mp4"
            if not destination.exists():
                os.link(source / f"videos/{pid}_0.mp4", destination)
            summaries.extend(read_json(source / "videos/summary.json")["rank_summaries"])
            reports.append(read_json(source / "memory_report.json"))
        write_json(videos / "summary.json", {"rank_summaries": summaries,
                   "summary_scope": "actual native summaries from three separately generated prompts"})
        command = list(cfg["original_command"])
        command[command.index("--data_path") + 1] = cfg["prompts"]
        job = dict(**condition, name=condition["condition"], folder=str(folder), command=command,
                   seed=0, expected_video_count=3, status="complete",
                   input_digests={"prompts": file_sha256(cfg["prompts"])})
        save_job_identities(job)
        write_json(folder / "memory_report.json", {"reports": reports,
                   "measured_peak_kv_bytes": max(r["peak_checked_bytes"] for r in reports),
                   "whole_device_limit_bytes": LIMIT})
        jobs.append(job)
    write_json(out / "non_temp_generation_plan.json", {"status": "available_branches_generation_complete",
               "jobs": jobs, "unique_video_count": 9, "experiment6_complete": False,
               "excluded_branch": "TempoKV K5/V3 blocked; no fourth repair attempt",
               "logical_comparisons": [row for row in cfg["logical_comparisons"] if row["method"] in ("bf16", "qvg")]})


def quality(out):
    from rebuttal.experiments.memory.identity import video_identities, identity_key
    from rebuttal.experiments.temporal_window import run
    cfg = read_json(out / "study_config.json")
    h = cfg["shared_history_frames"]
    options = {"prompts_path": Path(cfg["prompts"])}
    references = {identity_key(r): r for r in video_identities(out / f"generation/bf16_h{h}/videos", **options)[0]}
    pairs = []
    for candidate in video_identities(out / f"generation/qvg_h{h}/videos", **options)[0]:
        reference = references[identity_key(candidate)]
        assert reference["prompt"] == candidate["prompt"]
        pairs.append(dict(reference=reference["path"], candidate=candidate["path"],
            prompt_id=candidate["prompt_id"], seed=candidate["noise_seed"], sample_id=0,
            model="self_forcing", method="qvg", comparison_id="strict48GB_matched_context_qvg_only"))
    assert len(pairs) == 3
    run(dict(pairs=pairs, expected_frames=3837, metrics=["psnr", "ssim", "lpips"],
             window_seconds=15, video_decoder="pyav", bootstrap_samples=1000,
             bootstrap_seed=0, plots=True), "evaluate", out / "non_temp_matched_context_quality")


def main(out):
    from rebuttal.experiments.memory.strict_study import environment, sample_job, validate
    from rebuttal.experiments.memory.hard_memory_limit import gpu_rows, MIB
    owner = (out / "non_temp_controller.lock").open("a+")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = read_json(out / "study_config.json")
    conditions = [c for c in cfg["conditions"] if c["method"] in ("bf16", "qvg")]
    assert {c["condition"] for c in conditions} == {"bf16_h60", "qvg_h144", "qvg_h60"}
    active, states, peaks = {}, {}, {}
    violations, audit_errors = [], []
    lock, stop = threading.Lock(), threading.Event()

    def kill_own():
        with lock:
            processes = list(active.values())
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def audit():
        try:
            with (out / "non_temp_device_memory_audit.jsonl").open("a") as handle:
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
                        kill_own()
                    stop.wait(.25)
        except BaseException:
            audit_errors.append(traceback.format_exc())
            stop.set()
            kill_own()

    def state(phase, status="running", **extra):
        with lock:
            records = list(states.values())
        write_json(out / "non_temp_status.json", dict(status=status, phase=phase, pid=os.getpid(),
            expected_videos=9, completed_videos=sum(r["status"] == "complete" for r in records),
            jobs=records, experiment6_complete=False, tempokv="blocked_after_three_checks_not_retried",
            device_peak_bytes=dict(peaks), whole_device_limit_bytes=LIMIT, **extra))

    def execute(command, log, gpu):
        with log.open("a") as handle:
            with lock:
                if stop.is_set():
                    raise RuntimeError("Memory audit stopped admission")
                process = subprocess.Popen(command, cwd=ROOT, env=environment(out, gpu),
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = process
            code = process.wait()
            with lock:
                active.pop(gpu, None)
        if code:
            raise RuntimeError(f"Process exit {code}; see {log}")

    def stage(prompt_ids, gpus):
        tasks = queue.Queue()
        for pid in prompt_ids:
            for condition in conditions:
                tasks.put(sample_job(out, cfg, condition, pid))
        def lane(gpu):
            while not stop.is_set():
                try:
                    job = tasks.get_nowait()
                except queue.Empty:
                    return
                folder = Path(job["folder"])
                key = f"{job['condition']}/prompt_{job['prompt_id']}"
                error = None
                for attempt in range(1, 4):
                    try:
                        with lock:
                            states[key] = dict(status="running", gpu=gpu, attempt=attempt, **job)
                        write_json(folder / "non_temp_status.json", states[key])
                        execute(job["command"], folder / f"non_temp_attempt{attempt}.log", gpu)
                        validate(job)
                        error = None
                        break
                    except Exception:
                        error = traceback.format_exc()
                        write_json(folder / f"non_temp_failure{attempt}.json", {"traceback": error})
                        if stop.is_set():
                            break
                        stop.wait(5)
                with lock:
                    states[key] = dict(status="failed" if error else "complete", gpu=gpu,
                        attempts=attempt, **job, **({"traceback": error} if error else {}))
                write_json(folder / "non_temp_status.json", states[key])
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(lane, gpu) for gpu in gpus]
            while not all(f.done() for f in futures):
                state("generation_prompt0" if prompt_ids == (0,) else "generation_remaining_prompts")
                stop.wait(5)
                if stop.is_set():
                    kill_own()
                    raise RuntimeError("Whole-device memory audit failed")
            for future in futures:
                future.result()

    thread = threading.Thread(target=audit, daemon=True)
    thread.start()
    try:
        state("generation_prompt0")
        stage((0,), (0, 1, 2))
        stage((1, 2), (0, 1, 2, 3))
        if stop.is_set() or violations or audit_errors or any(s["status"] != "complete" for s in states.values()):
            raise RuntimeError("Not all available branch generation jobs completed successfully")
        aggregate(out, cfg, conditions)
        state("qvg_matched_context_quality")
        execute([sys.executable, str(ROOT / "rebuttal/experiments/memory/strict_python.py"),
                 str(Path(__file__).resolve()), "--quality", "--output", str(out)],
                 out / "non_temp_quality.log", 0)
        assert read_json(out / "non_temp_matched_context_quality/quality_completion.json")["status"] == "complete"
        state("available_branches_generation_and_quality_complete", status="available_branches_complete")
    except BaseException:
        stop.set()
        kill_own()
        state("available_branches", status="failed", traceback=traceback.format_exc())
        raise
    finally:
        stop.set()
        thread.join(timeout=3)
        write_json(out / "non_temp_device_memory_summary.json", dict(device_peak_bytes=peaks,
            violations=violations, audit_errors=audit_errors, whole_device_limit_bytes=LIMIT,
            enforcement="verified 40000MiB MPS CUDA limit; 39000MiB Torch limit; single client per GPU; NVML audit"))
        owner.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality", action="store_true")
    args = parser.parse_args()
    if args.quality:
        quality(args.output.resolve())
    else:
        main(args.output.resolve())
