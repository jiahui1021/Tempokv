"""Fail-closed MPS admission and direct CUDA allocation-limit verification.

MPS bounds CUDA allocations, including internal allocations. A separate per-GPU
file lock prevents this experiment from starting two clients on one device.
All experiment tasks share admission checks and locks. Default compute mode
is allowed; isolation from unrelated tasks is checked, not administratively enforced.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import glob
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

MIB = 1024 ** 2
CUDA_LIMIT_MIB = 40000
TORCH_LIMIT_MIB = 39000
WHOLE_DEVICE_LIMIT_BYTES = 48000000000


def control(command):
    result = subprocess.run(["nvidia-cuda-mps-control"], input=command + "\n",
                            text=True, capture_output=True, check=True, timeout=10)
    if "error" in (result.stdout + result.stderr).lower():
        raise RuntimeError(f"MPS command failed: {command}: {result.stdout} {result.stderr}")
    return result.stdout.strip()


def gpu_rows():
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,memory.used,compute_mode",
        "--format=csv,noheader,nounits"], text=True, timeout=10)
    return [{"uuid": row[0].strip(), "memory_used_mib": int(row[1]),
             "compute_mode": row[2].strip()}
            for line in text.splitlines() if (row := line.split(","))]


def limit_bytes(text):
    match = re.fullmatch(r"([0-9.]+)\s*([GM])(?:B)?", text.strip())
    if not match:
        raise RuntimeError(f"Unrecognized MPS limit: {text!r}")
    return int(float(match[1]) * (1024 ** (3 if match[2] == "G" else 2)))


def _cudart():
    paths = glob.glob(str(Path(sys.prefix) / "lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*"))
    if not paths:
        raise RuntimeError("Cannot locate CUDA runtime library")
    runtime = ctypes.CDLL(paths[0])
    runtime.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    runtime.cudaMalloc.restype = ctypes.c_int
    runtime.cudaFree.argtypes = [ctypes.c_void_p]
    runtime.cudaFree.restype = ctypes.c_int
    runtime.cudaSetDevice.argtypes = [ctypes.c_int]
    runtime.cudaSetDevice.restype = ctypes.c_int
    return runtime


def allocation_probe(output, excess_request_mib=2048):
    """Verify OOM directly; the default probe expects a 1024M MPS limit."""
    runtime = _cudart()
    result = {"pid": os.getpid(), "uuid": os.environ["CUDA_VISIBLE_DEVICES"],
              "pipe_directory": os.environ["CUDA_MPS_PIPE_DIRECTORY"],
              "before": gpu_rows(), "timestamp": time.time()}
    result["set_device_code"] = runtime.cudaSetDevice(0)
    result["initialize_code"] = runtime.cudaFree(None)
    pointer = ctypes.c_void_p()
    result["small_request_bytes"] = 64 * MIB
    result["small_allocation_code"] = runtime.cudaMalloc(ctypes.byref(pointer), 64 * MIB)
    result["small_pointer_nonnull"] = bool(pointer.value)
    result["during_small_allocation"] = gpu_rows()
    large = ctypes.c_void_p()
    result["excess_request_bytes"] = excess_request_mib * MIB
    result["excess_allocation_code"] = runtime.cudaMalloc(ctypes.byref(large), excess_request_mib * MIB)
    result["excess_pointer_nonnull"] = bool(large.value)
    result["mps_clients"] = control("ps")
    result["after_excess_request"] = gpu_rows()
    if large.value:
        runtime.cudaFree(large)
    if pointer.value:
        runtime.cudaFree(pointer)
    result["verified"] = (result["set_device_code"] == 0 and result["initialize_code"] == 0
                          and result["small_allocation_code"] == 0
                          and result["excess_allocation_code"] == 2
                          and not result["excess_pointer_nonnull"]
                          and str(os.getpid()) in result["mps_clients"])
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    if not result["verified"]:
        raise RuntimeError(f"CUDA MPS OOM verification failed: {output}")
    return result


def compute_processes():
    text = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True, timeout=10)
    return [{"uuid": fields[0].strip(), "pid": int(fields[1]),
             "name": fields[2].strip(), "memory_mib": fields[3].strip()}
            for line in text.splitlines() if (fields := line.split(","))]


def host_pid(pid="self"):
    # /proc/<namespace-pid>/sched reports the host task PID on this Linux host.
    first = Path(f"/proc/{pid}/sched").read_text().splitlines()[0]
    match = re.search(r"\((\d+),", first)
    if not match:
        raise RuntimeError("Cannot identify CUDA process across the PID namespace")
    return int(match[1])


def assert_no_other_clients(uuid, proof, *, require_self=False):
    clients = []
    for line in control("ps").splitlines():
        fields = line.split()
        if fields and fields[0].isdigit() and len(fields) >= 4:
            if uuid.startswith(fields[3]):
                clients.append(int(fields[0]))
    expected = {os.getpid()} if require_self else set()
    if set(clients) != expected:
        raise RuntimeError(f"Unexpected MPS clients on {uuid}: {clients}; expected {sorted(expected)}")
    server_host_pid = proof.get("nvml_mps_server_pid")
    if not isinstance(server_host_pid, int):
        raise RuntimeError("Verification lacks NVML host PID for the MPS server")
    if host_pid(proof["server_pid"]) != server_host_pid:
        raise RuntimeError("MPS server host PID changed since verification")
    allowed_host_pids = {server_host_pid}
    if require_self:
        allowed_host_pids.add(host_pid())
    foreign = [row for row in compute_processes()
               if row["uuid"] == uuid and row["pid"] not in allowed_host_pids]
    if foreign:
        raise RuntimeError(f"Other CUDA compute processes on {uuid}: {foreign}")
    row = next(row for row in gpu_rows() if row["uuid"] == uuid)
    if row["memory_used_mib"] * MIB >= WHOLE_DEVICE_LIMIT_BYTES:
        raise RuntimeError("Whole-device NVML audit reached the 48GB ceiling")
    return {"mps_clients": clients, "nvml": row}


def acquire_verified_limit():
    """Admit one experiment client after live MPS and foreign-process checks."""
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid):
        raise RuntimeError("Strict worker requires exactly one complete GPU UUID")
    pipe = Path(os.environ.get("CUDA_MPS_PIPE_DIRECTORY", ""))
    proof_path = os.environ.get("TEMPOKV_STRICT_MEMORY_PROOF")
    if not proof_path or not pipe.is_dir():
        raise RuntimeError("Strict worker requires an MPS directory and verification proof")
    proof = json.loads(Path(proof_path).read_text())
    devices = proof.get("devices", {})
    if not proof.get("cuda_hard_limit_verified") or len(devices) != 4 or uuid not in devices:
        raise RuntimeError("Four-GPU CUDA allocation hard-limit verification is required")
    for device in devices.values():
        for key, expected_request in (("small_limit_probe", 2048), ("production_limit_probe", 40001)):
            probe = device.get(key, {})
            if (not probe.get("verified") or probe.get("small_allocation_code") != 0
                    or probe.get("excess_allocation_code") != 2
                    or probe.get("excess_pointer_nonnull")
                    or probe.get("excess_request_bytes") != expected_request * MIB):
                raise RuntimeError("Missing successful CUDA OOM probe for one of the four GPUs")
    if proof.get("pipe_directory") != str(pipe):
        raise RuntimeError("Verification belongs to another MPS daemon")
    daemon_pid = int((pipe / "nvidia-cuda-mps-control.pid").read_text().strip())
    if daemon_pid != proof.get("control_pid"):
        raise RuntimeError("MPS daemon changed since allocation-limit verification")
    os.kill(daemon_pid, 0)
    lock_dir = Path(proof_path).parent / "gpu_client_locks"
    lock_dir.mkdir(exist_ok=True)
    lock = open(lock_dir / (uuid + ".lock"), "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        row = next(row for row in gpu_rows() if row["uuid"] == uuid)
        if row["memory_used_mib"] * MIB + CUDA_LIMIT_MIB * MIB >= WHOLE_DEVICE_LIMIT_BYTES:
            raise RuntimeError("MPS/driver baseline leaves insufficient 48GB headroom")
        servers = [int(x) for x in control("get_server_list").split()]
        if servers != [proof.get("server_pid")]:
            raise RuntimeError("Expected the one verified MPS server")
        server = servers[0]
        for command in (f"get_default_device_pinned_mem_limit {uuid}",
                        f"get_device_pinned_mem_limit {server} {uuid}"):
            value = limit_bytes(control(command))
            # NVIDIA's query rounds 40000M to the displayed integer 39G.
            if not TORCH_LIMIT_MIB * MIB <= value <= CUDA_LIMIT_MIB * MIB:
                raise RuntimeError(f"Unexpected live MPS CUDA limit: {value}")
        assert_no_other_clients(uuid, proof)
        os.environ["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] = f"{uuid}={CUDA_LIMIT_MIB}M"
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps({"pid": os.getpid(), "uuid": uuid, "started": time.time()}))
        lock.flush()
    except BaseException:
        lock.close()
        raise
    return lock


def verify_current_client():
    proof = json.loads(Path(os.environ["TEMPOKV_STRICT_MEMORY_PROOF"]).read_text())
    return assert_no_other_clients(os.environ["CUDA_VISIBLE_DEVICES"], proof, require_self=True)


def apply_torch_limit():
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Strict worker must enumerate exactly one CUDA device")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(TORCH_LIMIT_MIB * MIB / total, device=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocation-probe", required=True, type=Path)
    parser.add_argument("--excess-request-mib", type=int, default=2048)
    args = parser.parse_args()
    allocation_probe(args.allocation_probe, args.excess_request_mib)
