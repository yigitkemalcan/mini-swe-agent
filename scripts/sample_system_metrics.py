#!/usr/bin/env python3
"""Sample run-level Linux and NVIDIA resource metrics with a shared timeline."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

CPU_FIELDS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")


def utc_timestamp(timestamp_ns: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ns / 1_000_000_000, timezone.utc).isoformat().replace("+00:00", "Z")
    )


def parse_cpu_times(text: str) -> dict[str, int]:
    line = next((line for line in text.splitlines() if line.startswith("cpu ")), None)
    if line is None:
        raise ValueError("/proc/stat has no aggregate CPU line")
    values = [int(value) for value in line.split()[1 : len(CPU_FIELDS) + 1]]
    if len(values) != len(CPU_FIELDS):
        raise ValueError("/proc/stat aggregate CPU line is incomplete")
    return dict(zip(CPU_FIELDS, values, strict=True))


def parse_memory(text: str) -> dict[str, int]:
    values = {}
    for line in text.splitlines():
        name, separator, value = line.partition(":")
        if separator and name in {"MemTotal", "MemAvailable", "MemFree"}:
            amount, unit, *_ = value.split()
            if unit != "kB":
                raise ValueError(f"Unexpected unit for {name}: {unit}")
            values[name] = int(amount) * 1024
    missing = {"MemTotal", "MemAvailable", "MemFree"} - values.keys()
    if missing:
        raise ValueError(f"/proc/meminfo is missing: {', '.join(sorted(missing))}")
    return {
        "host_memory_used_bytes": values["MemTotal"] - values["MemAvailable"],
        "host_memory_available_bytes": values["MemAvailable"],
        "host_memory_free_bytes": values["MemFree"],
        "host_memory_total_bytes": values["MemTotal"],
    }


@dataclass
class CpuTracker:
    previous: dict[str, int]
    previous_monotonic_ns: int

    def sample(self, current: dict[str, int], monotonic_ns: int) -> dict[str, float | int]:
        delta = {name: current[name] - self.previous[name] for name in CPU_FIELDS}
        if any(value < 0 for value in delta.values()):
            raise ValueError("Aggregate CPU counters moved backwards")
        total = sum(delta.values())
        if total <= 0:
            raise ValueError("Aggregate CPU counters did not advance")
        interval_ns = monotonic_ns - self.previous_monotonic_ns
        self.previous = current
        self.previous_monotonic_ns = monotonic_ns
        return {
            "cpu_util_percent": 100.0 * (total - delta["idle"] - delta["iowait"]) / total,
            "cpu_user_percent": 100.0 * (delta["user"] + delta["nice"]) / total,
            "cpu_system_percent": 100.0 * (delta["system"] + delta["irq"] + delta["softirq"]) / total,
            "cpu_iowait_percent": 100.0 * delta["iowait"] / total,
            "cpu_steal_percent": 100.0 * delta["steal"] / total,
            "cpu_measurement_interval_ns": interval_ns,
        }


class NvmlError(RuntimeError):
    pass


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class Nvml:
    def __init__(self, expected_gpus: int):
        library = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
        self.library = ctypes.CDLL(library)
        self.initialized = False
        self._bind()
        self._check(self._init(), "nvmlInit")
        self.initialized = True
        try:
            count = ctypes.c_uint()
            self._check(self._get_count(ctypes.byref(count)), "nvmlDeviceGetCount")
            if count.value != expected_gpus:
                raise NvmlError(f"Expected {expected_gpus} GPUs, NVML reported {count.value}")
            self.handles = []
            for index in range(count.value):
                handle = ctypes.c_void_p()
                self._check(
                    self._get_handle(index, ctypes.byref(handle)), f"nvmlDeviceGetHandleByIndex({index})"
                )
                self.handles.append(handle)
        except Exception:
            self.shutdown()
            raise

    def _bind(self) -> None:
        self._init = getattr(self.library, "nvmlInit_v2", None) or self.library.nvmlInit
        self._get_count = getattr(self.library, "nvmlDeviceGetCount_v2", None) or self.library.nvmlDeviceGetCount
        self._get_handle = (
            getattr(self.library, "nvmlDeviceGetHandleByIndex_v2", None)
            or self.library.nvmlDeviceGetHandleByIndex
        )
        for function in (
            self._init,
            self._get_count,
            self._get_handle,
            self.library.nvmlDeviceGetUtilizationRates,
            self.library.nvmlDeviceGetMemoryInfo,
            self.library.nvmlDeviceGetUUID,
            self.library.nvmlShutdown,
        ):
            function.restype = ctypes.c_int

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result != 0:
            raise NvmlError(f"{operation} failed with NVML status {result}")

    def sample(self) -> list[dict]:
        samples = []
        for index, handle in enumerate(self.handles):
            utilization = _NvmlUtilization()
            memory = _NvmlMemory()
            uuid = ctypes.create_string_buffer(96)
            self._check(
                self.library.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(utilization)),
                f"nvmlDeviceGetUtilizationRates({index})",
            )
            self._check(
                self.library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)),
                f"nvmlDeviceGetMemoryInfo({index})",
            )
            self._check(
                self.library.nvmlDeviceGetUUID(handle, uuid, len(uuid)), f"nvmlDeviceGetUUID({index})"
            )
            samples.append(
                {
                    "index": index,
                    "uuid": uuid.value.decode(),
                    "utilization_percent": utilization.gpu,
                    "memory_used_bytes": memory.used,
                    "memory_total_bytes": memory.total,
                }
            )
        return samples

    def shutdown(self) -> None:
        if getattr(self, "initialized", False):
            self._check(self.library.nvmlShutdown(), "nvmlShutdown")
            self.initialized = False


class Summary:
    def __init__(self, run_id: str, interval_ns: int, start_wall_ns: int, start_monotonic_ns: int):
        self.run_id = run_id
        self.interval_ns = interval_ns
        self.start_wall_ns = start_wall_ns
        self.start_monotonic_ns = start_monotonic_ns
        self.samples = 0
        self.cpu_weighted_sum = 0.0
        self.cpu_interval_ns = 0
        self.max_cpu: float | None = None
        self.peak_memory: int | None = None
        self.gpus: dict[int, dict] = {}

    def add(self, event: dict) -> None:
        self.samples += 1
        interval_ns = event["cpu_measurement_interval_ns"]
        self.cpu_weighted_sum += event["cpu_util_percent"] * interval_ns
        self.cpu_interval_ns += interval_ns
        self.max_cpu = max(self.max_cpu or 0.0, event["cpu_util_percent"])
        self.peak_memory = max(self.peak_memory or 0, event["host_memory_used_bytes"])
        for gpu in event["gpus"]:
            aggregate = self.gpus.setdefault(
                gpu["index"],
                {
                    "index": gpu["index"],
                    "uuid": gpu["uuid"],
                    "memory_total_bytes": gpu["memory_total_bytes"],
                    "utilization_sum": 0,
                    "max_utilization_percent": 0,
                    "peak_memory_used_bytes": 0,
                },
            )
            aggregate["utilization_sum"] += gpu["utilization_percent"]
            aggregate["max_utilization_percent"] = max(
                aggregate["max_utilization_percent"], gpu["utilization_percent"]
            )
            aggregate["peak_memory_used_bytes"] = max(
                aggregate["peak_memory_used_bytes"], gpu["memory_used_bytes"]
            )

    def finish(self, end_wall_ns: int, end_monotonic_ns: int) -> dict:
        gpus = []
        for aggregate in self.gpus.values():
            utilization_sum = aggregate.pop("utilization_sum")
            gpus.append(aggregate | {"mean_utilization_percent": utilization_sum / self.samples})
        return {
            "event_type": "system_metrics_summary",
            "run_id": self.run_id,
            "sampling_interval_ns": self.interval_ns,
            "number_of_samples": self.samples,
            "start_timestamp_ns": self.start_wall_ns,
            "start_timestamp_utc": utc_timestamp(self.start_wall_ns),
            "end_timestamp_ns": end_wall_ns,
            "end_timestamp_utc": utc_timestamp(end_wall_ns),
            "start_monotonic_ns": self.start_monotonic_ns,
            "end_monotonic_ns": end_monotonic_ns,
            "elapsed_ns": end_monotonic_ns - self.start_monotonic_ns,
            "mean_cpu_util_percent": (
                self.cpu_weighted_sum / self.cpu_interval_ns if self.cpu_interval_ns else None
            ),
            "max_cpu_util_percent": self.max_cpu,
            "peak_host_memory_used_bytes": self.peak_memory,
            "gpus": gpus,
        }


def read_boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def parent_alive(parent_pid: int | None) -> bool:
    return parent_pid is None or os.getppid() == parent_pid


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def check(expected_gpus: int) -> dict:
    parse_cpu_times(Path("/proc/stat").read_text())
    memory = parse_memory(Path("/proc/meminfo").read_text())
    if expected_gpus == 0:
        return {"gpus": 0, "host_memory_total_bytes": memory["host_memory_total_bytes"]}
    nvml = Nvml(expected_gpus)
    try:
        nvml.sample()
        return {"gpus": len(nvml.handles), "host_memory_total_bytes": memory["host_memory_total_bytes"]}
    finally:
        nvml.shutdown()


def sample(
    output: Path,
    summary_path: Path,
    interval_seconds: float,
    expected_gpus: int,
    parent_pid: int | None,
    ready_file: Path | None,
) -> None:
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGHUP, lambda *_: stop.set())
    start_wall_ns = time.time_ns()
    start_monotonic_ns = time.monotonic_ns()
    cpu = CpuTracker(parse_cpu_times(Path("/proc/stat").read_text()), start_monotonic_ns)
    nvml = Nvml(expected_gpus) if expected_gpus else None
    run_id = output.parent.name
    summary = Summary(run_id, int(interval_seconds * 1_000_000_000), start_wall_ns, start_monotonic_ns)
    boot_id = read_boot_id()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", buffering=1) as destination:
            if ready_file:
                write_json(ready_file, {"pid": os.getpid(), "start_timestamp_ns": start_wall_ns})
            deadline_ns = start_monotonic_ns + summary.interval_ns
            while parent_alive(parent_pid) and not stop.wait(max(0, deadline_ns - time.monotonic_ns()) / 1e9):
                event_monotonic_ns = time.monotonic_ns()
                event_wall_ns = time.time_ns()
                event = {
                    "event_type": "system_metrics_sample",
                    "run_id": run_id,
                    "timestamp_ns": event_wall_ns,
                    "timestamp_utc": utc_timestamp(event_wall_ns),
                    "monotonic_ns": event_monotonic_ns,
                    "elapsed_ns": event_monotonic_ns - start_monotonic_ns,
                    "clock": "CLOCK_MONOTONIC",
                    "boot_id": boot_id,
                }
                event |= cpu.sample(parse_cpu_times(Path("/proc/stat").read_text()), event_monotonic_ns)
                event |= parse_memory(Path("/proc/meminfo").read_text())
                event["gpus"] = nvml.sample() if nvml else []
                destination.write(json.dumps(event, separators=(",", ":")) + "\n")
                summary.add(event)
                deadline_ns += summary.interval_ns
                if deadline_ns <= time.monotonic_ns():
                    deadline_ns = time.monotonic_ns() + summary.interval_ns
        end_monotonic_ns = time.monotonic_ns()
        write_json(summary_path, summary.finish(time.time_ns(), end_monotonic_ns))
    finally:
        if nvml:
            nvml.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    if args.expected_gpus < 0:
        parser.error("--expected-gpus must not be negative")
    if args.check:
        print(json.dumps(check(args.expected_gpus)))
        return
    if args.output is None or args.summary is None:
        parser.error("output and --summary are required unless --check is used")
    sample(args.output, args.summary, args.interval, args.expected_gpus, args.parent_pid, args.ready_file)


if __name__ == "__main__":
    main()
