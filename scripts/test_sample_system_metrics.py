import json
import os
import subprocess
import sys
import time
from pathlib import Path

from sample_system_metrics import CpuTracker, parse_cpu_times, parse_memory

SCRIPT = Path(__file__).with_name("sample_system_metrics.py")


def test_proc_parsers_and_cpu_semantics():
    previous = parse_cpu_times("cpu  100 10 40 800 20 5 5 20 0 0\ncpu0 1 2 3 4\n")
    current = parse_cpu_times("cpu  120 10 50 850 30 10 10 20 0 0\n")

    assert CpuTracker(previous, 1_000).sample(current, 2_000) == {
        "cpu_util_percent": 40.0,
        "cpu_user_percent": 20.0,
        "cpu_system_percent": 20.0,
        "cpu_iowait_percent": 10.0,
        "cpu_steal_percent": 0.0,
        "cpu_measurement_interval_ns": 1_000,
    }
    assert parse_memory("MemTotal: 1000 kB\nMemFree: 100 kB\nMemAvailable: 400 kB\n") == {
        "host_memory_used_bytes": 600 * 1024,
        "host_memory_available_bytes": 400 * 1024,
        "host_memory_free_bytes": 100 * 1024,
        "host_memory_total_bytes": 1000 * 1024,
    }


def test_sampler_stops_cleanly_and_writes_valid_log_and_summary(tmp_path):
    output = tmp_path / "system_metrics.jsonl"
    summary = tmp_path / "system_metrics_summary.json"
    ready = tmp_path / "ready.json"
    process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            str(output),
            "--summary",
            str(summary),
            "--interval",
            "0.02",
            "--expected-gpus",
            "0",
            "--parent-pid",
            str(os.getpid()),
            "--ready-file",
            str(ready),
        ]
    )
    deadline = time.monotonic() + 3
    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists()
    time.sleep(0.08)
    process.terminate()
    assert process.wait(timeout=3) == 0

    events = [json.loads(line) for line in output.read_text().splitlines()]
    report = json.loads(summary.read_text())
    assert len(events) >= 2
    assert all(
        {
            "event_type",
            "run_id",
            "timestamp_ns",
            "timestamp_utc",
            "monotonic_ns",
            "elapsed_ns",
            "cpu_util_percent",
            "cpu_user_percent",
            "cpu_system_percent",
            "cpu_iowait_percent",
            "cpu_steal_percent",
            "host_memory_used_bytes",
            "host_memory_available_bytes",
            "host_memory_free_bytes",
            "host_memory_total_bytes",
            "gpus",
        }
        <= event.keys()
        for event in events
    )
    assert all(event["event_type"] == "system_metrics_sample" and event["gpus"] == [] for event in events)
    assert all(event["run_id"] == tmp_path.name for event in events)
    assert [event["monotonic_ns"] for event in events] == sorted(event["monotonic_ns"] for event in events)
    assert report["event_type"] == "system_metrics_summary"
    assert report["run_id"] == tmp_path.name
    assert report["number_of_samples"] == len(events)
    assert report["sampling_interval_ns"] == 20_000_000
    assert 0 <= report["mean_cpu_util_percent"] <= report["max_cpu_util_percent"] <= 100
    assert report["peak_host_memory_used_bytes"] >= max(
        event["host_memory_used_bytes"] for event in events
    )
    assert report["gpus"] == []
