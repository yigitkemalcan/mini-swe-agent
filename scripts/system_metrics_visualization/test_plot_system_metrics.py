import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from plot_system_metrics import visualize


def sample(timestamp_ns: int, elapsed_ns: int, cpu: float, gpu_values: tuple[int, int]) -> dict:
    return {
        "event_type": "system_metrics_sample",
        "run_id": "test-run",
        "timestamp_ns": timestamp_ns,
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "monotonic_ns": 10_000 + elapsed_ns,
        "elapsed_ns": elapsed_ns,
        "cpu_util_percent": cpu,
        "cpu_user_percent": cpu / 2,
        "cpu_system_percent": cpu / 2,
        "cpu_iowait_percent": 0,
        "cpu_measurement_interval_ns": 500_000_000,
        "host_memory_used_bytes": 40,
        "host_memory_available_bytes": 60,
        "host_memory_total_bytes": 100,
        "gpus": [
            {
                "index": index,
                "uuid": f"GPU-{index}",
                "utilization_percent": utilization,
                "memory_used_bytes": 60 + index,
                "memory_total_bytes": 100,
            }
            for index, utilization in enumerate(gpu_values)
        ],
    }


def test_visualize_generates_parseable_graphs_and_manifest(tmp_path: Path) -> None:
    run = tmp_path / "test-run"
    instance = run / "results" / "example__repo-1"
    instance.mkdir(parents=True)
    samples = [sample(1_000_000_000, 0, 20, (10, 30)), sample(1_500_000_000, 500_000_000, 60, (50, 90))]
    (run / "system_metrics.jsonl").write_text("\n".join(json.dumps(value) for value in samples) + "\n")
    (instance / "example__repo-1.traj.model_events.jsonl").write_text(
        json.dumps({"start_time_ns": 1_100_000_000, "end_time_ns": 1_300_000_000, "step_id": 1, "status": "success"})
        + "\n"
    )
    (instance / "example__repo-1.traj.tool_events.jsonl").write_text(
        json.dumps({"start_time_ns": 1_350_000_000, "end_time_ns": 1_400_000_000, "step_id": 1}) + "\n"
    )

    output = tmp_path / "graphs"
    manifest = visualize(run, output)

    assert manifest["number_of_samples"] == 2
    assert manifest["model_events"] == manifest["tool_events"] == 1
    assert manifest["cpu_utilization_percent_mean"] == 40
    assert manifest["gpus"][1]["max_utilization_percent"] == 90
    assert json.loads((output / "visualization_summary.json").read_text())["run_id"] == "test-run"
    assert "resource_timeline.svg" in (output / "index.html").read_text()
    for name in ("resource_timeline.svg", "gpu_utilization_heatmap.svg", "run_summary.svg"):
        assert ET.parse(output / name).getroot().tag.endswith("svg")


def test_visualize_rejects_inconsistent_gpu_identity(tmp_path: Path) -> None:
    run = tmp_path / "test-run"
    run.mkdir()
    samples = [sample(1_000_000_000, 0, 20, (10, 30)), sample(1_500_000_000, 500_000_000, 60, (50, 90))]
    samples[1]["gpus"][1]["uuid"] = "different"
    (run / "system_metrics.jsonl").write_text("\n".join(json.dumps(value) for value in samples) + "\n")

    with pytest.raises(ValueError, match="GPU identities changed"):
        visualize(run, tmp_path / "graphs")
