#!/usr/bin/env python3
"""Create dependency-free SVG graphs from a run-level system metrics log."""

from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path
from statistics import mean

COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5", "#be123c")
BACKGROUND = "#f8fafc"
GRID = "#cbd5e1"
TEXT = "#0f172a"
MUTED = "#64748b"


def read_jsonl(path: Path) -> list[dict]:
    events = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
        if not isinstance(event, dict):
            raise ValueError(f"Expected a JSON object in {path}:{line_number}")
        events.append(event)
    return events


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_samples(samples: list[dict], path: Path) -> None:
    if not samples:
        raise ValueError(f"No system metrics samples in {path}")
    run_id = samples[0].get("run_id")
    gpu_identity: list[tuple[int, str, int]] | None = None
    previous_timestamp = -1
    required = (
        "timestamp_ns",
        "monotonic_ns",
        "elapsed_ns",
        "cpu_util_percent",
        "cpu_user_percent",
        "cpu_system_percent",
        "cpu_iowait_percent",
        "cpu_measurement_interval_ns",
        "host_memory_used_bytes",
        "host_memory_available_bytes",
        "host_memory_total_bytes",
    )
    for index, sample in enumerate(samples, 1):
        if sample.get("event_type") != "system_metrics_sample" or sample.get("run_id") != run_id:
            raise ValueError(f"Unexpected event identity in {path}, sample {index}")
        if any(not is_number(sample.get(field)) for field in required):
            raise ValueError(f"Missing or invalid metric in {path}, sample {index}")
        if sample["timestamp_ns"] <= previous_timestamp:
            raise ValueError(f"System metric timestamps are not increasing in {path}")
        previous_timestamp = sample["timestamp_ns"]
        for field in ("cpu_util_percent", "cpu_user_percent", "cpu_system_percent", "cpu_iowait_percent"):
            if not 0 <= sample[field] <= 100:
                raise ValueError(f"{field} is outside 0..100 in {path}, sample {index}")
        if not 0 <= sample["host_memory_used_bytes"] <= sample["host_memory_total_bytes"]:
            raise ValueError(f"Invalid host memory values in {path}, sample {index}")
        gpus = sample.get("gpus")
        if not isinstance(gpus, list) or not gpus:
            raise ValueError(f"No GPU metrics in {path}, sample {index}")
        identity = []
        for gpu in gpus:
            if not all(
                key in gpu
                for key in ("index", "uuid", "utilization_percent", "memory_used_bytes", "memory_total_bytes")
            ):
                raise ValueError(f"Incomplete GPU metric in {path}, sample {index}")
            if not 0 <= gpu["utilization_percent"] <= 100:
                raise ValueError(f"GPU utilization is outside 0..100 in {path}, sample {index}")
            if not 0 <= gpu["memory_used_bytes"] <= gpu["memory_total_bytes"]:
                raise ValueError(f"Invalid GPU memory values in {path}, sample {index}")
            identity.append((gpu["index"], gpu["uuid"], gpu["memory_total_bytes"]))
        if gpu_identity is None:
            gpu_identity = identity
        elif identity != gpu_identity:
            raise ValueError(f"GPU identities changed in {path}, sample {index}")


def event_spans(run: Path, pattern: str) -> list[dict]:
    spans = []
    for path in sorted((run / "results").glob(pattern)):
        for event in read_jsonl(path):
            start = event.get("start_time_ns")
            end = event.get("end_time_ns")
            if not isinstance(start, int) or not isinstance(end, int) or end < start:
                raise ValueError(f"Invalid event interval in {path}")
            spans.append(event)
    return spans


class Svg:
    def __init__(self, width: int, height: int, title: str):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f"<title>{escape(title)}</title>",
            f'<rect width="100%" height="100%" fill="{BACKGROUND}"/>',
            "<style>text{font-family:system-ui,-apple-system,sans-serif} .label{fill:#0f172a;font-size:13px} "
            ".small{fill:#64748b;font-size:11px}.title{fill:#0f172a;font-size:20px;font-weight:700} "
            ".subtitle{fill:#475569;font-size:12px}.legend{fill:#334155;font-size:11px}</style>",
        ]

    def add(self, value: str) -> None:
        self.parts.append(value)

    def text(self, x: float, y: float, value: object, css: str = "label", anchor: str = "start") -> None:
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" class="{css}" text-anchor="{anchor}">{escape(str(value))}</text>')

    def save(self, path: Path) -> None:
        path.write_text("\n".join((*self.parts, "</svg>\n")))


def scale(value: float, low: float, high: float, start: float, end: float) -> float:
    return start if high == low else start + (value - low) * (end - start) / (high - low)


def axes(svg: Svg, x: float, y: float, width: float, height: float, title: str, y_max: float = 100) -> None:
    svg.text(x, y - 10, title)
    for value in (0, y_max / 2, y_max):
        py = scale(value, 0, y_max, y + height, y)
        svg.add(f'<line x1="{x}" y1="{py:.1f}" x2="{x + width}" y2="{py:.1f}" stroke="{GRID}" stroke-width="1"/>')
        svg.text(x - 8, py + 4, f"{value:g}", "small", "end")


def polyline(svg: Svg, points: list[tuple[float, float]], color: str, width: float = 1.7, opacity: float = 1) -> None:
    coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    svg.add(
        f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}" stroke-linejoin="round"/>'
    )


def legend(svg: Svg, entries: list[tuple[str, str]], x: float, y: float, columns: int = 4) -> None:
    for index, (label, color) in enumerate(entries):
        px = x + (index % columns) * 150
        py = y + (index // columns) * 18
        svg.add(f'<line x1="{px}" y1="{py - 4}" x2="{px + 18}" y2="{py - 4}" stroke="{color}" stroke-width="3"/>')
        svg.text(px + 23, py, label, "legend")


def timeline_plot(
    run: Path, samples: list[dict], model_events: list[dict], tool_events: list[dict], output: Path
) -> None:
    svg = Svg(1400, 1000, f"Resource timeline: {run.name}")
    svg.text(55, 36, "Run-level resource timeline", "title")
    svg.text(
        55,
        57,
        f"{run.name} · {len(samples)} samples · {samples[0]['timestamp_utc']} to {samples[-1]['timestamp_utc']}",
        "subtitle",
    )
    left, width = 80, 1270
    t0, t1 = samples[0]["timestamp_ns"], samples[-1]["timestamp_ns"]
    xs = [scale(sample["timestamp_ns"], t0, t1, left, left + width) for sample in samples]

    panels = ((100, "CPU utilization (%)"), (310, "GPU compute utilization (%)"), (550, "Memory occupancy (%)"))
    for y, title in panels:
        axes(svg, left, y, width, 150, title)

    cpu_fields = (
        ("cpu_util_percent", "CPU active", "#0f172a", 2.6),
        ("cpu_user_percent", "user + nice", "#2563eb", 1.7),
        ("cpu_system_percent", "system + IRQ", "#dc2626", 1.7),
        ("cpu_iowait_percent", "I/O wait", "#ea580c", 1.7),
    )
    for field, _, color, line_width in cpu_fields:
        polyline(
            svg,
            list(zip(xs, [scale(sample[field], 0, 100, 250, 100) for sample in samples], strict=True)),
            color,
            line_width,
        )
    legend(svg, [(label, color) for _, label, color, _ in cpu_fields], left + 540, 91)

    gpu_count = len(samples[0]["gpus"])
    for index in range(gpu_count):
        polyline(
            svg,
            list(
                zip(
                    xs,
                    [scale(sample["gpus"][index]["utilization_percent"], 0, 100, 460, 310) for sample in samples],
                    strict=True,
                )
            ),
            COLORS[index % len(COLORS)],
            1.25,
            0.75,
        )
    gpu_mean = [mean(gpu["utilization_percent"] for gpu in sample["gpus"]) for sample in samples]
    polyline(svg, list(zip(xs, [scale(value, 0, 100, 460, 310) for value in gpu_mean], strict=True)), "#0f172a", 3)
    legend(
        svg,
        [(f"GPU {index}", COLORS[index % len(COLORS)]) for index in range(gpu_count)] + [("8-GPU mean", "#0f172a")],
        left + 260,
        491,
        5,
    )

    host_memory = [100 * sample["host_memory_used_bytes"] / sample["host_memory_total_bytes"] for sample in samples]
    gpu_memory = [
        mean(100 * gpu["memory_used_bytes"] / gpu["memory_total_bytes"] for gpu in sample["gpus"]) for sample in samples
    ]
    polyline(svg, list(zip(xs, [scale(value, 0, 100, 700, 550) for value in host_memory], strict=True)), "#2563eb", 2.2)
    polyline(svg, list(zip(xs, [scale(value, 0, 100, 700, 550) for value in gpu_memory], strict=True)), "#9333ea", 2.2)
    host_gib = samples[-1]["host_memory_used_bytes"] / 2**30
    gpu_gib = mean(gpu["memory_used_bytes"] for gpu in samples[-1]["gpus"]) / 2**30
    legend(
        svg,
        [(f"Host used ({host_gib:.1f} GiB at end)", "#2563eb"), (f"Mean GPU used ({gpu_gib:.1f} GiB/GPU)", "#9333ea")],
        left + 670,
        541,
        2,
    )

    activity_y = 780
    svg.text(left, activity_y - 24, "Agent activity aligned by wall-clock timestamp")
    svg.text(left - 8, activity_y + 15, "model", "small", "end")
    svg.text(left - 8, activity_y + 49, "tool", "small", "end")
    svg.add(f'<rect x="{left}" y="{activity_y}" width="{width}" height="22" fill="#e2e8f0" rx="3"/>')
    svg.add(f'<rect x="{left}" y="{activity_y + 34}" width="{width}" height="22" fill="#e2e8f0" rx="3"/>')
    for event in model_events:
        x_start = scale(max(event["start_time_ns"], t0), t0, t1, left, left + width)
        x_end = scale(min(event["end_time_ns"], t1), t0, t1, left, left + width)
        if x_end >= left and x_start <= left + width:
            color = "#ef4444" if event.get("status") == "error" else "#16a34a"
            svg.add(
                f'<rect x="{x_start:.1f}" y="{activity_y}" width="{max(1, x_end - x_start):.1f}" height="22" fill="{color}" opacity="0.76"><title>model step {escape(str(event.get("step_id")))} · {escape(str(event.get("status", "unknown")))}</title></rect>'
            )
    for event in tool_events:
        x_start = scale(max(event["start_time_ns"], t0), t0, t1, left, left + width)
        x_end = scale(min(event["end_time_ns"], t1), t0, t1, left, left + width)
        if x_end >= left and x_start <= left + width:
            svg.add(
                f'<rect x="{x_start:.1f}" y="{activity_y + 34}" width="{max(1, x_end - x_start):.1f}" height="22" fill="#2563eb" opacity="0.78"><title>tool step {escape(str(event.get("step_id")))}</title></rect>'
            )
    legend(
        svg,
        [("model success", "#16a34a"), ("model error", "#ef4444"), ("tool execution", "#2563eb")],
        left + 780,
        activity_y - 29,
        3,
    )

    for second in range(0, math.ceil((t1 - t0) / 1e9) + 1, 5):
        px = scale(t0 + second * 1e9, t0, t1, left, left + width)
        if px <= left + width:
            svg.add(
                f'<line x1="{px:.1f}" y1="{activity_y + 58}" x2="{px:.1f}" y2="{activity_y + 65}" stroke="{MUTED}"/>'
            )
            svg.text(px, activity_y + 81, f"{second}s", "small", "middle")
    svg.text(
        left,
        930,
        "CPU is host-wide; GPU utilization is NVML's recent-window measurement. Memory peaks are sampled, not instantaneous.",
        "subtitle",
    )
    svg.text(
        left,
        950,
        "Colored activity spans come from model_events and tool_events and use their wall-clock start/end timestamps.",
        "subtitle",
    )
    svg.save(output)


def heat_color(value: float) -> str:
    low = (239, 246, 255)
    high = (30, 64, 175)
    ratio = value / 100
    return "#" + "".join(f"{round(a + (b - a) * ratio):02x}" for a, b in zip(low, high, strict=True))


def heatmap_plot(run: Path, samples: list[dict], output: Path) -> None:
    svg = Svg(1400, 590, f"GPU utilization heatmap: {run.name}")
    svg.text(55, 36, "GPU utilization heatmap", "title")
    svg.text(55, 57, f"{run.name} · each column is one system sample", "subtitle")
    left, top, width, row_height = 90, 90, 1240, 45
    cell_width = width / len(samples)
    for gpu_index in range(len(samples[0]["gpus"])):
        y = top + gpu_index * row_height
        svg.text(left - 12, y + 28, f"GPU {gpu_index}", "label", "end")
        for sample_index, sample in enumerate(samples):
            value = sample["gpus"][gpu_index]["utilization_percent"]
            x = left + sample_index * cell_width
            svg.add(
                f'<rect x="{x:.2f}" y="{y}" width="{cell_width + 0.2:.2f}" height="34" fill="{heat_color(value)}"><title>{sample["timestamp_utc"]} · GPU {gpu_index}: {value}%</title></rect>'
            )
    duration = (samples[-1]["timestamp_ns"] - samples[0]["timestamp_ns"]) / 1e9
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        x = left + fraction * width
        svg.text(x, top + len(samples[0]["gpus"]) * row_height + 23, f"{duration * fraction:.1f}s", "small", "middle")
    for index, value in enumerate((0, 25, 50, 75, 100)):
        x = left + index * 55
        svg.add(f'<rect x="{x}" y="{top + 395}" width="40" height="16" fill="{heat_color(value)}"/>')
        svg.text(x + 20, top + 430, f"{value}%", "small", "middle")
    svg.save(output)


def bar(svg: Svg, x: float, y: float, width: float, value: float, color: str, label: str) -> None:
    svg.add(f'<rect x="{x}" y="{y}" width="{width}" height="22" fill="#e2e8f0" rx="3"/>')
    svg.add(f'<rect x="{x}" y="{y}" width="{width * value / 100:.1f}" height="22" fill="{color}" rx="3"/>')
    svg.text(x - 10, y + 16, label, "label", "end")
    svg.text(x + width + 10, y + 16, f"{value:.1f}%", "label")


def summary_plot(run: Path, samples: list[dict], output: Path) -> dict:
    svg = Svg(1200, 780, f"Resource summary: {run.name}")
    svg.text(55, 36, "Run-level resource summary", "title")
    svg.text(55, 57, run.name, "subtitle")
    interval_seconds = [(b["monotonic_ns"] - a["monotonic_ns"]) / 1e9 for a, b in zip(samples, samples[1:])]
    cpu_interval_total = sum(sample["cpu_measurement_interval_ns"] for sample in samples)
    cpu_mean = (
        sum(sample["cpu_util_percent"] * sample["cpu_measurement_interval_ns"] for sample in samples)
        / cpu_interval_total
    )
    cpu_max = max(sample["cpu_util_percent"] for sample in samples)
    svg.text(210, 105, "CPU utilization")
    bar(svg, 210, 125, 360, cpu_mean, "#2563eb", "mean")
    bar(svg, 210, 160, 360, cpu_max, "#0f172a", "max")

    svg.text(210, 225, "GPU compute utilization by device")
    gpu_summary = []
    for index in range(len(samples[0]["gpus"])):
        values = [sample["gpus"][index]["utilization_percent"] for sample in samples]
        gpu_summary.append(
            {
                "index": index,
                "mean_utilization_percent": mean(values),
                "max_utilization_percent": max(values),
                "peak_memory_used_bytes": max(sample["gpus"][index]["memory_used_bytes"] for sample in samples),
                "memory_total_bytes": samples[0]["gpus"][index]["memory_total_bytes"],
            }
        )
        y = 245 + index * 42
        bar(svg, 210, y, 360, gpu_summary[-1]["mean_utilization_percent"], COLORS[index], f"GPU {index}")
        marker = 210 + 360 * gpu_summary[-1]["max_utilization_percent"] / 100
        svg.add(
            f'<line x1="{marker:.1f}" y1="{y - 3}" x2="{marker:.1f}" y2="{y + 25}" stroke="#0f172a" stroke-width="3"><title>maximum {gpu_summary[-1]["max_utilization_percent"]}%</title></line>'
        )
    svg.text(210, 604, "Colored bar = mean; black marker = maximum", "subtitle")

    peak_host = max(sample["host_memory_used_bytes"] for sample in samples)
    host_total = samples[0]["host_memory_total_bytes"]
    svg.text(760, 105, "Sampled peak memory")
    bar(svg, 760, 125, 280, 100 * peak_host / host_total, "#2563eb", "host")
    svg.text(760, 169, f"{peak_host / 2**30:.2f} / {host_total / 2**30:.2f} GiB", "subtitle")
    for index, gpu in enumerate(gpu_summary):
        if index in (0, 7):
            y = 220 + (index // 7) * 75
            bar(
                svg,
                760,
                y,
                280,
                100 * gpu["peak_memory_used_bytes"] / gpu["memory_total_bytes"],
                COLORS[index],
                f"GPU {index}",
            )
            svg.text(
                760,
                y + 44,
                f"{gpu['peak_memory_used_bytes'] / 2**30:.2f} / {gpu['memory_total_bytes'] / 2**30:.2f} GiB",
                "subtitle",
            )
    svg.text(760, 395, "Run coverage")
    duration_seconds = (samples[-1]["timestamp_ns"] - samples[0]["timestamp_ns"]) / 1e9
    rows = (
        ("samples", len(samples)),
        ("sampled span", f"{duration_seconds:.2f} s"),
        ("mean spacing", f"{mean(interval_seconds):.6f} s" if interval_seconds else "n/a"),
        ("maximum spacing", f"{max(interval_seconds):.6f} s" if interval_seconds else "n/a"),
    )
    for index, (label, value) in enumerate(rows):
        svg.text(760, 430 + index * 34, label, "small")
        svg.text(1040, 430 + index * 34, value, "label", "end")
    svg.save(output)
    return {
        "run_id": run.name,
        "number_of_samples": len(samples),
        "sampled_duration_seconds": duration_seconds,
        "sample_spacing_seconds_mean": mean(interval_seconds) if interval_seconds else None,
        "sample_spacing_seconds_max": max(interval_seconds) if interval_seconds else None,
        "cpu_utilization_percent_mean": cpu_mean,
        "cpu_utilization_percent_max": cpu_max,
        "peak_host_memory_used_bytes": peak_host,
        "host_memory_total_bytes": host_total,
        "gpus": gpu_summary,
    }


def write_index(run: Path, output: Path, manifest: dict) -> None:
    output.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>System metrics visualization</title>"
        "<style>body{font-family:system-ui;margin:2rem;background:#f8fafc;color:#0f172a}"
        "img{display:block;max-width:100%;margin:1.5rem 0;background:white;border:1px solid #cbd5e1}"
        "code{background:#e2e8f0;padding:.2rem .4rem}</style></head><body>"
        f"<h1>System metrics: {escape(run.name)}</h1>"
        f"<p>{manifest['number_of_samples']} samples across {manifest['sampled_duration_seconds']:.2f} seconds. "
        "Hover over heatmap cells and activity spans for details.</p>"
        "<img src='resource_timeline.svg' alt='Resource timeline'>"
        "<img src='gpu_utilization_heatmap.svg' alt='GPU utilization heatmap'>"
        "<img src='run_summary.svg' alt='Run summary'></body></html>\n"
    )


def visualize(run: Path, output: Path) -> dict:
    metrics_path = run / "system_metrics.jsonl"
    if not metrics_path.is_file():
        raise ValueError(f"System metrics log does not exist: {metrics_path}")
    samples = read_jsonl(metrics_path)
    validate_samples(samples, metrics_path)
    model_events = event_spans(run, "*/*.traj.model_events.jsonl")
    tool_events = event_spans(run, "*/*.traj.tool_events.jsonl")
    output.mkdir(parents=True, exist_ok=True)
    timeline_plot(run, samples, model_events, tool_events, output / "resource_timeline.svg")
    heatmap_plot(run, samples, output / "gpu_utilization_heatmap.svg")
    manifest = summary_plot(run, samples, output / "run_summary.svg")
    manifest |= {
        "source": str(metrics_path.resolve()),
        "model_events": len(model_events),
        "tool_events": len(tool_events),
        "outputs": ["index.html", "resource_timeline.svg", "gpu_utilization_heatmap.svg", "run_summary.svg"],
    }
    (output / "visualization_summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_index(run, output / "index.html", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="SWE-Bench run directory containing system_metrics.jsonl")
    parser.add_argument("--output", type=Path, help="Output directory (default: this script's results/<run-id>)")
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output or Path(__file__).parent / "results" / run.name
    manifest = visualize(run, output)
    print(json.dumps({"output_directory": str(output.resolve()), **manifest}, indent=2))


if __name__ == "__main__":
    main()
