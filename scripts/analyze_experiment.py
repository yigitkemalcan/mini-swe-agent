"""Summarize one experiment without interpreting shell commands."""

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import typer


def distribution(values: list[int | float], prefix: str) -> dict:
    ordered = sorted(values)
    return {
        f"{prefix}_samples": len(values),
        f"{prefix}_total": sum(values) if values else None,
        f"{prefix}_mean": statistics.fmean(values) if values else None,
        **{
            f"{prefix}_p{percent}": ordered[math.ceil(percent / 100 * len(ordered)) - 1] if ordered else None
            for percent in (50, 90, 95, 99)
        },
        f"{prefix}_max": max(values) if values else None,
    }


def evaluation_statuses(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    statuses = {}
    for field, status in (
        ("resolved_ids", "solved"),
        ("unresolved_ids", "unsolved"),
        ("error_ids", "evaluation_error"),
        ("empty_patch_ids", "empty_patch"),
        ("incomplete_ids", "incomplete"),
    ):
        for iid in data.get(field, []):
            if iid not in statuses:
                statuses[iid] = status
    # Also accept one official per-instance report.json.
    if not statuses and not any(key.endswith("_ids") for key in data):
        statuses = {
            iid: "solved" if report["resolved"] else "unsolved"
            for iid, report in data.items()
            if isinstance(report, dict) and isinstance(report.get("resolved"), bool)
        }
    if not statuses and not any(field in data for field in ("resolved_ids", "unresolved_ids", "error_ids", "empty_patch_ids", "incomplete_ids")):
        raise ValueError("Evaluation report contains no recognized instance outcomes")
    return statuses


def read_instance(iid: str, folder: Path) -> tuple[dict, list[float], list[int]]:
    trajectory = folder / f"{iid}.traj.json"
    log = folder / f"{iid}.traj.tool_events.jsonl"
    data = json.loads(trajectory.read_text()) if trajectory.exists() else {}
    info = data.get("info", {})
    events = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.exists() else []
    durations = []
    for event in events:
        if event.get("event_type") != "tool_execution" or event.get("instance_id", iid) != iid:
            raise ValueError(f"Unexpected event in {log}")
        duration = event.get("duration_ns")
        if not isinstance(duration, int) or isinstance(duration, bool) or duration < 0:
            raise ValueError(f"Invalid duration in {log}")
        durations.append(duration / 1e9)

    tokens = []
    responses = 0
    seen = set()
    for message in data.get("messages", []):
        response = message.get("extra", {}).get("response")
        if not isinstance(response, dict):
            continue
        response_id = response.get("id")
        if response_id and response_id in seen:
            continue
        if response_id:
            seen.add(response_id)
        responses += 1
        usage = response.get("usage") or {}
        value = usage.get("completion_tokens", usage.get("output_tokens"))
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Invalid output token count in {trajectory}")
        tokens.append(value)

    api_calls = info.get("model_stats", {}).get("api_calls")
    return {
        "instance_id": iid,
        "agent_exit_status": info.get("exit_status"),
        "trajectory_present": trajectory.exists(),
        "tool_log_present": log.exists(),
        "tool_calls": len(events) if log.exists() else None,
        "model_calls": api_calls,
        "recorded_responses": responses,
        "token_usage_complete": bool(trajectory.exists() and api_calls is not None and api_calls == responses == len(tokens)),
        **distribution(durations, "tool_seconds"),
        **distribution(tokens, "output_tokens"),
    }, durations, tokens


def analyze(root: Path, evaluation: Path | None = None) -> tuple[dict, list[dict]]:
    results = root / "results" if (root / "results").is_dir() else root
    if not results.is_dir():
        raise ValueError(f"Results directory does not exist: {results}")
    predictions = results / "preds.json"
    ids = set(json.loads(predictions.read_text())) if predictions.exists() else set()
    ids.update(path.parent.name for path in results.glob("*/*.traj.json"))
    ids.update(path.parent.name for path in results.glob("*/*.traj.tool_events.jsonl"))
    if not ids:
        raise ValueError("No instance artifacts found; supply one run directory or its results directory")
    statuses = evaluation_statuses(evaluation)
    rows, all_durations, all_tokens = [], [], []
    for iid in sorted(ids):
        row, durations, tokens = read_instance(iid, results / iid)
        row["evaluation_status"] = statuses.get(iid, "unknown")
        rows.append(row)
        all_durations.extend(durations)
        all_tokens.extend(tokens)
    counts = Counter(row["evaluation_status"] for row in rows)
    return {
        "results_directory": str(results.resolve()),
        "evaluation_report": str(evaluation.resolve()) if evaluation else None,
        "instances": len(rows),
        "solved": counts["solved"],
        "unsolved": counts["unsolved"],
        "evaluation_unknown": counts["unknown"],
        "evaluation_other": {key: value for key, value in counts.items() if key not in ("solved", "unsolved", "unknown")},
        "missing_trajectories": sum(not row["trajectory_present"] for row in rows),
        "missing_tool_logs": sum(not row["tool_log_present"] for row in rows),
        "instances_with_incomplete_token_usage": sum(not row["token_usage_complete"] for row in rows),
        "tool_calls": len(all_durations),
        **distribution(all_durations, "tool_seconds"),
        **distribution(all_tokens, "output_tokens"),
    }, rows


def main(
    root: Path = typer.Argument(..., help="One experiment directory or its results/ directory."),
    evaluation: Path | None = typer.Option(None, help="Matching official SWE-Bench summary JSON or report.json."),
    output: Path | None = typer.Option(None, help="Report directory; defaults to analysis/ inside the supplied root."),
) -> None:
    experiment, instances = analyze(root, evaluation)
    destination = output or root / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    with (destination / "instances.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(instances[0]))
        writer.writeheader()
        writer.writerows(instances)
    print(json.dumps(experiment, indent=2))
    print(f"Reports written to {destination.resolve()}")


if __name__ == "__main__":
    typer.run(main)
