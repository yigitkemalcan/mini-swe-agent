"""Summarize mini-SWE-Agent model events and optional vLLM request events."""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import typer
from analyze_experiment import distribution, evaluation_statuses

MODEL_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
VLLM_DURATION_FIELDS = (
    "queue_duration_ns",
    "prefill_duration_ns",
    "decode_duration_ns",
    "inference_duration_ns",
)


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
            raise ValueError(f"Expected one JSON object in {path}:{line_number}")
        events.append(event)
    return events


def is_integer(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def trajectory_response_count(trajectory: dict) -> int:
    """Count persisted, unique model responses, including format-error responses."""
    count = 0
    seen = set()
    for message in trajectory.get("messages", []):
        response = message.get("extra", {}).get("response")
        if not isinstance(response, dict):
            continue
        response_id = response.get("id")
        if response_id and response_id in seen:
            continue
        if response_id:
            seen.add(response_id)
        count += 1
    return count


def validate_model_event(event: dict, iid: str, path: Path) -> None:
    if event.get("event_type") != "model_request" or event.get("instance_id") != iid:
        raise ValueError(f"Unexpected event in {path}: {event}")
    if not is_integer(event.get("step_id"), minimum=1):
        raise ValueError(f"Invalid step_id in {path}: {event.get('step_id')!r}")
    if not is_integer(event.get("duration_ns"), minimum=1):
        raise ValueError(f"Invalid duration_ns in {path}: {event.get('duration_ns')!r}")
    if not is_integer(event.get("start_time_ns")) or not is_integer(event.get("end_time_ns")):
        raise ValueError(f"Invalid request timestamps in {path}")
    if event["end_time_ns"] < event["start_time_ns"]:
        raise ValueError(f"Model request ends before it starts in {path}")
    if event.get("status") not in ("success", "error"):
        raise ValueError(f"Invalid status in {path}: {event.get('status')!r}")
    if not isinstance(event.get("request_id"), str) or not event["request_id"]:
        raise ValueError(f"Invalid request_id in {path}: {event.get('request_id')!r}")
    tokens = [event.get(field) for field in MODEL_TOKEN_FIELDS]
    if any(value is not None for value in tokens):
        if not all(is_integer(value) for value in tokens):
            raise ValueError(f"Partially missing or invalid token usage in {path}: {tokens}")
        if event["total_tokens"] != event["prompt_tokens"] + event["completion_tokens"]:
            raise ValueError(f"Token total does not add up in {path}: {tokens}")
    finish_reason = event.get("finish_reason")
    if finish_reason is not None and (not isinstance(finish_reason, str) or not finish_reason):
        raise ValueError(f"Invalid finish_reason in {path}: {finish_reason!r}")


def validate_vllm_event(event: dict, path: Path) -> None:
    if not isinstance(event.get("request_id"), str) or not event["request_id"]:
        raise ValueError(f"Invalid request_id in {path}: {event.get('request_id')!r}")
    for field in (*VLLM_DURATION_FIELDS, "prompt_tokens", "completion_tokens"):
        if not is_integer(event.get(field)):
            raise ValueError(f"Invalid {field} in {path}: {event.get(field)!r}")
    if abs(event["inference_duration_ns"] - event["prefill_duration_ns"] - event["decode_duration_ns"]) > 1:
        raise ValueError(f"vLLM inference duration does not equal prefill plus decode in {path}")
    if not isinstance(event.get("finish_reason"), str) or not event["finish_reason"]:
        raise ValueError(f"Invalid finish_reason in {path}: {event.get('finish_reason')!r}")


def discover_instance_ids(results: Path) -> set[str]:
    predictions = results / "preds.json"
    ids = set(json.loads(predictions.read_text())) if predictions.exists() else set()
    ids.update(path.parent.name for path in results.glob("*/*.traj.json"))
    ids.update(path.parent.name for path in results.glob("*/*.traj.model_events.jsonl"))
    return ids


def analyze(
    root: Path, vllm_log: Path | None = None, evaluation: Path | None = None
) -> tuple[dict, list[dict], list[dict]]:
    results = root / "results" if (root / "results").is_dir() else root
    if not results.is_dir():
        raise ValueError(f"Results directory does not exist: {results}")
    ids = discover_instance_ids(results)
    if not ids:
        raise ValueError("No instance artifacts found; supply one run directory or its results directory")

    statuses = evaluation_statuses(evaluation)
    vllm_events = read_jsonl(vllm_log) if vllm_log else []
    for event in vllm_events:
        validate_vllm_event(event, vllm_log)
    vllm_by_id: dict[str, list[dict]] = {}
    for event in vllm_events:
        vllm_by_id.setdefault(event["request_id"], []).append(event)

    run_root = results.parent if results.name == "results" else root
    run_prefix = f"mswea-{run_root.name}-"
    run_vllm_events = [event for event in vllm_events if event["request_id"].startswith(run_prefix)]

    rows = []
    request_rows = []
    all_model_events = []
    all_matched_vllm_events = []
    all_request_ids = []
    model_statuses: Counter = Counter()
    model_errors: Counter = Counter()
    model_finishes: Counter = Counter()
    vllm_finishes: Counter = Counter()
    evaluation_counts: Counter = Counter()
    count_mismatches = 0
    token_mismatches = 0
    finish_mismatches = 0
    duplicate_vllm_matches = 0

    for iid in sorted(ids):
        folder = results / iid
        trajectory_path = folder / f"{iid}.traj.json"
        model_path = folder / f"{iid}.traj.model_events.jsonl"
        trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.exists() else {}
        events = read_jsonl(model_path) if model_path.exists() else []
        for event in events:
            validate_model_event(event, iid, model_path)

        info = trajectory.get("info", {})
        logical_calls = info.get("model_stats", {}).get("api_calls")
        if logical_calls is not None and not is_integer(logical_calls):
            raise ValueError(f"Invalid trajectory model-call count in {trajectory_path}: {logical_calls!r}")
        recorded_responses = trajectory_response_count(trajectory)
        count_matches = model_path.exists() and logical_calls is not None and len(events) == logical_calls
        count_mismatches += not count_matches

        ids_for_instance = [event["request_id"] for event in events]
        duplicate_model_ids = len(ids_for_instance) - len(set(ids_for_instance))
        matched = 0
        unmatched = 0
        instance_token_mismatches = 0
        instance_finish_mismatches = 0
        instance_duplicate_vllm = 0
        matched_vllm = []

        for event in events:
            matches = vllm_by_id.get(event["request_id"], [])
            match = matches[0] if len(matches) == 1 else None
            if len(matches) > 1:
                duplicate_vllm_matches += 1
                instance_duplicate_vllm += 1
            if match is None:
                unmatched += 1
            else:
                matched += 1
                matched_vllm.append(match)
                all_matched_vllm_events.append(match)
                tokens_agree = all(
                    event.get(field) == match.get(field) for field in ("prompt_tokens", "completion_tokens")
                )
                if not tokens_agree:
                    token_mismatches += 1
                    instance_token_mismatches += 1
                finish_agrees = event.get("finish_reason") == match.get("finish_reason")
                if not finish_agrees:
                    finish_mismatches += 1
                    instance_finish_mismatches += 1
                vllm_finishes[match["finish_reason"]] += 1

            request_rows.append(
                {
                    "instance_id": iid,
                    "step_id": event["step_id"],
                    "request_id": event["request_id"],
                    "status": event["status"],
                    "error_type": event.get("error_type"),
                    "model_duration_ns": event["duration_ns"],
                    "prompt_tokens": event.get("prompt_tokens"),
                    "completion_tokens": event.get("completion_tokens"),
                    "total_tokens": event.get("total_tokens"),
                    "model_finish_reason": event.get("finish_reason"),
                    "vllm_matched": match is not None,
                    "queue_duration_ns": match.get("queue_duration_ns") if match else None,
                    "prefill_duration_ns": match.get("prefill_duration_ns") if match else None,
                    "decode_duration_ns": match.get("decode_duration_ns") if match else None,
                    "inference_duration_ns": match.get("inference_duration_ns") if match else None,
                    "vllm_finish_reason": match.get("finish_reason") if match else None,
                    "tokens_agree": tokens_agree if match else None,
                    "finish_reasons_agree": finish_agrees if match else None,
                }
            )

        complete_token_events = [
            event for event in events if all(event.get(field) is not None for field in MODEL_TOKEN_FIELDS)
        ]
        durations = [event["duration_ns"] / 1e9 for event in events]
        evaluation_status = statuses.get(iid, "unknown")
        evaluation_counts[evaluation_status] += 1
        instance_statuses = Counter(event["status"] for event in events)
        instance_errors = Counter(event.get("error_type", "unknown") for event in events if event["status"] == "error")
        instance_finishes = Counter(str(event.get("finish_reason")) for event in events)
        rows.append(
            {
                "instance_id": iid,
                "agent_exit_status": info.get("exit_status"),
                "evaluation_status": evaluation_status,
                "trajectory_present": trajectory_path.exists(),
                "model_log_present": model_path.exists(),
                "logical_model_calls": logical_calls,
                "trajectory_responses": recorded_responses if trajectory_path.exists() else None,
                "model_events": len(events) if model_path.exists() else None,
                "model_call_count_matches": count_matches,
                "successful_model_events": instance_statuses["success"],
                "error_model_events": instance_statuses["error"],
                "model_error_types": json.dumps(dict(sorted(instance_errors.items()))),
                "model_finish_reasons": json.dumps(dict(sorted(instance_finishes.items()))),
                "duplicate_model_request_ids": duplicate_model_ids,
                "complete_token_events": len(complete_token_events),
                "missing_token_events": len(events) - len(complete_token_events),
                "vllm_matched_requests": matched if vllm_log else None,
                "vllm_unmatched_requests": unmatched if vllm_log else None,
                "vllm_duplicate_matches": instance_duplicate_vllm if vllm_log else None,
                "vllm_token_mismatches": instance_token_mismatches if vllm_log else None,
                "vllm_finish_reason_mismatches": instance_finish_mismatches if vllm_log else None,
                **distribution(durations, "model_request_seconds"),
                **distribution([event["prompt_tokens"] for event in complete_token_events], "prompt_tokens"),
                **distribution([event["completion_tokens"] for event in complete_token_events], "completion_tokens"),
                **distribution([event["total_tokens"] for event in complete_token_events], "total_tokens"),
                **distribution([event["queue_duration_ns"] / 1e9 for event in matched_vllm], "vllm_queue_seconds"),
                **distribution([event["prefill_duration_ns"] / 1e9 for event in matched_vllm], "vllm_prefill_seconds"),
                **distribution([event["decode_duration_ns"] / 1e9 for event in matched_vllm], "vllm_decode_seconds"),
                **distribution(
                    [event["inference_duration_ns"] / 1e9 for event in matched_vllm], "vllm_inference_seconds"
                ),
            }
        )
        all_model_events.extend(events)
        all_request_ids.extend(ids_for_instance)
        model_statuses.update(event["status"] for event in events)
        model_errors.update(event.get("error_type", "unknown") for event in events if event["status"] == "error")
        model_finishes.update(str(event.get("finish_reason")) for event in events)

    complete_events = [
        event for event in all_model_events if all(event.get(field) is not None for field in MODEL_TOKEN_FIELDS)
    ]
    matched_ids = {event["request_id"] for event in all_matched_vllm_events}
    run_vllm_ids = {event["request_id"] for event in run_vllm_events}
    experiment = {
        "results_directory": str(results.resolve()),
        "evaluation_report": str(evaluation.resolve()) if evaluation else None,
        "vllm_log": str(vllm_log.resolve()) if vllm_log else None,
        "instances": len(rows),
        "evaluation_statuses": dict(sorted(evaluation_counts.items())),
        "missing_trajectories": sum(not row["trajectory_present"] for row in rows),
        "missing_model_logs": sum(not row["model_log_present"] for row in rows),
        "instances_with_model_call_count_mismatch": count_mismatches,
        "model_requests": len(all_model_events),
        "model_statuses": dict(sorted(model_statuses.items())),
        "model_error_types": dict(sorted(model_errors.items())),
        "model_finish_reasons": dict(sorted(model_finishes.items())),
        "duplicate_model_request_ids": len(all_request_ids) - len(set(all_request_ids)),
        "model_requests_with_complete_tokens": len(complete_events),
        "model_requests_missing_tokens": len(all_model_events) - len(complete_events),
        "vllm_requests_for_run": len(run_vllm_events) if vllm_log else None,
        "vllm_matched_requests": len(matched_ids) if vllm_log else None,
        "vllm_unmatched_model_requests": len(all_model_events) - len(matched_ids) if vllm_log else None,
        "vllm_orphan_requests": len(run_vllm_ids - set(all_request_ids)) if vllm_log else None,
        "vllm_duplicate_matches": duplicate_vllm_matches if vllm_log else None,
        "vllm_token_mismatches": token_mismatches if vllm_log else None,
        "vllm_finish_reason_mismatches": finish_mismatches if vllm_log else None,
        "vllm_finish_reasons": dict(sorted(vllm_finishes.items())) if vllm_log else None,
        **distribution([event["duration_ns"] / 1e9 for event in all_model_events], "model_request_seconds"),
        **distribution([event["prompt_tokens"] for event in complete_events], "prompt_tokens"),
        **distribution([event["completion_tokens"] for event in complete_events], "completion_tokens"),
        **distribution([event["total_tokens"] for event in complete_events], "total_tokens"),
        **distribution([event["queue_duration_ns"] / 1e9 for event in all_matched_vllm_events], "vllm_queue_seconds"),
        **distribution(
            [event["prefill_duration_ns"] / 1e9 for event in all_matched_vllm_events], "vllm_prefill_seconds"
        ),
        **distribution([event["decode_duration_ns"] / 1e9 for event in all_matched_vllm_events], "vllm_decode_seconds"),
        **distribution(
            [event["inference_duration_ns"] / 1e9 for event in all_matched_vllm_events], "vllm_inference_seconds"
        ),
    }
    return experiment, rows, request_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(
    root: Path = typer.Argument(..., help="One experiment directory or its results/ directory."),
    vllm_log: Path | None = typer.Option(None, help="Optional vLLM request_events.jsonl for correlation."),
    evaluation: Path | None = typer.Option(None, help="Matching official SWE-Bench summary JSON or report.json."),
    output: Path | None = typer.Option(
        None, help="Report directory; defaults to model-analysis/ inside the supplied root."
    ),
) -> None:
    experiment, instances, requests = analyze(root, vllm_log, evaluation)
    destination = output or root / "model-analysis"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    write_csv(destination / "instances.csv", instances)
    write_csv(destination / "requests.csv", requests)
    print(json.dumps(experiment, indent=2))
    print(f"Reports written to {destination.resolve()}")


if __name__ == "__main__":
    typer.run(main)
