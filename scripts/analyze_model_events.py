"""Summarize mini-SWE-Agent model events and optional vLLM request events."""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import typer
from analyze_experiment import distribution, evaluation_statuses

MODEL_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
MODEL_OUTCOMES = ("completed", "unparsed", "failed")
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
    if event.get("outcome") not in MODEL_OUTCOMES:
        raise ValueError(f"Invalid outcome in {path}: {event.get('outcome')!r}")
    durations = event.get("attempt_durations_ns")
    if durations is not None:
        if not isinstance(durations, list) or not all(is_integer(value) for value in durations):
            raise ValueError(f"Invalid attempt_durations_ns in {path}: {durations!r}")
        if len(durations) != len(attempt_ids(event)):
            raise ValueError(f"attempt_durations_ns does not cover every attempt in {path}: {durations!r}")


def normalize_model_event(event: dict) -> dict:
    """Derive `outcome` for logs written before the field existed.

    A pre-`outcome` error event that still carries usage came back from a completed, billed request
    that the agent could not parse; one without usage never produced a response at all.
    """
    if "outcome" in event:
        return event
    if event.get("status") == "success":
        return event | {"outcome": "completed"}
    return event | {"outcome": "unparsed" if event.get("prompt_tokens") is not None else "failed"}


def attempt_ids(event: dict) -> list[str]:
    """Every actual API attempt behind one logical model call, not only the one that answered."""
    return list(event.get("request_ids") or [event["request_id"]])


def failed_attempt_seconds(event: dict) -> float | None:
    """Seconds spent in attempts that did not produce the response, or None when the split is unknown.

    Zero for a single-attempt call. Retry backoff is not included: it is the remainder of
    `duration_ns` and cannot be separated from client post-processing.
    """
    durations = event.get("attempt_durations_ns")
    return sum(durations[:-1]) / 1e9 if isinstance(durations, list) and durations else None


def validate_vllm_event(event: dict, path: Path) -> None:
    if not isinstance(event.get("request_id"), str) or not event["request_id"]:
        raise ValueError(f"Invalid request_id in {path}: {event.get('request_id')!r}")
    for field in (*VLLM_DURATION_FIELDS, "prompt_tokens", "completion_tokens"):
        if not is_integer(event.get(field)):
            raise ValueError(f"Invalid {field} in {path}: {event.get(field)!r}")
    if abs(event["inference_duration_ns"] - event["prefill_duration_ns"] - event["decode_duration_ns"]) > 1:
        raise ValueError(f"vLLM inference duration does not equal prefill plus decode in {path}")
    finish_reason = event.get("engine_finish_reason", event.get("finish_reason"))
    if not isinstance(finish_reason, str) or not finish_reason:
        raise ValueError(f"Invalid engine_finish_reason in {path}: {finish_reason!r}")


def normalize_vllm_event(event: dict) -> dict:
    """Normalize legacy logs without implying equivalence to the client finish reason."""
    if "engine_finish_reason" in event:
        return event
    return event | {"engine_finish_reason": event["finish_reason"]}


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

    run_root = results.parent if results.name == "results" else root
    if vllm_log is None and (run_root / "request_events.jsonl").is_file():
        vllm_log = run_root / "request_events.jsonl"
    statuses = evaluation_statuses(evaluation)
    raw_vllm_events = read_jsonl(vllm_log) if vllm_log else []
    for event in raw_vllm_events:
        validate_vllm_event(event, vllm_log)
    vllm_events = [normalize_vllm_event(event) for event in raw_vllm_events]
    vllm_by_id: dict[str, list[dict]] = {}
    for event in vllm_events:
        vllm_by_id.setdefault(event["request_id"], []).append(event)

    # "-" is the separator used before request IDs became unambiguously parseable; logs predating
    # that still have to be readable, so both spellings select this run's requests.
    run_prefixes = (f"mswea~{run_root.name}~", f"mswea-{run_root.name}-")
    run_vllm_events = [event for event in vllm_events if event["request_id"].startswith(run_prefixes)]

    rows = []
    request_rows = []
    all_model_events = []
    all_matched_vllm_events = []
    all_request_ids = []
    model_statuses: Counter = Counter()
    model_outcomes: Counter = Counter()
    vllm_unmatched_outcomes: Counter = Counter()
    model_errors: Counter = Counter()
    model_finishes: Counter = Counter()
    vllm_engine_finishes: Counter = Counter()
    evaluation_counts: Counter = Counter()
    count_mismatches = 0
    token_mismatches = 0
    duplicate_vllm_matches = 0

    for iid in sorted(ids):
        folder = results / iid
        trajectory_path = folder / f"{iid}.traj.json"
        model_path = folder / f"{iid}.traj.model_events.jsonl"
        trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.exists() else {}
        events = [normalize_model_event(event) for event in read_jsonl(model_path)] if model_path.exists() else []
        for event in events:
            validate_model_event(event, iid, model_path)

        info = trajectory.get("info", {})
        logical_calls = info.get("model_stats", {}).get("api_calls")
        if logical_calls is not None and not is_integer(logical_calls):
            raise ValueError(f"Invalid trajectory model-call count in {trajectory_path}: {logical_calls!r}")
        recorded_responses = trajectory_response_count(trajectory)
        count_matches = model_path.exists() and logical_calls is not None and len(events) == logical_calls
        count_mismatches += not count_matches

        ids_for_instance = [request_id for event in events for request_id in attempt_ids(event)]
        duplicate_model_ids = len(ids_for_instance) - len(set(ids_for_instance))
        unmatched_attempts = sum(1 for request_id in ids_for_instance if request_id not in vllm_by_id)
        instance_failed_attempts = [seconds for event in events if (seconds := failed_attempt_seconds(event))]
        matched = 0
        unmatched = 0
        instance_token_mismatches = 0
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
                vllm_unmatched_outcomes[event["outcome"]] += 1
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
                vllm_engine_finishes[match["engine_finish_reason"]] += 1

            request_rows.append(
                {
                    "instance_id": iid,
                    "step_id": event["step_id"],
                    "request_id": event["request_id"],
                    "status": event["status"],
                    "outcome": event["outcome"],
                    "error_type": event.get("error_type"),
                    "model_duration_ns": event["duration_ns"],
                    "attempts": len(attempt_ids(event)),
                    "failed_attempt_seconds": failed_attempt_seconds(event),
                    "prompt_tokens": event.get("prompt_tokens"),
                    "completion_tokens": event.get("completion_tokens"),
                    "total_tokens": event.get("total_tokens"),
                    "model_finish_reason": event.get("finish_reason"),
                    "vllm_matched": match is not None,
                    "queue_duration_ns": match.get("queue_duration_ns") if match else None,
                    "prefill_duration_ns": match.get("prefill_duration_ns") if match else None,
                    "decode_duration_ns": match.get("decode_duration_ns") if match else None,
                    "inference_duration_ns": match.get("inference_duration_ns") if match else None,
                    "vllm_engine_finish_reason": match.get("engine_finish_reason") if match else None,
                    "tokens_agree": tokens_agree if match else None,
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
                "model_outcomes": json.dumps(dict(sorted(Counter(e["outcome"] for e in events).items()))),
                "model_attempts": len(ids_for_instance),
                "model_events_with_retries": sum(len(attempt_ids(event)) > 1 for event in events),
                "failed_attempt_seconds": sum(instance_failed_attempts) if instance_failed_attempts else None,
                "model_error_types": json.dumps(dict(sorted(instance_errors.items()))),
                "model_finish_reasons": json.dumps(dict(sorted(instance_finishes.items()))),
                "duplicate_model_request_ids": duplicate_model_ids,
                "complete_token_events": len(complete_token_events),
                "missing_token_events": len(events) - len(complete_token_events),
                "vllm_matched_requests": matched if vllm_log else None,
                "vllm_unmatched_requests": unmatched if vllm_log else None,
                "vllm_unmatched_attempts": unmatched_attempts if vllm_log else None,
                "vllm_duplicate_matches": instance_duplicate_vllm if vllm_log else None,
                "vllm_token_mismatches": instance_token_mismatches if vllm_log else None,
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
        model_outcomes.update(event["outcome"] for event in events)
        model_errors.update(event.get("error_type", "unknown") for event in events if event["status"] == "error")
        model_finishes.update(str(event.get("finish_reason")) for event in events)

    complete_events = [
        event for event in all_model_events if all(event.get(field) is not None for field in MODEL_TOKEN_FIELDS)
    ]
    all_failed_attempts = [seconds for event in all_model_events if (seconds := failed_attempt_seconds(event))]
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
        "model_attempts": len(all_request_ids),
        "model_requests_with_retries": sum(len(attempt_ids(event)) > 1 for event in all_model_events),
        "model_statuses": dict(sorted(model_statuses.items())),
        "model_outcomes": dict(sorted(model_outcomes.items())),
        "model_error_types": dict(sorted(model_errors.items())),
        "model_finish_reasons": dict(sorted(model_finishes.items())),
        "duplicate_model_request_ids": len(all_request_ids) - len(set(all_request_ids)),
        "model_requests_with_complete_tokens": len(complete_events),
        "model_requests_missing_tokens": len(all_model_events) - len(complete_events),
        "vllm_requests_for_run": len(run_vllm_events) if vllm_log else None,
        "vllm_matched_requests": len(matched_ids) if vllm_log else None,
        "vllm_unmatched_model_requests": len(all_model_events) - len(matched_ids) if vllm_log else None,
        "vllm_unmatched_attempts": len(set(all_request_ids) - set(vllm_by_id)) if vllm_log else None,
        "vllm_unmatched_by_outcome": dict(sorted(vllm_unmatched_outcomes.items())) if vllm_log else None,
        "vllm_orphan_requests": len(run_vllm_ids - set(all_request_ids)) if vllm_log else None,
        "vllm_duplicate_matches": duplicate_vllm_matches if vllm_log else None,
        "vllm_token_mismatches": token_mismatches if vllm_log else None,
        "vllm_engine_finish_reasons": dict(sorted(vllm_engine_finishes.items())) if vllm_log else None,
        **distribution([event["duration_ns"] / 1e9 for event in all_model_events], "model_request_seconds"),
        **distribution(all_failed_attempts, "failed_attempt_seconds"),
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
    vllm_log: Path | None = typer.Option(
        None, help="vLLM JSONL; defaults to request_events.jsonl in the experiment directory."
    ),
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
