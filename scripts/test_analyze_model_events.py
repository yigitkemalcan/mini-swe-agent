"""Synthetic tests; no model, vLLM, benchmark, or evaluator is started."""

import json

import pytest
from analyze_model_events import analyze


def write_jsonl(path, events):
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def model_event(iid, step, request_id, *, status="success", error_type=None):
    complete = status == "success"
    return {
        "event_type": "model_request",
        "instance_id": iid,
        "step_id": step,
        "start_time_ns": step * 2_000_000_000,
        "end_time_ns": step * 2_000_000_000 + step * 1_000_000_000,
        "duration_ns": step * 1_000_000_000,
        "prompt_tokens": 10 if complete else None,
        "completion_tokens": 4 if complete else None,
        "total_tokens": 14 if complete else None,
        "finish_reason": "tool_calls" if complete else None,
        "status": status,
        "error_type": error_type,
        "request_id": request_id,
    }


def vllm_event(request_id):
    return {
        "request_id": request_id,
        "queue_duration_ns": 100,
        "prefill_duration_ns": 200,
        "decode_duration_ns": 300,
        "inference_duration_ns": 500,
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "finish_reason": "stop",
    }


def test_correlates_requests_and_keeps_rejected_request_visible(tmp_path):
    run = tmp_path / "run-1"
    for iid, event in (
        ("a", model_event("a", 1, "mswea-run-1-a-step1-attempt1-ok")),
        (
            "b",
            model_event(
                "b",
                1,
                "mswea-run-1-b-step1-attempt1-rejected",
                status="error",
                error_type="ContextWindowExceededError",
            ),
        ),
    ):
        folder = run / "results" / iid
        folder.mkdir(parents=True)
        response = {"extra": {"response": {"id": event["request_id"]}}} if iid == "a" else {}
        (folder / f"{iid}.traj.json").write_text(
            json.dumps({"info": {"exit_status": "Submitted", "model_stats": {"api_calls": 1}}, "messages": [response]})
        )
        write_jsonl(folder / f"{iid}.traj.model_events.jsonl", [event])
    vllm = tmp_path / "vllm.jsonl"
    write_jsonl(vllm, [vllm_event("mswea-run-1-a-step1-attempt1-ok")])

    experiment, instances, requests = analyze(run, vllm)

    assert experiment["model_requests"] == 2
    assert experiment["model_statuses"] == {"error": 1, "success": 1}
    assert experiment["model_error_types"] == {"ContextWindowExceededError": 1}
    assert experiment["model_requests_with_complete_tokens"] == 1
    assert experiment["model_requests_missing_tokens"] == 1
    assert experiment["vllm_matched_requests"] == 1
    assert experiment["vllm_unmatched_model_requests"] == 1
    assert experiment["vllm_token_mismatches"] == 0
    assert experiment["vllm_finish_reason_mismatches"] == 1
    assert experiment["instances_with_model_call_count_mismatch"] == 0
    assert [row["model_call_count_matches"] for row in instances] == [True, True]
    assert [row["vllm_matched"] for row in requests] == [True, False]


def test_rejects_invalid_model_duration(tmp_path):
    folder = tmp_path / "results" / "a"
    folder.mkdir(parents=True)
    (folder / "a.traj.json").write_text(json.dumps({"info": {"model_stats": {"api_calls": 1}}}))
    event = model_event("a", 1, "request-1") | {"duration_ns": 0}
    write_jsonl(folder / "a.traj.model_events.jsonl", [event])

    with pytest.raises(ValueError, match="Invalid duration_ns"):
        analyze(tmp_path)


def test_reports_token_mismatch(tmp_path):
    run = tmp_path / "run-2"
    folder = run / "results" / "a"
    folder.mkdir(parents=True)
    (folder / "a.traj.json").write_text(json.dumps({"info": {"model_stats": {"api_calls": 1}}}))
    event = model_event("a", 1, "mswea-run-2-a-step1-attempt1-id")
    write_jsonl(folder / "a.traj.model_events.jsonl", [event])
    server = vllm_event(event["request_id"]) | {"completion_tokens": 5}
    write_jsonl(tmp_path / "vllm.jsonl", [server])

    experiment, instances, requests = analyze(run, tmp_path / "vllm.jsonl")

    assert experiment["vllm_token_mismatches"] == 1
    assert instances[0]["vllm_token_mismatches"] == 1
    assert requests[0]["tokens_agree"] is False
