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
    # A pre-outcome log is classified from recovered usage: the rejected request returned none.
    assert experiment["model_outcomes"] == {"completed": 1, "failed": 1}
    assert experiment["vllm_unmatched_by_outcome"] == {"failed": 1}
    assert experiment["model_error_types"] == {"ContextWindowExceededError": 1}
    assert experiment["model_requests_with_complete_tokens"] == 1
    assert experiment["model_requests_missing_tokens"] == 1
    assert experiment["vllm_matched_requests"] == 1
    assert experiment["vllm_unmatched_model_requests"] == 1
    assert experiment["vllm_token_mismatches"] == 0
    assert experiment["vllm_engine_finish_reasons"] == {"stop": 1}
    assert experiment["instances_with_model_call_count_mismatch"] == 0
    assert [row["model_call_count_matches"] for row in instances] == [True, True]
    assert [row["vllm_matched"] for row in requests] == [True, False]
    assert requests[0]["vllm_engine_finish_reason"] == "stop"
    assert "finish_reasons_agree" not in requests[0]


def test_discovers_run_local_vllm_log(tmp_path):
    run = tmp_path / "run-1"
    folder = run / "results" / "a"
    folder.mkdir(parents=True)
    event = model_event("a", 1, "mswea-run-1-a-step1-attempt1-id")
    (folder / "a.traj.json").write_text(json.dumps({"info": {"model_stats": {"api_calls": 1}}}))
    write_jsonl(folder / "a.traj.model_events.jsonl", [event])
    write_jsonl(run / "request_events.jsonl", [vllm_event(event["request_id"])])

    experiment, _, requests = analyze(run)

    assert experiment["vllm_log"] == str((run / "request_events.jsonl").resolve())
    assert experiment["vllm_matched_requests"] == 1
    assert requests[0]["vllm_matched"] is True


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


def test_every_attempt_is_correlated_not_only_the_one_that_answered(tmp_path):
    """A retried call keeps its lost attempts visible and its wasted time separable from the answer."""
    run = tmp_path / "run-1"
    folder = run / "results" / "a"
    folder.mkdir(parents=True)
    event = model_event("a", 1, "mswea-run-1-a-step1-attempt2-ok") | {
        "request_ids": ["mswea-run-1-a-step1-attempt1-lost", "mswea-run-1-a-step1-attempt2-ok"],
        "attempt_durations_ns": [600_000_000, 400_000_000],
    }
    (folder / "a.traj.json").write_text(
        json.dumps(
            {
                "info": {"exit_status": "Submitted", "model_stats": {"api_calls": 1}},
                "messages": [{"extra": {"response": {"id": "r"}}}],
            }
        )
    )
    write_jsonl(folder / "a.traj.model_events.jsonl", [event])
    vllm = tmp_path / "vllm.jsonl"
    write_jsonl(vllm, [vllm_event("mswea-run-1-a-step1-attempt2-ok")])

    experiment, _, requests = analyze(run, vllm)

    assert (experiment["model_requests"], experiment["model_attempts"]) == (1, 2)
    assert experiment["model_requests_with_retries"] == 1
    assert experiment["vllm_unmatched_attempts"] == 1  # the lost attempt never finished on the server
    assert experiment["vllm_unmatched_model_requests"] == 0  # the attempt that answered did match
    assert experiment["vllm_orphan_requests"] == 0
    assert experiment["failed_attempt_seconds_total"] == 0.6
    assert (requests[0]["attempts"], requests[0]["failed_attempt_seconds"]) == (2, 0.6)


def test_rejects_attempt_durations_that_do_not_cover_every_attempt(tmp_path):
    """A duration list out of step with the attempt list is a logging bug, not something to average over."""
    run = tmp_path / "run-1"
    folder = run / "results" / "a"
    folder.mkdir(parents=True)
    event = model_event("a", 1, "mswea-run-1-a-step1-attempt2-ok") | {
        "request_ids": ["mswea-run-1-a-step1-attempt1-lost", "mswea-run-1-a-step1-attempt2-ok"],
        "attempt_durations_ns": [600_000_000],
    }
    write_jsonl(folder / "a.traj.model_events.jsonl", [event])

    with pytest.raises(ValueError, match="does not cover every attempt"):
        analyze(run)


def test_selects_run_requests_under_either_request_id_separator(tmp_path):
    """Runs recorded before request IDs became unambiguous must stay analyzable alongside new ones."""
    for separator in ("~", "-"):
        run = tmp_path / f"run{separator.encode().hex()}"
        folder = run / "results" / "a"
        folder.mkdir(parents=True)
        request_id = f"mswea{separator}{run.name}{separator}a{separator}step1{separator}attempt1{separator}id"
        write_jsonl(folder / "a.traj.model_events.jsonl", [model_event("a", 1, request_id)])
        write_jsonl(run / "request_events.jsonl", [vllm_event(request_id)])

        experiment, _, _ = analyze(run)

        assert experiment["vllm_requests_for_run"] == 1, separator
        assert experiment["vllm_orphan_requests"] == 0, separator
