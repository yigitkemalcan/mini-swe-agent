import json
from types import SimpleNamespace

import pytest

from instrumentation.vllm_request_metrics import (
    flush_finished_request_metrics,
    is_consistent,
    log_finished_request_metrics,
    make_finished_request_event,
)


def stats(**overrides):
    return SimpleNamespace(
        **{
            "queued_ts": 10.0,
            "scheduled_ts": 10.25,
            "first_token_ts": 10.75,
            "last_token_ts": 12.0,
            "num_generation_tokens": 17,
        }
        | overrides
    )


def test_finished_request_event_uses_vllm_intervals_and_normalizes_chat_id():
    assert make_finished_request_event("chatcmpl-run-instance-step-attempt", "stop", 41, stats()) == {
        "request_id": "run-instance-step-attempt",
        "queue_duration_ns": 250_000_000,
        "prefill_duration_ns": 500_000_000,
        "decode_duration_ns": 1_250_000_000,
        "inference_duration_ns": 1_750_000_000,
        "prompt_tokens": 41,
        "completion_tokens": 17,
        "engine_finish_reason": "stop",
    }


def test_logger_routes_each_run_and_ignores_unrecognized_ids(tmp_path, monkeypatch, caplog):
    """Routing follows the request ID's own structure, so a run may be named anything at all."""
    first = tmp_path / "qwen-run-20260917T100000Z-abc123" / "request_events.jsonl"
    second = tmp_path / "some-other.naming_scheme-42" / "request_events.jsonl"
    monkeypatch.delenv("MSWEA_VLLM_RUNS_ROOT", raising=False)
    log_finished_request_metrics("chatcmpl-mswea~qwen-run-20260917T100000Z-abc123~i~step1", "stop", 1, stats())
    assert not first.exists()

    monkeypatch.setenv("MSWEA_VLLM_RUNS_ROOT", str(tmp_path))
    log_finished_request_metrics(
        "chatcmpl-mswea~qwen-run-20260917T100000Z-abc123~django__django-1~step1~attempt1~id", "length", 23, stats()
    )
    log_finished_request_metrics(
        "chatcmpl-mswea~some-other.naming_scheme-42~instance~step1~attempt1~id", "stop", 29, stats()
    )
    log_finished_request_metrics("chatcmpl-unrecognized", "stop", 31, stats())
    flush_finished_request_metrics()

    assert [json.loads(line)["prompt_tokens"] for line in first.read_text().splitlines()] == [23]
    assert [json.loads(line)["prompt_tokens"] for line in second.read_text().splitlines()] == [29]
    assert "Cannot determine run directory for vLLM request unrecognized" in caplog.text


def test_request_without_a_generated_token_is_dropped_not_written(tmp_path, monkeypatch, caplog):
    """An unset first_token_ts would mean a negative prefill, which must never reach the log."""
    monkeypatch.setenv("MSWEA_VLLM_RUNS_ROOT", str(tmp_path))
    assert not is_consistent(make_finished_request_event("r", "abort", 5, stats(first_token_ts=0.0)))

    log_finished_request_metrics(
        "chatcmpl-mswea~run-x~instance~step1~attempt1~id", "abort", 5, stats(first_token_ts=0.0, last_token_ts=0.0)
    )
    flush_finished_request_metrics()

    assert not (tmp_path / "run-x").exists()
    assert "unordered phases" in caplog.text


@pytest.mark.parametrize(
    "broken",
    [SimpleNamespace(), SimpleNamespace(queued_ts=None, scheduled_ts=1.0), "not-a-stats-object"],
)
def test_broken_stats_never_reach_vllm(tmp_path, monkeypatch, caplog, broken):
    """Whatever vLLM hands over, the hook reports the problem instead of raising into the engine."""
    monkeypatch.setenv("MSWEA_VLLM_RUNS_ROOT", str(tmp_path))

    log_finished_request_metrics("chatcmpl-mswea~run-x~instance~step1~attempt1~id", "stop", 5, broken)

    assert "Failed to record vLLM request metrics" in caplog.text


def test_agent_request_ids_route_to_their_run_directory(tmp_path, monkeypatch):
    """The ID minted by the agent and the one parsed by the server live in different processes.

    Nothing else checks that the two halves still agree on the format, and disagreement is silent:
    the server would simply drop every event of the run.
    """
    from minisweagent.models.litellm_model import LitellmModel

    monkeypatch.setenv("MSWEA_VLLM_RUNS_ROOT", str(tmp_path))
    model = LitellmModel(model_name="gpt-4")
    model.set_request_context(run_id="qwen-run-20260918T010203Z-Ab9zQ1", instance_id="django__django-11099", step_id=7)

    log_finished_request_metrics(f"chatcmpl-{model._make_request_id(1)}", "stop", 11, stats())
    flush_finished_request_metrics()

    written = tmp_path / "qwen-run-20260918T010203Z-Ab9zQ1" / "request_events.jsonl"
    assert json.loads(written.read_text())["prompt_tokens"] == 11
