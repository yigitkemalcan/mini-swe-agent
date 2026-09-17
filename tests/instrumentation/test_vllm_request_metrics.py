import json
from types import SimpleNamespace

from instrumentation.vllm_request_metrics import log_finished_request_metrics, make_finished_request_event


def stats():
    return SimpleNamespace(
        queued_ts=10.0,
        scheduled_ts=10.25,
        first_token_ts=10.75,
        last_token_ts=12.0,
        num_generation_tokens=17,
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
        "finish_reason": "stop",
    }


def test_logger_appends_jsonl_only_when_configured(tmp_path, monkeypatch):
    destination = tmp_path / "vllm" / "events.jsonl"
    monkeypatch.delenv("MSWEA_VLLM_EVENT_LOG", raising=False)
    log_finished_request_metrics("chatcmpl-not-written", "stop", 1, stats())
    assert not destination.exists()

    monkeypatch.setenv("MSWEA_VLLM_EVENT_LOG", str(destination))
    log_finished_request_metrics("chatcmpl-request-1", "length", 23, stats())
    log_finished_request_metrics("chatcmpl-request-2", "stop", 29, stats())

    events = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [event["request_id"] for event in events] == ["request-1", "request-2"]
    assert [event["prompt_tokens"] for event in events] == [23, 29]
