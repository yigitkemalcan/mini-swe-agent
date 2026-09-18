"""Tests for per-request model events written by agents."""

import json
import time
from pathlib import Path

import pytest
import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import FormatError
from minisweagent.models.test_models import DeterministicModel, make_output

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


@pytest.fixture
def agent_config():
    return yaml.safe_load(Path("src/minisweagent/config/default.yaml").read_text())["agent"]


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def response_output(actions: list[dict], *, prompt: int, completion: int, finish_reason: str = "stop") -> dict:
    output = make_output("thought", actions)
    output["extra"]["response"] = {
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
        "choices": [{"finish_reason": finish_reason}],
    }
    return output


class RaisingModel(DeterministicModel):
    def query(self, messages, **kwargs):
        raise RuntimeError("model unavailable")


class SlowModel(DeterministicModel):
    def query(self, messages, **kwargs):
        time.sleep(0.15)
        return super().query(messages, **kwargs)


class FormatErrorModel(DeterministicModel):
    def query(self, messages, **kwargs):
        raise FormatError(
            {
                "role": "user",
                "content": "invalid response",
                "extra": {
                    "response": {
                        "usage": {"input_tokens": 7, "output_tokens": 5, "total_tokens": 12},
                        "status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                    }
                },
            }
        )


class CorrelatedModel(DeterministicModel):
    def set_request_context(self, **context):
        self.context = context

    def query(self, messages, **kwargs):
        output = super().query(messages, **kwargs)
        output["extra"]["request_ids"] = [
            f"{self.context['run_id']}:{self.context['instance_id']}:{self.context['step_id']}:attempt"
        ]
        return output


class RetryingModel(DeterministicModel):
    """A model whose first two API attempts fail, as the retry loop inside query() would report them."""

    def query(self, messages, **kwargs):
        output = super().query(messages, **kwargs)
        output["extra"] |= {
            "request_ids": ["id-attempt1", "id-attempt2", "id-attempt3"],
            "attempt_durations_ns": [700_000_000, 300_000_000, 5_000_000],
        }
        return output


def test_logs_every_request_including_response_without_tool_call(tmp_path, agent_config):
    agent = DefaultAgent(
        DeterministicModel(
            outputs=[
                response_output([], prompt=10, completion=4),
                response_output([{"command": SUBMIT}], prompt=14, completion=3, finish_reason="tool_calls"),
            ]
        ),
        LocalEnvironment(),
        **agent_config | {"model_log_path": tmp_path / "events.jsonl"},
    )
    agent.instance_id = "django__django-11133"
    assert agent.run("task")["exit_status"] == "Submitted"

    events = read_events(tmp_path / "events.jsonl")
    assert [(event["instance_id"], event["step_id"], event["status"]) for event in events] == [
        ("django__django-11133", 1, "success"),
        ("django__django-11133", 2, "success"),
    ]
    assert [(event["prompt_tokens"], event["completion_tokens"], event["total_tokens"]) for event in events] == [
        (10, 4, 14),
        (14, 3, 17),
    ]
    assert [event["finish_reason"] for event in events] == ["stop", "tool_calls"]
    assert all(event["event_type"] == "model_request" and event["duration_ns"] > 0 for event in events)
    assert all(isinstance(event["start_time_ns"], int) and isinstance(event["end_time_ns"], int) for event in events)


def test_duration_covers_direct_model_request(tmp_path, agent_config):
    agent = DefaultAgent(
        SlowModel(outputs=[response_output([{"command": SUBMIT}], prompt=1, completion=1)]),
        LocalEnvironment(),
        **agent_config | {"model_log_path": tmp_path / "events.jsonl"},
    )
    agent.run("task")

    assert read_events(tmp_path / "events.jsonl")[0]["duration_ns"] >= 100_000_000


def test_model_event_carries_same_attempt_id_and_agent_context(tmp_path, agent_config):
    agent = DefaultAgent(
        CorrelatedModel(outputs=[response_output([{"command": SUBMIT}], prompt=2, completion=3)]),
        LocalEnvironment(),
        **agent_config | {"run_id": "run-42", "model_log_path": tmp_path / "events.jsonl"},
    )
    agent.instance_id = "instance-7"
    agent.run("task")

    event = read_events(tmp_path / "events.jsonl")[0]
    assert event["request_id"] == "run-42:instance-7:1:attempt"
    assert event["request_ids"] == [event["request_id"]]


def test_model_exception_is_logged_and_reraised(tmp_path, agent_config):
    agent = DefaultAgent(
        RaisingModel(outputs=[]),
        LocalEnvironment(),
        **agent_config | {"model_log_path": tmp_path / "events.jsonl"},
    )
    agent.instance_id = "broken-instance"
    with pytest.raises(RuntimeError, match="model unavailable"):
        agent.run("task")

    event = read_events(tmp_path / "events.jsonl")[0]
    assert event | {"start_time_ns": 0, "end_time_ns": 0, "duration_ns": 0} == {
        "event_type": "model_request",
        "instance_id": "broken-instance",
        "step_id": 1,
        "start_time_ns": 0,
        "end_time_ns": 0,
        "duration_ns": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "finish_reason": None,
        "status": "error",
        "outcome": "failed",
        "error_type": "RuntimeError",
        "error_message": "model unavailable",
    }
    assert event["duration_ns"] > 0


def test_format_error_preserves_response_usage_in_error_event(tmp_path, agent_config):
    agent = DefaultAgent(
        FormatErrorModel(outputs=[]),
        LocalEnvironment(),
        **agent_config | {"model_log_path": tmp_path / "events.jsonl"},
    )
    with pytest.raises(FormatError):
        agent.query()

    event = read_events(tmp_path / "events.jsonl")[0]
    assert (event["prompt_tokens"], event["completion_tokens"], event["total_tokens"]) == (7, 5, 12)
    assert event["finish_reason"] == "length"
    assert event["status"] == "error"
    assert event["outcome"] == "unparsed"  # the request completed and was billed; only parsing failed


def test_logging_disabled_does_not_create_model_events(tmp_path, agent_config):
    agent = DefaultAgent(
        DeterministicModel(outputs=[response_output([{"command": SUBMIT}], prompt=1, completion=1)]),
        LocalEnvironment(),
        **agent_config | {"output_path": tmp_path / "run.traj.json"},
    )
    agent.run("task")

    assert [path.name for path in tmp_path.iterdir()] == ["run.traj.json"]


def test_successful_event_is_classified_and_records_one_attempt(tmp_path, agent_config):
    """A parsed response is `completed`, and its single attempt is timed like any other."""
    agent = DefaultAgent(
        DeterministicModel(outputs=[response_output([{"command": SUBMIT}], prompt=3, completion=2)]),
        LocalEnvironment(),
        **agent_config | {"model_log_path": tmp_path / "events.jsonl"},
    )
    agent.run("task")

    event = read_events(tmp_path / "events.jsonl")[0]
    assert (event["outcome"], event["status"]) == ("completed", "success")
    assert "error_type" not in event


def test_retried_call_keeps_every_attempt_duration(tmp_path, agent_config):
    """One logical call spanning three attempts stays one event, but the failed attempts stay separable."""
    agent = DefaultAgent(
        RetryingModel(outputs=[response_output([{"command": SUBMIT}], prompt=3, completion=2)]),
        LocalEnvironment(),
        **agent_config | {"model_log_path": tmp_path / "events.jsonl"},
    )
    agent.run("task")

    events = read_events(tmp_path / "events.jsonl")
    assert len(events) == 1
    assert events[0]["attempt_durations_ns"] == [700_000_000, 300_000_000, 5_000_000]
    assert len(events[0]["request_ids"]) == len(events[0]["attempt_durations_ns"])
    assert events[0]["request_id"] == "id-attempt3"
    assert events[0]["outcome"] == "completed"
