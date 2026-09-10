"""Tests for the per-action tool execution events written by the agents."""

import concurrent.futures
import json
import time
from pathlib import Path

import pytest
import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, DeterministicToolcallModel, make_output

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


class RaisingEnvironment(LocalEnvironment):
    """Environment whose execute always raises, to exercise exceptions escaping `env.execute`."""

    def __init__(self, exception: Exception, **kwargs):
        super().__init__(**kwargs)
        self.exception = exception

    def execute(self, action: dict, cwd: str = "", **kwargs) -> dict:
        raise self.exception


class CountingEnvironment(LocalEnvironment):
    """Environment that records every command it was asked to execute."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.commands: list[str] = []

    def execute(self, action: dict, cwd: str = "", **kwargs) -> dict:
        self.commands.append(action["command"])
        return super().execute(action, cwd, **kwargs)


def make_flaky_time_ns(succeed_first: int):
    """A `time.time_ns` replacement that starts raising after `succeed_first` successful calls."""
    calls = []

    def flaky_time_ns() -> int:
        calls.append(1)
        if len(calls) > succeed_first:
            raise OSError("clock failure")
        return 1_000_000

    return flaky_time_ns


@pytest.fixture
def agent_config():
    return yaml.safe_load(Path("src/minisweagent/config/default.yaml").read_text())["agent"]


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def make_model(*action_batches: list[dict]) -> DeterministicModel:
    return DeterministicModel(outputs=[make_output("thought", actions) for actions in action_batches])


def test_events_written_for_each_action(tmp_path, agent_config):
    """One event per action, with step/action ids, exact commands, exit codes and output sizes."""
    agent = DefaultAgent(
        make_model(
            [{"command": "echo 'a  b' && exit 3"}, {"command": "echo second"}],
            [{"command": f"{SUBMIT} && echo done"}],
        ),
        LocalEnvironment(),
        **agent_config | {"tool_log_path": tmp_path / "events.jsonl"},
    )
    assert agent.run("task")["exit_status"] == "Submitted"

    events = read_events(tmp_path / "events.jsonl")
    assert [(e["event_type"], e["step_id"], e["action_index"]) for e in events] == [
        ("tool_execution", 1, 0),
        ("tool_execution", 1, 1),
        ("tool_execution", 2, 0),
    ]
    assert [e["raw_action"] for e in events] == ["echo 'a  b' && exit 3", "echo second", f"{SUBMIT} && echo done"]
    assert [e.get("return_code") for e in events] == [3, 0, None]
    assert [e.get("output_size_bytes") for e in events] == [len("a  b\n"), len("second\n"), None]
    assert all("tool_call_id" not in e and "instance_id" not in e for e in events)
    assert all(e["duration_ns"] > 0 for e in events)
    assert all(isinstance(e["start_time_ns"], int) and isinstance(e["end_time_ns"], int) for e in events)


def test_duration_covers_execution(tmp_path, agent_config):
    """The measured duration spans the whole env.execute call (generous lower bound)."""
    agent = DefaultAgent(
        make_model([{"command": "sleep 0.5"}], [{"command": SUBMIT}]),
        LocalEnvironment(),
        **agent_config | {"tool_log_path": tmp_path / "events.jsonl"},
    )
    agent.run("task")

    assert read_events(tmp_path / "events.jsonl")[0]["duration_ns"] > 100_000_000


def test_returned_timeout_is_recorded_as_returned_error(tmp_path, agent_config):
    """A timeout is returned by the environment, not raised, and must stay distinguishable."""
    agent = DefaultAgent(
        make_model([{"command": "sleep 30"}], [{"command": SUBMIT}]),
        LocalEnvironment(timeout=1),
        **agent_config | {"tool_log_path": tmp_path / "events.jsonl"},
    )
    assert agent.run("task")["exit_status"] == "Submitted"

    timed_out = read_events(tmp_path / "events.jsonl")[0]
    assert timed_out["return_code"] == -1
    assert timed_out["returned_exception_type"] == "TimeoutExpired"
    assert "timed out" in timed_out["returned_exception_info"]
    assert "exception_type" not in timed_out  # nothing escaped env.execute


def test_escaping_exception_is_recorded_and_reraised(tmp_path, agent_config):
    """An exception escaping the environment is recorded without inventing a return code."""
    agent = DefaultAgent(
        make_model([{"command": "echo boom"}]),
        RaisingEnvironment(RuntimeError("env exploded")),
        **agent_config | {"tool_log_path": tmp_path / "events.jsonl"},
    )
    with pytest.raises(RuntimeError, match="env exploded"):
        agent.run("task")

    event = read_events(tmp_path / "events.jsonl")[0]
    assert event["exception_type"] == "RuntimeError"
    assert event["exception_message"] == "env exploded"
    assert "return_code" not in event
    assert "output_size_bytes" not in event
    assert event["duration_ns"] > 0


def test_submitted_is_recorded_and_still_ends_the_run(tmp_path, agent_config):
    """Submitted escapes env.execute, so it is recorded as an exception and still exits cleanly."""
    agent = DefaultAgent(
        make_model([{"command": f"{SUBMIT} && echo the-patch"}]),
        LocalEnvironment(),
        **agent_config | {"tool_log_path": tmp_path / "events.jsonl"},
    )
    assert agent.run("task") == {"exit_status": "Submitted", "submission": "the-patch\n"}

    event = read_events(tmp_path / "events.jsonl")[0]
    assert event["exception_type"] == "Submitted"
    assert "return_code" not in event


def test_actions_after_submit_are_neither_executed_nor_logged(tmp_path, agent_config):
    """Submitting aborts the remaining actions of the same step, so they produce no events."""
    marker = tmp_path / "marker"
    agent = DefaultAgent(
        make_model([{"command": "echo first"}, {"command": SUBMIT}, {"command": f'touch "{marker}"'}]),
        LocalEnvironment(),
        **agent_config | {"tool_log_path": tmp_path / "events.jsonl"},
    )
    agent.run("task")

    assert [e["raw_action"] for e in read_events(tmp_path / "events.jsonl")] == ["echo first", SUBMIT]
    assert not marker.exists()


def test_output_size_counts_utf8_bytes(tmp_path, agent_config):
    """Output size is the byte length of the output, not its character count."""
    agent = DefaultAgent(
        make_model([{"command": "printf 'héllo'"}], [{"command": SUBMIT}]),
        LocalEnvironment(),
        **agent_config | {"tool_log_path": tmp_path / "events.jsonl"},
    )
    agent.run("task")

    assert read_events(tmp_path / "events.jsonl")[0]["output_size_bytes"] == 6  # 5 characters, 6 bytes


def test_instance_id_recorded_when_agent_has_one(tmp_path, agent_config):
    """Benchmark agents carry an instance_id (see ProgressTrackingAgent); it belongs in the events."""
    agent = DefaultAgent(
        make_model([{"command": SUBMIT}]),
        LocalEnvironment(),
        **agent_config | {"tool_log_path": tmp_path / "events.jsonl"},
    )
    agent.instance_id = "django__django-11133"
    agent.run("task")

    assert read_events(tmp_path / "events.jsonl")[0]["instance_id"] == "django__django-11133"


def test_tool_call_id_recorded(tmp_path, agent_config):
    """Toolcall actions carry their id into the event."""
    agent = DefaultAgent(
        DeterministicToolcallModel(
            outputs=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [],
                    "extra": {"actions": [{"command": "echo hi", "tool_call_id": "call_42"}], "cost": 1.0},
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [],
                    "extra": {"actions": [{"command": SUBMIT}], "cost": 1.0},
                },
            ]
        ),
        LocalEnvironment(),
        **agent_config | {"tool_log_path": tmp_path / "events.jsonl"},
    )
    agent.run("task")

    assert read_events(tmp_path / "events.jsonl")[0]["tool_call_id"] == "call_42"


def test_logging_disabled_writes_nothing(tmp_path, agent_config):
    """Without tool_log_path nothing is written, even when a trajectory is saved."""
    agent = DefaultAgent(
        make_model([{"command": SUBMIT}]),
        LocalEnvironment(),
        **agent_config | {"output_path": tmp_path / "run.traj.json"},
    )
    assert agent.run("task")["exit_status"] == "Submitted"
    assert [p.name for p in tmp_path.iterdir()] == ["run.traj.json"]


def test_logging_does_not_change_messages(tmp_path, agent_config):
    """Instrumentation must not alter actions, observations or control flow."""

    def run(tool_log_path) -> list[tuple]:
        agent = DefaultAgent(
            make_model([{"command": "echo hi"}], [{"command": f"{SUBMIT} && echo the-patch"}]),
            LocalEnvironment(),
            **agent_config | {"tool_log_path": tool_log_path},
        )
        agent.run("task")
        return [(m.get("role"), m.get("content")) for m in agent.messages]

    instrumented = run(tmp_path / "events.jsonl")
    assert instrumented == run(None)
    assert len(read_events(tmp_path / "events.jsonl")) == 2


def test_logging_failure_does_not_affect_run(tmp_path, agent_config):
    """An unwritable tool log must not mask a successful execution or a Submitted exception."""
    (blocker := tmp_path / "blocker").write_text("not a directory")
    agent = DefaultAgent(
        make_model([{"command": "echo hello"}], [{"command": f"{SUBMIT} && echo the-patch"}]),
        LocalEnvironment(),
        **agent_config | {"tool_log_path": blocker / "events.jsonl"},
    )
    assert agent.run("task") == {"exit_status": "Submitted", "submission": "the-patch\n"}
    assert "hello" in json.dumps(agent.messages)
    assert blocker.read_text() == "not a directory"


def test_concurrent_agents_write_separate_files(tmp_path, agent_config):
    """Agents running in parallel must not interleave events into one another's files."""

    def run_agent(name: str) -> None:
        agent = DefaultAgent(
            make_model([{"command": f"echo {name}"}], [{"command": SUBMIT}]),
            LocalEnvironment(),
            **agent_config | {"tool_log_path": tmp_path / name / "events.jsonl"},
        )
        agent.instance_id = name
        agent.run("task")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run_agent, ["inst-a", "inst-b"]))

    for name in ["inst-a", "inst-b"]:
        events = read_events(tmp_path / name / "events.jsonl")
        assert [e["instance_id"] for e in events] == [name, name]
        assert [e["raw_action"] for e in events] == [f"echo {name}", SUBMIT]


def test_interactive_agent_logs_actions(tmp_path, agent_config):
    """The InteractiveAgent override goes through the same shared wrapper."""
    agent = InteractiveAgent(
        make_model([{"command": "echo hi"}, {"command": "exit 7"}], [{"command": SUBMIT}]),
        LocalEnvironment(),
        **agent_config | {"tool_log_path": tmp_path / "events.jsonl", "mode": "yolo", "confirm_exit": False},
    )
    agent.run("task")

    events = read_events(tmp_path / "events.jsonl")
    assert [(e["step_id"], e["action_index"], e["raw_action"]) for e in events] == [
        (1, 0, "echo hi"),
        (1, 1, "exit 7"),
        (2, 0, SUBMIT),
    ]
    assert [e.get("return_code") for e in events] == [0, 7, None]


def test_interactive_agent_logging_disabled(tmp_path, agent_config):
    """Disabled logging is honoured by the InteractiveAgent path as well."""
    agent = InteractiveAgent(
        make_model([{"command": "echo hi"}], [{"command": SUBMIT}]),
        LocalEnvironment(),
        **agent_config | {"mode": "yolo", "confirm_exit": False},
    )
    agent.run("task")

    assert list(tmp_path.iterdir()) == []


def test_event_preparation_failure_runs_action_exactly_once(tmp_path, agent_config, monkeypatch):
    """If the event cannot even be built, the action still runs -- once -- and returns its output."""
    monkeypatch.setattr(time, "time_ns", make_flaky_time_ns(0))
    agent = DefaultAgent(
        make_model(), CountingEnvironment(), **agent_config | {"tool_log_path": tmp_path / "events.jsonl"}
    )

    assert agent.execute_action({"command": "echo hi"})["output"] == "hi\n"
    assert agent.env.commands == ["echo hi"]
    assert not (tmp_path / "events.jsonl").exists()


def test_event_preparation_failure_preserves_environment_exception(tmp_path, agent_config, monkeypatch):
    """The uninstrumented fallback must not swallow or retry an environment that raises."""
    monkeypatch.setattr(time, "time_ns", make_flaky_time_ns(0))
    agent = DefaultAgent(
        make_model(),
        RaisingEnvironment(RuntimeError("env exploded")),
        **agent_config | {"tool_log_path": tmp_path / "events.jsonl"},
    )

    with pytest.raises(RuntimeError, match="env exploded"):
        agent.execute_action({"command": "echo hi"})


def test_end_timing_failure_preserves_successful_result(tmp_path, agent_config, monkeypatch):
    """A clock failure while finishing the event must not affect the environment's return value."""
    monkeypatch.setattr(time, "time_ns", make_flaky_time_ns(1))  # the start reading works, the end one fails
    agent = DefaultAgent(
        make_model(), CountingEnvironment(), **agent_config | {"tool_log_path": tmp_path / "events.jsonl"}
    )

    assert agent.execute_action({"command": "echo hi"}) == {"output": "hi\n", "returncode": 0, "exception_info": ""}
    assert agent.env.commands == ["echo hi"]
    assert not (tmp_path / "events.jsonl").exists()


def test_end_timing_failure_preserves_environment_exception(tmp_path, agent_config, monkeypatch):
    """The same failure on the exception path must still re-raise the original exception."""
    monkeypatch.setattr(time, "time_ns", make_flaky_time_ns(1))
    agent = DefaultAgent(
        make_model(),
        RaisingEnvironment(RuntimeError("env exploded")),
        **agent_config | {"tool_log_path": tmp_path / "events.jsonl"},
    )

    with pytest.raises(RuntimeError, match="env exploded"):
        agent.execute_action({"command": "echo hi"})


def test_failing_warning_does_not_break_isolation(tmp_path, agent_config, monkeypatch):
    """Reporting an instrumentation failure must not itself become one."""
    (blocker := tmp_path / "blocker").write_text("not a directory")
    agent = DefaultAgent(
        make_model(), LocalEnvironment(), **agent_config | {"tool_log_path": blocker / "events.jsonl"}
    )

    def boom(*args, **kwargs):
        raise RuntimeError("logging handler is broken")

    monkeypatch.setattr(agent.logger, "warning", boom)

    assert agent.execute_action({"command": "echo hi"})["output"] == "hi\n"
    assert blocker.read_text() == "not a directory"
