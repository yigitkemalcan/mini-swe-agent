"""Tests for the shared benchmark runner utilities."""

from pathlib import Path

from minisweagent.run.benchmarks.utils.common import logger, with_default_tool_log_path


def _failing_unlink(self, missing_ok=False):
    raise PermissionError("read-only file system")


def test_default_path_is_per_instance_and_reset(tmp_path):
    """Each instance gets its own file, and a rerun of that instance starts from a clean one."""
    default = tmp_path / "inst-a" / "inst-a.traj.tool_events.jsonl"
    default.parent.mkdir()
    default.write_text('{"event_type": "tool_execution"}\n')

    assert with_default_tool_log_path({"cost_limit": 3.0}, default) == {"cost_limit": 3.0, "tool_log_path": default}
    assert not default.exists()


def test_custom_path_wins_and_is_never_deleted(tmp_path):
    """A custom path is appended to across runs, so it must survive; the unused default is left alone."""
    (default := tmp_path / "default.jsonl").write_text("stale default\n")
    (custom := tmp_path / "custom.jsonl").write_text("earlier run\n")

    assert with_default_tool_log_path({"tool_log_path": custom}, default)["tool_log_path"] == custom
    assert custom.read_text() == "earlier run\n"
    assert default.read_text() == "stale default\n"


def test_explicit_none_disables_logging_without_deleting(tmp_path):
    """An explicit None disables the instrumentation and must not remove existing events."""
    (default := tmp_path / "default.jsonl").write_text("previous events\n")

    assert with_default_tool_log_path({"tool_log_path": None}, default)["tool_log_path"] is None
    assert default.read_text() == "previous events\n"


def test_cleanup_failure_disables_logging_and_keeps_file(tmp_path, monkeypatch):
    """If the stale default cannot be cleared, this instance runs with logging off instead of appending."""
    (default := tmp_path / "default.jsonl").write_text("previous events\n")
    config = {"cost_limit": 3.0}
    monkeypatch.setattr(Path, "unlink", _failing_unlink)

    assert with_default_tool_log_path(config, default) == {"cost_limit": 3.0, "tool_log_path": None}
    assert default.read_text() == "previous events\n"
    assert config == {"cost_limit": 3.0}


def test_cleanup_failure_survives_a_broken_logger(tmp_path, monkeypatch):
    """Reporting the cleanup failure must not turn it into a crashed benchmark instance."""
    (default := tmp_path / "default.jsonl").write_text("previous events\n")

    def boom(*args, **kwargs):
        raise RuntimeError("logging handler is broken")

    monkeypatch.setattr(Path, "unlink", _failing_unlink)
    monkeypatch.setattr(logger, "warning", boom)

    assert with_default_tool_log_path({}, default)["tool_log_path"] is None
    assert default.read_text() == "previous events\n"


def test_shared_agent_config_is_not_mutated(tmp_path):
    """The agent config is shared across instances and worker threads, so it must not change in place."""
    config = {"cost_limit": 3.0}

    assert with_default_tool_log_path(config, tmp_path / "a.jsonl")["tool_log_path"] == tmp_path / "a.jsonl"
    assert with_default_tool_log_path(config, tmp_path / "b.jsonl")["tool_log_path"] == tmp_path / "b.jsonl"
    assert config == {"cost_limit": 3.0}
