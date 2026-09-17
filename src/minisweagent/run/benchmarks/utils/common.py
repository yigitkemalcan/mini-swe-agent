"""Shared agent utilities for benchmark runners."""

import contextlib
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import logger


def with_default_tool_log_path(agent_config: dict, default: Path) -> dict:
    """Return a copy of `agent_config` with `tool_log_path` defaulting to this instance's own file.

    An explicit `tool_log_path` wins, including `None` to disable logging. Tool logs are append-only,
    so the per-instance default is reset here to keep reruns of one instance clean; a custom path is
    never deleted and accumulates events across runs (and across instances if it is shared). If the
    stale default cannot be removed, logging is disabled for this instance rather than appending to it.
    """
    config = {"tool_log_path": default} | agent_config
    if config["tool_log_path"] == default:
        try:
            default.unlink(missing_ok=True)
        except Exception as e:
            config["tool_log_path"] = None
            with contextlib.suppress(Exception):
                logger.warning(f"Disabling tool logging for '{default}': stale log could not be cleared: {e}")
    return config


def with_default_model_log_path(agent_config: dict, default: Path) -> dict:
    """Return a copy of `agent_config` with a clean per-instance model event path by default."""
    config = {"model_log_path": default} | agent_config
    if config["model_log_path"] == default:
        try:
            default.unlink(missing_ok=True)
        except Exception as e:
            config["model_log_path"] = None
            with contextlib.suppress(Exception):
                logger.warning(f"Disabling model logging for '{default}': stale log could not be cleared: {e}")
    return config


class ProgressTrackingAgent(DefaultAgent):
    """Agent that reports per-step progress via :class:`RunBatchProgressManager`."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        return super().step()
