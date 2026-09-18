"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import contextlib
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded, Submitted, TimeExceeded
from minisweagent.models.utils.actions_toolcall_response import finish_reason_from_responses_api
from minisweagent.utils.serialize import recursive_merge


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    wall_time_limit_seconds: int = 0
    """Stop agent after this many seconds of wall-clock time. 0 means no limit."""
    max_consecutive_format_errors: int = 3
    """Exit after this many format errors in a row (0 = no limit)."""
    output_path: Path | None = None
    """Save the trajectory to this path."""
    tool_log_path: Path | None = None
    """Append one JSONL event per agent-triggered tool execution to this path.
    `None` (the default) disables the instrumentation entirely. Writing is append-only: reruns add to
    an existing file rather than replacing it, so reset the file yourself if you need a clean one."""
    model_log_path: Path | None = None
    """Append one JSONL event per model request to this path.
    Durations cover the direct `model.query(...)` call, including model-client and server overhead.
    `attempt_durations_ns` splits that out per actual API attempt, so time lost to failed attempts
    stays separable from the request that answered."""
    run_id: str = ""
    """Identifier shared by all instances in one benchmark run, used to correlate model API attempts."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "elapsed_seconds": int(time.time() - self._start_time),
            },
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        while True:
            try:
                self.step()
                self.n_consecutive_format_errors = 0  # reset on any clean step
            except FormatError as e:
                # The call was billed before parsing failed, so query() never got to charge it.
                self.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                self.n_consecutive_format_errors += 1
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        return self.execute_actions(self.query())

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        if set_request_context := getattr(self.model, "set_request_context", None):
            set_request_context(
                run_id=self.config.run_id,
                instance_id=getattr(self, "instance_id", ""),
                step_id=self.n_calls,
            )
        if not self.config.model_log_path:
            message = self.model.query(self.messages)
        else:
            prepared = None
            try:
                prepared = self._start_model_event()
            except Exception as e:
                self._warn_model_event_failure(e)
            if prepared is None:
                message = self.model.query(self.messages)
            else:
                event, start = prepared
                try:
                    message = self.model.query(self.messages)
                except Exception as e:
                    self._finish_model_event(event, start, exception=e)
                    raise
                self._finish_model_event(event, start, message=message)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def _start_model_event(self) -> tuple[dict, int]:
        """Build a model-request event and take the monotonic start reading immediately before the call."""
        return {
            "event_type": "model_request",
            "instance_id": getattr(self, "instance_id", ""),
            "step_id": self.n_calls,
            "start_time_ns": time.time_ns(),
        }, time.perf_counter_ns()

    def _finish_model_event(
        self, event: dict, start_ns: int, *, message: dict | None = None, exception: Exception | None = None
    ) -> None:
        """Finish and append a model-request event without changing model-call behavior."""
        try:
            event |= {
                "end_time_ns": time.time_ns(),
                "duration_ns": time.perf_counter_ns() - start_ns,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "finish_reason": None,
                "status": "error" if exception is not None else "success",
                "outcome": "completed",
            }
            if exception is not None:
                event |= {"error_type": type(exception).__name__, "error_message": str(exception)}
                if request_ids := getattr(exception, "model_request_ids", None):
                    event |= {"request_id": request_ids[-1], "request_ids": request_ids}
                if attempt_durations := getattr(exception, "model_attempt_durations_ns", None):
                    event["attempt_durations_ns"] = attempt_durations
                message = _message_from_model_exception(exception)
                # A recovered response means the request itself completed and was billed; the agent
                # merely could not parse it. Only a request that returned nothing failed outright.
                event["outcome"] = "unparsed" if message is not None else "failed"
            if message is not None:
                event |= _model_response_fields(message)
            line = json.dumps(event)
            path = self.config.model_log_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(line + "\n")
        except Exception as e:
            self._warn_model_event_failure(e)

    def _warn_model_event_failure(self, error: Exception) -> None:
        """Report a model instrumentation failure without affecting the request."""
        with contextlib.suppress(Exception):
            self.logger.warning(f"Failed to record model request event: {error}")

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        outputs = [
            self.execute_action(action, i) for i, action in enumerate(message.get("extra", {}).get("actions", []))
        ]
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def execute_action(self, action: dict, action_index: int = 0) -> dict[str, Any]:
        """Execute a single agent-triggered action, recording a tool execution event if enabled.

        The instrumentation is best effort: whatever happens inside it, the environment's return
        value -- or the exception it raised, e.g. `Submitted` -- is what reaches the caller.
        """
        if not self.config.tool_log_path:  # disabled: execute without building events or timing
            return self.env.execute(action)
        prepared = None
        try:
            prepared = self._start_tool_event(action, action_index)
        except Exception as e:
            self._warn_tool_event_failure(e)
        if prepared is None:  # instrumentation could not be set up: run the action once, uninstrumented
            return self.env.execute(action)
        event, start = prepared
        try:
            output = self.env.execute(action)
        except Exception as e:
            self._finish_tool_event(event, start, exception=e)
            raise
        self._finish_tool_event(event, start, output=output)
        return output

    def _start_tool_event(self, action: dict, action_index: int) -> tuple[dict, int]:
        """Build the event and take the start readings, the monotonic one as late as possible."""
        event = {
            "event_type": "tool_execution",
            "step_id": self.n_calls,
            "action_index": action_index,
            "raw_action": action.get("command", ""),
            "start_time_ns": time.time_ns(),
        }
        if instance_id := getattr(self, "instance_id", ""):
            event["instance_id"] = instance_id
        if "tool_call_id" in action:
            event["tool_call_id"] = action["tool_call_id"]
        return event, time.perf_counter_ns()

    def _finish_tool_event(
        self, event: dict, start_ns: int, *, output: dict | None = None, exception: Exception | None = None
    ) -> None:
        """Record the end of the execution, then serialize and append the event. Never raises."""
        try:
            event |= {"end_time_ns": time.time_ns(), "duration_ns": time.perf_counter_ns() - start_ns}
            if exception is None:
                event |= _returned_event_fields(output)
            else:
                event |= {"exception_type": type(exception).__name__, "exception_message": str(exception)}
                event |= _escaped_event_fields(exception)
            line = json.dumps(event)  # serialize before touching the filesystem
            path = self.config.tool_log_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(line + "\n")
        except Exception as e:
            self._warn_tool_event_failure(e)

    def _warn_tool_event_failure(self, error: Exception) -> None:
        """Report an instrumentation failure; reporting must not raise either."""
        with contextlib.suppress(Exception):
            self.logger.warning(f"Failed to record tool execution event: {error}")

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data


def _returned_event_fields(output: dict) -> dict[str, Any]:
    """Tool event fields for an execution that returned rather than raised.

    Environments report timeouts and other command failures in their return value (`returncode` -1
    plus `exception_info`), which must stay distinguishable from an exception escaping `env.execute`.
    """
    fields: dict[str, Any] = {"outcome": "returned", "return_code": output.get("returncode")}
    if isinstance(text := output.get("output"), str):
        fields["output_size_bytes"] = len(text.encode("utf-8"))
    if exception_info := output.get("exception_info"):
        fields["returned_exception_info"] = exception_info
    if exception_type := output.get("extra", {}).get("exception_type"):
        fields["returned_exception_type"] = exception_type
    return fields


def _escaped_event_fields(exception: Exception) -> dict[str, Any]:
    """Tool event fields for an exception that escaped `env.execute`.

    `Submitted` is the terminal submission call rather than a failure: the command itself ran, and
    every environment raises it only after a zero return code, so the code is known rather than
    guessed. Its payload is the output without the marker line, so it gets its own field instead of
    standing in for the output size of an observation that was never returned to the model.
    """
    if not isinstance(exception, Submitted):
        return {"outcome": "raised"}
    fields: dict[str, Any] = {"outcome": "submitted", "return_code": 0}
    messages = getattr(exception, "messages", ())
    if messages and isinstance(submission := messages[0].get("extra", {}).get("submission"), str):
        fields["submission_size_bytes"] = len(submission.encode("utf-8"))
    return fields


def _message_from_model_exception(exception: Exception) -> dict | None:
    """Return a persisted response message when a model error provides one (for example FormatError)."""
    messages = getattr(exception, "messages", None)
    return messages[0] if isinstance(messages, (list, tuple)) and messages and isinstance(messages[0], dict) else None


def _model_response_fields(message: dict) -> dict[str, Any]:
    """Extract provider-neutral usage and finish metadata from a model message's persisted response."""
    extra = message.get("extra", {})
    response = extra.get("response", {})
    if not isinstance(response, dict):
        return {}
    usage = response.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    choices = response.get("choices") or []
    finish_reason = choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None
    fields = {
        "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
        "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
        "total_tokens": usage.get("total_tokens"),
        "finish_reason": finish_reason or response.get("finish_reason") or finish_reason_from_responses_api(response),
    }
    if request_ids := extra.get("request_ids"):
        fields |= {"request_id": request_ids[-1], "request_ids": request_ids}
    if attempt_durations := extra.get("attempt_durations_ns"):
        fields["attempt_durations_ns"] = attempt_durations
    return fields
