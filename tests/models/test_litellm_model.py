from unittest.mock import MagicMock, patch

import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL


class TestLitellmModelConfig:
    def test_default_format_error_template(self):
        assert LitellmModelConfig(model_name="test").format_error_template == "{{ error }}"


def _mock_litellm_response(tool_calls):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = tool_calls
    mock_response.choices[0].message.model_dump.return_value = {"role": "assistant", "content": None}
    mock_response.model_dump.return_value = {}
    return mock_response


class TestLitellmModel:
    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_includes_bash_tool(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        model.set_request_context(run_id="run-7", instance_id="django__django-11133", step_id=4)
        result = model.query([{"role": "user", "content": "test"}])

        mock_completion.assert_called_once()
        assert mock_completion.call_args.kwargs["tools"] == [BASH_TOOL]
        request_id = mock_completion.call_args.kwargs["extra_headers"]["X-Request-Id"]
        assert request_id.startswith("mswea~run-7~django__django-11133~step4~attempt1~")
        assert mock_completion.call_args.kwargs["num_retries"] == 0
        assert result["extra"]["request_ids"] == [request_id]

    def test_attempt_ids_are_unique_and_keep_correlation_hierarchy(self):
        model = LitellmModel(model_name="gpt-4")
        model.set_request_context(run_id="run/unsafe", instance_id="inst:a", step_id=9)

        first = model._make_request_id(1)
        second = model._make_request_id(2)
        assert first.startswith("mswea~run_unsafe~inst_a~step9~attempt1~")
        assert second.startswith("mswea~run_unsafe~inst_a~step9~attempt2~")
        assert first != second
        # The separator is stripped from every part, so the run ID is always the second field.
        assert first.split("~")[1] == "run_unsafe"

    @patch("minisweagent.models.litellm_model.litellm.completion", side_effect=RuntimeError("API down"))
    def test_failed_api_attempt_exposes_its_request_id(self, mock_completion, monkeypatch):
        monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "1")
        model = LitellmModel(model_name="gpt-4")
        model.set_request_context(run_id="run-7", instance_id="instance-2", step_id=3)

        with pytest.raises(RuntimeError, match="API down") as exc:
            model.query([{"role": "user", "content": "test"}])

        request_id = mock_completion.call_args.kwargs["extra_headers"]["X-Request-Id"]
        assert exc.value.model_request_ids == [request_id]
        assert len(exc.value.model_attempt_durations_ns) == 1
        assert exc.value.model_attempt_durations_ns[0] > 0

    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost", return_value=0.001)
    @patch("minisweagent.models.litellm_model.litellm.completion")
    def test_each_attempt_is_timed_separately_across_retries(self, mock_completion, mock_cost, monkeypatch):
        """A retried call reports one duration per actual API attempt, failed ones included."""
        monkeypatch.setattr("minisweagent.models.utils.retry.wait_exponential", lambda **_: lambda *_a: 0)
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.side_effect = [RuntimeError("API down"), _mock_litellm_response([tool_call])]

        model = LitellmModel(model_name="gpt-4")
        model.set_request_context(run_id="run-7", instance_id="instance-2", step_id=3)
        result = model.query([{"role": "user", "content": "test"}])

        durations = result["extra"]["attempt_durations_ns"]
        assert len(durations) == len(result["extra"]["request_ids"]) == 2
        assert all(duration > 0 for duration in durations)

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_valid_tool_call(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "ls -la"}'
        tool_call.id = "call_abc"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        result = model.query([{"role": "user", "content": "list files"}])
        assert result["extra"]["actions"] == [{"command": "ls -la", "tool_call_id": "call_abc"}]

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_no_tool_calls_raises(self, mock_cost, mock_completion):
        mock_completion.return_value = _mock_litellm_response(None)
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        with pytest.raises(FormatError):
            model.query([{"role": "user", "content": "test"}])

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_finish_reason_threaded_into_format_error_template(self, mock_cost, mock_completion):
        """The response finish_reason is exposed to format_error_template via template_kwargs, so a
        config can report a max_tokens truncation instead of the misleading "no tool call" error."""
        response = _mock_litellm_response(None)
        response.choices[0].finish_reason = "length"
        mock_completion.return_value = response
        mock_cost.return_value = 0.001

        model = LitellmModel(
            model_name="gpt-4",
            format_error_template="{% if finish_reason == 'length' %}cut off{% else %}{{ error }}{% endif %}",
        )
        with pytest.raises(FormatError) as exc:
            model.query([{"role": "user", "content": "test"}])
        assert exc.value.messages[0]["content"] == "cut off"

    def test_format_observation_messages(self):
        model = LitellmModel(model_name="gpt-4", observation_template="{{ output.output }}")
        message = {"extra": {"actions": [{"command": "echo test", "tool_call_id": "call_1"}]}}
        outputs = [{"output": "test output", "returncode": 0}]
        result = model.format_observation_messages(message, outputs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "test output"

    def test_format_observation_messages_no_actions(self):
        model = LitellmModel(model_name="gpt-4")
        result = model.format_observation_messages({"extra": {}}, [])
        assert result == []
