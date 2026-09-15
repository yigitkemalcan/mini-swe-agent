"""Synthetic artifact tests; no model requests or environment execution."""

import json

from analyze_experiment import analyze, distribution


def test_pooled_metrics_include_format_errors_and_evaluation(tmp_path):
    for iid, seconds, tokens in (("a", [1], [10]), ("b", [3, 5, 7], [20, 30])):
        folder = tmp_path / "results" / iid
        folder.mkdir(parents=True)
        messages = [
            {"role": "assistant" if index == 0 else "user", "extra": {
                "response": {"id": f"{iid}-{index}", "usage": {"completion_tokens": value}},
                "interrupt_type": "FormatError" if index else "",
            }}
            for index, value in enumerate(tokens)
        ]
        (folder / f"{iid}.traj.json").write_text(json.dumps({
            "info": {"model_stats": {"api_calls": len(tokens)}, "exit_status": "Submitted"},
            "messages": messages + messages[:1],
        }))
        (folder / f"{iid}.traj.tool_events.jsonl").write_text("".join(
            json.dumps({"event_type": "tool_execution", "instance_id": iid, "duration_ns": value * 10**9}) + "\n"
            for value in seconds
        ))
    report = tmp_path / "official.json"
    report.write_text(json.dumps({"resolved_ids": ["a", "outside-experiment"], "unresolved_ids": ["b"]}))
    experiment, rows = analyze(tmp_path, report)
    assert (experiment["instances"], experiment["solved"], experiment["unsolved"]) == (2, 1, 1)
    assert (experiment["tool_calls"], experiment["tool_seconds_total"], experiment["tool_seconds_mean"]) == (4, 16, 4)
    assert (experiment["output_tokens_total"], experiment["output_tokens_mean"], experiment["output_tokens_p99"]) == (60, 20, 30)
    assert rows[1]["output_tokens_mean"] == 25
    assert all(row["token_usage_complete"] for row in rows)


def test_missing_artifacts_are_not_success_or_measured_zero(tmp_path):
    (tmp_path / "preds.json").write_text(json.dumps({"missing": {}}))
    experiment, rows = analyze(tmp_path)
    assert experiment["evaluation_unknown"] == experiment["missing_tool_logs"] == experiment["missing_trajectories"] == 1
    assert experiment["solved"] == experiment["unsolved"] == 0
    assert rows[0]["tool_calls"] is None
    assert rows[0]["output_tokens_total"] is None
    assert not rows[0]["token_usage_complete"]


def test_nearest_rank_and_empty_samples():
    assert distribution(list(range(1, 21)), "x")["x_p95"] == 19
    assert distribution(list(range(1, 21)), "x")["x_p99"] == 20
    assert distribution([], "x")["x_mean"] is None
    assert distribution([0], "x")["x_total"] == 0
