# Experiment analysis

Run from the repository root with the existing mini-SWE-Agent Python environment (requires `typer`). No model, Docker, or evaluator is launched by this script.

```bash
python scripts/analyze_experiment.py /path/to/experiment
python scripts/analyze_experiment.py /path/to/experiment --evaluation /path/to/official-summary.json
python scripts/analyze_experiment.py /path/to/experiment/results --output /path/to/analysis
```

Supply **one experiment at a time**, not the parent of several experiments. The script discovers all instances from `preds.json`, trajectories, and tool logs. Instances with no artifacts at all cannot be discovered. Output consists of `experiment.json` and `instances.csv` (one row per instance). Repeating analysis overwrites these report files, not source artifacts.

## Metrics

Both levels report tool-call count and tool duration total, mean, p50, p90, p95, p99, and maximum in seconds. Compound commands remain one call. Experiment averages and percentiles pool individual calls, rather than averaging per-instance statistics. Summed durations are aggregate tool time, not elapsed runtime.

Output tokens are **model-generated tokens per response**, not bytes printed by shell commands. Both levels report total tokens, mean, p50, p90, p95, p99, and maximum per response, with sample counts. Per-instance statistics use that instance's responses; experiment statistics pool responses across all instances. Format-error responses count too. Responses are read only from `messages[*].extra.response`, deduplicated by response ID when available, avoiding duplicate nested representations. The script uses recorded `completion_tokens` (or `output_tokens`); it does not estimate tokens from text.

Percentiles use nearest rank: sorted value at `ceil(p * n)`, with one-based indexing. High percentiles from small samples are not reliable estimates of a broader population's tail. Totals and percentiles are computed before rounding.

## Evaluation and missing data

Pass the official report **for this experiment** using `--evaluation`. Summary fields `resolved_ids` and `unresolved_ids` supply solved/unsolved counts. A per-instance `report.json` keyed by instance ID with a boolean `resolved` is also supported. Error, empty-patch, and incomplete IDs are kept separate when they are not already in the resolved/unresolved lists. Unlisted instances are unknown, not automatically unsolved. `Submitted` is never treated as proof of correctness. Only IDs present in this experiment's artifacts contribute to the report.

Missing trajectories and tool logs are reported. Missing token usage is excluded from token statistics and flagged by comparing response/sample counts with the trajectory model-call count. All aggregates describe **available recorded measurements**, not guaranteed complete workload totals. Missing metrics are null in JSON or blank in CSV; sample count zero indicates no observations. Malformed JSON or invalid duration/token values raise an error instead of being silently skipped. Valid but missing events cannot always be detected from these artifacts alone.

No command parsing, semantic categories, inferred LLM time, overhead subtraction, or total runtime is included.
