# Experiment analysis

## Run-level system sampling

`run-swebench.sh` automatically starts `scripts/sample_system_metrics.py` after
creating the run directory and stops it when the benchmark command exits,
including error and signal exits. The default interval is 0.5 seconds; override
it for a run with:

```bash
bash run-swebench.sh --slice 0:5 --system-metrics-interval 1.0
```

The sampler writes `system_metrics.jsonl` and
`system_metrics_summary.json` at the top of the run directory. It reads
aggregate CPU counters from `/proc/stat`, host memory from `/proc/meminfo`, and
all eight GPUs through the native NVML library. It does not invoke
`nvidia-smi`, vLLM, Docker, or a GPU workload.

Each JSONL sample has wall-clock `timestamp_ns`/`timestamp_utc` fields for
alignment with model and tool event timestamps, plus `monotonic_ns`,
`elapsed_ns`, `boot_id`, `clock`, and the enclosing directory's `run_id` for stable ordering and correlation. GPU
records use NVML indices and include UUIDs so identities remain explicit.

`cpu_util_percent` is aggregate active CPU time normalized to 0–100% across the
whole host: `(total - idle - iowait) / total`. `cpu_user_percent` includes
user and nice time; `cpu_system_percent` includes system, IRQ, and soft-IRQ
time; I/O wait is separate. CPU steal time is also recorded, so active
utilization can exceed user plus system time on a virtualized host. Memory used
is `MemTotal - MemAvailable`, not agent-process RSS. CPU percentages cover the interval since the preceding
sample. NVML GPU utilization is the driver's recent utilization sample, not
exclusive CUDA-kernel time. Memory peaks are sampled peaks and can miss spikes
shorter than the configured interval.

The summary reports a CPU-interval-weighted mean and maximum, sampled peak host
memory, and sampled mean/maximum utilization and peak memory for every GPU.
The run wrapper verifies access to exactly eight GPUs before launching the
benchmark and treats unexpected sampler termination as a run error.

## Tool and model-event analysis

Run from the repository root with the existing mini-SWE-Agent Python environment (requires `typer`). No model, Docker, or evaluator is launched by this script.

```bash
python scripts/analyze_experiment.py /path/to/experiment
python scripts/analyze_experiment.py /path/to/experiment --evaluation /path/to/official-summary.json
python scripts/analyze_experiment.py /path/to/experiment/results --output /path/to/analysis
```

Supply **one experiment at a time**, not the parent of several experiments. The script discovers all instances from `preds.json`, trajectories, and tool logs. Instances with no artifacts at all cannot be discovered. Output consists of `experiment.json` and `instances.csv` (one row per instance).
Every tool event carries an `outcome` of `returned`, `submitted`, or `raised`,
so a call never has to be classified from which fields happen to be present.
`submitted` is the terminal submission: the command itself succeeded, and the
environments raise only after a zero return code, so `return_code` is `0` and
the payload size is reported as `submission_size_bytes` rather than as
`output_size_bytes` -- the submission is never returned to the model as an
observation, so it must not enter the observation-size distribution. Repeating analysis overwrites these report files, not source artifacts.

## Metrics

Both levels report tool-call count and tool duration total, mean, p50, p90, p95, p99, and maximum in seconds. Compound commands remain one call. Experiment averages and percentiles pool individual calls, rather than averaging per-instance statistics. Summed durations are aggregate tool time, not elapsed runtime.

Output tokens are **model-generated tokens per response**, not bytes printed by shell commands. Both levels report total tokens, mean, p50, p90, p95, p99, and maximum per response, with sample counts. Per-instance statistics use that instance's responses; experiment statistics pool responses across all instances. Format-error responses count too. Responses are read only from `messages[*].extra.response`, deduplicated by response ID when available, avoiding duplicate nested representations. The script uses recorded `completion_tokens` (or `output_tokens`); it does not estimate tokens from text.

Percentiles use nearest rank: sorted value at `ceil(p * n)`, with one-based indexing. High percentiles from small samples are not reliable estimates of a broader population's tail. Totals and percentiles are computed before rounding.

## Evaluation and missing data

Pass the official report **for this experiment** using `--evaluation`. Summary fields `resolved_ids` and `unresolved_ids` supply solved/unsolved counts. A per-instance `report.json` keyed by instance ID with a boolean `resolved` is also supported. Error, empty-patch, and incomplete IDs are kept separate when they are not already in the resolved/unresolved lists. Unlisted instances are unknown, not automatically unsolved. `Submitted` is never treated as proof of correctness. Only IDs present in this experiment's artifacts contribute to the report.

Missing trajectories and tool logs are reported. Missing token usage is excluded from token statistics and flagged by comparing response/sample counts with the trajectory model-call count. All aggregates describe **available recorded measurements**, not guaranteed complete workload totals. Missing metrics are null in JSON or blank in CSV; sample count zero indicates no observations. Malformed JSON or invalid duration/token values raise an error instead of being silently skipped. Valid but missing events cannot always be detected from these artifacts alone.

No command parsing, semantic categories, inferred LLM time, overhead subtraction, or total runtime is included.

## Model-event and vLLM analysis

Use the dedicated model analyzer for the per-instance `*.traj.model_events.jsonl`
files. The analyzer automatically uses `request_events.jsonl` from the supplied
experiment directory when present. Supplying another vLLM JSONL explicitly
overrides it; supplying the official evaluation report adds outcome labels.

```bash
python scripts/analyze_model_events.py /path/to/experiment
python scripts/analyze_model_events.py /path/to/experiment \
  --vllm-log /path/to/legacy/request_events.jsonl \
  --evaluation /path/to/official-summary.json
```

It writes `experiment.json`, `instances.csv`, and `requests.csv` under
`model-analysis/` by default. Metrics include model/API request duration;
prompt, completion, and total tokens; request statuses and errors; finish
reasons; and vLLM queue, prefill, decode, and inference-phase durations.
Correlation reports missing, duplicate, and orphan request IDs plus token-count
mismatches.

Every model event carries an `outcome`, which separates what `status` conflates:

| `outcome` | meaning | inference happened | vLLM record expected |
| --- | --- | --- | --- |
| `completed` | response parsed into actions | yes | yes |
| `unparsed` | response complete and billed, parsing failed | yes | yes |
| `failed` | no response came back | no | no |

`completed` and `unparsed` are the events that belong in latency statistics and
that must correlate one-to-one with vLLM. A `failed` event without a vLLM record
is expected, not an anomaly, and `vllm_unmatched_by_outcome` says which of the
two a gap is. Logs written before the field existed are classified on read from
`status` and recovered usage, so earlier runs analyze under the same schema.

Correlation covers **every** API attempt, not only the one that answered:
`model_attempts` counts them and `vllm_unmatched_attempts` reports attempts with
no server-side record. An abandoned attempt leaves no vLLM event at all, so this
is how that gap becomes visible.

`failed_attempt_seconds` is the time spent in attempts that did not produce the
response -- exactly zero for a single-attempt call, and `None` for logs written
before `attempt_durations_ns` existed, where the split is genuinely unknown
rather than zero. Retry backoff is deliberately excluded: it is the remainder of
`duration_ns` and cannot be separated from client post-processing. Subtract
`failed_attempt_seconds_total` from `model_request_seconds_total` to state
serving time without retry waste. The model response's `finish_reason` and vLLM's
`engine_finish_reason` are reported separately because tool parsing happens
above the engine layer and makes the values semantically different.

`vllm_inference_seconds_*` uses vLLM's exact finished-request definition,
`last_token_time - first_scheduled_time`. It is server-side inference-phase
duration, including batching gaps and preemption, not exclusive CUDA/GPU-kernel
time. The analyzer does not estimate inference time from client request duration
and does not subtract tool time or server time from total runtime.
