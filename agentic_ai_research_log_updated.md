# Agentic AI Characterization — Research Log

> **Current checkpoint:** the measurement stack now covers tool execution, logical model-request latency, per-attempt request timing, token usage, per-attempt vLLM server timing, request-ID correlation, outcome classification on both tool and model events, and run-level CPU/GPU resource sampling. Two 100-instance SWE-Bench Verified characterization runs have been completed and officially evaluated. Two known gaps remain measured but uncorrected: the ~184 ms tool-execution floor and the absence of any server-side record for requests vLLM abandons.
>
> **Current analysis scope:** semantic classification of shell commands into tool categories remains postponed. Analysis should first use directly recorded counts, durations, token counts, identifiers, exit statuses, and benchmark outcomes.
>
> The operational two-terminal procedure has been moved to the separate `agentic_ai_runtime_setup.md` file.

---

## 1. Research Goal

Characterize agentic AI workloads from a systems perspective, focusing on the alternation between LLM serving and agent/tool execution.

| Component | Experimental target |
| --- | --- |
| Benchmark | SWE-Bench Verified |
| Agent | mini-SWE-Agent v2 (`2.4.6`) |
| Model | `Qwen/Qwen3-235B-A22B-Instruct-2507` |
| Inference server | vLLM `0.10.1.1`, V1 engine |
| Hardware | 8 x NVIDIA A100-SXM4-80GB |
| Current benchmark concurrency | `--workers 1` for clean characterization |

Current measurement objectives:

1. Record every agent-triggered tool/environment execution.
2. Measure tool execution latency.
3. Record every logical model call made by the agent.
4. Measure logical model-request latency.
5. Record prompt and completion/output-token counts.
6. Record every actual vLLM request attempt.
7. Measure vLLM queue, prefill, decode, and inference-phase durations.
8. Correlate model calls, retry attempts, tool calls, and benchmark instances.
9. Join systems measurements with official SWE-Bench outcomes.
10. Measure runtime density from a shared run-level CPU/GPU time series before attempting per-tool resource attribution.

---

## 2. Tool Measurement Semantics

### 2.1 Tool invocation

> **One agent-requested `env.execute(action)` call is one tool invocation.**

Compound shell commands are not split.

Example:

```bash
cd /testbed && rg "foo" src/ && pytest tests/
```

This remains one invocation with one measured duration.

The runtime logger records the exact `raw_action`. Semantic categories such as file search, read, edit, test, version control, or submission are not assigned at runtime.

### 2.2 Tool timing boundary

The timer surrounds the agent-triggered:

```text
env.execute(action)
```

The measured duration includes environment/Docker execution overhead experienced by the agent.

It excludes the preceding model request, subsequent JSONL writing, output-size calculation after the execution returns, and benchmark setup/evaluation operations that bypass the agent wrapper.

For concurrent workers, summed tool durations are aggregate execution time, not a non-overlapping share of wall-clock time.

---

## 3. Model and Inference Measurement Semantics

There are now **two distinct model-side timing levels**.

### 3.1 Logical model-request duration

mini-SWE-Agent records timing around:

```text
model.query(...)
```

This is the latency experienced by the agent for one logical model call.

It may include one or more API attempts, retry/backoff delay, LiteLLM/client overhead, vLLM queueing and serving, and response-return overhead.

Therefore this metric should be described as:

```text
model request duration
```

or:

```text
agent-visible model-call latency
```

It is **not** pure GPU time.

### 3.2 Retry behavior

Retries occur below the agent's logical `model.query()` call.

mini-SWE-Agent's LiteLLM model uses a Tenacity retry loop with:

```text
default maximum attempts: 10
exponential waiting: 4 s up to 60 s
```

The current correlated-instrumentation design disables LiteLLM-internal retries so that mini-SWE-Agent's retry loop is the single retry layer assigning one unique ID per actual API attempt.

Each actual attempt receives a request ID containing:

```text
run_id
instance_id
step_id
attempt_id
```

A logical model event stores the final `request_id`, all `request_ids` associated with its attempts, and `attempt_durations_ns` -- one duration per actual API attempt, failed attempts included.

This matters because a retried call is still **one** event with one `duration_ns`. In `qwen-run-20260917T043711Z-eQus52`, four calls out of 4995 (0.08%) were retried and contributed 3038 s: **20.4% of total model time** and 15.4% of the run's wall-clock span. They are invisible at p50-p95 and worth +4.3% at p99, so they distort *totals and means*, not latency percentiles. Report serving time as `model_request_seconds_total - failed_attempt_seconds_total`.

Retry backoff is deliberately not broken out: it is the remainder of `duration_ns` after the attempts and cannot be separated from client post-processing.

### 3.3 vLLM server-side inference duration

vLLM V1 records request-level server timing.

For each actual request attempt, the instrumentation records:

```text
queue_duration_ns
prefill_duration_ns
decode_duration_ns
inference_duration_ns
```

The inference metric is defined as:

```text
inference_duration_ns
    = last_token_time - first_scheduled_time
```

This reproduces vLLM's own request-inference-time definition at per-request granularity.

It includes the request's running interval: prefill, decode, possible gaps between engine iterations, and batching/preemption effects.

It is a genuine server-side inference-phase measurement, but it is **not exclusive CUDA-kernel time**.

Continuous batching means pure per-request GPU-active time is not uniquely attributable when multiple requests share GPU kernels.

### 3.4 Token measurements

The client-side model event records provider-returned usage:

```text
prompt_tokens
completion_tokens
total_tokens
```

`completion_tokens` is the LLM output-token count.

The vLLM event independently records prompt and generated-token counts.

The validated single-instance run showed exact client/server agreement for all correlated requests.

### 3.5 Run-level resource timeline

`run-swebench.sh` starts one lightweight sampler for the lifetime of the
benchmark command. Samples use wall-clock nanoseconds for alignment with model
and tool events and `CLOCK_MONOTONIC` nanoseconds/elapsed time for stable local
ordering. This first version intentionally measures the whole experiment rather
than attributing CPU or GPU activity to individual tools.

CPU utilization comes from deltas of the aggregate `/proc/stat` CPU counters.
It is normalized to 0–100% across the host and treats I/O wait as separate from
active utilization. Steal time is recorded separately and remains part of
active utilization. Host memory comes from `/proc/meminfo`; used memory is
`MemTotal - MemAvailable`, not process RSS. GPU utilization and memory come from
native NVML calls for all eight devices, identified by index and UUID. No
`nvidia-smi` subprocess is launched per sample.

At the default 500 ms interval, CPU percentages describe the interval between
readings. NVML utilization remains the driver's recent utilization sample and
must not be interpreted as exclusive per-request CUDA-kernel time. Host/GPU
memory maxima are observed sample maxima, so shorter spikes can be missed.

---

## 4. System and Storage

### 4.1 Remote machine

| Item | Recorded value |
| --- | --- |
| SSH target | `jovan@10.0.0.4` |
| Hostname | `vmss-a100000002` |
| GPUs | 8 x NVIDIA A100-SXM4-80GB |
| NVIDIA driver | `560.35.05` |
| Driver-reported CUDA | `12.6` |
| CPU | AMD EPYC 7V12, 96 CPUs |
| RAM | about 1.7 TiB |
| Docker root | `/mnt/raid0/docker` |

### 4.2 Project paths

| Purpose | Path |
| --- | --- |
| Project root | `/mnt/raid0/yigit/agent-characterization` |
| mini-SWE-Agent checkout | `/mnt/raid0/yigit/agent-characterization/mini-swe-agent` |
| SWE-Bench runs | `/mnt/raid0/yigit/agent-characterization/swebench-runs` |
| Evaluation logs | `/mnt/raid0/yigit/agent-characterization/logs/run_evaluation` |
| vLLM request log | `<SWE-Bench run>/request_events.jsonl` |
| SWE-Bench evaluator env | `/mnt/raid0/yigit/agent-characterization/envs/swebench-eval` |
| vLLM env | `/mnt/experiment-yigit/qwen-experiment/envs/vllm` |
| Qwen/HF cache | `/mnt/experiment-yigit/qwen-experiment/huggingface-cache` |

Compatibility symlink:

```text
/home/jovan/agent-characterization
    -> /mnt/raid0/yigit/agent-characterization
```

The Qwen snapshot is stored under `/mnt` and is about 438 GiB.

`/mnt` is temporary/ephemeral Azure storage; important experiment artifacts should also be preserved elsewhere when needed.

---

## 5. Software State

### mini-SWE-Agent

```text
version: 2.4.6
branch: characterization/tool-profiling
```

Tool-only instrumentation was previously validated at:

```text
28decad8c36e6d69f61cf6cf28fa914582facfc9
```

The current checkout now contains additional model/vLLM correlation instrumentation and helper scripts beyond that older tool-only commit.

### vLLM stack

| Component | Version |
| --- | --- |
| vLLM | `0.10.1.1` |
| Transformers | `4.55.2` |
| Tokenizers | `0.21.4` |
| PyTorch | `2.7.1+cu126` |
| Hugging Face Hub | `0.36.2` |

The vLLM instrumentation is preserved reproducibly in the research repository rather than existing only as an untracked `site-packages` edit.

Relevant tracked infrastructure includes:

```text
instrumentation/
scripts/install-vllm-instrumentation.sh
start-vllm.sh
run-swebench.sh
evaluate-swebench.sh
```

---

## 6. Instrumentation Artifacts

For each SWE-Bench instance, mini-SWE-Agent now produces:

```text
<instance>.traj.json
<instance>.traj.tool_events.jsonl
<instance>.traj.model_events.jsonl
```

Each benchmark run also produces:

```text
system_metrics.jsonl
system_metrics_summary.json
```

### 6.1 Tool-event log

Important fields include:

```text
event_type
instance_id
step_id
action_index
tool_call_id
raw_action
start_time_ns
end_time_ns
duration_ns
outcome
return_code
output_size_bytes
submission_size_bytes
exception metadata
```

`outcome` is `returned`, `submitted`, or `raised`, so a call is never classified from which fields happen to be present. `submitted` is the terminal submission call: the command itself succeeded, and every environment raises `Submitted` only after a zero return code, so `return_code` is `0` rather than absent. Its payload is recorded as `submission_size_bytes`, not `output_size_bytes`, because the submission is never returned to the model as an observation and must not enter the observation-size distribution.

A filter such as `return_code == 0` previously dropped every submission silently: 89 of 4937 events in `eQus52` and 68 of 4164 in `vyvcar`.

### 6.2 Model-event log

The model-event log records one logical mini-SWE-Agent model call, including fields such as:

```text
event_type
instance_id
step_id
start_time_ns
end_time_ns
duration_ns
status
outcome
finish_reason
prompt_tokens
completion_tokens
total_tokens
request_id
request_ids
attempt_durations_ns
```

The event is preserved even when the response produces no executable tool call.

`outcome` separates what `status` conflates:

| `outcome` | Meaning | Inference happened | vLLM record expected |
| --- | --- | --- | --- |
| `completed` | Response parsed into actions | Yes | Yes |
| `unparsed` | Response complete and billed, parsing failed | Yes | Yes |
| `failed` | No response came back | No | No |

A `FormatError` is a fully successful, billed request the agent could not parse; a `ContextWindowExceededError` is a request the server refused before any inference. Both were previously `status: error`. In `vyvcar` the 167 error events are 149 `unparsed` plus 18 `failed`, and all 18 unmatched vLLM records are exactly the `failed` ones, so no correlation gap is unexplained.

The classification is recovered on read for logs written before the field existed, so both 100-instance runs analyze under this schema without being re-run.

### 6.3 vLLM request-event log

Per-run path:

```text
/mnt/raid0/yigit/agent-characterization/swebench-runs/<RUN-FOLDER>/request_events.jsonl
```

Each actual API attempt records:

```text
request_id
queue_duration_ns
prefill_duration_ns
decode_duration_ns
inference_duration_ns
prompt_tokens
completion_tokens
engine_finish_reason
```

Request IDs correlate the vLLM records with mini-SWE-Agent run, instance,
logical step, and retry attempt. A background writer appends queued events so
filesystem I/O does not block vLLM's finished-request path.

Request IDs are formed as:

```text
mswea~<run>~<instance>~step<N>~attempt<K>~<uuid>
```

The `~` separator is stripped from every part by mini-SWE-Agent's sanitizer, so the run is whatever precedes the second separator whatever the run is named, and the server-side writer depends on no run-directory naming convention. Records written before this change use `-` and remain readable; the analyzer accepts either separator.

The hook cannot raise into vLLM: every failure inside it is logged and swallowed, and an event whose phases are not ordered is dropped rather than written, so the log never contains a record the analyzer would reject.

**Aborted requests are not recorded.** The hook sits on the finished-request path only, so a request vLLM gives up on leaves no server-side trace. In `eQus52` that is 5 attempts of 5000 by count but **3021 s, 15.3% of the run**, by time: vLLM explains only 17.7 s of the 3038 s spent inside retried calls. During that interval, with `--workers 1` and nothing else in flight, whether the GPU was busy or idle is unknown. `attempt_durations_ns` bounds the interval from the client side; closing it would require hooking vLLM's abort path.

### 6.4 System-metrics log

`system_metrics.jsonl` contains the run-level time series. Important fields are:

```text
event_type
run_id
timestamp_ns
timestamp_utc
monotonic_ns
elapsed_ns
cpu_util_percent
cpu_user_percent
cpu_system_percent
cpu_iowait_percent
cpu_steal_percent
host_memory_used_bytes
host_memory_available_bytes
host_memory_free_bytes
host_memory_total_bytes
gpus[]
```

Each GPU record contains its NVML index, UUID, utilization, memory used, and
total memory. `system_metrics_summary.json` records sample/run bounds and the
requested aggregate means and sampled maxima.

---

## 7. Validation History

### 7.1 Tool instrumentation

The earlier tool logger passed:

```text
26 focused tests
594 full-suite tests passed
51 skipped
1 pre-existing warning
```

It was also validated with a deterministic Docker smoke test and a real Qwen integration test.

### 7.2 New model/vLLM instrumentation

Focused validation after adding model-request and vLLM instrumentation:

```text
45 focused tests passed
Python linting passed
shell syntax checks passed
Git whitespace checks passed
```

### 7.3 Real single-instance correlation validation

Validated instance:

```text
django__django-11099
```

The latest instrumentation validation produced:

```text
model events: 10
request IDs correlated with vLLM: 10/10
model-event count matched trajectory calls: 10 = 10
token counts matched client/server: 10/10
final status: Submitted
```

Model steps:

```text
1-10
```

Tool steps:

```text
1-5, 7-10
```

Step 6 produced a `FormatError`, so there was a model event and vLLM event but no tool event. This confirmed that model calls and tool calls remain separate quantities.

All model `total_tokens` values equaled:

```text
prompt_tokens + completion_tokens
```

A finish-reason semantic difference was observed:

```text
mini-SWE-Agent/OpenAI serving layer: tool_calls
vLLM engine layer: stop
```

for tool-producing responses. This is expected because tool parsing occurs
above the engine's finish-reason layer. The analyzer reports the fields
separately and does not count this semantic difference as a mismatch.

---

## 8. 100-Instance Characterization Run

Run directory:

```text
/mnt/raid0/yigit/agent-characterization/swebench-runs/qwen-run-20260915T235912Z-vyvcar
```

Selection:

```text
first 100 SWE-Bench Verified test instances
workers = 1
```

Agent-run wall time:

```text
3:43:16
```

### 8.1 Agent exit statuses

Across 100 attempted instances:

| Agent exit status | Count | Percent |
| --- | ---: | ---: |
| Submitted | 68 | 68% |
| ContextWindowExceededError | 18 | 18% |
| RepeatedFormatError | 14 | 14% |
| **Total** | **100** | **100%** |

These exit statuses are part of the workload behavior and should not be discarded from characterization analysis.

### 8.2 Official SWE-Bench evaluation

Evaluation run ID:

```text
qwen-run-20260915T235912Z-vyvcar-evaluation-20260916T040257Z-6rOqDD
```

Official result:

| Evaluation outcome | Count |
| --- | ---: |
| Predictions submitted | 100 |
| Empty patches | 33 |
| Non-empty patches sent to evaluation | 67 |
| Completed normally | 62 |
| Resolved | 29 |
| Unresolved | 33 |
| Evaluation errors | 5 |
| Likely infrastructure failures | 0 |
| Ambiguous failures | 0 |
| Unstopped containers | 0 |

Consistency:

```text
29 resolved
+ 33 unresolved
+ 5 errors
= 67 non-empty patches

67 non-empty
+ 33 empty
= 100 submitted predictions
```

The report also lists:

```text
Total SWE-Bench Verified instances: 500
Incomplete: 400
```

because only 100 predictions were supplied from the 500-instance Verified split.

### 8.3 End-to-end resolved rate

Across all 100 attempted tasks:

```text
29 / 100 = 29%
```

This is the most direct end-to-end resolved rate for this batch because empty patches and agent failures are real outcomes of the agent.

For reference:

```text
resolved / non-empty patches = 29 / 67 ≈ 43.3%
resolved / normally completed evaluations = 29 / 62 ≈ 46.8%
```

These use different denominators and should not be substituted for the 29% attempted-task rate without explicitly stating the denominator.

### 8.4 Evaluation errors

Five predictions produced evaluation errors rather than a normal resolved/unresolved outcome.

At least one evaluator log explicitly referenced an instance such as:

```text
django__django-12708
```

These error cases remain to be inspected separately. They should not automatically be treated as unresolved or infrastructure failures.

---

## 9. Current Measurement Status

| Quantity | Status |
| --- | --- |
| Raw agent actions | Available |
| Tool invocation count | Available |
| Individual tool duration | Available |
| Total tool time | Available |
| Tool return/error metadata | Available |
| Tool output-size bytes | Available |
| Model-call count | Available |
| Logical model-request duration | Available |
| Prompt-token count | Available |
| Completion/output-token count | Available |
| Total-token count | Available |
| Per-attempt request ID | Available |
| Retry-attempt correlation | Available |
| Per-attempt request duration | Available (runs after 2026-09-18) |
| Time lost to failed attempts | Available (runs after 2026-09-18) |
| Tool outcome classification | Available |
| Model outcome classification | Available (recovered on read for earlier runs) |
| Submission payload size | Available (runs after 2026-09-18) |
| Server-side record of aborted requests | Not available |
| vLLM queue duration | Available |
| vLLM prefill duration | Available |
| vLLM decode duration | Available |
| vLLM inference-phase duration | Available |
| Run-level host CPU utilization time series | Available |
| Run-level host memory time series | Available |
| Per-GPU NVML utilization/memory time series | Available |
| Model/tool step alignment | Validated |
| Client/server token agreement | Validated |
| Official SWE-Bench outcome | Available for the 100-instance run |
| Semantic tool categories | Postponed |
| Frequency/time by semantic tool category | Not yet computed |
| Exclusive GPU-kernel time per request | Not available / not uniquely attributable |

---

## 10. Analysis Rules and Caveats

1. **Do not compute LLM time as run wall time minus summed tool time.**
2. `model_events.jsonl` duration is the logical agent-visible model-call latency. It can contain retries and backoff.
3. vLLM `inference_duration_ns` is the server-side scheduled-to-last-token interval. It is not exclusive CUDA-kernel time.
4. A model call can produce zero, one, or multiple tool calls.
5. A `FormatError` can therefore produce a model/vLLM event with no corresponding tool event.
6. `completion_tokens` is the LLM output-token count.
7. Each run has its own vLLM request log; correlate records with model events using request IDs.
8. With `--workers 1`, the characterization is easier to interpret because agent requests do not overlap across multiple benchmark workers.
9. Empty patches, context-window failures, repeated-format failures, and evaluation errors are meaningful outcomes and should remain represented in the dataset.
10. Semantic shell-command classification remains a separate offline analysis step and should not alter the raw measurement logs.
11. Classify tool calls by `outcome`, not by the presence of `return_code`. A `return_code == 0` filter drops every `submitted` event.
12. Classify model calls by `outcome`, not by `status`. `completed` and `unparsed` are real, billed inferences and belong in latency statistics; `failed` is not and does not. Only `failed` events are expected to have no vLLM record.
13. Do not report `model_request_seconds_total` or `_mean` as serving time without subtracting `failed_attempt_seconds_total`. Percentiles up to p99 are unaffected by retries; totals and means are not.
14. Tool-duration statistics contain a constant harness floor of about 184 ms per call (`docker exec` plus a `bash -lc` login shell), which is 94% of p50, 55% of p90, 19% of p99, and 54% of total tool time. It has not been calibrated out. Do not compare tool times across runs or configurations without accounting for it.
15. The run wall-clock span is not fully accounted for. For `eQus52`: 14630 s vLLM-accounted server time, 1667 s tool execution, 304 s client/frontend overhead, 3021 s unaccounted inside retried calls, 75 s residual agent overhead, of a 19697 s span.

---

## 11. Analysis Dataset Available Now

The current 100-instance run can be consolidated using:

```text
benchmark instance
official outcome
agent exit status

logical model-call duration
queue duration
prefill duration
decode duration
inference duration

prompt tokens
completion/output tokens
total tokens

tool invocation count
individual tool duration
total tool duration
raw tool action

request IDs
step IDs
retry attempt IDs
```

This is the current foundation for analyzing CPU/GPU alternation, token-length behavior, model-serving latency, tool activity, and task outcome across the 100-instance run.
