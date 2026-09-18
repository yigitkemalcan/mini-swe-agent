# vLLM request measurement

The project installs the two `vllm-0.10.1.1-request-metrics*.patch` hook
patches and `vllm_request_metrics.py` into the existing vLLM environment
before starting the server. The installer is version-gated and idempotent.

Each completed V1 request appends one JSON object to
`MSWEA_VLLM_RUNS_ROOT/<run-id>/request_events.jsonl`. The `request_id` is
supplied by mini-SWE-Agent and encodes the run, instance, step, and API attempt
as `mswea~<run>~<instance>~step<N>~attempt<K>~<uuid>`. The parts are separated
by `~` because mini-SWE-Agent's sanitizer strips that character from every part,
so the run is whatever precedes the second separator no matter how runs or
instances are named. Nothing here depends on a run-directory naming convention.
Token counts and all durations come from vLLM's own finished-request state.
Events are placed on a bounded in-process queue and written by one background
thread, so filesystem I/O does not block vLLM's finished-request path. The
queue is flushed during orderly interpreter shutdown; a full queue is reported
and drops the new event instead of blocking inference.

Nothing this instrumentation does may reach vLLM. Every failure inside the hook
is logged and swallowed: losing a measurement is acceptable, breaking the
serving path it measures is not. An event whose phases are not ordered -- a
request that finished without generating a token leaves `first_token_ts` unset,
which would otherwise be written as a negative prefill duration -- is reported
and dropped rather than written, so the log never contains a record the
analyzer would reject.

Requests that vLLM abandons are **not** recorded. The hook sits on the
finished-request path only, so a client disconnect or timeout leaves no
server-side trace. The client-side `attempt_durations_ns` in the model event
bounds that interval; see the research log for the size of the gap.

LiteLLM-internal retries are disabled at the model-call boundary. Retries are
performed by mini-SWE-Agent's existing retry loop instead, which assigns a new
`X-Request-Id` to every actual API attempt and records all attempt IDs in the
corresponding mini-SWE-Agent model event.

`inference_duration_ns` has exactly vLLM's request-inference definition:

```text
last_token_time - first_scheduled_time
```

It is server-side inference-phase duration, including prefill, decode,
continuous-batching gaps, and preemption. It is not exclusive CUDA/GPU-kernel
time and must not be described as such.

`engine_finish_reason` is vLLM's low-level request termination reason. It is
not equivalent to the OpenAI-compatible response's `finish_reason` after tool
parsing and must not be compared with it directly.
