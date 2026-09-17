# vLLM request measurement

The project installs the two `vllm-0.10.1.1-request-metrics*.patch` hook
patches and `vllm_request_metrics.py` into the existing vLLM environment
before starting the server. The installer is version-gated and idempotent.

Each completed V1 request appends one JSON object to `MSWEA_VLLM_EVENT_LOG`.
The `request_id` is supplied by mini-SWE-Agent and encodes the run, instance,
step, and API attempt. Token counts and all durations come from vLLM's own
finished-request state.

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
