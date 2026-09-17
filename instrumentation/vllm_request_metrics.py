"""Per-request V1 inference-phase event logging for vLLM 0.10.1.1.

`inference_duration_ns` is the monotonic interval from the request's first
scheduled time through its last generated token. It is server-side
inference-phase duration, not exclusive CUDA/GPU-kernel time.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("vllm.request_metrics_jsonl")


def make_finished_request_event(request_id, finish_reason, prompt_tokens, stats) -> dict:
    """Build one JSON-compatible event from vLLM V1's finished-request state."""
    return {
        "request_id": request_id.removeprefix("chatcmpl-"),
        "queue_duration_ns": int((stats.scheduled_ts - stats.queued_ts) * 1_000_000_000),
        "prefill_duration_ns": int((stats.first_token_ts - stats.scheduled_ts) * 1_000_000_000),
        "decode_duration_ns": int((stats.last_token_ts - stats.first_token_ts) * 1_000_000_000),
        "inference_duration_ns": int((stats.last_token_ts - stats.scheduled_ts) * 1_000_000_000),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": stats.num_generation_tokens,
        "finish_reason": str(finish_reason),
    }


def log_finished_request_metrics(request_id, finish_reason, prompt_tokens, stats) -> None:
    """Append a finished-request event when `MSWEA_VLLM_EVENT_LOG` is configured."""
    if not (destination := os.getenv("MSWEA_VLLM_EVENT_LOG")):
        return
    try:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(make_finished_request_event(request_id, finish_reason, prompt_tokens, stats)) + "\n"
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o664)
        try:
            os.write(descriptor, line.encode())
        finally:
            os.close(descriptor)
    except Exception as error:
        logger.warning("Failed to record vLLM request metrics: %s", error)
