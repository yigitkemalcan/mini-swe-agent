"""Per-request V1 inference-phase event logging for vLLM 0.10.1.1.

`inference_duration_ns` is the monotonic interval from the request's first
scheduled time through its last generated token. It is server-side
inference-phase duration, not exclusive CUDA/GPU-kernel time.
"""

import json
import logging
import os
import queue
import re
import threading
from atexit import register
from pathlib import Path

logger = logging.getLogger("vllm.request_metrics_jsonl")
# mini-SWE-Agent separates the parts of a request ID with "~", which its own sanitizer strips from
# every part, so the run ID is whatever precedes the second separator whatever the run is named.
# Nothing here may depend on how run directories happen to be named.
RUN_REQUEST_ID = re.compile(r"^mswea~([^~]+)~")
DURATION_FIELDS = ("queue_duration_ns", "prefill_duration_ns", "decode_duration_ns", "inference_duration_ns")
EVENT_QUEUE: queue.Queue[tuple[Path, dict] | None] = queue.Queue(maxsize=10_000)
WRITER_LOCK = threading.Lock()
writer_thread: threading.Thread | None = None


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
        "engine_finish_reason": str(finish_reason),
    }


def is_consistent(event: dict) -> bool:
    """Whether the request's phases are ordered, so the event is worth writing.

    A request that finished without generating a token leaves `first_token_ts` unset, which would
    otherwise be recorded as a negative prefill duration. The writer must never emit an event that
    the analyzer would reject: a dropped record is recoverable, a poisoned log is not.
    """
    return all(event[field] >= 0 for field in DURATION_FIELDS)


def _append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o664)
    try:
        os.write(descriptor, (json.dumps(event) + "\n").encode())
    finally:
        os.close(descriptor)


def _writer_loop() -> None:
    while True:
        item = EVENT_QUEUE.get()
        try:
            if item is None:
                return
            path, event = item
            try:
                _append_event(path, event)
            except Exception as error:
                logger.warning("Failed to record vLLM request metrics: %s", error)
        finally:
            EVENT_QUEUE.task_done()


def _ensure_writer() -> None:
    global writer_thread
    if writer_thread is not None and writer_thread.is_alive():
        return
    with WRITER_LOCK:
        if writer_thread is None or not writer_thread.is_alive():
            writer_thread = threading.Thread(target=_writer_loop, name="vllm-request-metrics", daemon=True)
            writer_thread.start()


def flush_finished_request_metrics() -> None:
    """Wait until queued events have been written. Intended for tests and orderly shutdown."""
    EVENT_QUEUE.join()


def _shutdown_writer() -> None:
    if writer_thread is None or not writer_thread.is_alive():
        return
    try:
        EVENT_QUEUE.put(None, timeout=5)
    except queue.Full:
        logger.warning("Timed out while stopping the vLLM request-metrics writer")
        return
    writer_thread.join(timeout=5)
    if writer_thread.is_alive():
        logger.warning("vLLM request-metrics writer did not stop within five seconds")


register(_shutdown_writer)


def log_finished_request_metrics(request_id, finish_reason, prompt_tokens, stats) -> None:
    """Queue a finished-request event for its experiment without blocking vLLM output processing.

    Nothing this instrumentation does may reach vLLM: losing a measurement is an acceptable outcome,
    breaking the serving path it measures is not. Every failure here is reported and swallowed.
    """
    if not (runs_root := os.getenv("MSWEA_VLLM_RUNS_ROOT")):
        return
    try:
        event = make_finished_request_event(request_id, finish_reason, prompt_tokens, stats)
        if not (match := RUN_REQUEST_ID.match(event["request_id"])):
            logger.warning("Cannot determine run directory for vLLM request %s", event["request_id"])
            return
        if not is_consistent(event):
            logger.warning("Dropping vLLM request %s with unordered phases: %s", event["request_id"], event)
            return
        _ensure_writer()
        EVENT_QUEUE.put_nowait((Path(runs_root) / match.group(1) / "request_events.jsonl", event))
    except queue.Full:
        logger.warning("vLLM request-metrics queue is full; dropping request %s", request_id)
    except Exception as error:
        logger.warning("Failed to record vLLM request metrics for %s: %s", request_id, error)
