# Agentic AI Characterization — Runtime Setup

> **Purpose:** This file contains the operational procedure for running Qwen3-235B through vLLM, mini-SWE-Agent on SWE-Bench Verified, the current timing/token instrumentation, and official SWE-Bench evaluation.
>
> The research design, measurement definitions, validation results, and experiment outcomes are kept in the separate research log.

---

## 1. Runtime Architecture

Use two remote SSH terminals, each inside its own `tmux` session:

```text
LOCAL MAC
├── Terminal 1 -> SSH -> tmux: qwen-vllm
│                    -> vLLM + Qwen3-235B
│                    -> server-side request/inference logging
│
└── Terminal 2 -> SSH -> tmux: swebench-agent
                     -> mini-SWE-Agent / SWE-Bench
                     -> per-instance model + tool logs
                     -> run-level CPU/GPU resource samples
                     -> official SWE-Bench evaluation
```

SSH target:

```bash
ssh jovan@10.0.0.4
```

Main checkout:

```text
/mnt/raid0/yigit/agent-characterization/mini-swe-agent
```

The preferred workflow uses the tracked helper scripts:

```text
start-vllm.sh
run-swebench.sh
evaluate-swebench.sh
scripts/install-vllm-instrumentation.sh
```

Run the scripts with `bash`, not `source`.

---

## 2. Canonical Paths

| Purpose | Path |
| --- | --- |
| Project root | `/mnt/raid0/yigit/agent-characterization` |
| mini-SWE-Agent checkout | `/mnt/raid0/yigit/agent-characterization/mini-swe-agent` |
| mini-SWE-Agent environment | `/mnt/raid0/yigit/agent-characterization/mini-swe-agent/.venv` |
| SWE-Bench run root | `/mnt/raid0/yigit/agent-characterization/swebench-runs` |
| Evaluation logs | `/mnt/raid0/yigit/agent-characterization/logs/run_evaluation` |
| vLLM request-event log | `<SWE-Bench run>/request_events.jsonl` |
| SWE-Bench evaluator environment | `/mnt/raid0/yigit/agent-characterization/envs/swebench-eval` |
| vLLM environment | `/mnt/experiment-yigit/qwen-experiment/envs/vllm` |
| Hugging Face cache | `/mnt/experiment-yigit/qwen-experiment/huggingface-cache` |
| vLLM cache | `/mnt/experiment-yigit/qwen-experiment/vllm-cache` |
| uv cache | `/mnt/experiment-yigit/qwen-experiment/uv-cache` |
| Docker root | `/mnt/raid0/docker` |

Compatibility symlink:

```text
/home/jovan/agent-characterization
    -> /mnt/raid0/yigit/agent-characterization
```

Do not remove this symlink; existing environments may still expose old-path entry points through it.

Model:

```text
Qwen/Qwen3-235B-A22B-Instruct-2507
```

Pinned snapshot:

```text
/mnt/experiment-yigit/qwen-experiment/huggingface-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e
```

---

# 3. Terminal 1 — Start vLLM

## 3.1 Create or attach to the vLLM tmux session

From the remote SSH shell:

```bash
tmux ls
```

If the session does not exist:

```bash
tmux new -s qwen-vllm
```

If it already exists:

```bash
tmux attach -t qwen-vllm
```

Do not start a second vLLM server if one is already listening on port 8000.

## 3.2 Start the instrumented server

Inside `qwen-vllm`:

```bash
cd /mnt/raid0/yigit/agent-characterization/mini-swe-agent
bash start-vllm.sh
```

Use the updated script rather than the old manual `vllm serve` command when collecting inference measurements. The updated startup path uses the tracked vLLM instrumentation needed for per-request server metrics.

The instrumentation is kept reproducibly through:

```text
instrumentation/
scripts/install-vllm-instrumentation.sh
```

Each run's server-side request log is:

```text
/mnt/raid0/yigit/agent-characterization/swebench-runs/<RUN-FOLDER>/request_events.jsonl
```

Wait until vLLM reports that the API server is listening before starting the agent.

## 3.3 Verify vLLM from another shell

```bash
ss -ltnp | grep ':8000'
```

Expected listener:

```text
127.0.0.1:8000
```

Then:

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

Expected model:

```text
Qwen/Qwen3-235B-A22B-Instruct-2507
```

## 3.4 Leave vLLM running

Detach without stopping it:

```text
Ctrl+B
D
```

Do **not** use `Ctrl+C` just to disconnect. `Ctrl+C` stops vLLM and releases the GPUs.

---

# 4. Terminal 2 — Run SWE-Bench

## 4.1 Create or attach to the agent tmux session

From a second SSH terminal:

```bash
tmux ls
```

If the session does not exist:

```bash
tmux new -s swebench-agent
```

If it already exists:

```bash
tmux attach -t swebench-agent
```

Run long benchmark jobs inside this tmux session so they survive VS Code, SSH, Wi-Fi, or local-machine disconnects.

## 4.2 Move to the checkout

Inside `swebench-agent`:

```bash
cd /mnt/raid0/yigit/agent-characterization/mini-swe-agent
```

The scripted workflow handles the mini-SWE-Agent environment and creates a fresh run directory.

Check usage when needed:

```bash
bash run-swebench.sh --help
```

## 4.3 Run one instance

Validated example:

```bash
bash run-swebench.sh --instance 'django__django-11099'
```

Use a single instance after instrumentation changes before scaling up.

## 4.4 Run the first N instances

First 5:

```bash
bash run-swebench.sh --slice 0:5 --workers 1
```

First 100:

```bash
bash run-swebench.sh --slice 0:100 --workers 1
```

`--slice 0:N` selects the first `N` entries using a zero-based, exclusive stop.

For characterization runs, keep `--workers 1` unless concurrency itself is intentionally being studied.

## 4.5 Run all SWE-Bench Verified test instances

The recorded Verified test split contains 500 instances.

```bash
bash run-swebench.sh --all --workers 1
```

Only increase worker count if overlapping requests/tools are part of the experiment.

Resource sampling starts automatically. The default interval is 0.5 seconds;
set it explicitly when needed:

```bash
bash run-swebench.sh --slice 0:100 --workers 1 --system-metrics-interval 0.5
```

---

# 5. Outputs Produced by a Run

`run-swebench.sh` creates a fresh timestamped run directory under:

```text
/mnt/raid0/yigit/agent-characterization/swebench-runs/
```

Typical structure:

```text
qwen-run-<timestamp>-<suffix>/
├── local-qwen.yaml
├── agent-command.txt
├── system_metrics.jsonl
├── system_metrics_summary.json
└── results/
    ├── preds.json
    ├── minisweagent.log
    ├── exit_statuses_*.yaml
    └── <instance-id>/
        ├── <instance-id>.traj.json
        ├── <instance-id>.traj.tool_events.jsonl
        └── <instance-id>.traj.model_events.jsonl
```

### Tool events

```text
<instance>.traj.tool_events.jsonl
```

Contains the exact agent-triggered actions and tool execution timing.

### Model events

```text
<instance>.traj.model_events.jsonl
```

Contains logical mini-SWE-Agent model-call measurements, token usage, status, finish reason, and request-ID correlation information.

### Run-level system metrics

```text
system_metrics.jsonl
system_metrics_summary.json
```

The sampler covers the benchmark command's complete lifetime and stays separate
from per-instance records. Each JSONL sample contains wall-clock and monotonic
timestamps, aggregate host CPU utilization and user/system/I/O-wait breakdown,
host memory, and utilization/memory for NVML GPU indices 0 through 7. The
summary contains the sampling interval, sample count, run bounds, mean/maximum
CPU utilization, sampled peak host memory, and per-GPU mean/maximum utilization
and sampled peak memory.

Aggregate CPU utilization is normalized to 0–100% for the whole host and is
defined as `(total - idle - iowait) / total` over consecutive `/proc/stat`
readings. User includes nice time; system includes IRQ and soft-IRQ time; steal
time is preserved separately. Host memory used is `MemTotal - MemAvailable`;
`MemFree` is also preserved. NVML GPU
utilization is the driver's recent utilization sample, not exclusive GPU-kernel
time. All reported memory peaks are sampled peaks and can miss sub-interval
spikes.

### Per-run vLLM request events

```text
<RUN-DIRECTORY>/request_events.jsonl
```

Contains one server-side record per actual vLLM request attempt, including:

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

`inference_duration_ns` is:

```text
last_token_time - first_scheduled_time
```

It is a **server-side inference-phase duration**, not exclusive CUDA-kernel/GPU-active time.

The vLLM server routes each record to the run directory encoded in its request
ID, so each file contains only one experiment's requests. A bounded background
queue writes the records without blocking vLLM's finished-request path.

---

# 6. Inspect a Completed Run

Set the run directory printed by the script:

```bash
export RUN_DIR="/mnt/raid0/yigit/agent-characterization/swebench-runs/<RUN-FOLDER>"
```

List artifacts:

```bash
find "$RUN_DIR/results" -maxdepth 3 -type f -print | sort
```

Find tool logs:

```bash
find "$RUN_DIR/results" -name '*.tool_events.jsonl' -print
```

Find model logs:

```bash
find "$RUN_DIR/results" -name '*.model_events.jsonl' -print
```

Count tool events:

```bash
find "$RUN_DIR/results" -name '*.tool_events.jsonl' -exec wc -l {} \;
```

Count model events:

```bash
find "$RUN_DIR/results" -name '*.model_events.jsonl' -exec wc -l {} \;
```

Inspect recent vLLM events:

```bash
tail -n 50 "$RUN_DIR/request_events.jsonl"
```

For validation, the same `request_id` should correlate the mini-SWE-Agent model event with its matching vLLM request event.

---

# 7. Official SWE-Bench Evaluation

Once the agent run finishes, vLLM is no longer needed for official evaluation. If other people need the GPUs, stop vLLM first.

The evaluator script usage is:

```text
bash evaluate-swebench.sh RUN_DIRECTORY [WORKERS]
```

Example:

```bash
cd /mnt/raid0/yigit/agent-characterization/mini-swe-agent

bash evaluate-swebench.sh \
  /mnt/raid0/yigit/agent-characterization/swebench-runs/<RUN-FOLDER>
```

Default: one evaluation worker.

Example with 3 evaluator workers:

```bash
bash evaluate-swebench.sh \
  /mnt/raid0/yigit/agent-characterization/swebench-runs/<RUN-FOLDER> \
  3
```

The evaluator script:

1. reads `results/preds.json`;
2. writes `results/preds.swebench.jsonl`;
3. creates a unique evaluation run ID;
4. runs official SWE-Bench Verified evaluation;
5. writes detailed evaluator logs under the project evaluation-log tree;
6. writes the official summary JSON under the project root.

Detailed evaluator logs:

```text
/mnt/raid0/yigit/agent-characterization/logs/run_evaluation/<run-id>/<model>/<instance>/
```

Official summary JSON pattern:

```text
/mnt/raid0/yigit/agent-characterization/<model-name>.<run-id>.json
```

If only 100 predictions are supplied, the report can show 500 total Verified instances and 400 incomplete entries. Those 400 are simply the unsubmitted members of the full split.

---

# 8. Stopping or Detaching

## Keep jobs running while disconnecting

Detach from either tmux session:

```text
Ctrl+B
D
```

Reconnect later:

```bash
tmux attach -t qwen-vllm
```

or:

```bash
tmux attach -t swebench-agent
```

## Stop vLLM and release the GPUs

Attach:

```bash
tmux attach -t qwen-vllm
```

Then press:

```text
Ctrl+C
```

After shutdown:

```bash
nvidia-smi
```

The eight vLLM tensor-parallel worker processes should disappear and GPU memory usage should drop substantially.

---

# 9. Useful tmux Controls

Scroll through terminal history:

```text
Ctrl+B
[
```

Then use arrow/Page-Up/Page-Down keys.

Exit copy mode:

```text
q
```

Detach without stopping the running process:

```text
Ctrl+B
D
```

---

# 10. Copy Results to the Local Mac

The server does not provide the SFTP subsystem expected by modern `scp`, so use legacy SCP mode with `-O`.

From the **local Mac**:

```bash
scp -O -r \
  jovan@10.0.0.4:/mnt/raid0/yigit/agent-characterization/swebench-runs/<RUN-FOLDER> \
  ~/Desktop/
```

To copy one instance only:

```bash
scp -O -r \
  jovan@10.0.0.4:/mnt/raid0/yigit/agent-characterization/swebench-runs/<RUN-FOLDER>/results/<INSTANCE-ID> \
  ~/Desktop/
```

---

# 11. Normal Experiment Order

```text
Terminal 1
  SSH
  -> tmux qwen-vllm
  -> bash start-vllm.sh
  -> wait until server is ready

Terminal 2
  SSH
  -> tmux swebench-agent
  -> bash run-swebench.sh <selection> --workers 1
  -> inspect exit-status summary
  -> inspect model/tool logs if needed

After agent run
  -> stop vLLM if GPUs should be released
  -> bash evaluate-swebench.sh <RUN_DIRECTORY>
  -> inspect official summary

Analysis
  -> correlate model_events.jsonl
     with <RUN_DIRECTORY>/request_events.jsonl using request_id
  -> combine with tool_events.jsonl and official evaluation outcome
```
