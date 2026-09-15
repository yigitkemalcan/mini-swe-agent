# Agentic AI Characterization — Research Log

> **Current analysis scope:** Semantic classification of tool calls is postponed. Classifying shell commands into categories requires heuristics whose correctness is not assured. For now, analysis uses recorded counts, durations, identifiers, outcomes, and other directly available fields without interpreting command purpose.
>
> **Compound commands:** One agent-requested `env.execute(action)` call counts as one tool invocation, even when its command contains multiple shell operations. Keep its measured duration as one value; do not split the command or attribute parts of its duration to individual operations.

> **Current checkpoint:** the end-to-end path from SWE-Bench Verified -> mini-SWE-Agent -> Qwen3-235B via vLLM -> agent tool execution -> tool-event logging -> patch submission -> official SWE-Bench evaluation has been validated on one real benchmark instance. The validated instance, `django__django-11099`, was **RESOLVED**. Tool-execution logging and timing are validated. The active project root is `/mnt/raid0/yigit/agent-characterization`. Explicit LLM/model-request timing has **not** yet been implemented.

---

## 1. Project Goal

Characterize agentic AI workloads from a systems perspective.

| Component        | Experimental target                        |
| ---------------- | ------------------------------------------ |
| Benchmark        | SWE-Bench / SWE-Bench Verified             |
| Agent            | mini-SWE-Agent v2, package version `2.4.6` |
| Model            | `Qwen/Qwen3-235B-A22B-Instruct-2507`       |
| Inference server | vLLM `0.10.1.1`                            |
| Hardware         | 8 x NVIDIA A100-SXM4-80GB                  |

Current measurement objectives:

1. Record each agent-requested tool/environment execution.
2. Preserve the exact raw action for later semantic classification.
3. Measure individual tool-execution duration.
4. Count tool invocations and aggregate tool time.
5. Keep enough identifiers to align tool events with benchmark instance and model-call step.
6. Eventually measure explicit LLM/model-request time separately. Do **not** infer LLM time as run wall time minus tool time.

---

## 2. Measurement Definition

### 2.1 Tool invocation

> **One agent-requested environment execution is one tool invocation.**

The runtime logger does not split compound shell commands.

Example:

```bash
cd /testbed && rg "foo" src/ && pytest tests/
```

This is one invocation with one measured duration.

Semantic categories such as `file_search`, `file_read`, `file_edit`, `test`, `version_control`, `submission`, `mixed`, and `unknown` are **not** assigned at runtime. The logger stores the exact `raw_action`; classification is performed later.

### 2.2 Tool timing boundary

The timer surrounds the agent-triggered `env.execute(action)` call.

It measures elapsed execution latency experienced by the agent, including Docker/environment execution overhead and the environment's completion checks.

It excludes the preceding model call, interactive confirmation, output-size calculation after execution, JSON serialization, tool-event file I/O, and benchmark setup/evaluation calls that bypass the agent wrapper.

Do not label all time outside tool events as "LLM time."

For concurrent workers, summed tool durations are aggregate execution time and are not a non-overlapping share of batch wall-clock time.

---

## 3. System, Storage, and Canonical Paths

This section is the central path reference for the project.

### 3.1 Remote machine

| Item                           | Recorded value            |
| ------------------------------ | ------------------------- |
| SSH target used from local Mac | `jovan@10.0.0.4`          |
| Hostname                       | `vmss-a100000002`         |
| Remote user                    | `jovan`                   |
| GPUs                           | 8 x NVIDIA A100-SXM4-80GB |
| NVIDIA driver                  | `560.35.05`               |
| Driver-reported CUDA           | `12.6`                    |
| CPU                            | AMD EPYC 7V12, 96 CPUs    |
| RAM                            | about 1.7 TiB             |
| Docker root                    | `/mnt/raid0/docker`       |

### 3.2 Project and experiment paths

| Purpose                   | Path                                                                    | Notes                                                                |
| ------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Main project root         | `/mnt/raid0/yigit/agent-characterization`                               | Active project, results, and control files                           |
| mini-SWE-Agent checkout   | `/mnt/raid0/yigit/agent-characterization/mini-swe-agent`                | Instrumented research checkout                                       |
| mini-SWE-Agent Python env | `/mnt/raid0/yigit/agent-characterization/mini-swe-agent/.venv`          | Used for `mini` and `mini-extra swebench`                            |
| SWE-Bench run root        | `/mnt/raid0/yigit/agent-characterization/swebench-runs`                 | Fresh timestamped run directories                                    |
| Tool-logger test evidence | `/mnt/raid0/yigit/agent-characterization/test-results/20260910T042416Z` | Focused/full tests + Docker smoke test                               |
| SWE-Bench evaluator env   | `/mnt/raid0/yigit/agent-characterization/envs/swebench-eval`            | Separate from mini-SWE-Agent env                                     |
| SWE-Bench evaluator logs  | `/mnt/raid0/yigit/agent-characterization/logs/run_evaluation`           | `run-id/model/instance/...`                                          |
| vLLM/Qwen experiment root | `/mnt/experiment-yigit/qwen-experiment`                                 | Large replaceable data under `/mnt`                                  |
| vLLM Python env           | `/mnt/experiment-yigit/qwen-experiment/envs/vllm`                       | vLLM serving environment                                             |
| Hugging Face cache        | `/mnt/experiment-yigit/qwen-experiment/huggingface-cache`               | Contains the Qwen snapshot                                           |
| vLLM cache                | `/mnt/experiment-yigit/qwen-experiment/vllm-cache`                      | vLLM cache root                                                      |
| uv cache                  | `/mnt/experiment-yigit/qwen-experiment/uv-cache`                        | uv cache                                                             |
| Model staging dir         | `/mnt/experiment-yigit/qwen-experiment/models`                          | Exists, but the current Qwen snapshot is used directly from HF cache |
| Experiment log dir        | `/mnt/experiment-yigit/qwen-experiment/logs`                            | Available for vLLM/experiment logs                                   |
| Docker storage            | `/mnt/raid0/docker`                                                     | Docker root                                                          |
| Persistent NFS            | `/mnt/hqiufilestoragenfs/hqiufileshare`                                 | Persistent storage; about 617 GB free at original inspection         |

The compatibility symlink `/home/jovan/agent-characterization` points to the active project root and must remain in place for the existing virtual environments. CLI paths may display this alias. Global mini-SWE-Agent settings remain at `/home/jovan/.config/mini-swe-agent/.env`.

### 3.3 Exact Qwen snapshot

Model:

```text
Qwen/Qwen3-235B-A22B-Instruct-2507
```

Pinned revision:

```text
ac9c66cc9b46af7306746a9250f23d47083d689e
```

Snapshot:

```text
/mnt/experiment-yigit/qwen-experiment/huggingface-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e
```

Download verification completed:

- 129 files present
- no broken symlinks
- no `*.incomplete` files
- model cache about 438 GiB

### 3.4 Storage notes

`/mnt` is Azure temporary/ephemeral storage. It is appropriate for the large model/cache because these files are replaceable, but irreplaceable experiment results should also be copied to the local machine or persistent storage.

The administrator described the machine as fixed and not normally reallocated, but that does not change `/mnt`'s temporary-storage designation.

The Qwen model is used directly from the Hugging Face cache. Do not create another approximately 438 GiB copy unless there is a specific reason.

Avoid broad Docker cleanup on this shared machine.

---

## 4. Validated Software State

### 4.1 mini-SWE-Agent

Remote checkout:

```text
/mnt/raid0/yigit/agent-characterization/mini-swe-agent
```

Research branch:

```text
characterization/tool-profiling
```

Unmodified baseline commit:

```text
04d809ceab9df28f9adaed044884180159172930
```

Validated tool-instrumentation commit:

```text
28decad8c36e6d69f61cf6cf28fa914582facfc9
```

Package version:

```text
2.4.6
```

Agent environment entry point (the old-path alias may still appear in `which python`):

```text
/mnt/raid0/yigit/agent-characterization/mini-swe-agent/.venv/bin/python
```

### 4.2 vLLM serving environment

| Component        | Validated version |
| ---------------- | ----------------- |
| vLLM             | `0.10.1.1`        |
| Transformers     | `4.55.2`          |
| Tokenizers       | `0.21.4`          |
| PyTorch          | `2.7.1+cu126`     |
| Hugging Face Hub | `0.36.2`          |

The Transformers/Tokenizers versions are intentional. The earlier Transformers 5.x environment was incompatible with this vLLM/model path.

### 4.3 mini-SWE-Agent local-model configuration

The global default model was set to:

```text
hosted_vllm/Qwen/Qwen3-235B-A22B-Instruct-2507
```

No external API key is required for the local vLLM server.

For benchmark runs, the local endpoint is still specified explicitly:

```yaml
model:
  model_name: hosted_vllm/Qwen/Qwen3-235B-A22B-Instruct-2507
  cost_tracking: ignore_errors
  model_kwargs:
    api_base: http://127.0.0.1:8000/v1
```

Do not permanently modify the repository benchmark YAML just to point it at the local server. Use a run-specific override file.

---

# 5. Complete Runtime Procedure — Two-Terminal Workflow

Choose one way to run the system:

- **Script workflow:** connect using Section 5.1, then follow Section 5.0. The scripts perform environment activation and the run/evaluation commands for you.
- **Manual workflow:** follow Sections 5.1–5.10 in order instead of running the scripts.
- For either workflow, Section 5.11 covers optional result copying and Section 5.12 covers detaching or stopping.

Command labels are consistent throughout this section:

| Label | Meaning |
| --- | --- |
| **Required action** | Performs part of the selected workflow. |
| **Conditional action** | Run only when the stated condition applies. |
| **Alternative action** | Choose one example; do not execute every alternative. |
| **Verification — required** | Check the stated result before continuing; resolve a failed check first. |
| **Verification — optional** | Diagnostic inspection; it does not perform a required setup step. |
| **Optional action** | Extra work you may skip, such as copying results. |

Blocks marked `text` show expected output, paths, or key presses; they are not shell commands. Replace angle-bracket placeholders before executing examples.

The normal setup uses two SSH terminals on the remote machine:

```text
LOCAL MAC
├── Terminal 1 -> SSH -> tmux: qwen-vllm -> vLLM/Qwen server environment
└── Terminal 2 -> SSH -> tmux: swebench-agent -> mini-SWE-Agent environment
                                            -> then SWE-Bench evaluator environment
```

The environments are different and should be activated only in the terminal where they are needed:

```text
Terminal 1:
    /mnt/experiment-yigit/qwen-experiment/envs/vllm

Terminal 2 while running the agent:
    /mnt/raid0/yigit/agent-characterization/mini-swe-agent/.venv

Terminal 2 after the agent run, while evaluating:
    /mnt/raid0/yigit/agent-characterization/envs/swebench-eval
```

Do not activate the vLLM and mini-SWE-Agent environments in the same shell.

---

## 5.0 Script shortcuts — same two-terminal workflow

The following scripts live beside this log in the `mini-swe-agent` repository root:

- `start-vllm.sh`: starts the pinned Qwen model using the existing vLLM environment and cache paths.
- `run-swebench.sh`: checks the served model, creates a fresh run directory and local override, then runs SWE-Bench Verified (`test` split).
- `evaluate-swebench.sh`: converts a run's predictions and evaluates them using the separate evaluator environment.

These scripts use installed environments; they do not install dependencies or create virtual environments. Run them with `bash`, not `source`. Their environment changes stay inside the script, so manually switching virtual environments in the parent terminal is unnecessary. The old-path compatibility symlink must remain in place.

After transferring the scripts to the remote checkout, use the following commands. All commands in this subsection run on the remote machine after SSH login.

**Terminal 1 — create the tmux session if it does not exist:**

**Conditional action — create this session only if it does not already exist.**

```bash
tmux new -s qwen-vllm
```

If the session already exists, attach instead:

**Conditional action — attach to this existing session when reconnecting; skip if already inside it.**

```bash
tmux attach -t qwen-vllm
```

If vLLM is already running in the session, leave it running and continue in Terminal 2. Otherwise, at the shell prompt inside tmux:

**Conditional action — start vLLM only if it is not already running.**

```bash
cd /mnt/raid0/yigit/agent-characterization/mini-swe-agent
bash start-vllm.sh
```

Wait for the server to finish loading. The script runs in the foreground and refuses to start if port 8000 is already listening. It does not create tmux sessions itself. Detach with Ctrl+B then D to keep serving, or stop the server with Ctrl+C.

**Terminal 2 — create its separate tmux session if it does not exist:**

**Conditional action — create this session only if it does not already exist.**

```bash
tmux new -s swebench-agent
```

If the session already exists, attach instead:

**Conditional action — attach to this existing session when reconnecting; skip if already inside it.**

```bash
tmux attach -t swebench-agent
```

If an agent run or evaluation is already running there, let it finish. Otherwise, inside this session, select one run command below. Keep evaluation in this same session.

**Required action — run this command block as part of the selected workflow.**

```bash
cd /mnt/raid0/yigit/agent-characterization/mini-swe-agent
```

Run one instance (replace the placeholder with its exact dataset ID):

**Alternative action — choose this instance selection for one experiment; do not run every example.**

```bash
bash run-swebench.sh --instance '<INSTANCE_ID>'
```

Run five instances with up to two running concurrently:

**Alternative action — choose this instance selection for one experiment; do not run every example.**

```bash
bash run-swebench.sh --slice 0:5 --workers 2
```

Run the first five instances:

**Alternative action — choose this instance selection for one experiment; do not run every example.**

```bash
bash run-swebench.sh --slice 0:5
```

Run all Verified test instances (500 in the recorded dataset), sequentially:

**Alternative action — choose this instance selection for one experiment; do not run every example.**

```bash
bash run-swebench.sh --all --workers 1
```

Run all instances with up to three benchmark workers:

**Alternative action — choose this instance selection for one experiment; do not run every example.**

```bash
bash run-swebench.sh --all --workers 3
```

`--slice` selects zero-based positions with an exclusive stop, after filtering. Use `--slice 0:N` for the first N instances. There is no default instance: running without a selection prints usage and exits. Repeat `--instance` for multiple exact IDs, or use `--filter` for a regex, or `--all` for the full split; these three selectors cannot be mixed. Worker count controls concurrency independently of how many instances are selected. The scripts keep per-instance tool logs separate. They do not resume an earlier run; each invocation creates a new timestamped directory with a unique suffix.

The run script prints the run directory and a complete evaluation command. It also saves `agent-command.txt` beside `local-qwen.yaml` to record the selection and worker count. **Verification — required:** inspect the runner's exit-status summary before evaluation; a finished runner does not guarantee every instance submitted successfully. **Verification — optional:** inspect individual tool logs for measurement details.

**Terminal 2, inside `swebench-agent` — evaluate after the agent finishes:**

Copy the evaluation command printed by the run script, or replace `<RUN-DIRECTORY>` below with its full path:

**Alternative action — choose one evaluation command after the agent run finishes.**

```bash
bash evaluate-swebench.sh '<RUN-DIRECTORY>'
```

To use three evaluator workers:

**Alternative action — choose one evaluation command after the agent run finishes.**

```bash
bash evaluate-swebench.sh '<RUN-DIRECTORY>' 3
```

The evaluator script processes the supplied predictions against Verified. It preserves `preds.json`, writes `results/preds.swebench.jsonl`, and creates a unique `evaluation-<timestamp>-<suffix>` folder within the run for `run-id.txt` and `evaluator.log`. Official detailed logs remain under the project root's `logs/run_evaluation/<run-id>/`, and the official summary JSON is written in the project root, matching the manual workflow.

For argument help:

**Verification — optional: display script usage without launching a workload.**

```bash
bash start-vllm.sh --help
bash run-swebench.sh --help
bash evaluate-swebench.sh --help
```

The scripts were reviewed locally and checked for shell syntax only. Remote execution is required to validate them. The manual commands below remain available for step-by-step operation.

---

## 5.1 Before starting — open both SSH terminals

On the **local Mac**, open two terminal windows.

In **Local Terminal 1**:

**Conditional action — connect from this Mac terminal if it is not already connected to the remote machine.**

```bash
ssh jovan@10.0.0.4
```

In **Local Terminal 2**:

**Conditional action — connect from this Mac terminal if it is not already connected to the remote machine.**

```bash
ssh jovan@10.0.0.4
```

After login:

```text
Remote Terminal 1 = vLLM server
Remote Terminal 2 = agent / benchmark / evaluation
```

---

## 5.2 Remote Terminal 1 — start the vLLM/Qwen server

### Step 1: make sure Terminal 1 is not using another virtual environment

If the shell prompt shows another environment, leave it first:

**Conditional action — leave an active environment before switching; this command also tolerates having none active.**

```bash
deactivate 2>/dev/null || true
```

### Step 2: check whether the persistent vLLM tmux session already exists

**Verification — optional: list existing sessions to decide whether to create or attach. A “no server running” message means no session is available to attach to.**

```bash
tmux ls
```

If `qwen-vllm` exists:

**Conditional action — attach to this existing session when reconnecting; skip if already inside it.**

```bash
tmux attach -t qwen-vllm
```

If the vLLM server is already running inside that session, **do not start another copy**. Leave it running and continue to Section 5.3 in Terminal 2.

If `qwen-vllm` does not exist:

**Conditional action — create this session only if it does not already exist.**

```bash
tmux new -s qwen-vllm
```

You are now inside the tmux session where vLLM should run.

### Step 3: export the vLLM/Qwen cache paths

Inside the `qwen-vllm` tmux session:

**Required action — run this command block as part of the selected workflow.**

```bash
export HF_HOME=/mnt/experiment-yigit/qwen-experiment/huggingface-cache
export VLLM_CACHE_ROOT=/mnt/experiment-yigit/qwen-experiment/vllm-cache
export UV_CACHE_DIR=/mnt/experiment-yigit/qwen-experiment/uv-cache
```

These variables are shell-local, so set them again whenever starting vLLM from a new shell/session.

### Step 4: activate the vLLM virtual environment

Still in **Terminal 1**:

**Required action — run this command block as part of the selected workflow.**

```bash
source /mnt/experiment-yigit/qwen-experiment/envs/vllm/bin/activate
```

Verify the environment before starting the server:

**Verification — optional: inspect the active Python environment and installed CLI. Resolve an unexpected environment before continuing.**

```bash
which python
python --version
vllm --version
```

The vLLM environment path should be:

```text
/mnt/experiment-yigit/qwen-experiment/envs/vllm
```

Validated stack:

```text
vLLM          0.10.1.1
Transformers  4.55.2
Tokenizers    0.21.4
PyTorch       2.7.1+cu126
```

### Step 5: set the exact pinned Qwen snapshot

**Required action — run this command block as part of the selected workflow.**

```bash
SNAPSHOT="/mnt/experiment-yigit/qwen-experiment/huggingface-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e"
```

Optional verification:

**Verification — optional: confirm the snapshot directory exists; expect “snapshot exists”. If it is missing, fix the path before launching vLLM.**

```bash
test -d "$SNAPSHOT" && echo "snapshot exists"
```

### Step 6: start vLLM

**Required action — run this command block as part of the selected workflow.**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve "$SNAPSHOT" \
  --served-model-name Qwen/Qwen3-235B-A22B-Instruct-2507 \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 127.0.0.1 \
  --port 8000
```

The two tool-calling flags are required:

```text
--enable-auto-tool-choice
--tool-call-parser hermes
```

Without them, vLLM rejects mini-SWE-Agent's `tool_choice="auto"` requests.

### Step 7: wait for startup to complete

**Verification — required: wait for the server startup logs to report that the API server is listening before continuing in Terminal 2.**

Do not start the agent yet.

Wait until vLLM has finished loading the model and the API server is listening.

Leave **Terminal 1** running. It is useful to keep it visible so vLLM logs can be monitored while the agent runs.

**Conditional action — if closing Terminal 1 while keeping vLLM alive, detach from tmux:**

```text
Ctrl+B
D
```

Do **not** use `Ctrl+C` merely to disconnect. `Ctrl+C` stops vLLM.

---

## 5.3 Remote Terminal 2 — open tmux and verify vLLM

Terminal 2 uses its own persistent tmux session for mini-SWE-Agent, SWE-Bench, result inspection, and official evaluation.

Before the checks below, create this session from the SSH shell if it does not exist:

**Conditional action — create this session only if it does not already exist.**

```bash
tmux new -s swebench-agent
```

If the session already exists, attach instead:

**Conditional action — attach to this existing session when reconnecting; skip if already inside it.**

```bash
tmux attach -t swebench-agent
```

If a run or evaluation is already active, let it finish rather than starting another. Perform all Terminal 2 commands in Sections 5.3–5.10 inside this session. Use `qwen-vllm` only for Terminal 1's model server; do not nest these sessions.

### Step 1: verify that vLLM from Terminal 1 is listening

Before starting an agent run:

**Verification — required for the manual workflow: confirm the server is listening before running the agent.**

```bash
ss -ltnp | grep ':8000'
```

Expected:

```text
127.0.0.1:8000
```

### Step 2: verify the served model

**Verification — required for the manual workflow: confirm the expected Qwen model is served.**

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

Confirm:

```text
model ID: Qwen/Qwen3-235B-A22B-Instruct-2507
max_model_len: 32768
```

If this check fails, stop here and fix Terminal 1. Do not start mini-SWE-Agent against an unhealthy server.

### Step 3: optional tool-calling sanity check

This optional check is useful after a fresh vLLM restart, a model-stack change, or a serving-flag change:

**Verification — optional: send a small model request to check tool-call formatting. This performs inference but does not execute the returned bash command.**

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "messages": [
      {
        "role": "user",
        "content": "Use the bash tool to print exactly TOOL_OK."
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "bash",
          "description": "Execute a bash command.",
          "parameters": {
            "type": "object",
            "properties": {
              "command": {"type": "string"}
            },
            "required": ["command"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 128
  }' | python3 -m json.tool
```

Validated behavior:

```text
finish_reason = tool_calls
bash tool call such as: echo TOOL_OK
```

---

## 5.4 Remote Terminal 2 — activate mini-SWE-Agent

### Step 1: leave any previously active environment

**Conditional action — leave an active environment before switching; this command also tolerates having none active.**

```bash
deactivate 2>/dev/null || true
```

### Step 2: enter the instrumented mini-SWE-Agent checkout

**Required action — run this command block as part of the selected workflow.**

```bash
cd /mnt/raid0/yigit/agent-characterization/mini-swe-agent
```

### Step 3: activate the mini-SWE-Agent virtual environment

**Required action — run this command block as part of the selected workflow.**

```bash
source .venv/bin/activate
```

Verify:

**Verification — optional: inspect the active Python environment and installed CLI. Resolve an unexpected environment before continuing.**

```bash
which python
python --version
mini --help
mini-extra swebench --help
```

Agent environment entry point (the old-path alias may still appear in `which python`):

```text
/mnt/raid0/yigit/agent-characterization/mini-swe-agent/.venv/bin/python
```

Verify the physical checkout and imported package location after activation:

**Verification — optional: confirm the physical checkout, compatibility shortcut, and imported package location.**

```bash
pwd -P
readlink -f /home/jovan/agent-characterization
python - <<'PYTHON'
from pathlib import Path
import minisweagent

location = Path(minisweagent.__file__).resolve()
print("Physical package location:", location)
assert location.is_relative_to(Path("/mnt/raid0/yigit/agent-characterization"))
PYTHON
```

The physical package location was verified as `/mnt/raid0/yigit/agent-characterization/mini-swe-agent/src/minisweagent/__init__.py`.

This is the environment that must remain active while running `mini` or `mini-extra swebench`.

---

## 5.5 Remote Terminal 2 — create a fresh benchmark run

Always create a new timestamped directory for a logically new run.

**Required action — run this command block as part of the selected workflow.**

```bash
export RUN_DIR="/mnt/raid0/yigit/agent-characterization/swebench-runs/qwen-run-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
echo "$RUN_DIR"
```

Create the local-Qwen override:

**Required action — write the run-specific local-Qwen configuration file.**

```bash
cat > "$RUN_DIR/local-qwen.yaml" <<'YAML'
model:
  model_name: hosted_vllm/Qwen/Qwen3-235B-A22B-Instruct-2507
  cost_tracking: ignore_errors
  model_kwargs:
    api_base: http://127.0.0.1:8000/v1
YAML
```

Keep this `swebench-agent` tmux shell through evaluation so `RUN_DIR` remains available. You may detach and reconnect without losing the shell or its variables; do not exit the shell.

Verify it:

**Verification — optional: inspect the file created in the preceding action.**

```bash
cat "$RUN_DIR/local-qwen.yaml"
```

Do not permanently edit the repository's benchmark YAML just to point it at the local vLLM server.

---

## 5.6 Remote Terminal 2 — run SWE-Bench Verified

Example using the already validated instance:

**Required action — launch the selected benchmark run (the command below is a single-instance example).**

```bash
mini-extra swebench \
  -c /mnt/raid0/yigit/agent-characterization/mini-swe-agent/src/minisweagent/config/benchmarks/swebench.yaml \
  -c "$RUN_DIR/local-qwen.yaml" \
  -m hosted_vllm/Qwen/Qwen3-235B-A22B-Instruct-2507 \
  -o "$RUN_DIR/results" \
  --subset verified \
  --split test \
  --filter '^(django__django-11099)$' \
  --workers 1
```

For another experiment, change the instance selection and, when intentionally testing concurrency, the worker count.

For early characterization runs, `--workers 1` gives the cleanest timing interpretation.

The SWE-Bench runner automatically creates one tool-event JSONL file per instance. Do not manually set one shared `agent.tool_log_path` for the benchmark.

Expected output layout:

```text
$RUN_DIR/
    local-qwen.yaml
    results/
        minisweagent.log
        preds.json
        exit_statuses_*.yaml
        <instance-id>/
            <instance-id>.traj.json
            <instance-id>.traj.tool_events.jsonl
```

---

## 5.7 Remote Terminal 2 — inspect the benchmark artifacts

After the agent run finishes, while the mini-SWE-Agent environment is still active:

**Verification — optional: inspect saved artifacts after the run; this does not execute another experiment.**

```bash
find "$RUN_DIR/results" -maxdepth 3 -type f -print | sort
```

List tool-event logs:

**Verification — optional: inspect saved artifacts after the run; this does not execute another experiment.**

```bash
find "$RUN_DIR/results" -name '*.tool_events.jsonl' -print
```

Count tool invocations:

**Verification — optional: inspect saved artifacts after the run; this does not execute another experiment.**

```bash
find "$RUN_DIR/results" -name '*.tool_events.jsonl' -exec wc -l {} \;
```

Inspect the JSONL:

**Verification — optional: inspect saved artifacts after the run; this does not execute another experiment.**

```bash
find "$RUN_DIR/results" -name '*.tool_events.jsonl' -exec cat {} \;
```

Important fields:

```text
instance_id
step_id
action_index
tool_call_id
raw_action
duration_ns
return_code / exception metadata
```

At this point the agent/benchmark run is complete.

---

## 5.8 Remote Terminal 2 — switch from mini-SWE-Agent to the SWE-Bench evaluator

The evaluator uses a **different** virtual environment. Evaluation can be performed later, after the agent run has finished; vLLM does not need to be running.

If returning in a new SSH terminal, reconnect to the agent/evaluator tmux session before continuing. Shell variables exist only in the shell where they were set. The run script does not export `RUN_DIR` to its parent shell.

**Conditional action — attach from the SSH shell if the session exists and you are not already inside it.**

```bash
tmux attach -t swebench-agent
```

**Conditional action — create the session instead if it does not exist.**

```bash
tmux new -s swebench-agent
```

**Verification — optional: list completed run prediction files to identify the experiment you want. Do not automatically choose the newest run.**

```bash
find /mnt/raid0/yigit/agent-characterization/swebench-runs -mindepth 3 -maxdepth 3 -type f -name preds.json -print | sort
```

**Required action — set the intended run directory explicitly; replace `<RUN-FOLDER>` with its actual name. Use the experiment folder, not its `results` subfolder.**

```bash
export RUN_DIR="/mnt/raid0/yigit/agent-characterization/swebench-runs/<RUN-FOLDER>"
```

**Verification — required: confirm the directory and predictions exist before continuing.**

```bash
printf 'Selected run: %s\n' "$RUN_DIR"
test -f "$RUN_DIR/results/preds.json" && echo "Predictions found"
```

If `Predictions found` is not printed, correct `RUN_DIR` first. If `PRED_JSON` or `PRED_JSONL` was previously assigned using an unset or different `RUN_DIR`, those variables must be assigned again; changing `RUN_DIR` does not update them automatically.

**Alternative action — use the evaluation script for all predictions in this Verified run. It activates the evaluator environment, converts predictions, and selects a unique evaluation ID automatically. Skip the remaining manual steps in Sections 5.8–5.10 when using this command.**

```bash
bash /mnt/raid0/yigit/agent-characterization/mini-swe-agent/evaluate-swebench.sh "$RUN_DIR"
```

The optional second script argument sets evaluation concurrency, for example `"$RUN_DIR" 3` for three evaluator workers. This applies to a single instance, a selected group, or all instances present in the run's predictions.

For manual evaluation instead, continue below.

### Step 1: deactivate mini-SWE-Agent

**Conditional action — leave an active environment before switching; this command also tolerates having none active.**

```bash
deactivate 2>/dev/null || true
```

### Step 2: activate the SWE-Bench evaluator environment

**Required action — run this command block as part of the selected workflow.**

```bash
source "/mnt/raid0/yigit/agent-characterization/envs/swebench-eval/bin/activate"
```

Verify:

**Verification — optional: inspect the active Python environment and installed CLI. Resolve an unexpected environment before continuing.**

```bash
which python
python --version
swebench --help
```

The evaluator environment path is:

```text
/mnt/raid0/yigit/agent-characterization/envs/swebench-eval
```

The old `/home/jovan/agent-characterization/envs/swebench-eval/bin/python` path in `which python` is expected because of the compatibility symlink. Do not run the official evaluator from the mini-SWE-Agent `.venv`.

---

## 5.9 Remote Terminal 2 — convert predictions for the official evaluator

mini-SWE-Agent writes:

```text
$RUN_DIR/results/preds.json
```

as a JSON object keyed by instance ID.

Convert it to one-line-per-prediction JSONL without modifying the original:

**Required action — run this command block as part of the selected workflow.**

```bash
: "${RUN_DIR:?Set RUN_DIR to the completed experiment directory first}"
export PRED_JSON="$RUN_DIR/results/preds.json"
export PRED_JSONL="$RUN_DIR/results/preds.swebench.jsonl"
```

Then:

**Required action — run this command block as part of the selected workflow.**

```bash
python - <<'PY'
import json
import os

src = os.environ["PRED_JSON"]
dst = os.environ["PRED_JSONL"]

with open(src) as f:
    data = json.load(f)

with open(dst, "w") as f:
    for pred in data.values():
        f.write(json.dumps(pred) + "\n")

print(dst)
PY
```

Verify:

**Verification — optional: inspect the file created in the preceding action.**

```bash
cat "$PRED_JSONL"
```

---

## 5.10 Remote Terminal 2 — run official SWE-Bench evaluation

Move to the project root so the summary JSON is written in a known location:

**Required action — run this command block as part of the selected workflow.**

```bash
cd "/mnt/raid0/yigit/agent-characterization"
```

Create a fresh evaluation ID. This command evaluates all instances represented in the prediction file against SWE-Bench Verified; there is no Django-specific selection.

**Required action — evaluate the selected run. `-j 1` uses one evaluator worker; change it to `-j 3`, for example, for up to three concurrent evaluations.**

```bash
EVAL_RUN_ID="$(basename "$RUN_DIR")-eval-$(date -u +%Y%m%dT%H%M%SZ)"

swebench eval verified \
  -p "$PRED_JSONL" \
  -j 1 \
  --run-id "$EVAL_RUN_ID"
```

Use a **new `--run-id`** for a different prediction or a logically distinct evaluation run.

**Verification — required: inspect the evaluation summary for completed instances and errors before treating the evaluation as successful.** The primary `swebench eval` output is authoritative.

Detailed evaluation logs are written under:

```text
/mnt/raid0/yigit/agent-characterization/logs/run_evaluation/<run-id>/<model>/<instance>/
```

Typical files:

```text
report.json
test_output.txt
```

The evaluation summary JSON is written in the current directory.

Summary filename pattern (use the actual run ID from this evaluation):

```text
/mnt/raid0/yigit/agent-characterization/<model-name>.<run-id>.json
```

---

## 5.11 Local Mac — copy benchmark results to the Desktop

Run these commands on the **local Mac**, not on the remote server.

The server does not expose the SFTP subsystem expected by modern `scp`, so use legacy SCP mode:

```text
scp -O
```

Copy one complete instance directory directly to the Desktop:

**Optional action — copy results from a local Mac terminal. Choose an instance copy or a whole-run copy.**

```bash
scp -O -r \
  jovan@10.0.0.4:/mnt/raid0/yigit/agent-characterization/swebench-runs/<RUN-DIR>/results/<INSTANCE-ID> \
  ~/Desktop/
```

Validated example:

**Optional action — copy results from a local Mac terminal. Choose an instance copy or a whole-run copy.**

```bash
scp -O -r \
  jovan@10.0.0.4:/mnt/raid0/yigit/agent-characterization/swebench-runs/qwen-first-20260910T052130Z/results/django__django-11099 \
  ~/Desktop/
```

The copied instance directory contains at least:

```text
<instance-id>.traj.json
<instance-id>.traj.tool_events.jsonl
```

To copy the whole run instead:

**Optional action — copy results from a local Mac terminal. Choose an instance copy or a whole-run copy.**

```bash
scp -O -r \
  jovan@10.0.0.4:/mnt/raid0/yigit/agent-characterization/swebench-runs/<RUN-DIR> \
  ~/Desktop/
```

---

## 5.12 Ending the session

### If more experiments will be run soon

Leave vLLM running.

**Conditional action — when disconnecting while preserving the processes, detach from each terminal’s tmux session:**

```text
Ctrl+B
D
```

Then both local terminal windows may be closed.

Later, reconnect and use:

**Conditional action — attach to this existing session when reconnecting; skip if already inside it.**

```bash
tmux attach -t qwen-vllm
```

In Terminal 2, reconnect to the agent/evaluator session:

**Conditional action — attach to this existing session when reconnecting; skip if already inside it.**

```bash
tmux attach -t swebench-agent
```

### If vLLM should be stopped and the GPUs released

Attach to the vLLM session:

**Conditional action — attach to this existing session when reconnecting; skip if already inside it.**

```bash
tmux attach -t qwen-vllm
```

**Conditional action — only when you want to stop vLLM and release its GPUs, press:**

```text
Ctrl+C
```

Wait for the shell prompt to return.

Optional GPU check:

**Verification — optional: inspect GPU usage after stopping your server.**

```bash
nvidia-smi
```

Terminal 2 can be closed after detaching with Ctrl+B then D, even while the agent or evaluator is running. The `swebench-agent` session keeps the process and shell alive. Ctrl+C interrupts the active job; do not use it just to disconnect.

---

## 6. Tool Instrumentation

The validated implementation touches:

| File                                              | Purpose                                               |
| ------------------------------------------------- | ----------------------------------------------------- |
| `src/minisweagent/agents/default.py`              | Shared tool-execution wrapper and JSONL logging       |
| `src/minisweagent/agents/interactive.py`          | Routes interactive actions through the shared wrapper |
| `src/minisweagent/run/benchmarks/utils/common.py` | Per-instance default tool-log path handling           |
| `src/minisweagent/run/benchmarks/swebench.py`     | Enables per-instance tool logging for SWE-Bench       |
| `src/minisweagent/run/benchmarks/programbench.py` | Enables the same pattern for ProgramBench             |
| `tests/agents/test_tool_logging.py`               | Tool-logging tests                                    |
| `tests/run/test_benchmark_utils.py`               | Benchmark-path/config helper tests                    |

For a generic agent, `agent.tool_log_path = None` means logging is disabled.

For SWE-Bench and ProgramBench, the benchmark runner automatically seeds a per-instance path.

Typical pair:

```text
django__django-11099.traj.json
django__django-11099.traj.tool_events.jsonl
```

Use fresh output directories for experiments. Do not point multiple workers to a single custom shared JSONL path.

### Tool-event schema

| Field                     | Meaning                                               |
| ------------------------- | ----------------------------------------------------- |
| `event_type`              | `tool_execution`                                      |
| `step_id`                 | model-call progression index used by the agent        |
| `action_index`            | zero-based action position within that model response |
| `raw_action`              | exact command sent to the environment                 |
| `start_time_ns`           | wall-clock start timestamp                            |
| `end_time_ns`             | wall-clock end timestamp                              |
| `duration_ns`             | monotonic elapsed execution duration                  |
| `instance_id`             | benchmark instance when available                     |
| `tool_call_id`            | model-provided tool-call ID when available            |
| `return_code`             | returned environment code                             |
| `output_size_bytes`       | UTF-8 byte length of returned output string           |
| `returned_exception_type` | returned timeout/error type when applicable           |
| `returned_exception_info` | returned timeout/error detail when applicable         |
| `exception_type`          | exception escaping `env.execute`, e.g. `Submitted`    |
| `exception_message`       | escaping exception message                            |

`Submitted` is expected completion behavior and is not automatically a failed tool execution.

---

## 9. Local Editing / Remote Execution Rule

The local coding agent can inspect and edit code, but it cannot execute the project.

The workflow is:

```text
LOCAL CODING AGENT
    inspect/edit source and tests only
        ->
USER ON LOCAL MAC
    review diff
    stage/commit/push
        ->
REMOTE GPU MACHINE
    pull approved commit
    run tests
    run Docker/integration/benchmark experiments
```

Do not ask the local coding agent to run shell commands, Python, tests, Git, Docker, SSH, vLLM, or SWE-Bench.

The user performs local Git operations manually.

When updating the remote checkout:

```bash
cd /mnt/raid0/yigit/agent-characterization/mini-swe-agent
git status --short
git fetch origin
git switch characterization/tool-profiling
git pull --ff-only origin characterization/tool-profiling
git rev-parse HEAD

source .venv/bin/activate
which python
```

If the working tree is unexpectedly dirty, stop rather than using a destructive reset.

---

## 10. Current Measurement Status

| Quantity                                       | Status                                                           |
| ---------------------------------------------- | ---------------------------------------------------------------- |
| Raw agent actions                              | Available                                                        |
| Tool invocation count                          | Available                                                        |
| Individual tool duration                       | Available                                                        |
| Total recorded tool time                       | Available                                                        |
| Tool return/error metadata                     | Available                                                        |
| Output-size bytes                              | Available                                                        |
| Benchmark instance linkage                     | Available                                                        |
| Model-call count                               | Available in trajectory model stats                              |
| Tool/model step alignment                      | Validated on one SWE-Bench instance                              |
| Official benchmark correctness                 | Validated on one instance                                        |
| Semantic tool categories                       | Raw commands available; offline classification not yet performed |
| Frequency by semantic tool category            | Not yet computed                                                 |
| Execution-time statistics by semantic category | Not yet computed                                                 |
| Explicit LLM/model-request duration            | **Not yet instrumented**                                         |

Current rule:

> **Do not compute LLM time as `run wall time - summed tool time`.**

The first validated SWE-Bench run had `4.813115 s` of recorded tool time across 20 tool executions. The remaining wall-clock interval includes model inference plus other agent/orchestration work and cannot yet be labeled precisely as LLM time.
