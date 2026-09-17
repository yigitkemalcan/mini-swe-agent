#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash run-swebench.sh [--instance ID ... | --all | --filter REGEX] [--slice START:STOP] [--workers N] [--shuffle] [--system-metrics-interval SECONDS]

Runs SWE-Bench Verified, test split, using the existing local Qwen server.
Choose instances explicitly, or supply a slice. Default worker count: one.
Repeat --instance to select multiple exact IDs. Replace the example placeholders below.

Examples:
  bash run-swebench.sh --instance '<INSTANCE_ID>'
  bash run-swebench.sh --instance '<ID_1>' --instance '<ID_2>' --workers 2
  bash run-swebench.sh --filter '<REGEX>'
  bash run-swebench.sh --slice 0:5 --workers 2
  bash run-swebench.sh --all --slice 0:5
  bash run-swebench.sh --all --slice 5:10
  bash run-swebench.sh --all --slice 0:5 --shuffle
  bash run-swebench.sh --all --workers 1
  bash run-swebench.sh --all --workers 3

Slices are zero-based and exclude STOP. They apply after filtering.
--shuffle uses the runner's fixed seed (42), before filtering and slicing.
Every invocation creates a fresh run directory and prints its evaluation command.
Run-level CPU, host-memory, and eight-GPU metrics are sampled every 0.5 seconds by default.
HELP
}

filter=''
selection=''
instances=()
slice=''
workers=1
system_metrics_interval=0.5
extra=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --instance)
            [[ -z "$selection" || "$selection" == --instance ]] || { echo 'Do not combine --instance with --all or --filter.' >&2; exit 1; }
            selection=--instance
            instances+=("${2:?--instance requires an instance ID}")
            shift 2
            ;;
        --all|--filter)
            [[ -z "$selection" ]] || { echo 'Choose --instance, --all, or --filter; do not mix selectors.' >&2; exit 1; }
            selection="$1"
            if [[ "$1" == --all ]]; then
                filter=''
                shift
            else
                filter="${2:?--filter requires a regex}"
                shift 2
            fi
            ;;
        --slice) slice="${2:?--slice requires START:STOP}"; shift 2 ;;
        --workers|-w) workers="${2:?--workers requires a positive integer}"; shift 2 ;;
        --shuffle) extra+=(--shuffle); shift ;;
        --system-metrics-interval)
            system_metrics_interval="${2:?--system-metrics-interval requires seconds}"
            shift 2
            ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done
[[ -n "$selection" || -n "$slice" ]] || { usage >&2; exit 1; }
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo 'Workers must be a positive integer.' >&2; exit 1; }
[[ -z "$slice" || "$slice" =~ ^[0-9]*:[0-9]*$ ]] || { echo 'Use a slice such as 0:5 or 5:10.' >&2; exit 1; }
if [[ "$selection" == --instance ]]; then
    for instance in "${instances[@]}"; do
        if [[ ! "$instance" =~ ^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-[0-9]+$ ]]; then
            echo "Invalid exact instance ID: $instance" >&2
            echo 'Pass a plain ID to --instance (for example, django__django-11099); use --filter for regex.' >&2
            exit 1
        fi
    done
fi

project=/mnt/raid0/yigit/agent-characterization
agent_root="$project/mini-swe-agent"
source "$agent_root/.venv/bin/activate"
cd "$agent_root"
if [[ "$selection" == --instance ]]; then
    filter=$("$agent_root/.venv/bin/python" -c 'import re, sys; print("^(?:" + "|".join(map(re.escape, sys.argv[1:])) + ")$")' "${instances[@]}")
fi
curl --fail --silent --show-error --max-time 15 http://127.0.0.1:8000/v1/models |
    "$agent_root/.venv/bin/python" -c 'import json, sys; assert any(m["id"] == "Qwen/Qwen3-235B-A22B-Instruct-2507" for m in json.load(sys.stdin)["data"]), "Expected Qwen model is not served"'

mkdir -p "$project/swebench-runs"
run_dir=$(mktemp -d "$project/swebench-runs/qwen-run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
cat > "$run_dir/local-qwen.yaml" <<YAML
agent:
  run_id: $(basename "$run_dir")
model:
  model_name: hosted_vllm/Qwen/Qwen3-235B-A22B-Instruct-2507
  cost_tracking: ignore_errors
  model_kwargs:
    api_base: http://127.0.0.1:8000/v1
    num_retries: 0
YAML

sampler="$agent_root/scripts/sample_system_metrics.py"
"$agent_root/.venv/bin/python" "$sampler" --check --interval "$system_metrics_interval" --expected-gpus 8 >/dev/null
sampler_ready="$run_dir/.system_metrics.ready"
sampler_pid=''
cleanup_sampler() {
    status=$?
    trap - EXIT
    if [[ -n "$sampler_pid" ]]; then
        if kill -0 "$sampler_pid" 2>/dev/null; then
            kill -TERM "$sampler_pid" 2>/dev/null || true
            wait "$sampler_pid" || [[ "$status" -ne 0 ]] || status=1
        else
            wait "$sampler_pid" || true
            [[ "$status" -ne 0 ]] || status=1
        fi
    fi
    rm -f "$sampler_ready"
    exit "$status"
}
trap cleanup_sampler EXIT
nice -n 10 "$agent_root/.venv/bin/python" "$sampler" \
    "$run_dir/system_metrics.jsonl" \
    --summary "$run_dir/system_metrics_summary.json" \
    --interval "$system_metrics_interval" \
    --expected-gpus 8 \
    --parent-pid "$$" \
    --ready-file "$sampler_ready" &
sampler_pid=$!
for _ in {1..100}; do
    [[ -s "$sampler_ready" ]] && break
    kill -0 "$sampler_pid" 2>/dev/null || { wait "$sampler_pid"; exit 1; }
    sleep 0.05
done
[[ -s "$sampler_ready" ]] || { echo 'System metrics sampler did not become ready.' >&2; exit 1; }
rm -f "$sampler_ready"

command=("$agent_root/.venv/bin/mini-extra" swebench
    -c "$agent_root/src/minisweagent/config/benchmarks/swebench.yaml"
    -c "$run_dir/local-qwen.yaml"
    -m hosted_vllm/Qwen/Qwen3-235B-A22B-Instruct-2507
    -o "$run_dir/results" --subset verified --split test
    --filter "$filter" --slice "$slice" --workers "$workers")
if [[ ${#extra[@]} -gt 0 ]]; then command+=("${extra[@]}"); fi
printf '%q ' "${command[@]}" > "$run_dir/agent-command.txt"
printf '\n' >> "$run_dir/agent-command.txt"
echo "Run directory: $run_dir"
echo "After inspecting the run, evaluate in Terminal 2 with:"
printf 'bash %q %q\n' "$agent_root/evaluate-swebench.sh" "$run_dir"
"${command[@]}"
echo "Runner finished. Inspect exit statuses and tool events in $run_dir/results."
