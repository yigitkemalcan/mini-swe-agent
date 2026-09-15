#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash run-swebench.sh [--instance ID ... | --all | --filter REGEX] [--slice START:STOP] [--workers N] [--shuffle]

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
HELP
}

filter=''
selection=''
instances=()
slice=''
workers=1
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
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done
[[ -n "$selection" || -n "$slice" ]] || { usage >&2; exit 1; }
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo 'Workers must be a positive integer.' >&2; exit 1; }
[[ -z "$slice" || "$slice" =~ ^[0-9]*:[0-9]*$ ]] || { echo 'Use a slice such as 0:5 or 5:10.' >&2; exit 1; }

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
cat > "$run_dir/local-qwen.yaml" <<'YAML'
model:
  model_name: hosted_vllm/Qwen/Qwen3-235B-A22B-Instruct-2507
  cost_tracking: ignore_errors
  model_kwargs:
    api_base: http://127.0.0.1:8000/v1
YAML

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
