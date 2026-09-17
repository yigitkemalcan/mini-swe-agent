#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash start-vllm.sh'
    echo 'Run inside tmux in Terminal 1. Uses the existing Qwen/vLLM environment.'
    exit 0
fi
[[ $# == 0 ]] || { echo 'Unexpected arguments; use --help.' >&2; exit 1; }

qwen_root=/mnt/experiment-yigit/qwen-experiment
agent_root=/mnt/raid0/yigit/agent-characterization/mini-swe-agent
export HF_HOME="$qwen_root/huggingface-cache"
export VLLM_CACHE_ROOT="$qwen_root/vllm-cache"
export UV_CACHE_DIR="$qwen_root/uv-cache"
export MSWEA_VLLM_EVENT_LOG=/mnt/raid0/yigit/agent-characterization/vllm-logs/request_events.jsonl
snapshot="$HF_HOME/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e"

[[ -d "$snapshot" ]] || { echo "Missing model snapshot: $snapshot" >&2; exit 1; }
if [[ -n $(ss -H -ltn 'sport = :8000') ]]; then
    echo 'Port 8000 is already listening. Check the existing server before starting another.' >&2
    exit 1
fi

source "$qwen_root/envs/vllm/bin/activate"
cd "$qwen_root"
bash "$agent_root/scripts/install-vllm-instrumentation.sh" "$qwen_root/envs/vllm"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
exec "$qwen_root/envs/vllm/bin/vllm" serve "$snapshot" \
    --served-model-name Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --host 127.0.0.1 \
    --port 8000
