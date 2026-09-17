#!/usr/bin/env bash
set -euo pipefail

[[ $# == 1 ]] || { echo 'Usage: install-vllm-instrumentation.sh VLLM_ENV' >&2; exit 1; }
env_root=$1
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
agent_root=$(cd "$script_dir/.." && pwd -P)
python="$env_root/bin/python"

[[ -x "$python" ]] || { echo "Missing vLLM Python: $python" >&2; exit 1; }
version=$($python -c 'from importlib.metadata import version; print(version("vllm"))')
[[ "$version" == 0.10.1.1 ]] || { echo "Expected vLLM 0.10.1.1, found $version" >&2; exit 1; }

site_packages=$($python -c 'import site; print(site.getsitepackages()[0])')
target="$site_packages/vllm/v1/engine/output_processor.py"
install -m 0644 "$agent_root/instrumentation/vllm_request_metrics.py" "$site_packages/vllm/request_metrics_jsonl.py"

if ! grep -Fqx 'from vllm.request_metrics_jsonl import log_finished_request_metrics' "$target"; then
    patch --batch --forward -d "$site_packages" -p1 \
        < "$agent_root/instrumentation/vllm-0.10.1.1-request-metrics-import.patch"
fi
if ! grep -Fq '        log_finished_request_metrics(' "$target"; then
    patch --batch --forward -d "$site_packages" -p1 \
        < "$agent_root/instrumentation/vllm-0.10.1.1-request-metrics.patch"
fi
