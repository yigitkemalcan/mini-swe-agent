# System metrics visualization

Create standalone graphs from one completed run without starting SWE-Bench,
vLLM, Docker, or a GPU workload:

```bash
python scripts/system_metrics_visualization/plot_system_metrics.py \
  /path/to/qwen-run-YYYYMMDDTHHMMSSZ-ID
```

By default, outputs go to
`scripts/system_metrics_visualization/results/<run-id>/`. Use `--output` to
choose another directory.

The generated `index.html` combines three standalone SVG graphs:

- `resource_timeline.svg`: host CPU, per-GPU compute utilization, memory
  occupancy, and model/tool intervals on one wall-clock timeline.
- `gpu_utilization_heatmap.svg`: one row per GPU and one column per sample.
- `run_summary.svg`: mean/maximum CPU and GPU utilization, sampled peak
  memory, and sampling coverage.

`visualization_summary.json` records the derived values and source path used to
make the graphs. The script validates timestamps, ranges, memory bounds, and
stable GPU identities before plotting. It uses only the Python standard
library; no plotting package is required.

CPU values are host-wide. GPU compute utilization is the NVML recent-window
metric, not exclusive CUDA kernel time. Model and tool activity bands are
aligned using wall-clock nanosecond timestamps. Memory maxima are sampled peaks
and can miss spikes shorter than the sampling interval.
