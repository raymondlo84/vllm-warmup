---
name: vllm-warmup
description: Warm up vLLM engines with 10-phase comprehensive benchmarking.
version: 3.0.0
author: Raymond Lo (raymondlo84), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    skillDir: ~/.hermes/skills/vllm-warmup
    tags: [vllm, warmup, benchmark, cuda, jit, kv-cache]
---

# vLLM Warmup v3

Comprehensive vLLM warmup and benchmark covering CUDA graph capture, Triton JIT compilation, KV cache priming, streaming latency, and batch scheduling. 10 phases, ~20s total, zero pip dependencies.

## When to Use

- After restarting a vLLM server — prime all inference paths before production traffic
- Before benchmarking — skip cold-start skew with `--num-warmups`-style preflight
- After model updates — re-capture all CUDA graphs and JIT kernels
- When diagnosing latency regressions — compare phase-by-phase stats over time
- Don't use for: production load testing (not a stress test; use `vllm bench serve` instead)

## Prerequisites

- vLLM server running and reachable (default `http://localhost:8000`)
- Python 3.11+ (stdlib only — `urllib`, `concurrent.futures`, no pip deps)
- `gh` CLI installed if you want to push results to GitHub (optional)

## How to Run

### Full warmup (all 10 phases, ~20s)

```bash
python3 scripts/vllm_warmup_v3.py
```

### Dry run (preview, no network calls)

```bash
python3 scripts/vllm_warmup_v3.py --dry-run
```

### Single phase

```bash
python3 scripts/vllm_warmup_v3.py --phase 6
```

### Remote server

```bash
python3 scripts/vllm_warmup_v3.py --base-url http://remote:8000
```

## Phases

| # | Phase | What it exercises | Runs |
|---|---|---|---|
| 1 | Triton JIT Warmup | Kernel compilation (minimal prefill→decode) | 3 |
| 2 | CUDA Graph Prefill | Prefill shapes (64→4096 tokens) | 6 |
| 3 | CUDA Graph Decode | Decode shapes (32→512 tokens) | 4 |
| 4 | Streaming TTFT | Time-to-first-token measurement | 4 |
| 5 | Chat API | Multi-turn + reasoning paths | 4 |
| 6 | Concurrent Batch Scheduler | Ramp 2→4→8→16 concurrent | 30 |
| 7 | Chat Streaming | Realistic multi-turn with streaming | 3 |
| 8 | Sampling | Temperature × top-p grid (5×2) | 10 |
| 9 | CUDA Graph Verify | Pre- vs post-capture comparison | 2 |
| 10 | KV Cache Reuse | Repeated identical prompts | 3 |

**Total: 69 requests in ~20s**

## Output

Prints per-request timing + a summary table (N, avg, p50, p99, TPS). Saves:

- `results/YYYYMMDD_HHMMSS/warmup_report.json` — structured JSON with all data
- `results/YYYYMMDD_HHMMSS/warmup_log.txt` — human-readable log

Each run gets a new timestamped folder. Results are gitignored.

## Quick Reference

```bash
python3 scripts/vllm_warmup_v3.py               # all phases
python3 scripts/vllm_warmup_v3.py --dry-run     # preview only
python3 scripts/vllm_warmup_v3.py --phase 6     # batch scheduler only
python3 scripts/vllm_warmup_v3.py --base-url URL  # remote server
python3 scripts/vllm_warmup_v3.py --model MODEL  # override model name
```

## Pitfalls

- **Streaming TTFT shows 0 completion_tokens on some vLLM versions** — the tokenizer counts from `text`/`reasoning` chunks, but some servers omit these. TTFT timing is still accurate.
- **Batch scheduler latency spikes at higher concurrency** — expected. The warmup exposes where the scheduler stalls. Use this data to tune `--max-concurrency` in production.
- **CUDA Graph verify (phase 9) may show slower "post" time** — GPU memory pressure from earlier phases can cause this. Run in isolation (`--phase 9`) if you need clean numbers.
- **KV cache reuse (phase 10) measures warm vs warm, not cold vs warm** — the KV cache stays allocated between runs. The metric shows consistency, not cold-start cost.

## Verification

1. Script exits with code 0 and prints `TOTAL: 69 runs` in the summary
2. `results/` contains a new timestamped folder with `warmup_report.json` and `warmup_log.txt`
3. All phases show `N > 0` in the summary table (no silent failures)
4. No `ERROR:` lines in the per-request output
