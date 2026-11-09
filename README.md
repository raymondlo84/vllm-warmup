# vLLM Warmup & Benchmark

Comprehensive vLLM warmup script. 10 phases covering CUDA graph capture, Triton JIT compilation, KV cache priming, streaming latency, and batch scheduling. Zero pip dependencies.

## Quick Start

```bash
# Run all 10 phases (~20s)
python3 scripts/vllm_warmup_v3.py

# Dry run — preview without network calls
python3 scripts/vllm_warmup_v3.py --dry-run

# Single phase
python3 scripts/vllm_warmup_v3.py --phase 6

# Remote server
python3 scripts/vllm_warmup_v3.py --base-url http://remote:8000
```

## What It Does

The script runs **10 phases** (69 requests total) to exhaustively warm up a vLLM engine:

| # | Phase | Exercises |
|---|---|---|
| 1 | Triton JIT Warmup | Kernel compilation |
| 2 | CUDA Graph Prefill | Prefill shapes (64→4096 tokens) |
| 3 | CUDA Graph Decode | Decode shapes (32→512 tokens) |
| 4 | Streaming TTFT | Time-to-first-token measurement |
| 5 | Chat API | Multi-turn + reasoning paths |
| 6 | Concurrent Batch Scheduler | Ramp 2→4→8→16 concurrent |
| 7 | Chat Streaming | Realistic multi-turn streaming |
| 8 | Sampling | Temperature × top-p grid |
| 9 | CUDA Graph Verify | Pre- vs post-capture comparison |
| 10 | KV Cache Reuse | Repeated identical prompts |

## Output Example

```
================================================================================
  TOTAL: 69 runs  |  19.5s  |  nvidia/Qwen3.6-35B-A3B-NVFP4
================================================================================
  Phase                                  N     Avg ms     P50 ms     P99 ms      TPS
  ------------------------------------  ---   --------   --------   --------   ------
  1. Triton JIT Warmup                    3       82.3       87.2       87.3     49.0
  2. CUDA Graph Prefill (lengths)         6      203.0      232.8      265.6     43.1
  3. CUDA Graph Decode (lengths)          4     1489.8     1453.2     3273.8    146.7
  4. Streaming TTFT (lengths)             4      514.5      559.0      561.1     38.2
  5. Chat API (multi-turn + reasoning)    4      345.9      389.4      429.4     97.6
  6. Concurrent Batch Scheduler          30      515.2      547.5     1029.4     41.2
  7. Chat Streaming (realistic)           3      555.6      562.9      608.5     37.5
  8. Sampling (temp/top-p variation)     10      263.6      253.8      338.5    123.1
  9. CUDA Graph Warmup → Verify           2      254.2      264.7      264.7    125.0
  10. KV Cache Reuse                      3      395.9      399.5      404.5    164.5
================================================================================
```

## Requirements

- Python 3.11+ (stdlib only, no pip dependencies)
- vLLM server running (default: `http://localhost:8000`)

## Output Storage

Each run saves timestamped results to `results/YYYYMMDD_HHMMSS/`:

- `warmup_report.json` — structured JSON (loadable by scripts)
- `warmup_log.txt` — human-readable log

The `results/` directory is gitignored.

## Using as a Skill

This repo includes `SKILL.md` for use with Hermes Agent and OpenClaw:

1. Read `SKILL.md` to understand what it provides
2. Copy into your workspace:
   ```bash
   # For Hermes Agent
   mkdir -p ~/.hermes/skills/vllm-warmup
   cp -r ~/Desktop/vllm_warmup_v2/* ~/.hermes/skills/vllm-warmup/

   # For OpenClaw
   cp -r ~/Desktop/vllm_warmup_v2/* ~/.openclaw/skills/vllm-warmup/
   ```
3. Reference it in your `AGENTS.md` or skill index

## License

MIT
