# vLLM Warmup & Benchmark

Self-contained vLLM warmup and benchmark script. Measures TTFT (time to first token), throughput, and validates KV cache health.

## Quick Start

```bash
# Run the script (no pip install needed)
python3 scripts/vllm_warmup.py
```

## What It Does

The script runs 3 phases:

1. **Health Check** — waits up to 60s for vLLM to become ready
2. **Warmup** — fires 3 non-streaming requests to initialize CUDA kernels & KV cache
3. **Benchmark** — fires 3 streaming requests measuring TTFT, tokens/sec, total time

## Output Example

```
vLLM Warmup + Benchmark
========================================

[1/3] Checking server health...
  Server is ready.

[2/3] Sending warmup requests...
  Warmup 1/3: Say hello.
    -> Hello!

[3/3] Benchmarking (streaming, TTFT + throughput)...
  Bench 1/3...
    TTFT: 0.0719s  |  Tokens: 110  |  Time: 2.4962s  |  TPS: 44.07

========================================
BENCHMARK SUMMARY
  Avg TTFT : 0.0739s
  Avg TPS  : 44.21 tok/s
  Requests : 3
```

## Requirements

- Python 3.11+ (stdlib only, no pip dependencies)
- vLLM server running (default: `http://localhost:8000`)

## Configuration

Edit `BASE_URL` at the top of `scripts/vllm_warmup.py` to change the vLLM endpoint:

```python
BASE_URL = "http://localhost:8000/v1/chat/completions"
```

## Troubleshooting

- **Connection refused**: vLLM isn't running or on a different port
- **Timeout**: Server is slow to start; check GPU memory with `nvidia-smi`
- **Empty results**: Verify model name matches your deployed model

## License

MIT
