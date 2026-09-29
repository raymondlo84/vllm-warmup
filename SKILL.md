# vLLM Warmup & Benchmark

## When to use

Use when starting or restarting a vLLM server and needing to (1) warm up CUDA kernels / KV cache before production traffic, and (2) measure TTFT (time to first token) and tokens/sec for baseline performance.

## Prerequisites

- vLLM server running and reachable at `http://localhost:8000`
- Python 3.11+ (no extra pip deps — uses stdlib only)

## Quick start

```bash
# Run the bundled script (self-contained, no pip install needed)
python3 scripts/vllm_warmup.py
```

## Configuration

The script connects to `http://localhost:8000` by default (the vLLM OpenAI-compatible API port). To change the port or host, edit the `BASE_URL` variable at the top of the script:

```python
BASE_URL = "http://localhost:8000/v1/chat/completions"
```

## What it does

The script runs 3 phases sequentially:

### Phase 1: Health check
Sends GET to `/health` and waits up to 60s for vLLM to become ready.

### Phase 2: Warmup
Fires 3 non-streaming chat requests (short prompts) to initialize:
- CUDA kernels
- KV cache blocks
- Any model-specific internal state

### Phase 3: Benchmark
Fires 3 streaming chat requests and measures:
- **TTFT** (Time To First Token) — latency before first token arrives
- **Total generation time** — full response duration
- **Tokens/sec** — throughput

Outputs a summary with averages.

## Output example

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
  ...

========================================
BENCHMARK SUMMARY
  Avg TTFT : 0.0739s
  Avg TPS  : 44.21 tok/s
  Requests : 3
Done.
```

## Customization

### Custom prompts
Edit the `warmup_prompts` and `bench_prompts` lists in the script's `__main__` block.

### Custom parameters
- `max_tokens` in `benchmark_request()` (default: 256) — control output length
- `temperature` (default: 0.5) — control randomness
- Number of warmup/bench requests — adjust the list lengths

### Reasoning models
The script handles models that return content in the `reasoning` field instead of `content` (e.g. Qwen3.6). No special config needed.

## Troubleshooting

**Connection refused**: vLLM server isn't running or isn't on the configured port. Start it first:
```bash
vllm serve MODEL --port 8000
```

**Timeout**: Server is slow to respond (first request after cold start). Give it more time or check GPU memory with `nvidia-smi`.

**Empty results**: Check that the model name matches your deployed model. Some models require the `model` field to be set in requests.

## Files

| File | Purpose |
|------|---------|
| `scripts/vllm_warmup.py` | Main warmup + benchmark script |
