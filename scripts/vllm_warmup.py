#!/usr/bin/env python3
"""vLLM warmup + benchmark script."""

import json
import urllib.request
import urllib.error
import time
import sys

BASE_URL = "http://localhost:8000/v1/chat/completions"


def health_check(timeout: int = 60) -> bool:
    """Wait until vLLM is ready."""
    for attempt in range(timeout // 2):
        try:
            req = urllib.request.Request(
                "http://localhost:8000/health",
                method="GET",
            )
            urllib.request.urlopen(req, timeout=5)
            print("  Server is ready.")
            return True
        except Exception:
            time.sleep(2)
    print("  ERROR: Server not ready after 60s")
    return False


def benchmark_request(prompt: str, model: str, max_tokens: int = 256, long_form: bool = False) -> dict:
    """Stream a request and measure TTFT, total time, tokens/sec."""
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens if not long_form else 1024,
        "temperature": 0.7,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        BASE_URL,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    ttft = None
    token_count = 0
    start = time.time()

    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            text = line.decode().strip()
            if text.startswith("data: ") and text != "data: [DONE]":
                payload = json.loads(text[6:])
                delta = payload.get("choices", [{}])[0].get("delta", {})
                if delta:
                    if ttft is None:
                        ttft = time.time() - start
                    token_count += 1

    total_time = time.time() - start
    tps = token_count / total_time if total_time > 0 else 0

    return {
        "prompt": prompt,
        "ttft_s": round(ttft, 4),
        "total_s": round(total_time, 4),
        "tokens": token_count,
        "tokens_per_sec": round(tps, 2),
    }


def warmup(prompt: str, model: str = "", max_tokens: int = 10) -> str:
    """Non-streaming warmup — just fires one request."""
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        BASE_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
        msg = result["choices"][0]["message"]
        # Some models (e.g. Qwen3.6 with reasoning) put text in 'reasoning'
        text = msg.get("content") or msg.get("reasoning", "N/A")
        return text


if __name__ == "__main__":
    print("vLLM Warmup + Benchmark\n" + "=" * 40)

    # --- wait for readiness ---
    print("\n[1/3] Checking server health...")
    if not health_check():
        sys.exit(1)

    # --- warmup requests (no-streaming) ---
    print("\n[2/3] Sending warmup requests...")
    warmup_prompts = [
        "Say hello.",
        "What is the capital of France?",
        "Write a haiku about AI.",
    ]
    for i, p in enumerate(warmup_prompts, 1):
        print(f"  Warmup {i}/{len(warmup_prompts)}: {p}")
        try:
            resp = warmup(p)
            print(f"    -> {resp}")
        except Exception as e:
            print(f"    -> ERROR: {e}")

    # --- benchmark ---
    use_long = "--long" in sys.argv
    bench_type = "streaming + long-form" if use_long else "streaming (TTFT + throughput)"
    print(f"\n[3/3] Benchmarking ({bench_type})...")

    bench_prompts = [
        "Write a short story about a robot.",
        "Explain quantum computing in simple terms.",
        "Summarize the plot of Hamlet in 5 sentences.",
    ]

    long_prompts = [
        """Write a comprehensive technical analysis of machine learning model serving at production scale. 
        Cover the following topics in detail: (1) throughput optimization strategies including continuous batching and KV cache management, 
        (2) latency reduction techniques such as speculative decoding and prefix caching, (3) GPU memory optimization approaches 
        including quantization methods (INT8, INT4, FP8) and their accuracy trade-offs, (4) multi-GPU scaling patterns 
        including tensor parallelism and pipeline parallelism, (5) serving framework comparisons between vLLM, TGI, and TensorRT-LLM, 
        and (6) production monitoring and alerting best practices. Provide concrete numbers and examples where possible. 
        Aim for approximately 800-1000 words of thorough analysis. """,
        """Create a detailed guide for deploying large language models on edge devices like the NVIDIA Jetson Orin.
        Discuss hardware constraints (memory, compute, power), software stack choices (CUDA, TensorRT, llama.cpp),
        quantization strategies suitable for edge deployment, inference optimization techniques, and real-world latency benchmarks.
        Include specific model recommendations, memory profiling techniques, and thermal management considerations.
        Write approximately 800-1000 words covering all aspects. """,
        """Compose an extensive tutorial on building an AI-powered code review system.
        Describe the architecture from data collection through model serving, including: prompt engineering for code review,
        context window management for large codebases, evaluation methodologies for reviewing quality, 
        integration with CI/CD pipelines, handling different programming languages and frameworks,
        and strategies for reducing false positives and improving contextual understanding.
        Include practical examples and code snippets. Target 800-1000 words of comprehensive content. """,
    ]

    prompts = long_prompts if use_long else bench_prompts
    results = []
    for i, p in enumerate(prompts, 1):
        label = "Bench (long)" if use_long else f"Bench {i}/{len(prompts)}"
        print(f"  {label}...")
        try:
            r = benchmark_request(p, model="", long_form=use_long)
            r["idx"] = i
            results.append(r)
            print(f"    TTFT: {r['ttft_s']}s  |  Tokens: {r['tokens']}  |  "
                  f"Time: {r['total_s']}s  |  TPS: {r['tokens_per_sec']}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # --- summary ---
    print("\n" + "=" * 40)
    print("BENCHMARK SUMMARY")
    if results:
        avg_ttft = sum(r["ttft_s"] for r in results) / len(results)
        avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
        print(f"  Avg TTFT : {avg_ttft:.4f}s")
        print(f"  Avg TPS  : {avg_tps:.2f} tok/s")
        print(f"  Requests : {len(results)}")
    else:
        print("  No results (all requests failed).")

    print("Done.")
