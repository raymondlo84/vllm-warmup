#!/usr/bin/env python3
"""
Comprehensive vLLM Warmup v3 — CUDA Graph + Triton JIT + Concurrency + KV Cache

Exercises all major inference paths per vLLM docs (CUDA Graph capture,
Triton JIT compilation, KV cache priming, batch scheduling, streaming).

Usage:
    python3 run_warmup.py                      # full run with defaults
    python3 run_warmup.py --dry-run            # preview without requests
    python3 run_warmup.py --target-dir /tmp/wu # custom output dir
    python3 run_warmup.py --base-url URL       # remote server

Writes per-run results to:  <target_dir>/results/YYYYMMDD_HHMMSS/
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

# ── Constants ──────────────────────────────────────────────────────────

DEFAULT_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
DEFAULT_URL = "http://localhost:8000"
DEFAULT_TARGET = Path.home() / "Desktop" / "vllm_warmup_v2"
REQUEST_TIMEOUT = 120
HEARTBEAT_INTERVAL = 0.1  # seconds — show progress every N s


class HeartbeatTimer:
    """Print dots to stderr every HEARTBEAT_INTERVAL s while work runs."""

    def __init__(self) -> None:
        self._stop_event = False
        self._thread: Optional[Thread] = None

    # import inside method to avoid top-level circular import at parse time
    def start(self, max_duration: float = 300.0) -> None:  # noqa: A003
        import threading
        self._stop_event = False

        def _pulse() -> None:
            start_t = time.monotonic()
            while not self._stop_event and (time.monotonic() - start_t) < max_duration:
                time.sleep(HEARTBEAT_INTERVAL)
                if not self._stop_event:
                    sys.stderr.write(".")
                    sys.stderr.flush()
                    time.sleep(0.3)

        self._thread = threading.Thread(target=_pulse, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event = True
        if self._thread:
            self._thread.join(timeout=1)


# ── HTTP helpers ───────────────────────────────────────────────────────

def _request_json(
    url: str,
    payload: Dict[str, Any],
    timeout: int = REQUEST_TIMEOUT,
) -> Dict[str, Any]:
    """POST JSON and return the response body dict.  Raises on non-2xx."""
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
    )
    resp = urlopen(req, timeout=timeout)
    return json.loads(resp.read())


def _stream_request(
    url: str,
    payload: Dict[str, Any],
    timeout: int = REQUEST_TIMEOUT,
) -> Tuple[Dict[str, Any], float, float]:
    """Streaming POST → returns (usage_estimate, ttft_ms, total_ms)."""
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    })
    import io
    resp = urlopen(req, timeout=timeout)
    reader = io.TextIOWrapper(resp, encoding="utf-8", errors="replace")

    start = time.monotonic()
    ttft_ms: float = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for line in reader:
        line = line.rstrip("\n\r")
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk_obj = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        # Extract usage if present (some vLLM versions include it)
        if "usage" in chunk_obj:
            u = chunk_obj["usage"]
            prompt_tokens = u.get("prompt_tokens", prompt_tokens)
            completion_tokens = u.get("completion_tokens", completion_tokens)
            total_tokens = u.get("total_tokens", completion_tokens + prompt_tokens)
        # Count tokens from text chunks
        if ttft_ms == 0.0 and chunk_obj.get("choices"):
            ttft_ms = (time.monotonic() - start) * 1000
        # Count tokens from text chunks (handle chat `reasoning`/`content` and completion `text`)
        for choice in chunk_obj.get("choices", []):
            delta = choice.get("delta", {})
            text = delta.get("content", delta.get("text", ""))
            if not text:
                text = delta.get("reasoning", "")  # Chat models embed text in 'reasoning'
            if not text:
                text = choice.get("text", "")  # Completion streaming uses top-level 'text'
            if text:
                completion_tokens += 1

    usage: Dict[str, Any] = {}
    if prompt_tokens > 0 or completion_tokens > 0:
        usage["prompt_tokens"] = prompt_tokens
        usage["completion_tokens"] = completion_tokens
        usage["total_tokens"] = prompt_tokens + completion_tokens

    return usage, ttft_ms, (time.monotonic() - start) * 1000


# ── Phases ─────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    name: str = ""
    phase: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    elapsed_ms: float = 0.0
    ttft_ms: float = 0.0  # time-to-first-token (streaming)
    tps: float = 0.0  # tokens/sec
    finish_reason: str = ""
    error: str = ""


def _safe_run(fn, *args, **kwargs):  # noqa: ANN002
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


# ── Phase implementations ─────────────────────────────────────────────

def _make_completion_req(
    base_url: str, model: str, prompt: str,
    max_tokens: int = 32, temperature: float = 0.0, stream: bool = False,
) -> Dict[str, Any]:
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    return {"url": url, "payload": payload}


def _make_chat_req(
    base_url: str, model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 32, temperature: float = 0.0, stream: bool = False,
) -> Dict[str, Any]:
    url = f"{base_url}/v1/chat/completions"
    return {
        "url": url,
        "payload": {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        },
    }


def _run_single(url: str, payload: Dict[str, Any]) -> RunResult:
    """Non-streaming POST → return RunResult."""
    start = time.monotonic()
    try:
        resp = _request_json(url, payload)
        elapsed = (time.monotonic() - start) * 1000
        usage = resp.get("usage", {})
        choice = (resp.get("choices") or [{}])[0]
        ct = usage.get("completion_tokens", 0)
        return RunResult(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=ct,
            total_tokens=usage.get("total_tokens", 0),
            elapsed_ms=round(elapsed, 1),
            tps=round(ct / (elapsed / 1000), 1) if elapsed > 0 else 0,
            finish_reason=choice.get("finish_reason", "unknown"),
        )
    except Exception as exc:
        elapsed = (time.monotonic() - start) * 1000
        return RunResult(
            elapsed_ms=round(elapsed, 1),
            error=str(exc)[:200],
        )


def _run_stream_single(url: str, payload: Dict[str, Any]) -> RunResult:
    """Streaming POST → returns TTFT + usage from last chunk."""
    start = time.monotonic()
    try:
        usage, ttft, total_ms = _stream_request(url, payload)
        ct = usage.get("completion_tokens", 0)
        pt = usage.get("prompt_tokens", 0)
        tt = usage.get("total_tokens", ct + pt)
        return RunResult(
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=tt,
            elapsed_ms=round(total_ms, 1),
            ttft_ms=round(ttft, 1),
            tps=round(ct / (total_ms / 1000), 1) if total_ms > 0 else 0,
            reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
        )
    except Exception as exc:
        return RunResult(elapsed_ms=round((time.monotonic() - start) * 1000, 1), error=str(exc)[:200])


# ── Phase runners ─────────────────────────────────────────────────────

class PhaseContext:
    """Shared state across a phase."""

    def __init__(self, base_url: str, model: str, dry_run: bool) -> None:  # noqa: ANN001
        self.base_url = base_url
        self.model = model
        self.dry_run = dry_run
        self.results: List[RunResult] = []
        self.hb = HeartbeatTimer()
        self.phase_ctx_total_elapsed: float = 0.0


def _dry(name: str, prompt: int = 0, completion: int = 0) -> RunResult:  # noqa: ANN001
    return RunResult(name=name, prompt_tokens=prompt, completion_tokens=completion, elapsed_ms=0, tps=0)


def phase_1_jit(ctx: PhaseContext) -> List[RunResult]:
    """Phase 1 — Triton JIT kernel compilation (warmup.py: minimal prefill+decode)."""
    name = "1. Triton JIT Warmup"
    ctx.hb.start(max_duration=10)
    if ctx.dry_run:
        return [_dry("jit_min", 1, 4) for _ in range(3)]
    results: List[RunResult] = []
    for i in range(3):
        req = _make_completion_req(ctx.base_url, ctx.model, "Hi", max_tokens=4, temperature=0.0)
        r = _run_single(req["url"], req["payload"])
        r.name = f"jit_min_{i+1}"
        r.phase = name
        results.append(r)
    ctx.hb.stop()
    ctx.results.extend(results)
    return results


def phase_2_prefill_lengths(ctx: PhaseContext) -> List[RunResult]:
    """Phase 2 — CUDA Graph prefill capture (varying input lengths)."""
    name = "2. CUDA Graph Prefill (lengths)"
    ctx.hb.start(max_duration=30)
    if ctx.dry_run:
        return [_dry(f"prefill_{n}", n, 8) for n in [64, 256, 512, 1024, 2048, 4096]]
    results: List[RunResult] = []
    for plen in [64, 256, 512, 1024, 2048, 4096]:
        prompt = "Hello world. " * plen
        req = _make_completion_req(ctx.base_url, ctx.model, prompt, max_tokens=8, temperature=0.0)
        r = _run_single(req["url"], req["payload"])
        r.name = f"prefill_{plen}"
        r.phase = name
        r.input_tokens = plen
        results.append(r)
    ctx.hb.stop()
    ctx.results.extend(results)
    return results


def phase_3_decode_lengths(ctx: PhaseContext) -> List[RunResult]:
    """Phase 3 — CUDA Graph decode capture (varying output lengths)."""
    name = "3. CUDA Graph Decode (lengths)"
    ctx.hb.start(max_duration=60)
    if ctx.dry_run:
        return [_dry(f"decode_{n}", 5, n) for n in [32, 128, 256, 512]]
    results: List[RunResult] = []
    for olen in [32, 128, 256, 512]:
        req = _make_completion_req(ctx.base_url, ctx.model, "Write a detailed explanation.", max_tokens=olen, temperature=0.0)
        r = _run_single(req["url"], req["payload"])
        r.name = f"decode_{olen}"
        r.phase = name
        results.append(r)
    ctx.hb.stop()
    ctx.results.extend(results)
    return results


def phase_4_streaming(ctx: PhaseContext) -> List[RunResult]:
    """Phase 4 — Streaming TTFT measurement."""
    name = "4. Streaming TTFT (lengths)"
    ctx.hb.start(max_duration=60)
    if ctx.dry_run:
        return [_dry(f"stream_{n}", 0, 0) for n in [32, 128, 256, 512]]
    results: List[RunResult] = []
    for plen in [32, 128, 256, 512]:
        prompt = "System warmup request. Process this. " * plen
        req = _make_completion_req(ctx.base_url, ctx.model, prompt, max_tokens=64, temperature=0.0, stream=True)
        r = _run_stream_single(req["url"], req["payload"])
        r.name = f"stream_{plen}"
        r.phase = name
        r.input_tokens = plen
        results.append(r)
    ctx.hb.stop()
    ctx.results.extend(results)
    return results


def phase_5_chat(ctx: PhaseContext) -> List[RunResult]:
    """Phase 5 — Chat API (system/user/assistant + reasoning)."""
    name = "5. Chat API (multi-turn + reasoning)"
    ctx.hb.start(max_duration=60)
    if ctx.dry_run:
        return [_dry(f"chat_{n}", 0, 0) for n in [32, 256, 512, 1024]]
    results: List[RunResult] = []
    for plen in [32, 256, 512, 1024]:
        ctx_text = "This is context. " * plen
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": ctx_text + " Summarize the above text."},
        ]
        req = _make_chat_req(ctx.base_url, ctx.model, msgs, max_tokens=32, temperature=0.7)
        r = _run_single(req["url"], req["payload"])
        r.name = f"chat_{plen}"
        r.phase = name
        r.input_tokens = plen
        results.append(r)
    ctx.hb.stop()
    ctx.results.extend(results)
    return results


def phase_6_concurrent_batch(ctx: PhaseContext) -> List[RunResult]:
    """Phase 6 — Concurrent batch scheduling (ramp up 1→N)."""
    name = "6. Concurrent Batch Scheduler"
    ctx.hb.start(max_duration=60)
    if ctx.dry_run:
        return [_dry(f"conc_{i}", 9, 16) for i in range(8)]
    results: List[RunResult] = []
    prompt_text = "Concurrent warmup request. Process now."
    sizes = [2, 4, 8, 16]

    for size in sizes:
        with ThreadPoolExecutor(max_workers=size) as pool:
            futures = [
                pool.submit(
                    _run_single,
                    _make_completion_req(ctx.base_url, ctx.model, prompt_text, max_tokens=16)["url"],
                    _make_completion_req(ctx.base_url, ctx.model, prompt_text, max_tokens=16)["payload"],
                )
                for _ in range(size)
            ]
            for i, fut in enumerate(as_completed(futures)):
                r = fut.result()
                r.name = f"batch_{size}_run_{i+1}"
                r.phase = name
                r.input_tokens = 9
                results.append(r)
    ctx.hb.stop()
    ctx.results.extend(results)
    return results


def phase_7_chat_stream(ctx: PhaseContext) -> List[RunResult]:
    """Phase 7 — Chat streaming (realistic multi-turn conversation)."""
    name = "7. Chat Streaming (realistic multi-turn)"
    ctx.hb.start(max_duration=60)
    if ctx.dry_run:
        return [_dry(f"chat_stream_{n}", 0, 0) for n in [128, 512, 2048]]
    results: List[RunResult] = []
    for plen in [128, 512, 2048]:
        ctx_text = "Context paragraph. " * plen
        msgs = [
            {"role": "system", "content": "You are a technical expert."},
            {"role": "user", "content": ctx_text + " What is the key takeaway?"},
        ]
        req = _make_chat_req(ctx.base_url, ctx.model, msgs, max_tokens=64, temperature=0.5, stream=True)
        r = _run_stream_single(req["url"], req["payload"])
        r.name = f"chat_stream_{plen}"
        r.phase = name
        r.input_tokens = plen
        results.append(r)
    ctx.hb.stop()
    ctx.results.extend(results)
    return results


def phase_8_sampling(ctx: PhaseContext) -> List[RunResult]:
    """Phase 8 — Different sampling strategies (temperature/top-p)."""
    name = "8. Sampling (temp/top-p variation)"
    ctx.hb.start(max_duration=30)
    if ctx.dry_run:
        return [
            _dry(f"sample_t{t}_p{p}", 10, 32)
            for t in [0.0, 0.3, 0.7, 0.95, 1.0]
            for p in [0.5, 1.0]
        ]
    results: List[RunResult] = []
    for temp in [0.0, 0.3, 0.7, 0.95, 1.0]:
        for top_p in [0.5, 1.0]:
            req = _make_completion_req(
                ctx.base_url, ctx.model,
                "Generate creative text.",
                max_tokens=32, temperature=temp,
            )
            payload = req["payload"]
            payload["top_p"] = top_p
            r = _run_single(req["url"], payload)
            r.name = f"sample_t{temp}_p{top_p}"
            r.phase = name
            results.append(r)
    ctx.hb.stop()
    ctx.results.extend(results)
    return results


def phase_9_cudagraph_verify(ctx: PhaseContext) -> List[RunResult]:
    """Phase 9 — CUDA Graph warmup → verify speedup (graph-cached should be faster)."""
    name = "9. CUDA Graph Warmup → Verify"
    ctx.hb.start(max_duration=30)
    if ctx.dry_run:
        return [
            _dry("cudagraph_pre", 50, 32),
            _dry("cudagraph_post", 50, 32),
        ]
    results: List[RunResult] = []
    prompt_text = "Cuda graph verification request. " * 50

    # Pre-warmup: force graph capture
    req = _make_completion_req(ctx.base_url, ctx.model, prompt_text, max_tokens=32)
    r1 = _run_single(req["url"], req["payload"])
    r1.name = "cudagraph_pre"
    r1.phase = name
    results.append(r1)

    # Post-warmup: graph should be captured → should be similar or faster
    r2 = _run_single(req["url"], req["payload"])
    r2.name = "cudagraph_post"
    r2.phase = name
    results.append(r2)

    ctx.hb.stop()
    ctx.results.extend(results)
    return results


def phase_10_kv_cache(ctx: PhaseContext) -> List[RunResult]:
    """Phase 10 — KV cache reuse warmup (repeat similar prompts)."""
    name = "10. KV Cache Reuse"
    ctx.hb.start(max_duration=60)
    if ctx.dry_run:
        return [
            _dry("kv_cache_1", 20, 64),
            _dry("kv_cache_2", 20, 64),
            _dry("kv_cache_3", 20, 64),
        ]
    results: List[RunResult] = []
    prompt_text = "KV cache warmup test request. " * 20

    # First run — populates KV cache
    req = _make_completion_req(ctx.base_url, ctx.model, prompt_text, max_tokens=64)
    r1 = _run_single(req["url"], req["payload"])
    r1.name = "kv_cache_first"
    r1.phase = name
    results.append(r1)

    # Repeated runs — should be similar latency (same prompt → same path)
    for i in range(2, 4):
        r = _run_single(req["url"], req["payload"])
        r.name = f"kv_cache_{i}"
        r.phase = name
        results.append(r)

    ctx.hb.stop()
    ctx.results.extend(results)
    return results


# Registry
PHASES: List[Tuple[str, callable]] = [
    ("1. Triton JIT Warmup", phase_1_jit),
    ("2. CUDA Graph Prefill (lengths)", phase_2_prefill_lengths),
    ("3. CUDA Graph Decode (lengths)", phase_3_decode_lengths),
    ("4. Streaming TTFT (lengths)", phase_4_streaming),
    ("5. Chat API (multi-turn + reasoning)", phase_5_chat),
    ("6. Concurrent Batch Scheduler", phase_6_concurrent_batch),
    ("7. Chat Streaming (realistic)", phase_7_chat_stream),
    ("8. Sampling (temp/top-p)", phase_8_sampling),
    ("9. CUDA Graph Verify", phase_9_cudagraph_verify),
    ("10. KV Cache Reuse", phase_10_kv_cache),
]

# ── Stats & Summary ────────────────────────────────────────────────────

def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0, "p50": 0, "avg": 0, "p95": 0, "p99": 0, "max": 0, "count": 0}
    s = sorted(values)
    n = len(s)
    return {
        "min": round(s[0], 1),
        "p50": round(s[n // 2], 1),
        "avg": round(sum(s) / n, 1),
        "p95": round(s[min(int(n * 0.95), n - 1)], 1),
        "p99": round(s[min(int(n * 0.99), n - 1)], 1),
        "max": round(s[-1], 1),
        "count": n,
    }


def print_summary(ctx: PhaseContext) -> None:
    """Print a compact summary table with latency + TPS stats."""
    # Group by phase
    by_phase: Dict[str, List[RunResult]] = {}
    for r in ctx.results:
        by_phase.setdefault(r.phase, []).append(r)

    total_runs = len(ctx.results)
    total_elapsed = main_ctx.total_elapsed
    print(f"\n{'=' * 80}")
    print(f"  TOTAL: {total_runs} runs  |  {total_elapsed:.1f}s  |  {main_ctx.server_model}")
    print(f"{'=' * 80}")
    print(f"  {'Phase':<36} {'N':>4} {'Avg ms':>10} {'P50 ms':>10} {'P99 ms':>10} {'TPS':>8}")
    print(f"  {'-' * 36} {'---':>4} {'--------':>10} {'--------':>10} {'--------':>10} {'------':>8}")

    for name, results_list in by_phase.items():
        elapsed_vals = [r.elapsed_ms for r in results_list if r.elapsed_ms > 0]
        tps_vals = [r.tps for r in results_list if r.tps > 0]
        st = _stats(elapsed_vals)
        tps_st = _stats(tps_vals)
        print(
            f"  {name:<36} {st['count']:>4} {st['avg']:>10.1f} {st['p50']:>10.1f} {st['p99']:>10.1f} {tps_st['avg']:>8.1f}"
        )

    print(f"{'=' * 80}")


# ── Main ───────────────────────────────────────────────────────────────

class MainCtx:
    """Global context for the main run."""
    server_model: str = ""
    server_url: str = ""
    total_elapsed: float = 0.0


main_ctx = MainCtx()


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive vLLM Warmup v3")
    parser.add_argument("--base-url", default=DEFAULT_URL, help="vLLM API base URL")
    parser.add_argument("--model", default=None, help="Model name (default: auto-detect)")
    parser.add_argument("--target-dir", default=None, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--phase", default=None, help="Run only this phase by number (1-10)")
    args = parser.parse_args()

    # Detect server model
    print("Checking vLLM server...")
    try:
        model_info = _request_json(f"{args.base_url}/v1/models", {})
        server_model = model_info["data"][0]["id"] if model_info.get("data") else "unknown"
    except Exception:
        server_model = DEFAULT_MODEL

    model = args.model or server_model
    print(f"  Server: {args.base_url}")
    print(f"  Model:  {server_model}\n")

    main_ctx.server_model = server_model
    main_ctx.server_url = args.base_url

    # Filter phases
    phases_to_run = PHASES
    if args.phase:
        try:
            phase_num = int(args.phase)
            phases_to_run = [p for p in PHASES if p[0].startswith(f"{phase_num}.")]
            if not phases_to_run:
                print(f"ERROR: Phase {phase_num} not found. Available: {', '.join(p[0] for p in PHASES)}")
                sys.exit(1)
        except ValueError:
            print(f"ERROR: --phase must be a number (1-{len(PHASES)}), got '{args.phase}'")
            sys.exit(1)

    ctx = PhaseContext(args.base_url, model, args.dry_run)
    start = time.monotonic()

    for desc, run_fn in phases_to_run:
        print(f"\n[{desc}]")
        sys.stdout.flush()
        results = run_fn(ctx)
        for r in results:
            tag = f"{r.prompt_tokens}p→{r.completion_tokens}c @ {r.elapsed_ms:.0f}ms"
            if r.ttft_ms > 0:
                tag += f"  (TTFT {r.ttft_ms:.0f}ms)"
            if r.error:
                tag = f"ERROR: {r.error[:80]}"
            print(f"  {r.name}: {tag}")
        sys.stdout.flush()

    ctx.phase_ctx_total_elapsed = (time.monotonic() - start)
    main_ctx.total_elapsed = ctx.phase_ctx_total_elapsed

    print_summary(ctx)

    if args.dry_run:
        print("\n[Dry run — no results saved]\n")
        return

    # ── Save results ─────────────────────────────────────────────────
    base_dir = Path(args.target_dir) if args.target_dir else DEFAULT_TARGET
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / "results" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    report = {
        "version": "3",
        "timestamp": timestamp,
        "server_url": args.base_url,
        "server_model": server_model,
        "total_elapsed_s": round(main_ctx.total_elapsed, 2),
        "total_runs": len(ctx.results),
        "phases": {},
    }
    by_phase: Dict[str, List[Dict]] = {}
    for r in ctx.results:
        d = {k: v for k, v in asdict(r).items() if v not in (0, 0.0, "", None) and k != "error"}
        if r.error:
            d["error"] = r.error
        by_phase.setdefault(r.phase, []).append(d)
    report["phases"] = by_phase

    json_path = out_dir / "warmup_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  JSON report → {json_path}")

    # Text log
    log_path = out_dir / "warmup_log.txt"
    with open(log_path, "w") as f:
        f.write(f"vLLM Warmup v3 — {timestamp}\n")
        f.write(f"Server: {args.base_url}  Model: {server_model}\n")
        f.write(f"Total time: {main_ctx.total_elapsed:.1f}s  Runs: {len(ctx.results)}\n\n")
        for name, plist in by_phase.items():
            f.write(f"=== {name} ({len(plist)} runs) ===\n")
            for d in plist:
                parts = [f"{d.get('prompt_tokens', '?')}p→{d.get('completion_tokens', '?')}c"]
                if "elapsed_ms" in d:
                    parts.append(f"{d['elapsed_ms']:.0f}ms")
                if "ttft_ms" in d:
                    parts.append(f"TTFT {d['ttft_ms']:.0f}ms")
                if "tps" in d:
                    parts.append(f"{d['tps']} tok/s")
                if "error" in d:
                    parts.append(f"ERROR: {d['error']}")
                f.write(f"  {d['name']}: {'  '.join(parts)}\n")
            f.write("\n")
    print(f"  Text log   → {log_path}\n")


if __name__ == "__main__":
    # Catch SIGINT for graceful stop
    signal.signal(signal.SIGINT, lambda *_: sys.exit(1))
    main()
