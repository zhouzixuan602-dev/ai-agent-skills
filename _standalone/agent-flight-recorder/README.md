# 📼 Agent Flight Recorder — a black box for AI agents, so crashes debug themselves

[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/zhouzixuan602-dev/agent-flight-recorder?style=social)]()

> **TL;DR**: When your agent fails in production, all you have is a wall of print statements and a vague stack trace pointing into the SDK. Flight Recorder logs every LLM call, tool call and error as structured JSONL spans — timing, tokens, cost, inputs, stack — and `replay()` reconstructs the timeline and names the exact failing step with the exact inputs that triggered it.

## The Problem

Your agent ran fine for two weeks. Tonight it crashed on run #4,817:

```
KeyError: 'rate'
  File "agent.py", line 212, in run_loop
```

Which tool call? With what arguments? What did the LLM say the step before? How much did the run burn before dying? You don't know — that context lived in memory and died with the process. So you add more `print()` calls, redeploy, and wait for it to crash again.

Airplanes solved this decades ago: the black box records everything, all the time, so the post-mortem is forensics instead of guesswork.

## How It Works

```
 agent code                flight recorder                 trace file
┌──────────────┐  span()  ┌───────────────────┐  JSONL   ┌──────────────────┐
│  llm call    │ ───────▶ │ timing · tokens   │ ───────▶ │ run-a1b2c3.jsonl │
│  tool call   │          │ cost · in/output  │          └────────┬─────────┘
│  any step    │          │ errors · stack    │                   │ replay()
└──────────────┘          └───────────────────┘                   ▼
                                          ✅ [ llm] claude-haiku     812ms $0.0004
                                          ✅ [tool] get_exchange_rate  51ms
                                          💥 [tool] get_exchange_rate  50ms
                                          🔍 VERDICT: KeyError — inputs ("EUR","JPZ")
```

Three pieces, ~150 lines total:

1. **`span()`** — a context manager that wraps any step. Records start/end, duration, truncated inputs/outputs. On exception it captures the error + stack **and re-raises** — the recorder never changes program behavior.
2. **`rec.llm()` / `rec.tool()`** — recorded drop-ins for `messages.create()` and any tool function. LLM spans automatically get token counts and cost in USD.
3. **`replay(path)`** — post-mortem: prints the full timeline with per-span cost, then a VERDICT block naming the failing step, its error, its inputs, and the code location.

Traces are append-only JSONL: one file per run, greppable, tailable, shippable to Datadog/S3/wherever.

## Quick Start

```bash
git clone https://github.com/zhouzixuan602-dev/agent-flight-recorder.git
cd agent-flight-recorder
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY
python main.py         # tool-crash demo works even without a key
```

## Usage

```python
from main import FlightRecorder, replay

rec = FlightRecorder("support-agent")
resp = rec.llm(client, messages=[...])          # LLM call → tokens + cost recorded
data = rec.tool(search_orders, user_id=42)      # tool call → args + result recorded
with rec.span("logic", "rank_results", n=9):    # wrap any custom step
    ...
replay(rec.path)                                 # timeline + verdict after the run
```

## Before vs After

| Metric | `print()` debugging | Flight Recorder |
|---|---|---|
| Time to locate failing step | 30–60 min re-reading logs | seconds — `replay()` names it |
| Context at crash point | whatever you remembered to print | inputs, output, stack, timing for **every** span |
| Cost visibility | none until the invoice | per-call + per-run USD in the trace |
| Reproducing the bug | guess the inputs, hope | exact recorded inputs at the crash |
| Overhead | n/a | ~1 file write per span, zero deps beyond stdlib |

## Contributing

PRs welcome! ⭐ Star this repo if it saved you a 2am debugging session.

---
*Part of [AI Agent Skills](https://github.com/zhouzixuan602-dev/ai-agent-skills) — daily production-ready tools for AI engineers.*
