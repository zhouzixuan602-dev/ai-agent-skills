"""
agent-flight-recorder — A black-box flight recorder for AI agents.

Every LLM call, tool call and error becomes a structured JSONL span with
timing, tokens, cost and stack traces. When a run crashes, `replay()`
reconstructs the timeline and points at the exact failing step with the
exact inputs that triggered it — no more grepping print statements at 2am.
"""
import json
import os
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

import anthropic

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# $ per MTok (input, output). Extend for the models you use.
PRICING = {"claude-haiku-4-5-20251001": (1.00, 5.00)}
FALLBACK_PRICE = (3.00, 15.00)


def _truncate(value, limit: int = 400) -> str:
    """Keep traces small: store a preview, never the full 50KB payload."""
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + f"…(+{len(s) - limit} chars)"


class FlightRecorder:
    """Records one agent run as append-only JSONL spans — the black box."""

    def __init__(self, run_name: str, trace_dir: str = "traces"):
        os.makedirs(trace_dir, exist_ok=True)
        self.run_id = f"{run_name}-{uuid.uuid4().hex[:8]}"
        self.path = os.path.join(trace_dir, f"{self.run_id}.jsonl")
        self._seq = 0
        self._write({"type": "run_start", "run": self.run_id})

    def _write(self, event: dict) -> None:
        self._seq += 1
        event.update(seq=self._seq, ts=datetime.now(timezone.utc).isoformat())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    @contextmanager
    def span(self, kind: str, name: str, **inputs):
        """Wrap ANY step. Records duration + result; captures errors but never
        swallows them — the black box records, the plane still reports."""
        sid = uuid.uuid4().hex[:8]
        self._write({"type": "span_start", "id": sid, "kind": kind, "name": name,
                     "inputs": _truncate(inputs)})
        t0, holder = time.perf_counter(), {}
        try:
            yield holder  # caller stashes output/tokens/cost into holder
            self._write({"type": "span_end", "id": sid, "kind": kind, "name": name,
                         "status": "ok", "ms": round((time.perf_counter() - t0) * 1000, 1),
                         "output": _truncate(holder.get("output", "")),
                         **{k: v for k, v in holder.items() if k != "output"}})
        except Exception as e:
            self._write({"type": "span_end", "id": sid, "kind": kind, "name": name,
                         "status": "error", "ms": round((time.perf_counter() - t0) * 1000, 1),
                         "error": f"{type(e).__name__}: {e}",
                         "stack": traceback.format_exc(limit=3)})
            raise

    def llm(self, client: anthropic.Anthropic, **kwargs):
        """Recorded drop-in for client.messages.create(): adds tokens + cost."""
        model = kwargs.setdefault("model", DEFAULT_MODEL)
        with self.span("llm", model, messages=kwargs.get("messages")) as s:
            resp = client.messages.create(**kwargs)
            pi, po = PRICING.get(model, FALLBACK_PRICE)
            usage = resp.usage
            s["output"] = resp.content[0].text if resp.content else ""
            s["tokens"] = {"in": usage.input_tokens, "out": usage.output_tokens}
            s["cost_usd"] = round(usage.input_tokens / 1e6 * pi + usage.output_tokens / 1e6 * po, 6)
            return resp

    def tool(self, fn, *args, **kwargs):
        """Recorded wrapper for any tool function."""
        with self.span("tool", fn.__name__, args=args, kwargs=kwargs) as s:
            s["output"] = fn(*args, **kwargs)
            return s["output"]


# ------------------------------------------------------------- post-mortem
def replay(path: str) -> dict:
    """Reconstruct a trace: timeline, cost total, and the point of failure."""
    with open(path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f]
    starts = {e["id"]: e for e in events if e["type"] == "span_start"}
    ends = [e for e in events if e["type"] == "span_end"]

    print(f"📼 FLIGHT RECORDER — {os.path.basename(path)}")
    total = 0.0
    for s in ends:
        icon = "✅" if s["status"] == "ok" else "💥"
        cost = f"  ${s['cost_usd']:.4f}" if "cost_usd" in s else ""
        print(f"  {icon} [{s['kind']:>4}] {s['name']:<32} {s['ms']:>8.1f}ms{cost}")
        total += s.get("cost_usd", 0)
    print(f"  Σ  {len(ends)} spans · total cost ${total:.4f}")

    crash = next((s for s in ends if s["status"] == "error"), None)
    if crash:
        start = starts.get(crash["id"], {})
        frames = [l.strip() for l in crash["stack"].splitlines() if l.strip().startswith("File")]
        last_frame = frames[-1] if frames else crash["stack"].strip().splitlines()[-1]
        print("\n  🔍 VERDICT — exact failing step, no grepping required")
        print(f"  step   : {crash['kind']}:{crash['name']}  (seq {crash['seq']})")
        print(f"  error  : {crash['error']}")
        print(f"  inputs : {start.get('inputs')}")
        print(f"  where  : {last_frame}")
    return {"spans": len(ends), "cost_usd": round(total, 6),
            "crash": crash["name"] if crash else None}


# -------------------------------------------------------------------- demo
RATES = {("USD", "EUR"): 0.92, ("EUR", "JPY"): 162.40}


def get_exchange_rate(src: str, dst: str) -> float:
    time.sleep(0.05)  # simulate network latency
    if (src, dst) not in RATES:
        raise KeyError(f"no rate for {src}->{dst}")
    return RATES[(src, dst)]


def convert(amount: float, src: str, dst: str) -> float:
    return round(amount * RATES[(src, dst)], 2)


if __name__ == "__main__":
    rec = FlightRecorder("currency-agent")
    print(f"🛫 Recording run to {rec.path}\n")

    try:  # LLM step is optional so the demo also runs without an API key
        rec.llm(anthropic.Anthropic(), max_tokens=100, messages=[{
            "role": "user",
            "content": "Plan in one line: convert 100 USD to EUR, then to JPY."}])
    except Exception as e:
        print(f"(LLM step skipped: {type(e).__name__} — tool demo continues)\n")

    try:
        rec.tool(get_exchange_rate, "USD", "EUR")
        rec.tool(convert, 100, "USD", "EUR")
        rec.tool(get_exchange_rate, "EUR", "JPZ")  # typo'd currency code → 💥
        rec.tool(convert, 92, "EUR", "JPZ")        # never reached
    except Exception as e:
        print(f"💥 Agent crashed mid-run: {e}\n{'=' * 64}")

    replay(rec.path)  # post-mortem: timeline + verdict from the black box
