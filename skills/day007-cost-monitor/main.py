"""
day007_cost_monitor.py
实时 token 成本监控与预算熔断
支持 OpenAI / Anthropic / 自定义模型定价
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import time


# ── 模型价格表（USD per 1M tokens） ─────────────────────────────
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-6":         {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-6":       {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5":        {"input": 0.8,   "output": 4.0},
    # OpenAI
    "gpt-4o":                  {"input": 5.0,   "output": 15.0},
    "gpt-4o-mini":             {"input": 0.15,  "output": 0.6},
    "gpt-4-turbo":             {"input": 10.0,  "output": 30.0},
    "gpt-3.5-turbo":           {"input": 0.5,   "output": 1.5},
    # 默认（未知模型）
    "_default":                {"input": 5.0,   "output": 15.0},
}


@dataclass
class CallRecord:
    timestamp: float
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    label: str = ""


@dataclass
class CostMonitor:
    """
    实时 token 成本监控与预算熔断。

    使用示例：
        monitor = CostMonitor(hard_limit_usd=1.0, soft_limit_usd=0.5)
        response = client.messages.create(...)
        monitor.track(response, model="claude-sonnet-4-6", label="summarize")
    """
    hard_limit_usd: float = 5.0
    soft_limit_usd: float = 2.0
    on_soft_limit: Optional[Callable] = None
    custom_pricing: dict = field(default_factory=dict)

    _records: list[CallRecord] = field(default_factory=list)
    _total_cost: float = 0.0
    _total_prompt_tokens: int = 0
    _total_completion_tokens: int = 0
    _soft_warned: bool = False

    def _get_price(self, model: str) -> dict[str, float]:
        if model in self.custom_pricing:
            return self.custom_pricing[model]
        for key in MODEL_PRICING:
            if model.startswith(key) or key.startswith(model):
                return MODEL_PRICING[key]
        return MODEL_PRICING["_default"]

    def _calc_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        price = self._get_price(model)
        return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000

    def track(
        self,
        response,
        model: str = "",
        label: str = "",
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> float:
        """解析 response token 用量，累计成本，检查预算。返回本次调用成本（USD）。"""
        if prompt_tokens is None or completion_tokens is None:
            usage = getattr(response, "usage", None)
            if usage:
                prompt_tokens = (
                    getattr(usage, "input_tokens", None)
                    or getattr(usage, "prompt_tokens", 0)
                )
                completion_tokens = (
                    getattr(usage, "output_tokens", None)
                    or getattr(usage, "completion_tokens", 0)
                )
            else:
                prompt_tokens = prompt_tokens or 0
                completion_tokens = completion_tokens or 0

        if not model:
            model = getattr(response, "model", "_default")

        cost = self._calc_cost(model, prompt_tokens, completion_tokens)

        record = CallRecord(
            timestamp=time.time(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            label=label,
        )
        self._records.append(record)
        self._total_cost += cost
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        if not self._soft_warned and self._total_cost >= self.soft_limit_usd:
            self._soft_warned = True
            msg = (f"⚠️  [CostMonitor] 成本已达 ${self._total_cost:.4f}，"
                   f"逼近预算上限 ${self.hard_limit_usd:.2f}")
            if self.on_soft_limit:
                self.on_soft_limit(msg, self)
            else:
                print(msg)

        if self._total_cost >= self.hard_limit_usd:
            raise BudgetExceeded(
                f"预算已耗尽: ${self._total_cost:.4f} >= 上限 ${self.hard_limit_usd:.2f}",
                monitor=self,
            )

        return cost

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def call_count(self) -> int:
        return len(self._records)

    def remaining_budget(self) -> float:
        return max(0.0, self.hard_limit_usd - self._total_cost)

    def reset(self):
        """开始新任务前重置计数器。"""
        self._records.clear()
        self._total_cost = 0.0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._soft_warned = False

    def report(self) -> str:
        """生成文本摘要报告。"""
        lines = [
            "━" * 48,
            "📊 CostMonitor 使用报告",
            "━" * 48,
            f"  总调用次数:  {self.call_count}",
            f"  输入 tokens: {self._total_prompt_tokens:,}",
            f"  输出 tokens: {self._total_completion_tokens:,}",
            f"  总成本:      ${self._total_cost:.6f} USD",
            f"  剩余预算:    ${self.remaining_budget():.6f} USD",
            "  ── 调用明细 ──────────────────────────────",
        ]
        for i, r in enumerate(self._records, 1):
            lines.append(
                f"  [{i:02d}] {r.model:<25} "
                f"in={r.prompt_tokens:>6} out={r.completion_tokens:>5} "
                f"${r.cost_usd:.6f}"
                + (f"  [{r.label}]" if r.label else "")
            )
        lines.append("━" * 48)
        report_str = "\n".join(lines)
        print(report_str)
        return report_str


class BudgetExceeded(Exception):
    def __init__(self, message: str, monitor: CostMonitor):
        super().__init__(message)
        self.monitor = monitor


# ── 演示 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 模拟几次 LLM 调用（无需真实 API key）
    class FakeUsage:
        def __init__(self, inp, out):
            self.input_tokens = inp
            self.output_tokens = out

    class FakeResponse:
        def __init__(self, model, inp, out):
            self.model = model
            self.usage = FakeUsage(inp, out)

    monitor = CostMonitor(hard_limit_usd=0.01, soft_limit_usd=0.005)

    calls = [
        ("claude-sonnet-4-6", 2000, 500, "initial-analysis"),
        ("claude-sonnet-4-6", 1500, 300, "summarize"),
        ("claude-haiku-4-5",  800,  200, "classify"),
        ("claude-sonnet-4-6", 3000, 800, "final-report"),   # 可能触发软/硬限制
    ]

    for model, inp, out, label in calls:
        try:
            resp = FakeResponse(model, inp, out)
            cost = monitor.track(resp, label=label)
            print(f"✅ [{label}] cost=${cost:.6f}  remaining=${monitor.remaining_budget():.6f}")
        except BudgetExceeded as e:
            print(f"🛑 {e}")
            break

    monitor.report()
