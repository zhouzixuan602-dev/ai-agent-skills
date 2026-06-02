# Day 007: Cost Monitor — Token 成本监控与预警

**痛点**: Agent 跑完才发现烧了几十美元，缺乏实时成本可见性和预算熔断机制。

---

## 问题描述

在生产环境中运行 AI Agent，成本失控是最常见的事故之一。常见场景：
- 循环调用中 prompt 越来越长，token 数指数增长
- 用户触发了意外的高成本路径（如上传大文件 + 多轮对话）
- 批量任务跑了一半，账单已超预算
- 没有告警，只能月底看账单

开发者需要的是：**实时 token 计数 + 成本估算 + 预算熔断 + 使用报告**，而不是事后复盘。

## 解决思路

```
每次 LLM 调用
      │
      ▼
  CostMonitor.track(response)
      │
      ├─ 累计 prompt_tokens / completion_tokens
      ├─ 按模型价格换算成本（USD）
      ├─ 检查是否超过 soft_limit → 发出警告
      └─ 检查是否超过 hard_limit → 抛出 BudgetExceeded

  任意时刻可调用：
  monitor.report()   → 打印使用摘要
  monitor.reset()    → 重置计数（新任务）
```

**价格表** 内置主流模型定价，支持自定义扩展。

## 实现代码

```python
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
    label: str = ""          # 可选标签，便于区分调用来源


@dataclass
class CostMonitor:
    """
    使用示例：
        monitor = CostMonitor(hard_limit_usd=1.0, soft_limit_usd=0.5)
        response = client.messages.create(...)
        monitor.track(response, model="claude-sonnet-4-6", label="summarize")
    """
    hard_limit_usd: float = 5.0          # 超过此值 → 抛出异常，熔断
    soft_limit_usd: float = 2.0          # 超过此值 → 调用警告回调
    on_soft_limit: Optional[Callable] = None   # 警告回调，默认 print
    custom_pricing: dict = field(default_factory=dict)

    # 内部状态
    _records: list[CallRecord] = field(default_factory=list)
    _total_cost: float = 0.0
    _total_prompt_tokens: int = 0
    _total_completion_tokens: int = 0
    _soft_warned: bool = False

    def _get_price(self, model: str) -> dict[str, float]:
        if model in self.custom_pricing:
            return self.custom_pricing[model]
        # 模糊匹配（前缀）
        for key in MODEL_PRICING:
            if model.startswith(key) or key.startswith(model):
                return MODEL_PRICING[key]
        return MODEL_PRICING["_default"]

    def _calc_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        price = self._get_price(model)
        return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000

    def track(
        self,
        response,                          # SDK response 对象（任意格式）
        model: str = "",
        label: str = "",
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> float:
        """
        解析 response 中的 token 用量，累计成本，检查预算。
        返回本次调用的成本（USD）。
        """
        # ── 自动从 response 提取 token 数 ─────────────────────────
        if prompt_tokens is None or completion_tokens is None:
            usage = getattr(response, "usage", None)
            if usage:
                # Anthropic: input_tokens / output_tokens
                # OpenAI:    prompt_tokens / completion_tokens
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

        # ── 自动提取 model ────────────────────────────────────────
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

        # ── 软限制：警告 ──────────────────────────────────────────
        if not self._soft_warned and self._total_cost >= self.soft_limit_usd:
            self._soft_warned = True
            msg = (f"⚠️  [CostMonitor] 成本已达 ${self._total_cost:.4f}，"
                   f"逼近预算上限 ${self.hard_limit_usd:.2f}")
            if self.on_soft_limit:
                self.on_soft_limit(msg, self)
            else:
                print(msg)

        # ── 硬限制：熔断 ──────────────────────────────────────────
        if self._total_cost >= self.hard_limit_usd:
            raise BudgetExceeded(
                f"预算已耗尽: ${self._total_cost:.4f} >= 上限 ${self.hard_limit_usd:.2f}",
                monitor=self,
            )

        return cost

    # ── 快速查询 ──────────────────────────────────────────────────
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
```

## 集成示例

```python
import anthropic
from day007_cost_monitor import CostMonitor, BudgetExceeded

client = anthropic.Anthropic()

# 初始化：硬限制 $1，软限制 $0.5
monitor = CostMonitor(hard_limit_usd=1.0, soft_limit_usd=0.5)

def agent_step(user_msg: str, label: str = "") -> str:
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": user_msg}],
        )
        # 只需一行接入
        monitor.track(response, label=label)
        return response.content[0].text

    except BudgetExceeded as e:
        print(f"🛑 熔断: {e}")
        e.monitor.report()
        return "[任务终止：预算耗尽]"

# 批量任务
tasks = [
    ("分析这份报告...", "report-analysis"),
    ("总结以上内容...", "summarize"),
    ("生成行动建议...", "action-items"),
]

for msg, label in tasks:
    result = agent_step(msg, label=label)
    print(f"[{label}] 剩余预算: ${monitor.remaining_budget():.4f}")

monitor.report()
```

**Webhook 告警集成**：

```python
import httpx

def slack_alert(msg: str, monitor: CostMonitor):
    httpx.post("https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
               json={"text": msg})

monitor = CostMonitor(
    hard_limit_usd=10.0,
    soft_limit_usd=7.0,
    on_soft_limit=slack_alert,   # 超软限制时自动发 Slack
)
```

## 效果

| 指标 | 改进前 | 改进后 |
|------|--------|--------|
| 成本可见性 | 月底账单 | 每次调用实时累计 |
| 超支发现时机 | 事后 | 实时熔断，立即停止 |
| 调试定位 | 无法追溯 | 每次调用有 label + token 明细 |
| 预算控制 | 全凭经验估算 | 硬性上限，不可突破 |
| 报警响应 | 手动查看 | 自动触发 Slack / 邮件 / 回调 |

## 延伸阅读

- [Anthropic Token Counting API](https://docs.anthropic.com/en/docs/build-with-claude/token-counting) — 调用前预估 token 数
- [OpenAI Usage API](https://platform.openai.com/docs/api-reference/usage) — 官方用量接口
- [LangSmith Cost Tracking](https://docs.smith.langchain.com/) — 生产级成本追踪平台
- [litellm](https://github.com/BerriAI/litellm) — 统一多模型接口，内置成本追踪

---
*Day 007 · AI Agent Skills Daily · Melbourne, Australia*
