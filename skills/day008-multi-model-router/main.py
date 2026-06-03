"""
Day 008: Multi-Model Router
按任务复杂度和类型自动选择最合适的 LLM，平衡性能与成本。
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Complexity(str, Enum):
    LOW = "low"        # 简单问答、格式转换
    MEDIUM = "medium"  # 逻辑推理、代码生成
    HIGH = "high"      # 多步推理、复杂分析


class TaskType(str, Enum):
    FACTUAL = "factual"       # 事实查询
    REASONING = "reasoning"   # 推理分析
    CODING = "coding"         # 代码生成/调试
    CREATIVE = "creative"     # 创意写作
    CLASSIFICATION = "classification"  # 分类/路由
    SUMMARIZATION = "summarization"    # 摘要压缩


@dataclass
class ModelConfig:
    name: str
    cost_per_1k_input: float   # USD
    cost_per_1k_output: float  # USD
    context_window: int
    strengths: list[TaskType]
    max_complexity: Complexity


# 模型注册表（可按需扩展）
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "claude-haiku-3": ModelConfig(
        name="claude-haiku-3",
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        context_window=200_000,
        strengths=[TaskType.FACTUAL, TaskType.CLASSIFICATION, TaskType.SUMMARIZATION],
        max_complexity=Complexity.MEDIUM,
    ),
    "claude-sonnet-4": ModelConfig(
        name="claude-sonnet-4",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        context_window=200_000,
        strengths=[TaskType.CODING, TaskType.REASONING, TaskType.CREATIVE],
        max_complexity=Complexity.HIGH,
    ),
    "claude-opus-4": ModelConfig(
        name="claude-opus-4",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        context_window=200_000,
        strengths=[TaskType.REASONING, TaskType.CODING, TaskType.CREATIVE],
        max_complexity=Complexity.HIGH,
    ),
}

# 路由规则：(complexity, task_type) -> preferred model key
ROUTING_TABLE: dict[tuple[Complexity, TaskType], str] = {
    (Complexity.LOW,    TaskType.FACTUAL):         "claude-haiku-3",
    (Complexity.LOW,    TaskType.CLASSIFICATION):  "claude-haiku-3",
    (Complexity.LOW,    TaskType.SUMMARIZATION):   "claude-haiku-3",
    (Complexity.LOW,    TaskType.CODING):          "claude-haiku-3",
    (Complexity.LOW,    TaskType.CREATIVE):        "claude-haiku-3",
    (Complexity.LOW,    TaskType.REASONING):       "claude-haiku-3",
    (Complexity.MEDIUM, TaskType.FACTUAL):         "claude-haiku-3",
    (Complexity.MEDIUM, TaskType.CLASSIFICATION):  "claude-haiku-3",
    (Complexity.MEDIUM, TaskType.SUMMARIZATION):   "claude-haiku-3",
    (Complexity.MEDIUM, TaskType.CODING):          "claude-sonnet-4",
    (Complexity.MEDIUM, TaskType.CREATIVE):        "claude-sonnet-4",
    (Complexity.MEDIUM, TaskType.REASONING):       "claude-sonnet-4",
    (Complexity.HIGH,   TaskType.FACTUAL):         "claude-sonnet-4",
    (Complexity.HIGH,   TaskType.CLASSIFICATION):  "claude-sonnet-4",
    (Complexity.HIGH,   TaskType.SUMMARIZATION):   "claude-sonnet-4",
    (Complexity.HIGH,   TaskType.CODING):          "claude-sonnet-4",
    (Complexity.HIGH,   TaskType.CREATIVE):        "claude-opus-4",
    (Complexity.HIGH,   TaskType.REASONING):       "claude-opus-4",
}

# ── 启发式分类器 ──────────────────────────────────────────────

_CODING_PATTERNS = re.compile(
    r"\b(code|debug|implement|function|class|algorithm|script|程序|代码|实现|函数)\b",
    re.IGNORECASE,
)
_REASONING_PATTERNS = re.compile(
    r"\b(analyze|reason|explain why|compare|evaluate|分析|推理|为什么|对比|评估)\b",
    re.IGNORECASE,
)
_CREATIVE_PATTERNS = re.compile(
    r"\b(write|story|poem|creative|imagine|blog|写作|故事|诗|创意)\b",
    re.IGNORECASE,
)
_FACTUAL_PATTERNS = re.compile(
    r"\b(what is|who is|when|where|how many|定义|是什么|多少|哪里)\b",
    re.IGNORECASE,
)
_CLASSIFY_PATTERNS = re.compile(
    r"\b(classify|categorize|label|route|分类|归类|标签)\b",
    re.IGNORECASE,
)
_SUMMARY_PATTERNS = re.compile(
    r"\b(summarize|summary|tldr|摘要|总结|概括)\b",
    re.IGNORECASE,
)

# 高复杂度关键词
_HIGH_COMPLEXITY_SIGNALS = re.compile(
    r"\b(step.by.step|multi.step|chain of thought|complex|sophisticated|"
    r"一步一步|多步|复杂|深入|系统性)\b",
    re.IGNORECASE,
)
_LOW_COMPLEXITY_SIGNALS = re.compile(
    r"\b(simple|quick|brief|one.line|单行|简单|快速|简短)\b",
    re.IGNORECASE,
)


def classify_task(prompt: str) -> tuple[Complexity, TaskType]:
    """从 prompt 文本推断任务类型和复杂度。"""
    # 任务类型
    if _CODING_PATTERNS.search(prompt):
        task_type = TaskType.CODING
    elif _REASONING_PATTERNS.search(prompt):
        task_type = TaskType.REASONING
    elif _CREATIVE_PATTERNS.search(prompt):
        task_type = TaskType.CREATIVE
    elif _CLASSIFY_PATTERNS.search(prompt):
        task_type = TaskType.CLASSIFICATION
    elif _SUMMARY_PATTERNS.search(prompt):
        task_type = TaskType.SUMMARIZATION
    else:
        task_type = TaskType.FACTUAL

    # 复杂度：按长度和关键词
    word_count = len(prompt.split())
    if _HIGH_COMPLEXITY_SIGNALS.search(prompt) or word_count > 200:
        complexity = Complexity.HIGH
    elif _LOW_COMPLEXITY_SIGNALS.search(prompt) or word_count < 20:
        complexity = Complexity.LOW
    else:
        complexity = Complexity.MEDIUM

    return complexity, task_type


# ── 路由器核心 ────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    model: str
    complexity: Complexity
    task_type: TaskType
    estimated_cost_per_1k_tokens: float
    reason: str


def route(
    prompt: str,
    force_complexity: Complexity | None = None,
    force_task_type: TaskType | None = None,
    budget_cap_per_1k: float | None = None,  # 超过此成本则降级
) -> RoutingDecision:
    """
    根据 prompt 内容路由到最优模型。

    Args:
        prompt: 用户输入
        force_complexity: 手动覆盖复杂度（可选）
        force_task_type: 手动覆盖任务类型（可选）
        budget_cap_per_1k: 每千 token 的成本上限（USD）

    Returns:
        RoutingDecision 包含选定模型和路由原因
    """
    complexity, task_type = classify_task(prompt)
    if force_complexity:
        complexity = force_complexity
    if force_task_type:
        task_type = force_task_type

    model_key = ROUTING_TABLE.get((complexity, task_type), "claude-sonnet-4")
    model = MODEL_REGISTRY[model_key]

    # 预算降级：如果成本超限，降到 haiku
    avg_cost = (model.cost_per_1k_input + model.cost_per_1k_output) / 2
    if budget_cap_per_1k and avg_cost > budget_cap_per_1k:
        fallback = MODEL_REGISTRY["claude-haiku-3"]
        fallback_cost = (fallback.cost_per_1k_input + fallback.cost_per_1k_output) / 2
        return RoutingDecision(
            model=fallback.name,
            complexity=complexity,
            task_type=task_type,
            estimated_cost_per_1k_tokens=fallback_cost,
            reason=f"预算上限 ${budget_cap_per_1k}/1k tokens，从 {model_key} 降级到 haiku",
        )

    return RoutingDecision(
        model=model.name,
        complexity=complexity,
        task_type=task_type,
        estimated_cost_per_1k_tokens=avg_cost,
        reason=f"复杂度={complexity.value}, 类型={task_type.value} → {model_key}",
    )


# ── 使用示例 ──────────────────────────────────────────────────

def demo():
    test_cases = [
        "What is the capital of France?",
        "Implement a binary search tree with insertion, deletion, and traversal",
        "Analyze step-by-step the geopolitical implications of AI regulation across G7 nations",
        "Write a creative short story about a robot learning to dream",
        "Summarize this text: ...",
        "分类这条客服消息：我的订单还没到",
    ]

    print(f"{'Prompt':<55} {'Model':<20} {'Type':<16} {'Complexity'}")
    print("-" * 105)
    for prompt in test_cases:
        d = route(prompt)
        print(f"{prompt[:53]:<55} {d.model:<20} {d.task_type.value:<16} {d.complexity.value}")

    # 带预算上限
    print("\n── 预算上限示例 (cap=$0.005/1k) ──")
    d = route(
        "Step-by-step reasoning about complex AI ethics",
        budget_cap_per_1k=0.005,
    )
    print(f"选定模型: {d.model} | 原因: {d.reason}")


if __name__ == "__main__":
    demo()
