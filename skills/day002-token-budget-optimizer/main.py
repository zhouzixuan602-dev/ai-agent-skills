# main.py — Token Budget Optimizer
# Day 002 · AI Agent Skills Daily
#
# 解决痛点：AI Agent 每次调用消耗大量 token，成本失控
# 方案：语义缓存 + 对话压缩 + 预算门控 + 自动降级
#
# 安装依赖：pip install anthropic tiktoken numpy

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import tiktoken

# ─────────────────────────────────────────
# 1. Token 计数工具
# ─────────────────────────────────────────

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """快速估算 token 数（适用于 GPT-4 / Claude 系列）"""
    return len(_enc.encode(text))


def count_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        total += count_tokens(msg.get("content", ""))
        total += 4  # 每条消息的固定 overhead
    return total + 2  # 对话起始/结束 overhead


# ─────────────────────────────────────────
# 2. 语义缓存（内存版，生产可换 Redis + pgvector）
# ─────────────────────────────────────────

@dataclass
class CacheEntry:
    question_embedding: list[float]
    answer: str
    tokens_saved: int
    created_at: float = field(default_factory=time.time)


class SemanticCache:
    """
    对相似问题命中缓存，避免重复调用 LLM。
    生产环境建议替换 _embed 为真实 embedding（如 text-embedding-3-small）。
    """

    def __init__(self, similarity_threshold: float = 0.92, max_size: int = 1000):
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.entries: list[CacheEntry] = []
        self.hit_count = 0
        self.total_tokens_saved = 0

    def _embed(self, text: str) -> list[float]:
        """
        演示用简化 embedding。
        生产环境替换为：client.embeddings.create(model="text-embedding-3-small", input=text)
        """
        tokens = text.lower().split()
        vec = np.zeros(64)
        for i, token in enumerate(tokens[:64]):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % 64] += 1.0 / (i + 1)
        norm = np.linalg.norm(vec)
        return (vec / norm if norm > 0 else vec).tolist()

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    def lookup(self, question: str) -> Optional[str]:
        """查询缓存，返回命中的答案或 None"""
        emb = self._embed(question)
        best_sim, best_entry = 0.0, None
        for entry in self.entries:
            sim = self._cosine_sim(emb, entry.question_embedding)
            if sim > best_sim:
                best_sim, best_entry = sim, entry
        if best_sim >= self.threshold and best_entry:
            self.hit_count += 1
            self.total_tokens_saved += best_entry.tokens_saved
            return best_entry.answer
        return None

    def store(self, question: str, answer: str, tokens_used: int):
        """存入缓存（LRU 超容量时淘汰最旧条目）"""
        emb = self._embed(question)
        self.entries.append(CacheEntry(emb, answer, tokens_used))
        if len(self.entries) > self.max_size:
            self.entries.pop(0)


# ─────────────────────────────────────────
# 3. 对话历史压缩器
# ─────────────────────────────────────────

class ConversationCompressor:
    """
    保留最近 N 轮对话，超出部分用摘要替代，防止上下文窗口溢出。
    """

    def __init__(self, max_recent_turns: int = 6, max_tokens: int = 2000):
        self.max_recent_turns = max_recent_turns
        self.max_tokens = max_tokens

    def compress(
        self, messages: list[dict], summarizer_fn=None
    ) -> list[dict]:
        """
        Args:
            messages: 完整对话历史（不含 system prompt）
            summarizer_fn: 接受消息列表，返回摘要字符串的函数（可传 LLM 调用）
        Returns:
            压缩后的消息列表
        """
        if len(messages) <= self.max_recent_turns * 2:
            return messages

        cutoff = len(messages) - self.max_recent_turns * 2
        old_messages = messages[:cutoff]
        recent_messages = messages[cutoff:]

        if summarizer_fn:
            summary_text = summarizer_fn(old_messages)
        else:
            # 无 summarizer 时：提取最后一句用户消息作为简单摘要
            user_msgs = [m["content"] for m in old_messages if m["role"] == "user"]
            summary_text = (
                f"[之前 {len(old_messages)} 条消息摘要] 用户主要询问了：{'; '.join(user_msgs[-3:])}"
            )

        summary_message = {
            "role": "system",
            "content": f"以下是之前对话的摘要，供参考：\n{summary_text}",
        }
        return [summary_message] + recent_messages


# ─────────────────────────────────────────
# 4. Token 预算门控器
# ─────────────────────────────────────────

@dataclass
class BudgetConfig:
    max_tokens_per_request: int = 4000         # 单次请求 token 上限
    daily_token_budget: int = 100_000          # 每日总预算（所有请求之和）
    primary_model: str = "claude-sonnet-4-6"
    fallback_model: str = "claude-haiku-4-5-20251001"  # 超预算时降级


class TokenBudgetGate:
    def __init__(self, config: BudgetConfig):
        self.config = config
        self.daily_used = 0
        self.request_count = 0
        self._day_start = time.time()

    def _reset_if_new_day(self):
        if time.time() - self._day_start > 86400:
            self.daily_used = 0
            self.request_count = 0
            self._day_start = time.time()

    def check_and_select_model(self, estimated_tokens: int) -> tuple[str, bool]:
        """
        Returns:
            (model_name, should_proceed)
            should_proceed=False 时建议拒绝请求
        """
        self._reset_if_new_day()

        if self.daily_used + estimated_tokens > self.config.daily_token_budget:
            print(
                f"⚠️  日预算超限: 已用 {self.daily_used:,} / {self.config.daily_token_budget:,}"
            )
            return self.config.fallback_model, False

        if estimated_tokens > self.config.max_tokens_per_request:
            print(
                f"⚠️  单次超限 ({estimated_tokens} > {self.config.max_tokens_per_request})，"
                f"降级到 {self.config.fallback_model}"
            )
            return self.config.fallback_model, True

        return self.config.primary_model, True

    def record_usage(self, actual_tokens: int):
        self.daily_used += actual_tokens
        self.request_count += 1

    def stats(self) -> dict:
        return {
            "daily_used": self.daily_used,
            "daily_budget": self.config.daily_token_budget,
            "utilization_pct": round(
                self.daily_used / self.config.daily_token_budget * 100, 1
            ),
            "request_count": self.request_count,
        }


# ─────────────────────────────────────────
# 5. 组合入口：TokenBudgetOptimizer
# ─────────────────────────────────────────

class TokenBudgetOptimizer:
    """
    三合一优化器：语义缓存 + 对话压缩 + 预算门控。

    用法：
        optimizer = TokenBudgetOptimizer()
        req = optimizer.prepare_request(user_q, history, system_prompt)
        if req["cache_hit"]: return req["answer"]
        # ... 调用 API ...
        optimizer.record_result(user_q, answer, actual_tokens)
    """

    def __init__(
        self,
        budget_config: Optional[BudgetConfig] = None,
        cache_threshold: float = 0.92,
        max_recent_turns: int = 6,
    ):
        self.cache = SemanticCache(similarity_threshold=cache_threshold)
        self.compressor = ConversationCompressor(max_recent_turns=max_recent_turns)
        self.gate = TokenBudgetGate(budget_config or BudgetConfig())

    def prepare_request(
        self,
        user_question: str,
        conversation_history: list[dict],
        system_prompt: str = "",
        summarizer_fn=None,
    ) -> dict:
        """
        优化请求，返回字典：
        - cache_hit=True:  {"cache_hit": True, "answer": str, "tokens_used": 0}
        - cache_hit=False: {"cache_hit": False, "model": str, "should_proceed": bool,
                            "messages": list, "estimated_tokens": int}
        """
        # 1. 语义缓存检查
        cached = self.cache.lookup(user_question)
        if cached:
            return {"cache_hit": True, "answer": cached, "tokens_used": 0}

        # 2. 压缩对话历史
        compressed = self.compressor.compress(conversation_history, summarizer_fn)

        # 3. 组装消息 + 估算 token
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(compressed)
        messages.append({"role": "user", "content": user_question})
        estimated_tokens = count_messages_tokens(messages)

        # 4. 预算检查
        model, should_proceed = self.gate.check_and_select_model(estimated_tokens)

        return {
            "cache_hit": False,
            "model": model,
            "should_proceed": should_proceed,
            "messages": messages,
            "estimated_tokens": estimated_tokens,
        }

    def record_result(self, question: str, answer: str, actual_tokens: int):
        """API 调用完成后调用，写缓存 + 更新统计"""
        self.cache.store(question, answer, actual_tokens)
        self.gate.record_usage(actual_tokens)

    def report(self) -> dict:
        return {
            "budget": self.gate.stats(),
            "cache": {
                "hit_count": self.cache.hit_count,
                "tokens_saved": self.cache.total_tokens_saved,
                "entries": len(self.cache.entries),
            },
        }


# ─────────────────────────────────────────
# 快速演示（不需要真实 API Key）
# ─────────────────────────────────────────

if __name__ == "__main__":
    optimizer = TokenBudgetOptimizer(
        budget_config=BudgetConfig(
            max_tokens_per_request=3000,
            daily_token_budget=50_000,
        ),
        cache_threshold=0.88,
        max_recent_turns=4,
    )

    # 模拟一些对话
    test_cases = [
        ("退款政策是什么？", 820),
        ("你们怎么退款？", 820),         # 语义相似 → 命中缓存
        ("如何申请退货退款？", 820),      # 语义相似 → 命中缓存
        ("产品质量问题怎么处理？", 650),
        ("收到的商品有损坏，怎么办？", 650),  # 相似 → 命中缓存
    ]

    history = []
    system_prompt = "你是专业客服，回答简洁。"

    print("=" * 50)
    print("Token Budget Optimizer 演示")
    print("=" * 50)

    for question, mock_tokens in test_cases:
        print(f"\n❓ {question}")
        req = optimizer.prepare_request(question, history, system_prompt)

        if req["cache_hit"]:
            print(f"  ✅ 缓存命中！节省约 {mock_tokens} tokens")
        else:
            print(f"  🤖 模型: {req['model']} | 预估: {req['estimated_tokens']} tokens")
            mock_answer = f"[模拟回答：{question}的处理方案]"
            optimizer.record_result(question, mock_answer, mock_tokens)
            history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": mock_answer},
            ])

    print("\n" + "=" * 50)
    print("📊 优化报告:")
    report = optimizer.report()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("=" * 50)
