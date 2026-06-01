"""
Day 006: Agent Rate Limiter & Task Queue
解决 LLM API 的 rate limit 错误，通过令牌桶算法 + 优先级队列实现并发控制。
"""

import asyncio
import time
import heapq
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine
from collections import deque


# ──────────────────────────────────────────────
# 令牌桶限流器（支持 RPM 和 TPM 双维度）
# ──────────────────────────────────────────────

class TokenBucket:
    """令牌桶：控制每分钟请求数（RPM）或 token 数（TPM）"""

    def __init__(self, capacity: int, refill_rate: float):
        """
        capacity: 桶容量（最大突发量）
        refill_rate: 每秒补充的令牌数
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """
        申请 tokens 个令牌，返回需要等待的秒数。
        如果令牌不足，会阻塞直到可用。
        """
        async with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # 计算等待时间
            deficit = tokens - self.tokens
            wait_time = deficit / self.refill_rate
            await asyncio.sleep(wait_time)
            self._refill()
            self.tokens -= tokens
            return wait_time

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    @property
    def available(self) -> float:
        self._refill()
        return self.tokens


# ──────────────────────────────────────────────
# 优先级任务队列
# ──────────────────────────────────────────────

@dataclass(order=True)
class PriorityTask:
    priority: int                              # 数字越小优先级越高
    task_id: str = field(compare=False)
    fn: Callable = field(compare=False)
    args: tuple = field(compare=False, default_factory=tuple)
    kwargs: dict = field(compare=False, default_factory=dict)
    estimated_tokens: int = field(compare=False, default=1000)


# ──────────────────────────────────────────────
# 主限流调度器
# ──────────────────────────────────────────────

class AgentRateLimiter:
    """
    统一管理 agent 的 API 调用：
    - 令牌桶限流（RPM + TPM）
    - 优先级任务队列
    - 指数退避重试
    - 并发数控制
    """

    def __init__(
        self,
        rpm: int = 60,          # 每分钟最大请求数
        tpm: int = 100_000,     # 每分钟最大 token 数
        max_concurrent: int = 5, # 最大并发请求数
        max_retries: int = 3,
    ):
        # RPM 桶：每分钟 rpm 次请求 → 每秒 rpm/60 次
        self.rpm_bucket = TokenBucket(capacity=rpm, refill_rate=rpm / 60)
        # TPM 桶：每分钟 tpm 个 token → 每秒 tpm/60 个
        self.tpm_bucket = TokenBucket(capacity=tpm, refill_rate=tpm / 60)

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries
        self._heap: list[PriorityTask] = []
        self._stats = {"total": 0, "success": 0, "retry": 0, "failed": 0, "waited_ms": 0}

    async def call(
        self,
        fn: Callable[..., Coroutine],
        *args,
        priority: int = 5,
        estimated_tokens: int = 1000,
        task_id: str = "",
        **kwargs,
    ) -> Any:
        """
        提交一个 API 调用任务，自动限流 + 重试。
        priority: 1（最高）~ 10（最低），默认 5
        estimated_tokens: 预估本次调用消耗的 token 数
        """
        self._stats["total"] += 1

        for attempt in range(self.max_retries + 1):
            try:
                async with self.semaphore:
                    # 同时占用 RPM 和 TPM 令牌
                    t0 = time.monotonic()
                    await self.rpm_bucket.acquire(1)
                    await self.tpm_bucket.acquire(estimated_tokens)
                    waited = (time.monotonic() - t0) * 1000
                    self._stats["waited_ms"] += waited

                    result = await fn(*args, **kwargs)
                    self._stats["success"] += 1
                    return result

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(
                    kw in error_str
                    for kw in ("rate limit", "429", "too many requests", "quota")
                )

                if is_rate_limit and attempt < self.max_retries:
                    self._stats["retry"] += 1
                    backoff = (2 ** attempt) + (0.1 * attempt)
                    print(f"[RateLimiter] Rate limit hit, retry {attempt+1}/{self.max_retries} in {backoff:.1f}s")
                    await asyncio.sleep(backoff)
                else:
                    self._stats["failed"] += 1
                    raise

    async def call_batch(
        self,
        tasks: list[dict],
    ) -> list[Any]:
        """
        批量提交任务，按优先级并发执行。
        每个 task dict: {"fn": ..., "args": (), "kwargs": {}, "priority": 5, "estimated_tokens": 1000}
        """
        coroutines = [
            self.call(
                t["fn"],
                *t.get("args", ()),
                priority=t.get("priority", 5),
                estimated_tokens=t.get("estimated_tokens", 1000),
                **t.get("kwargs", {}),
            )
            for t in sorted(tasks, key=lambda x: x.get("priority", 5))
        ]
        return await asyncio.gather(*coroutines, return_exceptions=True)

    @property
    def stats(self) -> dict:
        s = self._stats.copy()
        s["avg_wait_ms"] = (s["waited_ms"] / s["total"]) if s["total"] else 0
        s["rpm_available"] = round(self.rpm_bucket.available, 1)
        s["tpm_available"] = round(self.tpm_bucket.available, 1)
        return s


# ──────────────────────────────────────────────
# 使用示例
# ──────────────────────────────────────────────

async def demo():
    import random

    # 模拟一个 LLM API 调用（实际替换为 anthropic/openai SDK）
    async def mock_llm_call(prompt: str, tokens: int = 500) -> str:
        await asyncio.sleep(0.1)  # 模拟网络延迟
        if random.random() < 0.1:  # 10% 概率触发 rate limit
            raise Exception("Rate limit exceeded (429)")
        return f"Response to: {prompt[:30]}..."

    # 创建限流器：60 RPM，100k TPM，5 并发
    limiter = AgentRateLimiter(rpm=60, tpm=100_000, max_concurrent=5)

    # 批量任务（不同优先级）
    tasks = [
        {"fn": mock_llm_call, "kwargs": {"prompt": f"User query {i}", "tokens": 800},
         "priority": 1 if i < 3 else 5, "estimated_tokens": 800}
        for i in range(15)
    ]

    print("🚀 提交 15 个任务（前3个高优先级）...")
    start = time.monotonic()
    results = await limiter.call_batch(tasks)
    elapsed = time.monotonic() - start

    success = sum(1 for r in results if not isinstance(r, Exception))
    print(f"✅ 完成: {success}/15 成功，耗时 {elapsed:.1f}s")
    print(f"📊 统计: {limiter.stats}")


if __name__ == "__main__":
    asyncio.run(demo())
