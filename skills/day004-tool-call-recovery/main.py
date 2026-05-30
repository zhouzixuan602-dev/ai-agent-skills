"""
Day 004: Tool Call Failure Recovery
工具调用失败恢复 — 重试策略、降级方案、错误分类
"""

import asyncio
import random
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class FailureType(Enum):
    TRANSIENT = "transient"      # 可重试（网络抖动、超时、限流）
    DEGRADABLE = "degradable"    # 可降级（服务不可用，有备用方案）
    FATAL = "fatal"              # 不可重试（参数错误、权限不足）


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0       # 初始等待秒数
    max_delay: float = 30.0       # 最大等待秒数
    exponential_base: float = 2.0
    jitter: bool = True           # 加随机抖动，避免惊群效应


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    attempts: int = 1
    fallback_used: bool = False


def classify_error(exc: Exception) -> FailureType:
    """将异常分类为可重试 / 可降级 / 致命三种"""
    error_str = str(exc).lower()
    type_name = type(exc).__name__

    # 致命错误 — 重试无意义
    fatal_keywords = ["invalid", "unauthorized", "forbidden", "not found",
                      "bad request", "validation", "schema"]
    if any(k in error_str for k in fatal_keywords):
        return FailureType.FATAL
    if type_name in ("ValueError", "TypeError", "AttributeError"):
        return FailureType.FATAL

    # 可降级 — 服务整体不可用
    degradable_keywords = ["service unavailable", "503", "circuit breaker",
                           "maintenance", "quota exceeded"]
    if any(k in error_str for k in degradable_keywords):
        return FailureType.DEGRADABLE

    # 其余默认为可重试（网络、超时、限流等）
    return FailureType.TRANSIENT


def compute_delay(attempt: int, config: RetryConfig) -> float:
    """指数退避 + 可选随机抖动"""
    delay = min(
        config.base_delay * (config.exponential_base ** (attempt - 1)),
        config.max_delay
    )
    if config.jitter:
        delay *= (0.5 + random.random() * 0.5)  # 50%–100% 随机
    return delay


async def call_with_recovery(
    tool_fn: Callable,
    *args,
    retry_config: RetryConfig = None,
    fallback_fn: Optional[Callable] = None,
    tool_name: str = "tool",
    **kwargs,
) -> ToolResult:
    """
    带恢复策略的工具调用封装。

    流程:
      尝试调用 tool_fn
        ├─ 成功 → 返回
        ├─ TRANSIENT → 指数退避重试
        ├─ DEGRADABLE → 跳重试，直接降级到 fallback_fn
        └─ FATAL → 立即停止，返回错误
    """
    config = retry_config or RetryConfig()
    attempt = 0

    while attempt < config.max_attempts:
        attempt += 1
        try:
            logger.info(f"[{tool_name}] attempt {attempt}/{config.max_attempts}")
            result = await tool_fn(*args, **kwargs) if asyncio.iscoroutinefunction(tool_fn) \
                     else tool_fn(*args, **kwargs)
            return ToolResult(success=True, data=result, attempts=attempt)

        except Exception as exc:
            failure_type = classify_error(exc)
            logger.warning(f"[{tool_name}] {failure_type.value} error: {exc}")

            if failure_type == FailureType.FATAL:
                logger.error(f"[{tool_name}] Fatal error, stopping immediately.")
                return ToolResult(success=False, error=str(exc), attempts=attempt)

            if failure_type == FailureType.DEGRADABLE:
                logger.warning(f"[{tool_name}] Service unavailable, trying fallback.")
                break  # 跳出重试循环，走降级

            # TRANSIENT — 等待后重试
            if attempt < config.max_attempts:
                delay = compute_delay(attempt, config)
                logger.info(f"[{tool_name}] Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
            else:
                # 重试耗尽，尝试降级
                break

    # 降级逻辑
    if fallback_fn:
        try:
            logger.info(f"[{tool_name}] Using fallback.")
            fb_result = await fallback_fn(*args, **kwargs) if asyncio.iscoroutinefunction(fallback_fn) \
                        else fallback_fn(*args, **kwargs)
            return ToolResult(success=True, data=fb_result, attempts=attempt, fallback_used=True)
        except Exception as fb_exc:
            return ToolResult(success=False, error=f"Fallback also failed: {fb_exc}", attempts=attempt)

    return ToolResult(success=False, error="All attempts exhausted, no fallback.", attempts=attempt)


# ─── 示例：模拟真实场景 ─────────────────────────────────────────

class FlakyWeatherAPI:
    """模拟一个时常失败的第三方天气 API"""
    def __init__(self, fail_rate=0.6):
        self.fail_rate = fail_rate
        self.call_count = 0

    def get_weather(self, city: str) -> dict:
        self.call_count += 1
        roll = random.random()
        if roll < self.fail_rate * 0.3:
            raise ConnectionError("timeout: upstream took too long")
        if roll < self.fail_rate * 0.6:
            raise RuntimeError("503 Service Unavailable")
        if self.call_count == 1:
            raise ConnectionError("rate limit exceeded, retry after 2s")
        return {"city": city, "temp": 22, "condition": "sunny", "source": "primary"}


def weather_fallback(city: str) -> dict:
    """静态缓存降级：返回昨日数据"""
    return {"city": city, "temp": 20, "condition": "unknown", "source": "cache (stale)"}


async def demo():
    api = FlakyWeatherAPI(fail_rate=0.5)
    config = RetryConfig(max_attempts=4, base_delay=0.1, max_delay=1.0)

    result = await call_with_recovery(
        api.get_weather,
        "Melbourne",
        retry_config=config,
        fallback_fn=weather_fallback,
        tool_name="WeatherAPI",
    )

    print("\n=== Result ===")
    print(f"Success     : {result.success}")
    print(f"Data        : {result.data}")
    print(f"Attempts    : {result.attempts}")
    print(f"Fallback    : {result.fallback_used}")
    if result.error:
        print(f"Error       : {result.error}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    random.seed(42)
    asyncio.run(demo())
