"""
day009-structured-logger/main.py

Agent 结构化日志系统
可直接复制使用，依赖仅 Python 标准库

用法:
    from main import AgentLogger

    log = AgentLogger("my-agent")
    log.info("Started")
    with log.tool_context("web_search"):
        results = search(query)
    log.close()
"""

import json
import time
import uuid
import logging
import traceback
import functools
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
from contextlib import contextmanager
from typing import Any, Optional


class AgentLogger:
    """
    结构化日志记录器，专为 AI Agent 设计。

    特性：
    - 自动附加 run_id、step、tool 等上下文
    - JSON Lines 格式，兼容所有日志平台
    - 内置 breadcrumbs，出错时自动附加最近日志
    - 计时装饰器，追踪每个操作耗时
    """

    def __init__(
        self,
        agent_name: str,
        log_dir: str = "logs",
        breadcrumb_size: int = 20,
        console_output: bool = True,
    ):
        self.agent_name = agent_name
        self.run_id = str(uuid.uuid4())[:8]
        self.step = 0
        self._context: dict[str, Any] = {}
        self._breadcrumbs: deque = deque(maxlen=breadcrumb_size)

        # 日志文件（按日期命名）
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        self._log_file = open(log_path / f"{agent_name}_{date_str}.jsonl", "a")

        # 控制台输出（人类可读格式）
        self._console = console_output
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        self._logger = logging.getLogger(agent_name)

    # ── 核心日志方法 ──────────────────────────────────────

    def _write(self, level: str, message: str, **kwargs):
        """底层写入，自动注入 context"""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "agent": self.agent_name,
            "run_id": self.run_id,
            "step": self.step,
            "msg": message,
            **self._context,
            **kwargs,
        }

        # 写入文件（JSON Lines）
        self._log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._log_file.flush()

        # 保存到 breadcrumbs
        self._breadcrumbs.append(entry)

        # 控制台输出（带颜色）
        if self._console:
            color = {
                "DEBUG": "\033[37m",
                "INFO":  "\033[32m",
                "WARN":  "\033[33m",
                "ERROR": "\033[31m",
            }.get(level, "")
            reset = "\033[0m"
            ctx_str = f" [{self._context.get('tool', '')}]" if "tool" in self._context else ""
            print(f"{color}[{level}] [{self.run_id}] step={self.step}{ctx_str} {message}{reset}")

    def debug(self, msg: str, **kwargs): self._write("DEBUG", msg, **kwargs)
    def info(self, msg: str, **kwargs):  self._write("INFO",  msg, **kwargs)
    def warn(self, msg: str, **kwargs):  self._write("WARN",  msg, **kwargs)

    def error(self, msg: str, exc: Optional[Exception] = None, **kwargs):
        """错误日志，自动附加 breadcrumbs 和 traceback"""
        extra = dict(kwargs)
        if exc:
            extra["exception"] = type(exc).__name__
            extra["traceback"] = traceback.format_exc()
        # 附加最近的 breadcrumbs，方便追溯
        extra["breadcrumbs"] = list(self._breadcrumbs)[-10:]
        self._write("ERROR", msg, **extra)

    # ── Context 管理 ──────────────────────────────────────

    def next_step(self, description: str = ""):
        """进入下一步，自动递增 step"""
        self.step += 1
        self._context.pop("tool", None)  # 清除上一步的 tool
        self.info(f"→ Step {self.step}: {description}")

    @contextmanager
    def tool_context(self, tool_name: str):
        """工具调用上下文，自动记录开始/结束和耗时"""
        self._context["tool"] = tool_name
        start = time.perf_counter()
        self.debug(f"Tool call started: {tool_name}")
        try:
            yield self
            elapsed = (time.perf_counter() - start) * 1000
            self.info(f"Tool call completed: {tool_name}", duration_ms=round(elapsed, 2))
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self.error(f"Tool call failed: {tool_name}", exc=e, duration_ms=round(elapsed, 2))
            raise
        finally:
            self._context.pop("tool", None)

    @contextmanager
    def llm_context(self, model: str, prompt_tokens: int = 0):
        """LLM 调用上下文，记录模型、token、耗时"""
        self._context["model"] = model
        start = time.perf_counter()
        self.debug(f"LLM call: {model}", prompt_tokens=prompt_tokens)
        try:
            yield self
            elapsed = (time.perf_counter() - start) * 1000
            self.info("LLM call completed", duration_ms=round(elapsed, 2))
        except Exception as e:
            self.error(f"LLM call failed: {model}", exc=e)
            raise
        finally:
            self._context.pop("model", None)

    # ── 装饰器 ────────────────────────────────────────────

    def log_duration(self, label: str = ""):
        """装饰器：自动记录函数耗时"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                name = label or func.__name__
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed = (time.perf_counter() - start) * 1000
                    self.debug(f"{name} finished", duration_ms=round(elapsed, 2))
                    return result
                except Exception as e:
                    elapsed = (time.perf_counter() - start) * 1000
                    self.error(f"{name} raised", exc=e, duration_ms=round(elapsed, 2))
                    raise
            return wrapper
        return decorator

    # ── 查询工具 ──────────────────────────────────────────

    def get_run_summary(self) -> dict:
        """返回本次运行的摘要统计"""
        all_logs = list(self._breadcrumbs)
        errors = [l for l in all_logs if l["level"] == "ERROR"]
        durations = [l.get("duration_ms", 0) for l in all_logs if "duration_ms" in l]
        return {
            "run_id": self.run_id,
            "total_steps": self.step,
            "total_logs": len(all_logs),
            "error_count": len(errors),
            "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0,
        }

    def close(self):
        """关闭日志，写入运行摘要"""
        summary = self.get_run_summary()
        self.info("Run completed", **summary)
        self._log_file.close()


# ── 演示用法 ──────────────────────────────────────────────

def demo():
    """演示 AgentLogger 的基本用法"""
    import random

    log = AgentLogger(agent_name="demo-agent", log_dir="logs")

    try:
        log.info("Agent started", query="demo query")

        # Step 1: 模拟工具调用
        log.next_step("fetch data")
        with log.tool_context("http_fetch"):
            time.sleep(0.05)  # 模拟网络请求
            log.debug("Fetched 10 results", count=10)

        # Step 2: 模拟 LLM 调用
        log.next_step("llm summarize")
        with log.llm_context("claude-haiku", prompt_tokens=500):
            time.sleep(0.1)  # 模拟 LLM 延迟
            log.info("Summary generated", output_tokens=150)

        # Step 3: 模拟偶发错误
        log.next_step("validate output")
        try:
            if random.random() < 0.3:  # 30% 概率触发错误
                raise ValueError("Output validation failed: missing 'sources' field")
            log.info("Validation passed")
        except ValueError as e:
            log.error("Validation error", exc=e)
            log.warn("Falling back to unvalidated output")

    finally:
        print("\n── Run Summary ──")
        import json as _json
        print(_json.dumps(log.get_run_summary(), indent=2))
        log.close()


if __name__ == "__main__":
    demo()
