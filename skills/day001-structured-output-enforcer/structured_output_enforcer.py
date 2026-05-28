"""
Day 001: Structured Output Enforcer
AI Agent Skills Daily - https://github.com/YOUR_USERNAME/ai-agent-skills

问题: AI agent 输出格式不稳定，下游解析失败
方案: 三层防护机制确保 LLM 输出可靠解析
"""

import re
import json
import ast
from typing import Any, Optional, Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class StructuredOutputEnforcer:
    """
    三层防护：确保 LLM 输出可靠解析为结构化数据

    Layer 1: 正则提取 JSON（处理 markdown 包裹）
    Layer 2: 宽松解析（修复单引号、末尾逗号等常见错误）
    Layer 3: LLM 自我修复（传入错误信息，要求重新输出）
    """

    def __init__(self, llm_client: LLMClient, max_repair_attempts: int = 2):
        self.llm = llm_client
        self.max_repair_attempts = max_repair_attempts

    def parse(self, raw_output: str, expected_schema: dict = None) -> Any:
        result = self._layer1_extract_json(raw_output)
        if result is not None:
            return result

        result = self._layer2_lenient_parse(raw_output)
        if result is not None:
            return result

        result = self._layer3_llm_repair(raw_output, expected_schema)
        if result is not None:
            return result

        raise ValueError(
            f"StructuredOutputEnforcer: 三层解析全部失败\n"
            f"原始输出（前500字符）: {raw_output[:500]}"
        )

    def _layer1_extract_json(self, text: str) -> Optional[Any]:
        patterns = [
            r'```(?:json)?\s*\n?([\s\S]*?)\n?```',
            r'`([\s\S]*?)`',
        ]
        candidates = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            candidates.extend(matches)
        candidates.append(text.strip())

        for candidate in candidates:
            try:
                return json.loads(candidate.strip())
            except json.JSONDecodeError:
                continue
        return None

    def _layer2_lenient_parse(self, text: str) -> Optional[Any]:
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                candidate = text[start:end + 1]

                fixed = re.sub(r"(?<![\\])'", '"', candidate)
                fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
                fixed = (fixed
                         .replace('True', 'true')
                         .replace('False', 'false')
                         .replace('None', 'null'))
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass

                try:
                    return ast.literal_eval(candidate)
                except (ValueError, SyntaxError):
                    pass
        return None

    def _layer3_llm_repair(self, bad_output: str, schema: dict = None) -> Optional[Any]:
        schema_hint = f"\n期望的 JSON 结构: {json.dumps(schema)}" if schema else ""
        repair_prompt = (
            f"你之前的输出无法被解析为有效 JSON。\n\n"
            f"原始输出:\n{bad_output}\n"
            f"{schema_hint}\n\n"
            f"请只输出有效的 JSON，不要有任何其他文字、markdown 格式或解释。"
            f"直接从 {{ 或 [ 开始输出。"
        )
        for _ in range(self.max_repair_attempts):
            try:
                repaired = self.llm.complete(repair_prompt)
                result = self._layer1_extract_json(repaired)
                if result is not None:
                    return result
            except Exception:
                continue
        return None


# ── 快速测试 ──────────────────────────────────────────────────
if __name__ == "__main__":
    class MockLLM:
        def complete(self, prompt: str) -> str:
            return '{"status": "ok", "count": 3}'

    enforcer = StructuredOutputEnforcer(llm_client=MockLLM())

    tests = [
        ('markdown wrap',  '```json\n{"status": "ok", "count": 3}\n```'),
        ('prefix text',    'Here is the result: {"status": "ok", "count": 3}'),
        ('trailing comma', '{"status": "ok", "count": 3,}'),
        ('python dict',    "{'status': True, 'count': 3}"),
    ]

    print("=== Structured Output Enforcer Tests ===\n")
    for name, dirty in tests:
        result = enforcer.parse(dirty)
        print(f"✅ [{name}] → {result}")
