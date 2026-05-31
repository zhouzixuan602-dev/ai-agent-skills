"""
Day 005: Long Conversation Memory Manager
三层记忆架构，解决长对话上下文溢出问题

用法:
    python main.py
"""

import json
from dataclasses import dataclass, field
from typing import Any
from anthropic import Anthropic

client = Anthropic()


@dataclass
class Message:
    role: str
    content: str


@dataclass
class MemoryState:
    # 短期：完整保留最近 N 轮
    working_memory: list[Message] = field(default_factory=list)
    # 中期：旧对话的压缩摘要
    summary: str = ""
    # 长期：从对话中提取的关键实体
    entities: dict[str, Any] = field(default_factory=dict)


class ConversationMemoryManager:
    def __init__(
        self,
        working_memory_limit: int = 10,   # 保留最近几轮完整对话
        token_threshold: int = 4000,       # 触发压缩的 token 估算阈值
        model: str = "claude-haiku-4-5-20251001",
    ):
        self.working_memory_limit = working_memory_limit
        self.token_threshold = token_threshold
        self.model = model
        self.state = MemoryState()

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数（4字符≈1token）"""
        return len(text) // 4

    def _working_memory_tokens(self) -> int:
        total = sum(
            self._estimate_tokens(m.content)
            for m in self.state.working_memory
        )
        return total

    def _compress_old_messages(self, messages: list[Message]) -> tuple[str, dict]:
        """调用 LLM 压缩旧消息为摘要 + 实体"""
        text = "\n".join(f"{m.role}: {m.content}" for m in messages)
        existing_summary = self.state.summary
        existing_entities = json.dumps(self.state.entities, ensure_ascii=False)

        prompt = f"""请分析以下对话历史，完成两件事：

1. 将对话压缩为简洁摘要（100字以内），与现有摘要合并
2. 提取/更新关键实体（用户名、订单号、技术栈、决策等）为 JSON

现有摘要：{existing_summary or '无'}
现有实体：{existing_entities}

新对话：
{text}

请严格按以下 JSON 格式返回（不要有多余文字）：
{{
  "summary": "合并后的摘要",
  "entities": {{"key": "value"}}
}}"""

        response = client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            result = json.loads(response.content[0].text)
            return result.get("summary", ""), result.get("entities", {})
        except json.JSONDecodeError:
            # 解析失败时降级：直接截断摘要
            return existing_summary + " | " + text[:200], self.state.entities

    def add_message(self, role: str, content: str):
        """添加新消息，必要时自动压缩"""
        self.state.working_memory.append(Message(role=role, content=content))

        # 触发压缩条件：超出轮数限制 或 token 估算超阈值
        should_compress = (
            len(self.state.working_memory) > self.working_memory_limit * 2
            or self._working_memory_tokens() > self.token_threshold
        )

        if should_compress:
            # 压缩前半部分，保留后半部分作为 working memory
            keep_count = self.working_memory_limit
            to_compress = self.state.working_memory[:-keep_count]
            self.state.working_memory = self.state.working_memory[-keep_count:]

            summary, entities = self._compress_old_messages(to_compress)
            self.state.summary = summary
            self.state.entities.update(entities)

    def build_context(self) -> list[dict]:
        """构建发给 LLM 的完整 messages 列表"""
        messages = []

        # 1. 注入长期记忆（实体）
        if self.state.entities:
            entity_text = json.dumps(self.state.entities, ensure_ascii=False, indent=2)
            messages.append({
                "role": "user",
                "content": f"[系统记忆 - 关键信息]\n{entity_text}"
            })
            messages.append({
                "role": "assistant",
                "content": "已记录关键信息，继续对话。"
            })

        # 2. 注入中期记忆（摘要）
        if self.state.summary:
            messages.append({
                "role": "user",
                "content": f"[对话摘要 - 历史背景]\n{self.state.summary}"
            })
            messages.append({
                "role": "assistant",
                "content": "已了解历史背景，继续。"
            })

        # 3. 注入短期记忆（完整 working memory）
        for msg in self.state.working_memory:
            messages.append({"role": msg.role, "content": msg.content})

        return messages

    def get_stats(self) -> dict:
        """返回当前记忆状态统计"""
        return {
            "working_memory_turns": len(self.state.working_memory),
            "working_memory_tokens": self._working_memory_tokens(),
            "has_summary": bool(self.state.summary),
            "entity_count": len(self.state.entities),
        }


def run_agent_with_memory():
    """展示如何在 Agent 对话循环中使用记忆管理器"""
    memory = ConversationMemoryManager(
        working_memory_limit=8,
        token_threshold=3000,
    )
    system_prompt = "你是一个智能客服助手。"

    print("Agent 已启动（输入 'quit' 退出，'stats' 查看记忆状态）\n")

    while True:
        user_input = input("用户: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "stats":
            print(f"记忆统计: {memory.get_stats()}\n")
            continue

        # 添加用户消息到记忆
        memory.add_message("user", user_input)

        # 构建包含记忆的完整上下文
        context_messages = memory.build_context()

        # 调用 LLM
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            messages=context_messages,
        )

        assistant_reply = response.content[0].text

        # 将 assistant 回复也存入记忆
        memory.add_message("assistant", assistant_reply)

        print(f"Agent: {assistant_reply}\n")


if __name__ == "__main__":
    run_agent_with_memory()
