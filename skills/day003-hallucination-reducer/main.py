"""
Day 003: Hallucination Reducer
降低 AI Agent 幻觉率：事实核验 + 置信度评分 + 来源引用
"""

import re
import json
from dataclasses import dataclass, field
from typing import Optional
from anthropic import Anthropic

client = Anthropic()

# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class Claim:
    text: str
    confidence: float       # 0.0 - 1.0
    verifiable: bool        # 是否可通过外部工具核验
    source: Optional[str] = None
    verified: Optional[bool] = None

@dataclass
class VerifiedResponse:
    original: str
    claims: list[Claim] = field(default_factory=list)
    overall_confidence: float = 0.0
    safe_answer: str = ""       # 过滤掉低置信度内容后的安全回答
    flagged_parts: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Step 1: 提取 claims 并评分
# ─────────────────────────────────────────────

EXTRACT_PROMPT = """你是一个事实核查专家。分析以下 AI 回答，提取其中所有"可验证的事实性声明"，并对每条声明的置信度评分。

规则：
- 只提取具体的事实声明（数字、时间、名称、因果关系等）
- 不提取观点性表述
- 置信度基于：训练数据的覆盖程度、声明的具体性、是否有内在矛盾

以 JSON 格式返回，结构如下：
{
  "claims": [
    {
      "text": "声明原文片段",
      "confidence": 0.85,
      "verifiable": true,
      "reason": "置信度理由"
    }
  ]
}

AI 回答：
<answer>
{answer}
</answer>
"""


def extract_claims(answer: str) -> list[Claim]:
    """从 AI 回答中提取事实声明并评分"""
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",   # 用便宜模型做元分析
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": EXTRACT_PROMPT.format(answer=answer)
        }]
    )

    raw = resp.content[0].text.strip()
    # 提取 JSON（防止模型在 JSON 前后加解释文字）
    match = re.search(r'\{[\s\S]*\}', raw)
    if not match:
        return []

    data = json.loads(match.group())
    claims = []
    for c in data.get("claims", []):
        claims.append(Claim(
            text=c["text"],
            confidence=float(c["confidence"]),
            verifiable=bool(c["verifiable"]),
        ))
    return claims


# ─────────────────────────────────────────────
# Step 2: 对高价值 claim 做二次核验（Self-consistency）
# ─────────────────────────────────────────────

VERIFY_PROMPT = """请独立回答以下问题，不要参考任何已有说法：

"{claim}"

请直接回答"是"或"否"，并给出 1-2 句理由。
格式：
VERDICT: 是/否/不确定
REASON: ...
"""


def self_verify_claim(claim: Claim) -> Claim:
    """用独立问答核验 claim，若矛盾则降低置信度"""
    if claim.confidence > 0.9 or not claim.verifiable:
        # 高置信度或不可验证的 claim 跳过核验
        return claim

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": VERIFY_PROMPT.format(claim=claim.text)
        }]
    )

    text = resp.content[0].text
    if "VERDICT: 否" in text or "VERDICT: 不确定" in text:
        claim.confidence = max(0.0, claim.confidence - 0.3)
        claim.verified = False
    elif "VERDICT: 是" in text:
        claim.confidence = min(1.0, claim.confidence + 0.05)
        claim.verified = True

    return claim


# ─────────────────────────────────────────────
# Step 3: 生成安全回答（过滤低置信声明）
# ─────────────────────────────────────────────

SAFE_ANSWER_PROMPT = """以下是一个 AI 回答，以及其中被标记为低置信度的片段（可能是幻觉）。

原始回答：
<answer>{answer}</answer>

低置信度片段（请在回答中移除或加上不确定性修辞）：
{flagged}

请生成一个修订版回答：
- 移除或软化低置信度的断言
- 对不确定内容加上"据我了解"、"可能"、"建议查证"等措辞
- 保持回答流畅自然
- 不要添加新内容
"""


def build_safe_answer(answer: str, claims: list[Claim], threshold: float = 0.6) -> tuple[str, list[str]]:
    """生成移除幻觉内容的安全版回答"""
    flagged = [c.text for c in claims if c.confidence < threshold]

    if not flagged:
        return answer, []

    flagged_block = "\n".join(f"- {f}" for f in flagged)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": SAFE_ANSWER_PROMPT.format(
                answer=answer,
                flagged=flagged_block
            )
        }]
    )

    return resp.content[0].text.strip(), flagged


# ─────────────────────────────────────────────
# 主入口：完整流水线
# ─────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.6   # 低于此分值视为幻觉风险

def reduce_hallucination(user_question: str) -> VerifiedResponse:
    """
    三步降幻觉流水线：
    1. 生成原始回答
    2. 提取并核验事实声明
    3. 输出安全版本
    """
    # Step 1: 生成原始回答
    raw_resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": user_question}]
    )
    original = raw_resp.content[0].text

    # Step 2: 提取 claims + 评分
    claims = extract_claims(original)

    # Step 3: 对中低置信度 claim 做自我核验
    verified_claims = []
    for c in claims:
        if c.confidence < 0.85:
            c = self_verify_claim(c)
        verified_claims.append(c)

    # Step 4: 计算整体置信度
    if verified_claims:
        overall = sum(c.confidence for c in verified_claims) / len(verified_claims)
    else:
        overall = 1.0

    # Step 5: 生成安全回答
    safe, flagged = build_safe_answer(original, verified_claims, CONFIDENCE_THRESHOLD)

    return VerifiedResponse(
        original=original,
        claims=verified_claims,
        overall_confidence=overall,
        safe_answer=safe,
        flagged_parts=flagged
    )


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    question = "Python 是什么时候发布的？它的创始人是谁？现在最新版本是多少？"

    print("🔍 正在分析回答中的幻觉风险...\n")
    result = reduce_hallucination(question)

    print("=" * 60)
    print("📝 原始回答：")
    print(result.original)

    print("\n📊 事实声明分析：")
    for c in result.claims:
        icon = "✅" if c.confidence >= CONFIDENCE_THRESHOLD else "⚠️"
        print(f"  {icon} [{c.confidence:.0%}] {c.text}")

    print(f"\n🎯 整体置信度：{result.overall_confidence:.0%}")

    if result.flagged_parts:
        print(f"\n🚩 标记为高风险的声明（{len(result.flagged_parts)} 条）：")
        for f in result.flagged_parts:
            print(f"  - {f}")

        print("\n✅ 安全版回答：")
        print(result.safe_answer)
    else:
        print("\n✅ 未检测到高风险幻觉，原始回答可信度高")
