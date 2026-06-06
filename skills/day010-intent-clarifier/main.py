"""
Day 010: Intent Clarifier — 用户意图澄清器
用法: python main.py
"""

from intent_clarifier import IntentClarifier, clarify_and_execute, RiskLevel


def demo_rule_based():
    """演示基于规则的意图解析（无需 LLM）。"""
    print("=" * 55)
    print("Demo 1: 规则解析")
    print("=" * 55)

    clarifier = IntentClarifier()
    test_cases = [
        "帮我写个报告",
        "delete all old log files",
        "translate this document to English",
        "send the summary to my team",
        "analyze this CSV briefly",
    ]

    for req in test_cases:
        intent = clarifier.parse(req)
        questions = clarifier.get_clarification_questions(intent)
        print(f"\n📝 请求: {req!r}")
        print(f"   动作: {intent.action}  |  对象: {intent.target!r}  |  风险: {intent.risk.value}")
        print(f"   缺失: {intent.missing}")
        if questions:
            for q in questions:
                opts = f" {q.options}" if q.options else ""
                print(f"   ❓ {q.question}{opts}")
        else:
            print("   ✅ 意图清晰，可直接执行")


def demo_interactive():
    """交互式演示：完整澄清 → 执行流程。"""
    print("\n" + "=" * 55)
    print("Demo 2: 交互式澄清")
    print("=" * 55)

    def mock_executor(intent):
        print(f"\n🚀 执行完成!")
        print(f"   操作: {intent.action} {intent.target}")
        print(f"   参数: {intent.constraints}")
        return "success"

    # 模拟用户回答（自动化测试用，真实场景会等待用户输入）
    answers_queue = ["Markdown", "中等（300-800字）"]

    def mock_ask_user(question, options, default):
        ans = answers_queue.pop(0) if answers_queue else (default or "")
        print(f"\n❓ {question}")
        if options:
            print(f"   选项: {options}")
        print(f"   用户回答: {ans!r}")
        return ans

    clarify_and_execute(
        user_request="帮我写个报告",
        executor=mock_executor,
        ask_user=mock_ask_user,
    )


def demo_high_risk():
    """演示高风险操作的确认机制。"""
    print("\n" + "=" * 55)
    print("Demo 3: 高风险操作确认")
    print("=" * 55)

    clarifier = IntentClarifier()
    intent = clarifier.parse("delete all user records from the database")
    print(f"\n📝 请求: {intent.raw_request!r}")
    print(f"   风险级别: {intent.risk.value.upper()}")

    questions = clarifier.get_clarification_questions(intent)
    print(f"\n生成了 {len(questions)} 个澄清问题:")
    for q in questions:
        print(f"  ❓ {q.question}")
        if q.options:
            print(f"     选项: {q.options}")
        print(f"     默认: {q.default}")


if __name__ == "__main__":
    demo_rule_based()
    demo_interactive()
    demo_high_risk()

    print("\n" + "=" * 55)
    print("✅ Intent Clarifier Demo 完成")
    print("   导入 IntentClarifier 或 clarify_and_execute 即可集成")
    print("=" * 55)
