# 🤖 AI Agent Skills Daily

> 每天一个 skill，解决 AI agent 开发中最真实的痛点

**每天墨尔本时间 2am 自动更新** · 持续更新中

---

## 为什么有这个项目？

AI agent 在真实生产环境中面临的问题远比 demo 复杂：
- 输出格式不稳定，下游解析崩溃
- Token 消耗失控，成本爆炸
- 幻觉率居高不下，可信度低
- 工具调用失败后不知如何恢复
- 长对话记忆丢失，上下文断裂
- ...

这个项目每天针对一个具体痛点，提供**可以直接用的解决方案**：完整代码 + 原理解释 + 集成示例。

---

## Skills 列表

| # | Skill | 解决的痛点 | 日期 |
|---|-------|-----------|------|
| 001 | [Structured Output Enforcer](skills/day001-structured-output-enforcer/SKILL.md) | LLM 输出格式不稳定，JSON 解析失败 | 2026-05-28 |
| 002 | [Token Budget Optimizer](skills/day002-token-budget-optimizer/SKILL.md) | Token 消耗失控，API 成本爆炸 | 2026-05-29 |

---

## 使用方式

每个 skill 包含：
- `SKILL.md` — 问题描述、解决思路、完整代码、集成示例
- `*.py` / `*.ts` — 可直接运行的代码文件

直接复制代码到你的项目中使用即可。

---

## 话题覆盖

- 🔧 **稳定性**: 输出格式、错误恢复、重试策略
- 💰 **成本**: Token 压缩、缓存、批处理
- 🎯 **准确性**: 幻觉检测、事实核验、置信度评估
- 🧠 **记忆**: 上下文管理、长对话压缩、知识持久化
- 🔄 **流程**: 工具调用、并发、状态机

---

*Built in Melbourne, Australia 🇦🇺 · Powered by Claude*
