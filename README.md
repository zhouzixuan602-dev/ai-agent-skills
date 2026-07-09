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
| 003 | [Hallucination Reducer](skills/day003-hallucination-reducer/SKILL.md) | AI 自信给出错误事实，幻觉率居高不下 | 2026-05-30 |
| 004 | [Tool Call Recovery](skills/day004-tool-call-recovery/SKILL.md) | 工具调用失败时缺乏分类处理和降级兜底 | 2026-05-31 |
| 005 | [Memory Manager](skills/day005-memory-manager/SKILL.md) | 长对话超出上下文窗口，Agent "失忆" | 2026-06-01 |
| 006 | [Rate Limiter & Task Queue](skills/day006-rate-limiter/SKILL.md) | 并发任务触发 API rate limit，重试风暴导致大量请求失败 | 2026-06-02 |
| 007 | [Cost Monitor](skills/day007-cost-monitor/SKILL.md) | Agent 跑完才发现烧了几十美元，缺乏实时成本可见性和预算熔断 | 2026-06-03 |
| 008 | [Multi-Model Router](skills/day008-multi-model-router/SKILL.md) | 所有请求都打到最贵的模型，成本高却没有额外收益 | 2026-06-04 |
| 009 | [Structured Logger](skills/day009-structured-logger/SKILL.md) | Agent 出错时日志杂乱，无法快速定位哪个工具调用或 LLM 请求出了问题 | 2026-06-06 |
| 010 | [Intent Clarifier](skills/day010-intent-clarifier/SKILL.md) | 模糊请求导致 Agent 猜错需求，浪费多轮 token 甚至产出完全偏离的结果 | 2026-06-07 |
| 011 | [Prompt Injection Guard](skills/day011-prompt-injection-guard/SKILL.md) | Agent 处理外部内容时被恶意指令劫持，执行未授权操作 | 2026-06-08 |
| 012 | [Streaming Handler](skills/day012-streaming-handler/SKILL.md) | 流式响应中断无法恢复、Partial JSON 崩溃、进度完全不透明 | 2026-06-09 |
| 013 | [Tool Result Validator](_standalone/tool-result-validator/README.md) | 工具返回值校验缺失，null/类型错误/注入内容直接污染 LLM 上下文 | 2026-06-11 |
| 014 | [Context Window Manager](_standalone/context-window-manager/README.md) | 长对话超出上下文限制导致 BadRequestError，历史压缩不当丢失关键信息 | 2026-06-12 |
| 015 | [Agent Test Framework](_standalone/agent-test-framework/README.md) | Agent 每次改 prompt 都靠手测，无法自动捕获工具调用回归和行为漂移 | 2026-06-13 |
| 016 | [LLM Hallucination Detector](_standalone/llm-hallucination-detector/README.md) | LLM 用同等自信心输出错误事实，置信度评分+来源核查+自洽性检测三层防护 | 2026-06-14 |
| 017 | [Agent Cost Guard](_standalone/agent-cost-guard/README.md) | Agent 静默耗尽 API 预算，三层限额（单次/会话/累计）在调用前拦截超支 | 2026-06-15 |
| 018 | [Semantic Cache](_standalone/semantic-cache/README.md) | 相同意图的问题重复付费，语义相似度缓存让 LLM 调用成本降低 60-80% | 2026-06-16 |
| 019 | [Prompt Compressor](_standalone/prompt-compressor/README.md) | Agent prompt 因重复指令和冗余历史膨胀至数万 token，多策略压缩节省 60-75% 费用 | 2026-06-23 |
| 020 | [Response Evaluator](_standalone/response-evaluator/README.md) | Agent 自信输出错误答案你却不知道，LLM-as-Judge 四维评分在响应到达用户前自动拦截劣质输出 | 2026-07-03 |
| 021 | [Prompt Injection Shield](_standalone/prompt-injection-shield/README.md) | Agent 读取网页/邮件时被隐藏指令劫持，四层防护：归一化反混淆+签名扫描+LLM 裁判+随机边界围栏 | 2026-07-06 |
| 022 | [Streaming Checkpoint Recovery](_standalone/streaming-checkpoint-recovery/README.md) | 流式响应中途断连后重试从零开始，输出 token 重复付费且用户看到答案重来；checkpoint + assistant prefill 让模型从断点续写 | 2026-07-07 |
| 023 | [Agent Circuit Breaker](_standalone/agent-circuit-breaker/README.md) | 一个 flaky 工具拖垮整个 agent run：重试风暴打挂已宕机的 API、fatal 错误被无谓重试；错误分类+指数退避+熔断器+降级链三层防护 | 2026-07-08 |
| 024 | [LLM Cascade Router](_standalone/llm-cascade-router/README.md) | 所有请求都走最贵的模型，90% 流量为简单任务多付 3-15 倍费用；启发式预路由+便宜模型先试+低置信度自动升级，省 50-75% 成本 | 2026-07-10 |

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
