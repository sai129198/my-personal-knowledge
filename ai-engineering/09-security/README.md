# 09 - 安全与合规

#topic/security #topic/safety #topic/compliance #year/2026 #status/canonical

> **一句话定位**：AI 系统的安全防线——从 Prompt 注入防御到伦理治理的完整安全体系。

---

## 目录

| 文件 | 状态 | 一句话摘要 |
|------|------|-----------|
| [prompt-injection-defense.md](prompt-injection-defense.md) | #status/canonical | Prompt 注入检测、输入隔离与多层防御架构 |
| [data-privacy-protection.md](data-privacy-protection.md) | #status/canonical | 差分隐私、联邦学习、数据脱敏与合规 |
| [model-safety-alignment.md](model-safety-alignment.md) | #status/canonical | RLHF、DPO、红队测试与持续对齐 |
| [content-moderation.md](content-moderation.md) | #status/canonical | 多层审核系统、图像审核与策略管理 |
| [ai-ethics-governance.md](ai-ethics-governance.md) | #status/canonical | 公平性、可解释性、AI 治理框架与法规合规 |
| [red-teaming-guide.md](red-teaming-guide.md) | #status/canonical | 攻击面分析、自动化红队测试与防御策略 |

---

## 核心概念

### AI 安全层次模型

```
┌─────────────────────────────────────────┐
│  Layer 5: 伦理治理 (Ethics & Governance) │
│  - 公平性、透明性、问责制                 │
├─────────────────────────────────────────┤
│  Layer 4: 内容安全 (Content Safety)      │
│  - 审核、过滤、人工复核                   │
├─────────────────────────────────────────┤
│  Layer 3: 模型对齐 (Model Alignment)     │
│  - RLHF、DPO、价值观对齐                  │
├─────────────────────────────────────────┤
│  Layer 2: 输入防御 (Input Defense)       │
│  - 注入检测、输入隔离、语义过滤            │
├─────────────────────────────────────────┤
│  Layer 1: 数据隐私 (Data Privacy)        │
│  - 脱敏、差分隐私、联邦学习               │
└─────────────────────────────────────────┘
```

### 安全开发流程

```
设计阶段 → 风险评估 + 伦理审查
    ↓
开发阶段 → 安全编码 + 红队测试
    ↓
部署阶段 → 监控告警 + 人工复核
    ↓
运营阶段 → 持续监控 + 迭代改进
```

---

## 关键决策矩阵

| 场景 | 推荐方案 | 注意事项 |
|------|----------|----------|
| Prompt 注入防御 | 多层防御（规则+语义+隔离） | 没有绝对安全 |
| 数据隐私保护 | 差分隐私 + 数据脱敏 | 合规是底线 |
| 模型对齐 | DPO（简单有效） | 持续迭代 |
| 内容审核 | 规则 + ML + 人工 | 平衡安全与体验 |
| 伦理治理 | 跨职能委员会 | 不只是合规 |
| 红队测试 | 自动化 + 人工 | 持续进行 |

---

## 与其他板块的关系

- **01-foundations** → 理解模型能力与局限
- **03-prompt-engineering** → Prompt 安全设计
- **05-agent-design** → Agent 安全边界
- **06-ai-safety** → 安全基础与护栏
- **08-data-engineering** → 数据隐私保护

---

## 待办

- [ ] 补充供应链安全（模型、依赖库）
- [ ] 补充安全事件响应流程
- [ ] 补充 AI 保险与风险转移

---

*最后更新：2026-06-12*
