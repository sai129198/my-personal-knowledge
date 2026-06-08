#topic/llm #topic/decision-framework #year/2026 #status/draft

# 模型选型决策框架

> 基于场景、成本、性能的多维度模型选型指南。

---

## 1. 决策树

```
开始选型
    │
    ├─ 预算敏感？
    │   ├─ 是 → DeepSeek-V3 / Gemini 2.0 Flash / GPT-4.1 nano
    │   └─ 否 → 继续
    │
    ├─ 需要多模态（图像/视频）？
    │   ├─ 是 → Gemini 2.5 Pro / GPT-4o
    │   └─ 否 → 继续
    │
    ├─ 需要超长上下文（>200K）？
    │   ├─ 是 → Gemini 2.5 Pro / GPT-4.1（1M tokens）
    │   └─ 否 → 继续
    │
    ├─ 主要任务是编码？
    │   ├─ 是 → Claude 4 Sonnet / GPT-4.1 / DeepSeek-R1
    │   └─ 否 → 继续
    │
    ├─ 需要深度推理？
    │   ├─ 是 → o1 / o3-mini / DeepSeek-R1
    │   └─ 否 → 继续
    │
    ├─ 需要高安全性？
    │   ├─ 是 → Claude 4 Sonnet
    │   └─ 否 → 继续
    │
    └─ 默认推荐 → GPT-4o（通用场景）
```

---

## 2. 场景对照表

| 场景 | 首选 | 备选 | 理由 |
|------|------|------|------|
| **通用对话** | GPT-4o | Claude 4 Sonnet | 平衡性能与成本 |
| **代码生成** | Claude 4 Sonnet | GPT-4.1 | 代码能力最强 |
| **代码审查** | GPT-4.1 | Claude 4 Sonnet | 1M 上下文 + 编码优化 |
| **数学推理** | o1 / o3-mini | DeepSeek-R1 | 深度推理能力 |
| **长文档分析** | Gemini 2.5 Pro | GPT-4.1 | 1M 上下文 |
| **视频理解** | Gemini 2.5 Pro | GPT-4o | 原生多模态 |
| **实时搜索** | Gemini 2.5 Pro | - | Grounding 功能 |
| **中文内容** | DeepSeek-V3 | Claude 4 Sonnet | 中文优化 |
| **创意写作** | Claude 4 Opus | GPT-4.5 | 文学性强 |
| **Agent 系统** | Claude 4 Sonnet | GPT-4.1 | 工具调用稳定 |
| **高频低延迟** | GPT-4.1 nano | Gemini 2.0 Flash | 极速响应 |
| **私有化部署** | DeepSeek-V3 | Gemma 3 | 开源可本地运行 |
| **教育/研究** | DeepSeek-R1 | Gemma 3 | 开源可复现 |

---

## 3. 成本对比

### 每 1M tokens 价格（输入/输出）

| 模型 | 价格 | 性价比评级 |
|------|------|-----------|
| DeepSeek-V3 | $0.14 / $0.28 | ⭐⭐⭐⭐⭐ |
| Gemini 2.0 Flash | $0.075 / $0.30 | ⭐⭐⭐⭐⭐ |
| GPT-4.1 nano | $0.10 / $0.40 | ⭐⭐⭐⭐⭐ |
| GPT-4o-mini | $0.15 / $0.60 | ⭐⭐⭐⭐ |
| GPT-4.1 mini | $0.40 / $1.60 | ⭐⭐⭐⭐ |
| Gemini 2.5 Flash | $0.15 / $0.60 | ⭐⭐⭐⭐ |
| GPT-4o | $2.50 / $10.00 | ⭐⭐⭐ |
| GPT-4.1 | $2.00 / $8.00 | ⭐⭐⭐ |
| Claude 4 Sonnet | $3.00 / $15.00 | ⭐⭐ |
| Gemini 2.5 Pro | $1.25 / $10.00 | ⭐⭐⭐ |
| o3-mini | $1.10 / $4.40 | ⭐⭐⭐ |
| o1 | $15.00 / $60.00 | ⭐ |
| Claude 4 Opus | $15.00 / $75.00 | ⭐ |

### 月度成本估算（100M tokens/月）

| 方案 | 成本 | 适用场景 |
|------|------|----------|
| DeepSeek-V3 | ~$21K | 初创公司、预算敏感 |
| GPT-4o-mini | ~$38K | 高频调用、简单任务 |
| GPT-4o | ~$625K | 通用应用 |
| Claude 4 Sonnet | ~$900K | 企业级应用 |
| o1 | ~$3.75M | 深度研究 |

---

## 4. 混合策略

### 4.1 路由策略（Routing）

根据任务复杂度自动选择模型：

```python
def route_query(query: str, complexity: float) -> str:
    if complexity < 0.3:
        return "gpt-4.1-nano"  # 简单任务，极速响应
    elif complexity < 0.7:
        return "gpt-4o"        # 中等任务，平衡性能
    elif complexity < 0.9:
        return "claude-4-sonnet"  # 复杂任务，代码/推理
    else:
        return "o1"            # 深度推理
```

### 4.2 降级策略（Fallback）

主模型失败时自动降级：

```
主模型: Claude 4 Sonnet
  ↓ 超时/失败
降级 1: GPT-4o
  ↓ 失败
降级 2: GPT-4o-mini
  ↓ 失败
本地模型: DeepSeek-V3
```

### 4.3 缓存策略

- **Prompt Caching**：重复请求复用已计算的 KV Cache
- **结果缓存**：相同输入直接返回缓存结果
- **预计可节省 30-50% 成本**

---

## 5. 评估 checklist

### 5.1 功能评估

- [ ] 是否支持所需的语言（中文/英文/其他）？
- [ ] 是否支持所需的模态（文本/图像/音频/视频）？
- [ ] 上下文长度是否满足需求？
- [ ] 是否支持工具调用（Function Calling）？
- [ ] 是否支持结构化输出（JSON Schema）？

### 5.2 性能评估

- [ ] 在目标数据集上测试准确率
- [ ] 测试延迟（首 token 时间、总生成时间）
- [ ] 测试吞吐量（tokens/秒）
- [ ] 测试长上下文下的"Lost in the Middle"问题

### 5.3 成本评估

- [ ] 计算月度 token 消耗量
- [ ] 对比不同模型的成本
- [ ] 评估是否需要缓存策略
- [ ] 预留 20% 的成本缓冲

### 5.4 合规评估

- [ ] 数据隐私要求（是否需要私有化部署）？
- [ ] 行业合规要求（金融/医疗/法律）？
- [ ] 地域限制（国内/海外）？
- [ ] 内容安全要求？

---

## 💡 我的思考

1. **没有"最好"的模型，只有"最合适"的模型**：每个模型都有独特的优势和适用场景。选型时要综合考虑功能、性能、成本、合规等多个维度。

2. **混合策略是未来趋势**：单一模型难以满足所有需求。通过路由、降级、缓存等策略，可以在成本和性能之间找到最佳平衡。

3. **成本优化是持续过程**：随着新模型发布，价格会快速变化。建议每季度重新评估一次模型选型。

4. **开源模型正在缩小差距**：DeepSeek、Qwen、Gemma 等开源模型在快速追赶，对于预算敏感的场景，开源是不错的选择。

5. **长上下文是新的竞争维度**：1M tokens 上下文让整本书分析、大型代码库理解成为可能，这在法律、金融、软件工程领域是革命性的。

---

## 参考来源

- [LMSYS Chatbot Arena](https://chat.lmsys.org/) — 访问日期：2026-06-06
- [Artificial Analysis](https://artificialanalysis.ai/) — 访问日期：2026-06-06
- [各厂商官方定价页](https://openai.com/api/pricing/) — 访问日期：2026-06-06
