# 留学笔记：大模型原理、训练与微调

#topic/study-notes #topic/llm #topic/training #topic/finetuning #year/2026 #status/draft

**留学时间**: 2026-06-06
**留学主题**: 大模型原理、训练流程、微调技术
**资料来源**: Huyen Chip 博客、DeepLearning.AI The Batch、Anthropic Research

---

## 今日收获

### 1. 大模型核心原理

**关键洞察**：
- 大模型的能力主要来自"下一个词预测"这一简单目标的规模化
- 当模型参数达到某个阈值时，会突然出现上下文学习、推理等涌现能力
- 数据质量和多样性比单纯的数据量更重要

**Scaling Laws**：
- 模型性能与参数量、数据量、计算量呈幂律关系
- 这意味着：更大的模型通常更好，但边际收益递减

### 2. 训练三阶段

```
预训练（Pre-training）→ 学习通用知识和语言规律
    ↓
监督微调（SFT）→ 学习遵循指令和对话格式
    ↓
对齐训练（RLHF/DPO）→ 符合人类价值观
```

**关键发现**：
- 预训练是最耗资源但最关键的阶段
- SFT 数据质量比数量更重要（通常 10K-100K 高质量样本）
- DPO 是 RLHF 的有效替代，更简单且效果接近

### 3. 微调技术对比

| 方法 | 训练参数量 | 显存需求 | 适用场景 |
|------|-----------|---------|---------|
| Full Fine-tuning | 100% | 极高 | 大量数据、深度定制 |
| LoRA | <1% | 中等 | 大多数场景的最佳选择 |
| QLoRA | <1% | 低（4-bit） | 消费级 GPU 微调大模型 |
| Prompt Tuning | <0.1% | 极低 | 简单任务适配 |

**实践建议**：
- 优先尝试 LoRA/QLoRA
- rank 通常 8-64 足够
- 学习率 1e-4 左右
- 1-3 个 epoch 即可，避免过拟合

### 4. 行业动态

从 DeepLearning.AI The Batch 了解到：
- **AI Forward Deployed Engineer (FDE)** 成为新兴热门职位
- AI Engineer 需求激增，未来可能分化出更多专业角色
- 编码 Agent（Claude Code、Codex 等）正在改变软件开发方式

---

## 待深入研究

- [ ] LLM 推理优化技术详解（KV Cache、PageAttention、Continuous Batching）
- [ ] 模型评估指标详解（Perplexity、BLEU、ROUGE、HumanEval）
- [ ] 多模态模型架构（CLIP、Flamingo、LLaVA）
- [ ] 模型量化技术（INT8、INT4、GPTQ、AWQ）
- [ ] 分布式训练框架（DeepSpeed、FSDP、Megatron-LM）

---

## 参考资料

1. Huyen Chip - Building A Generative AI Platform (2024-07-25)
2. Huyen Chip - Multimodality and Large Multimodal Models (2023-10-10)
3. DeepLearning.AI The Batch - Issue 355 (2026-05-22)
4. Anthropic Research - Teaching Claude why (2026-05-08)

---

*访问日期: 2026-06-06*

## 💡 我的思考

1. **预训练是根基，微调是适配**：理解大模型的能力来源有助于设定合理的微调预期。不要期望通过微调让模型获得全新的能力，微调更多是调整行为模式。

2. **LoRA 改变了游戏规则**：QLoRA 让个人开发者也能在消费级硬件上微调 70B 模型，这极大地降低了 AI 应用的门槛。

3. **数据质量是微调成功的关键**：与其追求数据量，不如花时间清洗和审核数据。1000 条高质量数据 > 10000 条低质量数据。

4. **评估是容易被忽视但至关重要的环节**：没有好的评估体系，就无法判断微调是否成功，也无法迭代优化。
