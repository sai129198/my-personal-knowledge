#topic/personal #year/2026 #status/draft

# 留学笔记：AI 领域知识体系梳理

> 2026-06-05 留学记录。本次研究目标是梳理 AI 领域完整知识体系，为个人知识库建立基础框架。

---

## 研究过程

### 知识缺口识别

在搭建知识库时，我发现自己对 AI 领域的整体知识结构不够清晰：
- 知道 Transformer，但说不清它为什么成为现代 AI 的基石
- 用过 RAG，但对其架构组件和优化方法了解不深
- 听说过 Agent，但分不清 ReAct 和 Plan-and-Execute 的区别
- 对多模态 AI 的发展现状缺乏系统认知

### 搜索路径

1. **基础框架**：Microsoft AI For Beginners → 确认 AI 知识的分层结构
2. **行业动态**：DeepLearning.AI The Batch → 了解 2026 年最新趋势
3. **技术深度**：Huyen Chip 博客 → RAG 架构、多模态技术细节
4. **厂商研究**：Anthropic Research → AI 安全、Agent 最新进展
5. **论文跟踪**：arXiv cs.AI → 确认研究方向和热点

### 关键发现

1. **AI Engineer 角色正在分化**：Andrew Ng 预测将从通才分化为 Evals Engineer、LLMOps Engineer 等专门角色
2. **RAG 仍是企业应用首选**：比微调更实用，解决知识更新和幻觉问题
3. **Agent 可靠性是最大瓶颈**：当前成功率 60-80%，错误累积问题未解决
4. **多模态是下一个突破点**：文本 AI 已成熟，视觉-语言融合和具身智能是前沿

---

## 跨文章联系

| 主题 | 联系点 |
|------|--------|
| Transformer → LLM | Transformer 是 LLM 的架构基础，自注意力机制让模型能处理长序列 |
| LLM → RAG | RAG 解决 LLM 的知识时效性和幻觉问题，是 LLM 应用的核心架构 |
| RAG → Agent | Agent 可以使用 RAG 作为知识检索工具，增强决策能力 |
| Agent → Multi-Agent | 单 Agent 能力有限，Multi-Agent 模拟团队协作解决复杂问题 |
| 多模态 → Agent | Computer Use 让 Agent 能操作计算机，从文本走向物理世界 |

## 矛盾点

1. **安全性 vs. 能力**：Anthropic 的 Constitutional AI 提升安全性，但可能过度限制输出（过度拒绝问题）
2. **开源 vs. 闭源**：开源模型（Llama、DeepSeek）快速追赶，但闭源模型（GPT-4o、Claude）仍领先
3. **Agent 自主性 vs. 可靠性**：越自主的 Agent 越容易出错，如何在两者之间平衡是核心挑战

---

## 💡 我的思考

1. **知识分层很重要**：将 AI 知识分为 7 层（基础概念 → 深度学习 → LLM → 应用架构 → 多模态 → 工程实践 → 前沿研究），有助于建立清晰的学习路径。

2. **实践 > 理论**：RAG 和 Agent 的知识不能只停留在概念层面，需要通过实际项目来深化理解。下一步应该搭建一个完整的 RAG 系统。

3. **关注评估**：随着 AI 应用成熟，评估（Evaluation）变得越来越重要。这是目前知识库中缺失的部分，需要补充。

4. **中文资料不足**：虽然留学规则要求用英文搜索，但发现高质量的中文 AI 技术资料确实稀缺。这也是个人知识库的价值所在——将英文资料消化后用中文整理。

5. **知识库本身也是 RAG**：这个知识库未来可以作为个人 RAG 系统的知识源，实现"用 AI 管理 AI 知识"的闭环。

---

## 待深入研究

- [ ] 模型评估指标详解（Perplexity、BLEU、ROUGE、HumanEval 等）
- [ ] 模型微调（Fine-tuning）vs. RAG 的决策框架
- [ ] 向量数据库的 ANN 算法原理（HNSW、IVF 等）
- [ ] LLM 推理优化（KV Cache、Quantization、Speculative Decoding）
- [ ] AI 安全与对齐技术（RLHF、DPO、Constitutional AI）

---

## 参考来源

- [Microsoft AI For Beginners](https://github.com/microsoft/AI-For-Beginners) — 访问日期：2026-06-05
- [DeepLearning.AI The Batch Issue 355](https://www.deeplearning.ai/the-batch/issue-355) — 访问日期：2026-06-05
- [Anthropic Research](https://www.anthropic.com/research) — 访问日期：2026-06-05
- [Huyen Chip: Multimodality and Large Multimodal Models](https://huyenchip.com/2023/10/10/multimodal.html) — 访问日期：2026-06-05
- [Huyen Chip: Building A Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html) — 访问日期：2026-06-05
- [arXiv AI Papers](https://arxiv.org/list/cs.AI/recent) — 访问日期：2026-06-05
