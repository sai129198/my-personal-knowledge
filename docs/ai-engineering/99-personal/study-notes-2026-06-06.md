#topic/personal #year/2026 #status/draft

# 留学笔记：AI Infra 与应用层知识梳理

> 2026-06-06 留学记录。本次研究目标是梳理 AI Infra 层和应用层的知识模块，补全知识库的分层架构。

---

## 研究过程

### 知识缺口识别

上次留学梳理了 AI 领域的七层知识架构（基础概念 → 深度学习 → LLM → 应用架构 → 多模态 → 工程实践 → 前沿研究），但发现：
- Infra 层知识分散，缺乏系统性梳理
- 应用层知识更新快，需要结构化整理
- 知识库缺少对 AI 生态全景的覆盖

### 搜索路径

1. **Infra 架构**：Huyen Chip 的生成式 AI 平台架构分析 → 确认 Infra 核心组件
2. **行业动态**：DeepLearning.AI The Batch → 了解应用层最新趋势
3. **技术深度**：Anthropic Research → Agent、Computer Use 等前沿应用
4. **多模态**：Huyen Chip 多模态博客 → 视觉-语言融合应用

### 关键发现

1. **Infra 层正在快速成熟**：
   - 推理优化（KV Cache、Quantization、Speculative Decoding）成为 2026 年主战场
   - 向量数据库从"nice to have"变为"must have"
   - LLMOps 从 MLOps 演进，更关注 prompt 版本管理、RAG 评估

2. **应用层创新速度远超底层**：
   - 2024-2026 年 AI 应用呈现爆发式增长
   - 垂直领域（法律、医疗、金融）还有很大空间
   - 多模态应用（图像、视频、语音）还在早期

3. **Agent 是终极形态，但可靠性仍是挑战**：
   - 从"工具"到"助手"再到"代理"的演进趋势明显
   - 当前成功率 60-80%，错误累积是主要问题
   - Computer Use、Operator 等能力标志 Agent 进入物理世界

---

## 跨文章联系

| 主题 | 联系点 |
|------|--------|
| Infra 计算层 → 模型层 | GPU/TPU 等硬件决定模型训练规模和推理速度 |
| Infra 数据层 → RAG | 向量数据库是 RAG 的核心基础设施 |
| Infra 平台层 → 应用层 | MLOps/LLMOps 工具链支撑应用快速迭代 |
| 应用层内容生成 → 多模态 | 文本、图像、音视频生成技术融合 |
| Agent → 垂直行业 | 专业 Agent 需要领域知识 + 工具调用能力 |

## 矛盾点

1. **通用 vs. 垂直**：通用 AI 助手（ChatGPT）已红海，但垂直领域应用（法律、医疗）门槛高、机会大
2. **AI Native vs. AI Enhanced**：AI Native 产品（Perplexity）颠覆性强，但 AI Enhanced（Office Copilot）更容易变现
3. **自主 vs. 可控**：Agent 越自主效率越高，但可靠性越低，如何在两者之间平衡是核心挑战

---

## 💡 我的思考

1. **Infra 是护城河，应用是价值放大器**：模型能力同质化后，Infra 优化（延迟、成本、稳定性）成为竞争关键；应用层则决定技术能否转化为用户价值。

2. **"套壳"创造真实价值**：Perplexity、Cursor 都是基于现有模型，但通过深刻理解用户场景和优秀的产品体验，创造了巨大价值。技术不是唯一壁垒。

3. **多模态应用是下一个爆发点**：文本应用已相对成熟，图像、视频、语音的 AI 应用还在早期，机会更多。特别是具身智能（机器人、自动驾驶）可能带来最大变革。

4. **Agent 需要"人在回路"**：当前 Agent 可靠性不足，生产环境应设置人工检查点。完全自主的 Agent 还需要 2-3 年才能成熟。

5. **知识库本身也是 AI 应用**：这个知识库可以作为个人 RAG 系统的知识源，实现"用 AI 管理 AI 知识"的闭环。下一步可以考虑接入本地 LLM 实现问答功能。

---

## 待深入研究

- [ ] LLM 推理优化技术详解（PageAttention、Continuous Batching 等）
- [ ] 向量数据库的 ANN 算法原理（HNSW、IVF、ScaNN）
- [ ] 模型微调技术对比（Full Fine-tuning vs. LoRA vs. Prefix Tuning）
- [ ] AI 安全与对齐技术（RLHF、DPO、Constitutional AI）
- [ ] 多模态模型架构（CLIP、Flamingo、LLaVA）
- [ ] AI 硬件生态（GPU、TPU、NPU、专用芯片）

---

## 参考来源

- [Huyen Chip: Building A Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html) — 访问日期：2026-06-05
- [DeepLearning.AI The Batch Issue 355](https://www.deeplearning.ai/the-batch/issue-355) — 访问日期：2026-06-05
- [Anthropic Research](https://www.anthropic.com/research) — 访问日期：2026-06-05
- [Huyen Chip: Multimodality and Large Multimodal Models](https://huyenchip.com/2023/10/10/multimodal.html) — 访问日期：2026-06-05
- [OpenAI Products](https://openai.com/products) — 访问日期：2026-06-05
