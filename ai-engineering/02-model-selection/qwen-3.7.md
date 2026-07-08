# Qwen 3.7 深度研究报告

> 研究时间: 2026-07-08
> 来源: 阿里云百炼官方文档, Qwen 官方博客 (qwen.ai), Qwen GitHub, OpenRouter, Qwen readthedocs, Alibaba Cloud Summit 2026
> 标签: #status/canonical #topic/model-selection #topic/llm #topic/qwen #year/2026

---

## 一句话总结

**Qwen 3.7 是阿里云通义千问团队的旗舰大模型系列**，于 2026 年 5 月在阿里云峰会上正式发布。包含两个变体：**Qwen3.7-Max**（闭源旗舰推理模型，纯文本）和 **Qwen3.7-Plus**（多模态模型，文本+视觉输入），两者均支持 1M token 上下文窗口。**未开放权重**，仅通过 API 提供服务。在阿里云百炼的推荐中，它被定位为对位 GPT-5.5 和 Claude Opus 4.7 的最高能力档模型。

---

## 模型定位与演进

### Qwen 系列完整演进路线

| 版本 | 时间 | 参数量 | 核心特性 | 开放情况 |
|------|------|--------|----------|----------|
| Qwen2.5 | 2024-2025 | 多种规格 (0.5B-72B) | 大系列，MoE + Dense | Apache 2.0 开源 |
| Qwen2.5-Max | 2025.01 | 未公开 | 旗舰闭源模型 | 仅 API |
| **Qwen3** (2504) | 2025.04 | Dense: 0.6B-32B; MoE: 30B-A3B, 235B-A22B | 混合思考模式（thinking/non-thinking 自由切换），119 语言，预训练 36T tokens | Apache 2.0 开源 |
| Qwen3.5 | 2026.02 | MoE: 397B-A17B, 122B-A10B, 35B-A3B; Dense: 27B, 9B, 4B, 2B, 0.8B | 统一视觉-语言基础，Gated Delta Networks，201 语言，近 100% 多模态训练效率 | Apache 2.0 开源 |
| **Qwen3.6** | 2026.04 | MoE: 35B-A3B; Dense: 27B | Agentic Coding，Thinking 上下文保留（跨会话），稳定性与实用性优先 | Apache 2.0 开源 |
| **Qwen3.7-Max** | **2026.05.20** | **未公开（闭源）** | **1M 上下文，旗舰推理，纯文本，Intelligence Index 56.6** | **仅 API（闭源）** |
| **Qwen3.7-Plus** | **2026.06** | **未公开（闭源）** | **1M 上下文，多模态（文本+图像+视频），性价比均衡** | **仅 API（闭源）** |

### 定位分析

Qwen 3.7 是 Qwen 系列的一个重要转折点：**从"开源为主"转向"闭源旗舰 + 开源工具体系"双轨策略**。

- **Qwen3.7-Max**：在阿里云百炼中，被定位为对位 GPT-5.5 / Claude Opus 4.7 / Gemini 3.1 Pro 的「高能力」档
- **Qwen3.7-Plus**：对位 GPT-5.4 / Claude Sonnet 4.6 / Gemini 3 Pro 的「平衡」档，同时兼具多模态能力
- **Qwen3.6-Flash**：对位 GPT-5.4-mini / Claude Haiku 4.5 的「轻量低成本」档，1M 上下文，整体性价比突出

> **我的判断**：Qwen 3.7 未开源的策略和 GLM-5.2（MIT 开源）形成鲜明对比。阿里云选择了"用开源小模型（Qwen3.6 及以下）建立生态，用闭源旗舰（Qwen3.7）商业化变现"的路径，这与 OpenAI/GPT 的策略类似，但与 Meta/智谱的开源路线不同。

---

## Qwen 3.7 模型变体

### Qwen3.7-Max

| 属性 | 详情 |
|------|------|
| **类型** | 闭源旗舰推理模型 |
| **模态** | 纯文本（无视觉输入） |
| **上下文窗口** | **1M tokens**（~70 万汉字） |
| **思考模式** | 支持（通过 `enable_thinking` / `reasoning.effort` 控制） |
| **Function Calling** | ✅ 支持 |
| **内置工具** | ✅ 支持（联网搜索、代码解释器、网页抓取） |
| **结构化输出** | ❌ 不支持（截至 2026-07） |
| **批量调用** | ✅ 支持 |
| **发布** | 2026-05-20（阿里云峰会），快照版本：2026-05-17, 2026-05-20, 2026-06-08 |
| **Intelligence Index** | 56.6（发布时全球排名第 5） |
| **API 定价（阿里云百炼）** | ¥12/百万 tokens 输入, ¥36/百万 tokens 输出（~$2.50/$7.50 USD） |
| **API 定价（Cached Input）** | ¥1.2/百万 tokens（~$0.25，90% 折扣） |
| **第三方定价（OpenRouter/Together AI）** | $1.25 输入 / $3.75 输出（约 50% 折扣） |

### Qwen3.7-Plus

| 属性 | 详情 |
|------|------|
| **类型** | 多模态闭源模型 |
| **模态** | 文本 + 图像 + 视频输入 |
| **上下文窗口** | **1M tokens** |
| **思考模式** | 支持 |
| **Function Calling** | ✅ 支持 |
| **内置工具** | ✅ 支持（联网搜索、代码解释器、网页抓取） |
| **结构化输出** | ✅ 支持 |
| **批量调用** | ✅ 支持 |
| **发布** | 2026-06，快照版本：2026-05-26 |
| **API 定价（阿里云百炼）** | ~$0.40/百万 tokens 输入, ~$1.60/百万 tokens 输出 |
| **API 定价（Cached Input）** | ~$0.04/百万 tokens |
| **第三方定价（OpenRouter/Together AI）** | ~$0.32 输入 / $1.28 输出 |

> **性价比判断**：Qwen3.7-Plus 的输入价格仅为 Max 的 ~16%（官方价），是日常开发和高吞吐场景的首选。与 GLM-5.2 的 $1.4/$4.4 和 DeepSeek-V4-Pro 定价相比，Plus 在同类中性价比极具竞争力。

---

## 性能基准

> ⚠️ **重要说明**：Qwen3.7-Max 和 Plus 均为闭源模型，阿里云未公开完整的 benchmark 数据表格。以下数据来自现有 GLM-5.2 报告中的对比数据和 LM Arena 排行榜。

### 从 GLM-5.2 基准对比表中提取的 Qwen3.7-Max 数据

| 基准 | Qwen3.7-Max | GLM-5.2 | Claude Opus 4.8 | GPT-5.5 | DeepSeek-V4-Pro |
|------|-------------|---------|-----------------|---------|-----------------|
| HLE | **41.4** | 40.5 | 49.8 | 41.4 | - |
| HLE (w/ Tools) | 53.5 | 54.7 | **57.9** | 52.2 | - |
| CritPt | 13.4 | 20.9 | 20.9 | **27.1** | - |
| AIME 2026 | 97 | **99.2** | 95.7 | 98.3 | 95.2 |
| HMMT Nov. 2025 | 95 | 94.4 | **96.5** | **96.5** | - |
| HMMT Feb. 2026 | **97.1** | 92.5 | 96.7 | 96.7 | - |
| IMOAnswerBench | 90 | **91.0** | 83.5 | - | - |
| GPQA-Diamond | 90 | 91.2 | **93.6** | **93.6** | - |

### 分析与判断

1. **推理能力（HLE/GPQA）**：Qwen3.7-Max 在 HLE（41.4）上略领先 GLM-5.2（40.5），但落后 Claude Opus 4.8（49.8）。这说明超复杂推理仍是 Claude 的护城河，但 Qwen 已接近第一梯队。

2. **竞赛数学（AIME/HMMT/IMO）**：Qwen3.7-Max 在 HMMT Feb. 2026（97.1）上领先所有模型，AIME 2026（97）仅次于 GLM-5.2（99.2）。**数学推理是 Qwen 3.7 的核心竞争力之一**。

3. **Agent 和代码**：Qwen3.7-Plus 被百炼官方明确定位为 "AI 编程或 Agent 开发" 的推荐模型，强调其"能力与成本均衡，完整工具调用支持，1M 上下文适合大型代码库"。这表明其在 Agentic 和代码任务上的表现至少处于一线水平。

4. **多模态能力**：Qwen3.7-Plus 支持文本+图像+视频输入，这是其相对于 Qwen3.7-Max（纯文本）和 GLM-5.2（纯文本）的差异化优势。

> **核心洞察**：
> - 🥇 **数学第一梯队**：HMMT Feb. 2026（97.1）全球第一
> - 🥈 **综合推理接近前沿**：HLE（41.4）领先 GLM-5.2 和 GPT-5.5，仅次于 Claude
> - 📊 **闭源意味着数据不透明**：阿里云未公开 Qwen3.7 的完整 benchmark 和架构参数，选型决策中这是一个需要考虑的不确定性

---

## 架构推测与分析

由于 Qwen 3.7 是闭源模型，官方未公开具体架构参数。但从 Qwen 系列的演进路径可以做出合理推测：

### 基于 Qwen 系列演进的技术推演

1. **从 Qwen3.5 继承**：
   - **Gated Delta Networks**：Qwen3.5 引入的高效混合架构，结合稀疏 MoE，实现高吞吐低延迟推理
   - **统一视觉-语言基础**（Plus 专属）：在 Qwen3.5 中已实现 early fusion 训练，Qwen3.7-Plus 应基于此进一步优化
   - **强化学习规模化**：Qwen3.5 的 RL 在百万 agent 环境中训练，Qwen3.7 应在此基础上有进一步扩展

2. **从 Qwen3.6 继承**：
   - **Agentic Coding 能力**：Qwen3.6 特别优化的前端工作流和仓库级推理
   - **Thinking 上下文保留**：跨会话的思考上下文保持
   - **社区反馈驱动**：Qwen3.6 的开发方向直接受社区反馈影响

3. **Qwen3.7 独有**：
   - **1M 上下文窗口**（vs Qwen3.6 的 256K）：这是质的飞跃
   - **闭源意味着不受开源基础设施限制**：可以使用更大参数规模、更复杂的推理架构
   - **与 Qwen3.6-Flash 的协同**：Qwen3.6-Flash 在同一 API 平台提供低成本替代，形成产品矩阵

> **我的判断**：Qwen3.7-Max 很可能是一个超大规模 MoE 模型（估计激活参数量在 50B-100B 范围），基于 Gated Delta Networks 和扩展的 RL 训练。Plus 版本可能是类似架构但激活参数更少（30B-50B），并增加了视觉编码器。

---

## 阿里云百炼生态与工具链

Qwen 3.7 不仅仅是两个模型，而是嵌入阿里云百炼（Model Studio）生态的一部分：

### 核心功能支持

| 功能 | Qwen3.7-Max | Qwen3.7-Plus | Qwen3.6-Flash |
|------|:-----------:|:------------:|:-------------:|
| Thinking Mode | ✅ | ✅ | ✅ |
| Function Calling | ✅ | ✅ | ✅ |
| 内置工具（搜索/代码/抓取） | ✅ | ✅ | ✅ |
| 结构化输出 | ❌ | ✅ | ✅ |
| 批量推理 | ✅ | ✅ | ✅ |
| 多模态输入 | ❌ | ✅ | ❌ |
| 1M 上下文 | ✅ | ✅ | ✅ |

### 相关生态组件

| 组件 | 功能 | 说明 |
|------|------|------|
| **Qwen Studio** (chat.qwen.ai) | 官方聊天界面 | 支持 Deep Research, Web Dev, 自适应工具使用 |
| **Qwen Code** | 终端 AI Agent | 针对 Qwen 模型优化，理解大型代码库 |
| **Qwen-Agent** | Agent 开发框架 | 支持 MCP、工具调用、规划、记忆 |
| **Qwen-Image-Edit** | 图像编辑 | 基于 Qwen-Image 20B，支持精确文本编辑 |
| **Qwen-MT** | 翻译模型 | 92 语言，基于 Qwen3，万亿级翻译 token |
| **Qwen3Guard** | 安全护栏 | 首个 Qwen 系列安全分类模型 |
| **Qwen3-Rerank** | 重排序 | 向量检索精度提升 |

---

## 与主要竞品对比

### Qwen3.7-Max vs 同级旗舰

| 维度 | Qwen3.7-Max | GLM-5.2 | DeepSeek-V4-Pro | Claude Opus 4.8 |
|------|-------------|---------|-----------------|-----------------|
| **开闭源** | 闭源（仅 API） | **MIT 开源** | 开源 | 闭源 |
| **上下文** | 1M tokens | 1M tokens | 1M tokens | 未知 |
| **推理 (HLE)** | 41.4 | 40.5 | - | **49.8** 🥇 |
| **数学 (AIME 2026)** | 97 | **99.2** 🥇 | 95.2 | 95.7 |
| **定价 (输入/输出)** | $2.50/$7.50 | $1.4/$4.4 | - | ~$15/$75 |
| **多模态** | ❌ (仅文本) | ❌ (仅文本) | - | 支持 |
| **官方推荐场景** | 最强推理 | 代码/Agent | 知识通才 | 全能型 |

### Qwen3.7-Plus vs 同级均衡

| 维度 | Qwen3.7-Plus | GLM-4.7 | DeepSeek-V4-Flash |
|------|-------------|---------|-------------------|
| **上下文** | 1M tokens | 198K | 1M |
| **多模态** | ✅ 文本+图像+视频 | - | - |
| **定价 (输入/输出)** | $0.40/$1.60 🥇 | $0.6/$2.2 | - |
| **结构化输出** | ✅ | ✅ | ❌ |
| **内置工具** | ✅ | ❌ | ❌ |

> **核心洞察**：Qwen3.7-Plus 在「多模态 + 低价格 + 完整工具链」这个组合上形成了独特竞争力。如果你需要"一个模型同时处理文本、图像、调用外部工具、输出 JSON"，Qwen3.7-Plus 是目前最便宜的选择。

---

## 使用方式

### API 调用（百炼平台）

Qwen3.7 兼容 OpenAI API 格式：

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_DASHSCOPE_API_KEY",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Qwen3.7-Max（纯文本，最强推理）
response = client.chat.completions.create(
    model="qwen3.7-max",
    messages=[{"role": "user", "content": "解释量子计算的基本原理"}],
    max_tokens=32768,
    temperature=0.6,
    top_p=0.95,
)

# Qwen3.7-Plus（多模态，支持视觉输入）
response = client.chat.completions.create(
    model="qwen3.7-plus",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                {"type": "text", "text": "描述这张图片的内容"}
            ]
        }
    ],
)
```

### Thinking 模式控制

所有 Qwen3 及以上模型支持混合思考模式：

```python
# 启用思考模式（Responses API）
response = client.responses.create(
    model="qwen3.7-max",
    input="证明勾股定理",
    reasoning={"effort": "high"}  # low/medium/high
)

# 或通过 Chat Completions API 控制
response = client.chat.completions.create(
    model="qwen3.7-plus",
    messages=[{"role": "user", "content": "分析这个代码的性能瓶颈"}],
    extra_body={
        "enable_thinking": True,
        "chat_template_kwargs": {"enable_thinking": True}
    }
)
```

### 阿里云百炼平台特有功能

- **Thinking Budget 控制**：仅在百炼 API 支持，开源框架不支持
- **内置工具**：联网搜索、代码解释器、网页抓取无需配置
- **批量推理**：降低大批量请求成本
- **多区域部署**：北京、新加坡、美东、法兰克福

---

## 关键洞察与判断

### 1. 闭源策略：商业化的必然

Qwen3.7 不开源的决定意味着阿里云从"开源引流 → 云服务变现"的模型，转向"闭源旗舰 + 开源基础"的双轨制。这与 Google（Gemini 闭源 + Gemma 开源）和 OpenAI 的路线一致。对用户的影响：
- ✅ 通过 API 使用更简单，无需部署
- ✅ 模型能力更强（不受开源基础设施限制）
- ❌ 无法本地部署，不能微调
- ❌ 数据隐私需依赖阿里云的信誉

### 2. Plus 的性价比是关键武器

Qwen3.7-Plus 以 $0.40/$1.60 的价格提供多模态 + 1M 上下文 + 完整工具链，这个组合在市场上几乎没有直接竞争对手。对于需要高吞吐量、多模态输入、工具调用的场景（如客服、内容审核、RAG 系统），Plus 可能是 2026 年性价比最高的选择。

### 3. 1M 上下文的战略意义

Qwen3.7 全系列支持 1M tokens 上下文（vs Qwen3.6 的 256K），这在代码库理解、长文档处理、长对话历史等场景中具有明显的实用价值。

### 4. "Thinking 保留"的工程价值

从 Qwen3.6 引入的 Thinking 上下文保留功能（跨会话保持思考状态）是一个非常务实的设计。对于迭代式编程（Claude Code、OpenClaw Coding、Qwen Code 等场景），这能显著减少重复推理的 token 消耗。

### 5. 与阿里云基础设施的深度绑定

Qwen3.7 → 百炼平台 → 通义千问 → 阿里云全栈，这条链路意味着选择 Qwen3.7 不仅是选择模型，也是选择阿里云生态。如果你已经深度使用阿里云（ECS、OSS、RDS 等），Qwen3.7 是最自然的 AI 层选择。

### 6. 开源替代：Qwen3.6 的作用

如果你需要使用 Qwen 系列但不想被 API 绑定：**Qwen3.6-35B-A3B**（Apache 2.0 开源，MoE，性能接近 Plus，256K 上下文）是当前最佳选择。它可以直接替代 Qwen3.7-Plus 在自部署场景中的角色。

---

## 选型建议矩阵

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| 最强推理（数学/逻辑） | Qwen3.7-Max | HMMT Feb. 2026 第一，AIME 2026 97 分 |
| AI 编程 / Agent 开发 | **Qwen3.7-Plus** | 官方推荐，1M 上下文 + 工具调用 + 低成本 |
| 多模态理解应用 | **Qwen3.7-Plus** | 支持文本+图像+视频，$0.40 起 |
| 高吞吐 / 简单任务 | Qwen3.6-Flash | 接近旗舰能力，相同上下文和功能支持 |
| 自部署 / 数据隐私 | Qwen3.6-35B-A3B | Apache 2.0 开源，性能接近 Plus |
| 代码专家 / 开源 | GLM-5.2 | MIT 开源，代码 SOTA，可本地部署 |
| 预算极低 / 原型验证 | GLM-4.7-Flash / 免费层模型 | 零成本起步 |

> **我推荐的默认选择**：在预算允许的前提下，**Qwen3.7-Plus** 是目前综合来看最优的"全能型"模型——多模态、低成本、长上下文、完整工具链。只在确实需要极致推理能力时切换到 Max。

---

## 待研究问题

- [ ] Qwen3.7 的具体架构参数（总参数量、激活参数量、层数、注意力头数等）——官方迄今未公开
- [ ] Qwen3.7-Max 在 SWE-bench、Terminal Bench 等 Agent/代码基准上的具体分数
- [ ] Qwen3.7-Plus 的视觉理解基准（与 GPT-5.5、Claude Opus 4.8 的对比）
- [ ] Qwen3.7 是否会有开源版本（参考 Qwen2.5-Max → Qwen3-235B 的先例）
- [ ] Qwen3.7 的多语言支持范围是否继承了 Qwen3.5 的 201 语言

---

## 参考资源

- **阿里云百炼模型大全**: https://help.aliyun.com/zh/model-studio/models
- **Qwen 官方文档**: https://qwen.readthedocs.io/
- **Qwen 官方博客**: https://qwen.ai/blog
- **Qwen GitHub**: https://github.com/QwenLM
- **Qwen3.6 GitHub**: https://github.com/QwenLM/Qwen3.6
- **Qwen3 博客**: https://qwenlm.github.io/blog/qwen3/
- **Qwen Studio**: https://chat.qwen.ai
- **Qwen-Agent**: https://github.com/QwenLM/Qwen-Agent
- **阿里云百炼（Model Studio）**: https://bailian.console.aliyun.com/
- **OpenRouter Qwen3.7**: https://openrouter.ai/qwen/qwen3.7-max
