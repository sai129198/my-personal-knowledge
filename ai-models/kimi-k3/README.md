# Kimi K3 — 深度研究笔记

> 学习时间：2026-07-17
> 状态：API 已上线 | 权重 2026-07-27 开源 | 技术报告待发布
> 官方博客：https://www.kimi.com/blog/kimi-k3
> API 文档：https://platform.kimi.com/docs

---

## 一句话总结

Kimi K3 是 Moonshot AI（月之暗面）在 2026 年 7 月 16 日发布的旗舰模型，**2.8 万亿参数**，全球首个开源 3T 级模型，主打**长程编程 + 端到端知识工作**，整体性能紧追 Claude Fable 5 和 GPT 5.6 Sol，但**成本只有前者的 1/3 左右**。

---

## 1. 背景与定位

### Moonshot AI 的战略反击

| 时间 | 事件 |
|------|------|
| 2025.07 | Kimi K2 系列发布（1T 参数，MoE） |
| 2025.01 | DeepSeek R1 冲击中国市场，Kimi 月活从第三滑至第七 |
| 2025-2026 | Kimi 快速迭代：K2 → K2.5 → K2.6 → K3 |
| 2026.07.16 | Kimi K3 API 正式上线 |
| 2026.07.27 | 完整模型权重计划开源 |

过去12个月中有9个月，Kimi 模型保持着开源模型的规模上限。

### 产品矩阵

| 模型 | 定位 | 上下文 |
|------|------|--------|
| `kimi-k3` | 旗舰，长程编程+知识工作+推理 | 1M token |
| `kimi-k2.7-code` | Coding 专用 | 256K |
| `kimi-k2.7-code-highspeed` | Coding 高速版（~180 tok/s） | 256K |
| `kimi-k2.6` | 通用多模态 | 256K |

---

## 2. 核心架构

### 整体架构图

```
┌─────────────────────────────────────┐
│           Kimi K3 (2.8T)            │
├─────────────────────────────────────┤
│  Attention: KDA + AttnRes           │
│  FFN: Stable LatentMoE (16/896)    │
│  Activation: SiTU                   │
│  Attention: Gated MLA               │
│  Optimizer: Per-Head Muon           │
├─────────────────────────────────────┤
│  Weight: MXFP4 | Activation: MXFP8 │
│  Context: 1M tokens                 │
│  Vision: Native multimodal          │
└─────────────────────────────────────┘
```

### 2.1 Kimi Delta Attention (KDA)

KDA 是 Kimi K3 最核心的架构创新，一种**混合线性注意力机制**，目的是让注意力在超长序列（1M token）上高效扩展。

- 提供了高效扩展 attention 的基础
- 对传统 prefix caching 提出了新挑战，需要自定义实现
- Moonshot 已经向 vLLM 社区贡献了 KDA prefix caching 实现

**我的理解**：KDA 类似于在标准 attention 和线性 attention 之间做了某种"增量式"（delta）折中，既能保留标准 attention 的表达能力，又能线性扩展。具体数学细节要等技术报告。

### 2.2 Attention Residuals (AttnRes)

**选择性地跨深度检索表示**，而不是均匀累积。

传统 Transformer 每一层都在残差流上累加，AttnRes 的做法是让模型**有选择地**从不同深度拉取信息。这有点像"不是每层的信息都一样重要，我挑重点的用"。

与 Anthropic 在 Claude 中探索的 cross-layer attention 有异曲同工之妙，但 Moonshot 的实现路径不同。

### 2.3 Stable LatentMoE

- **896 个专家**，每 token 激活 **16 个**
- 核心技巧：**Quantile Balancing**（分位数均衡）
  - 从 router-score 的分位数直接导出专家分配
  - **消除了启发式更新**和敏感的均衡超参数
  - 这是训练稳定性的关键——在 2.8T 参数规模下，负载不均衡会导致严重的训练效率问题

**我的判断**：Quantile Balancing 可能是 K3 训练中最被低估的创新。在 896 专家这个稀疏度下，传统的 auxiliary loss 方法很难稳定工作。直接从分位数推导分配，把负载均衡从"优化目标"变成了"确定性操作"，思路很漂亮。

### 2.4 其他架构组件

| 组件 | 作用 | 我的理解 |
|------|------|----------|
| **Per-Head Muon** | 扩展 Muon optimizer，独立优化每个 attention head | 不同的 head 可能需要不同的学习动态，独立优化更灵活 |
| **SiTU** (Sigmoid Tanh Unit) | 改进的激活函数 | 替代 GELU/SwiGLU，可能在高稀疏度 MoE 下有更好的梯度特性 |
| **Gated MLA** | 增强的 attention 选择性 | MLA（Multi-head Latent Attention）基础上加门控，应该是 DeepSeek V2/V3 路线的延续 |

---

## 3. 关键性能数据

### 3.1 综合智能指数

根据 Artificial Analysis Intelligence Index：
- **57.1 分，排名第 4**
- 仅次于 Claude Fable 5、GPT 5.6 Sol，差距约 3%
- 每任务成本约 **$0.94**，比 Claude Fable 5 低 **65.8%**

### 3.2 Coding Benchmarks（与 Fable 5 / GPT 5.6 Sol 对比）

| Benchmark | K3 (max) | Fable 5 | GPT 5.6 Sol | 评价 |
|-----------|----------|---------|-------------|------|
| DeepSWE | 67.5 | 70.0 | 73.0 | 紧追 |
| Program Bench | 77.8 | 76.8 | 77.6 | **打平/略胜** |
| Terminal Bench 2.1 | 88.3 | 84.6 | 88.8 | 同一梯队 |
| FrontierSWE | 81.2 | 86.6 | 71.3 | 超 GPT5.6 |
| SWE Marathon | 42.0 | 35.0 | 39.0 | **最高** |
| GPU Kernel 优化实测 | 与 Fable 5 竞争 | — | 大幅超越 | K3 实际参与了自身开发 |

**亮点**：SWE Marathon（长程软件工程）上 K3 拿了最高分，这验证了它在"长时间自主完成任务"这个方向的优势。GPU Kernel 优化测试中 K3 和 Fable 5 打得有来有回，远超 GPT 5.6 和 Opus 4.8。

### 3.3 Agentic Benchmarks

| Benchmark | K3 (max) | Fable 5 | GPT 5.6 Sol |
|-----------|----------|---------|-------------|
| GDPval-AA v2 (Elo) | 1668 | 1760 | 1748 |
| BrowseComp | 91.2 | 88.0 | 90.4 |
| Toolathlon-Verified | 73.2 | 77.9 | 74.9 |
| MCP Atlas | 84.2 | 84.7 | 83.6 |

Agent 能力整体处于第一梯队，BrowseComp（长网页浏览理解）上拿了最高分。

### 3.4 Reasoning & Knowledge

| Benchmark | K3 (max) | Fable 5 | GPT 5.6 Sol |
|-----------|----------|---------|-------------|
| GPQA-Diamond | 93.5 | 92.6 | 94.1 |
| HLE-Full | 43.5 | 53.3 | 44.5 |
| HLE-Full w/ tools | 56.0 | 63.0 | 58.0 |

推理能力与顶级闭源模型在同一水平线，但 HLE（人类最后考试）上 Anthropic 有明显优势。

### 3.5 Vision

| Benchmark | K3 (max) | Fable 5 | GPT 5.6 Sol |
|-----------|----------|---------|-------------|
| MMMU-Pro | 81.6 | 81.2 | 83.0 |
| MMMU-Pro w/ python | 83.4 | 86.5 | 84.6 |
| CharXiv (RQ) w/ python | 91.3 | 93.5 | 89.1 |
| MathVision w/ python | **97.8** | 98.6 | 97.8 |
| ZeroBench_main w/ python | 41.0 | 46.0 | 35.0 |

视觉能力相当强，原生多模态（不是外挂模块），支持图片+视频输入。

---

## 4. 训练与推理优化

### 4.1 训练

- **量化感知训练（QAT）**：从 SFT 阶段开始就用 MXFP4/MXFP8
- **全均衡专家并行**：静态 shape，关键路径上无 host 同步
- **扩展效率**：相比 K2 提升约 **2.5 倍**（同等算力 → 更高的智力产出）

### 4.2 推理部署

- 推荐 **64+ 加速器的 supernode 配置**
- 受益于更大的高带宽通信域
- Mooncake 分离式推理架构 → 官方 API 在编程场景下 **>90% 缓存命中率**

### 4.3 vLLM 社区贡献

- KDA prefix caching 实现已贡献给 vLLM
- KDA + prefill cache 是控制推理成本的关键

---

## 5. API 与开发者生态

### 5.1 定价（人民币）

| 类型 | 价格 |
|------|------|
| 输入（缓存命中） | ¥2.00 / 1M tokens |
| 输入（缓存未命中） | ¥20.00 / 1M tokens |
| 输出 | ¥100.00 / 1M tokens |

美元定价：$0.30 / $3.00 / $15.00 per MTok

### 5.2 API 特性

- ✅ 兼容 OpenAI API 格式（`base_url: https://api.moonshot.cn/v1`）
- ✅ 流式输出（reasoning_content + content 双通道）
- ✅ 视觉输入（图片 base64、视频 ms:// 协议）
- ✅ 结构化输出（JSON Schema + strict mode）
- ✅ 工具调用（tool_choice、动态加载工具）
- ✅ 1M 上下文自动缓存（无需手动管理 cache ID）
- ✅ Partial Mode（前缀续写）
- ✅ 思考力度 `reasoning_effort`（当前仅 `max`）

### 5.3 K3 独有的工具调用模式

K3 的杀手级特性之一是**动态工具加载**（Dynamic Tool Loading）：

1. 顶层只声明一个 `search_tools` 工具
2. 用 `tool_choice: "required"` 强制模型先搜索工具
3. 通过 system message 动态注入工具定义
4. 解决了"工具太多导致 prompt 爆炸 + 选错工具"的问题

这是一个非常务实的工程创新——不是在模型层面解决工具选择，而是用一个轻量的"工具搜索引擎"来解耦。

### 5.4 重要限制

- `temperature`、`top_p` 等采样参数为固定值，不建议显式传入
- 视觉输入不支持公网 URL，必须用 base64 或 `ms://<file-id>`
- 联网搜索正在更新，近期不可用
- 多轮对话必须原样回传完整 assistant message（含 reasoning_content）
- `reasoning_effort` 当前仅 `max`，更多档位待上线

### 5.5 客户端生态

| 平台 | 说明 |
|------|------|
| Kimi.com | Web 端 |
| Kimi Work | 桌面端（Windows/Mac），知识工作专用 |
| Kimi Code | 终端编程 Agent（支持 /model 切换） |
| Kimi App | iOS/Android/HarmonyOS |
| Kimi Enterprise | 企业版，数据隔离 |

---

## 6. 竞争格局分析

### 6.1 定位

```
                    性能高
                      ↑
          Fable 5 ●  │  ● GPT 5.6 Sol
                      │
              K3  ●  │
                      │
     Opus 4.8 ●       │  ● GPT 5.5
                      │
                      │     ● GLM-5.2
                      │
         ──────────────────────→ 成本低
```

K3 占据的是 **"接近顶级能力 + 显著更便宜"** 这个甜蜜点。

### 6.2 核心优势

1. **长程编程**：SWE Marathon 最高分，GPU Kernel 优化实测与 Fable 5 竞争
2. **性价比**：综合成本比 Fable 5 低 65.8%
3. **开源**：首个 3T 级开源模型，全权重 7 月 27 日释放
4. **原生多模态**：视觉理解内建在模型里，不是外挂
5. **1M 上下文 + 自动缓存**：工程上很省心

### 6.3 核心劣势

1. **HLE（难推理）**：落后 Fable 5 约 10 个百分点
2. **Agent 综合能力**：GDPval-AA 上落后 Fable 5 约 100 Elo
3. **思考历史敏感性**：切换模型或丢失 thinking history 会导致质量大幅下降
4. **过度主动性**：可能在用户意图不明确时擅自做决定

### 6.4 我的判断

K3 的战略意义大于技术突破。Moonshot AI 在 DeepSeek R1 冲击后，通过**快速开源迭代**重新站稳脚跟。K3 不是要去"打败" Fable 5 或 GPT 5.6 Sol——它的目标是证明：**开源模型可以在接近顶级闭源能力的同时，提供更大的灵活性和更低的成本**。

从工程角度看，KDA + AttnRes + Stable LatentMoE 这个架构组合在扩展效率上有明显优势（2.5x vs K2），但这需要等技术报告才能判断是"架构创新"还是"训练 recipe 优化"的贡献更大。

---

## 7. 待追踪

- [ ] 2026-07-27：完整权重开源 → 社区能跑起来吗？硬件需求多大？
- [ ] 技术报告发布 → KDA/AttnRes/Quantile Balancing 的数学细节
- [ ] `reasoning_effort` 更多档位上线 → 低档位下的性能/成本权衡
- [ ] 联网搜索功能恢复
- [ ] 第三方推理提供商（Together AI 等）的 K3 部署方案
- [ ] 社区 Fine-tune 生态发展
- [ ] Kimi K3 在 vLLM/SGLang 等框架上的实际推理性能

---

## 参考来源

- 官方技术博客：https://www.kimi.com/blog/kimi-k3
- API 文档：https://platform.kimi.com/docs
- K3 快速开始：https://platform.kimi.com/docs/guide/kimi-k3-quickstart
- K3 定价：https://platform.kimi.com/docs/pricing/chat-k3
- 工具调用最佳实践：https://platform.kimi.com/docs/guide/kimi-k3-tool-calling-best-practice
- Artificial Analysis：https://artificialanalysis.ai
