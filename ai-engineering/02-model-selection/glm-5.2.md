# GLM-5.2 深度研究报告

> 研究时间: 2026-06-22
> 来源: zai-org/GLM-5 GitHub, Hugging Face, arXiv, 社区部署实践
> 标签: #status/canonical #topic/model-selection #topic/llm #topic/inference #year/2026

---

## 一句话总结

**GLM-5.2 是智谱 AI (Z.ai) 发布的 744B 参数（40B 激活）开源旗舰模型**，采用 DeepSeek Sparse Attention (DSA) + MoE 架构，支持 1M token 上下文，在代码、推理和 Agent 任务上达到开源 SOTA，并以 MIT 许可证完全开源。

---

## 模型定位与演进

### GLM 系列演进路线

| 版本 | 时间 | 参数量 | 核心特性 |
|------|------|--------|----------|
| GLM-4.5 | 2025 | 355B (32B 激活) | Agentic, Reasoning, Coding (ARC) 基础 |
| GLM-5 | 2026.02 | 744B (40B 激活) | + DSA 稀疏注意力，+ slime 异步 RL 训练 |
| GLM-5.1 | 2026.04 | 744B (40B 激活) | 长周期 Agent 能力，持续优化不 plateau |
| **GLM-5.2** | **2026.06** | **744B (40B 激活)** | **1M 上下文，IndexShare，MTP 优化，开源 SOTA** |

### 核心设计理念

从 "Vibe Coding"（随性编程）进化到 "Agentic Engineering"（代理工程）——模型不仅能写代码，更能作为自主 Agent 完成长周期复杂工程任务。

---

## 架构创新

### 1. DeepSeek Sparse Attention (DSA)

- **稀疏注意力机制**：大幅降低长上下文推理成本
- **IndexShare**：每 4 个稀疏注意力层共享同一个 indexer，1M 上下文下每 token FLOPs 降低 **2.9×**
- **78 层注意力模式**：21 层 Full + 57 层 Shared/Skip 的混合模式（`indexer_types` 数组定义）

### 2. MoE (Mixture of Experts)

- **256 个路由专家 + 1 个共享专家**
- **每 token 激活 8 个专家**
- **路由评分函数**：Sigmoid（非 Softmax）
- **Top-k 方法**：`noaux_tc`（无辅助损失的负载均衡）

### 3. MTP (Multi-Token Prediction) 推测解码

- GLM-5.2 改进了 MTP 层，推测解码接受长度提升 **20%**
- 配合 DSA 实现高效长文本生成

### 4. 关键配置参数

```json
{
  "hidden_size": 6144,
  "num_hidden_layers": 78,
  "num_attention_heads": 64,
  "head_dim": 192,
  "intermediate_size": 12288,
  "moe_intermediate_size": 2048,
  "n_routed_experts": 256,
  "num_experts_per_tok": 8,
  "max_position_embeddings": 1048576,
  "vocab_size": 154880,
  "rope_theta": 8000000
}
```

---

## 性能基准

### 推理能力

| 基准 | GLM-5.2 | GLM-5.1 | Qwen3.7-Max | Claude Opus 4.8 | GPT-5.5 |
|------|---------|---------|-------------|-----------------|---------|
| HLE | 40.5 | 31 | 41.4 | **49.8** | 41.4 |
| AIME 2026 | **99.2** | 95.3 | 97 | 95.7 | 98.3 |
| HMMT Nov. 2025 | 94.4 | 94 | 95 | **96.5** | **96.5** |
| GPQA-Diamond | 91.2 | 86.2 | 90 | **93.6** | **93.6** |

### 代码能力

| 基准 | GLM-5.2 | GLM-5.1 | Claude Opus 4.8 | GPT-5.5 |
|------|---------|---------|-----------------|---------|
| SWE-bench Pro | **62.1** | 58.4 | **69.2** | 58.6 |
| DeepSWE | **46.2** | 18 | 58 | 70 |
| Terminal Bench 2.1 | **81.0** | 63.5 | **85.0** | 84.0 |
| FrontierSWE | **74.4** | 30.5 | **75.1** | 72.6 |
| PostTrainBench | **34.3** | 20.1 | **37.2** | 28.4 |

### Agent 能力

| 基准 | GLM-5.2 | GLM-5.1 | Claude Opus 4.8 |
|------|---------|---------|-----------------|
| MCP-Atlas | **76.8** | 71.8 | **77.8** |
| Tool-Decathlon | **48.2** | 40.7 | **59.9** |

> **我的判断**：GLM-5.2 在开源模型中全面领先，在代码和 Agent 任务上已接近 Claude Opus 4.8，但推理能力仍有差距。DeepSWE 46.2 的分数相比 Claude 的 58 和 GPT-5.5 的 70 说明长周期软件工程仍是闭源模型的优势领域。

---

## 模型变体与下载

| 模型 | 精度 | 大小 | 适用场景 |
|------|------|------|----------|
| GLM-5.2 | BF16 | ~1.5TB | 训练、研究 |
| **GLM-5.2-FP8** | **FP8** | **~753GB** | **推理部署（推荐）** |
| GLM-5.2-NVFP4-REAP-469B | NVFP4 | ~313GB | 极限压缩部署 |

- **Hugging Face**: https://huggingface.co/zai-org/GLM-5.2-FP8
- **ModelScope**: 同步上架
- **许可证**: MIT（完全开源，无地域限制）

---

## 推理部署方案

### 官方支持框架

| 框架 | 最低版本 | 特点 |
|------|----------|------|
| **SGLang** | v0.5.13.post1+ | 性能最佳，支持 DSA 原生优化 |
| **vLLM** | v0.23.0+ | 生态成熟，recipe 丰富 |
| Transformers | v0.5.12+ | 开发调试 |
| KTransformers | v0.5.12+ | 消费级 GPU 友好 |
| Unsloth | v0.1.47-beta+ | 微调优化 |

### 硬件部署配置

#### 方案 A：数据中心级（H100/H200）

- 原生支持 DSA（sm_90）
- 单节点或多节点 TP/PP 并行

#### 方案 B：工作站级（RTX PRO 6000 Blackwell）

- **4× RTX PRO 6000** (96GB × 4, SM120)
- DCP=4（Decode Context Parallel）
- 250K 上下文，~60 tok/s 解码速度
- 参考实现：[0xSero/glm-5.2-sm120](https://github.com/0xSero/glm-5.2-sm120)

#### 方案 C：消费级（RTX 4090）⭐ 社区突破

- **24× RTX 4090-48GB**（3 节点 × 8 卡）
- TP=8 × PP=3，Pipeline + Tensor 并行
- ~10 tok/s 单流（CUDA Graph）
- 关键：社区移植了 DSA kernel 到 sm_89 (Ada)
- 参考实现：[renning22/glm-5.2-4090](https://github.com/renning22/glm-5.2-4090)

> **重要**：官方 vLLM/SGLang 对 GLM-5.2 的 DSA kernel 硬要求 sm_90+/sm_100，社区通过 Triton + tilelang 实现了 Ada 回退。

### 昇腾 NPU 部署

华为昇腾平台已官方支持 GLM-5.2：
- MoE Mega-Fusion 算子
- 通信-计算融合
- Attention 预处理 + MTP 优化
- PD 分离 + Prefix Caching
- W8A8 混合量化

参考：[example/ascend.md](https://github.com/zai-org/GLM-5/blob/main/example/ascend.md)

---

## 训练基础设施：slime

GLM-5 系列使用 **slime**（智谱自研异步 RL 框架）进行后训练：

- **异步 RL**：解耦生成与训练，大幅提升吞吐量
- **Agent RL**：支持多轮交互、工具调用、长周期轨迹
- **Variable Batch Size**：支持动态全局 batch size
- **最新版本**: v0.3.0（Agent-first 强化学习）

GitHub: https://github.com/THUDM/slime

---

## 思考模式控制

GLM-5.2 是推理模型，支持思考预算控制：

| 模式 | 参数 | 说明 |
|------|------|------|
| Max (默认) | `reasoning_effort="max"` 或不设置 | 完整推理，效果最佳 |
| High | `reasoning_effort="high"` | 快速推理，延迟更低 |
| Off | `enable_thinking=false` | 直接回答，无思考过程 |

> 注意：思考过程消耗 token，若 `max_tokens` 设置过小可能导致 `content` 为空（思考占用了全部预算）。建议 `max_tokens ≥ 2000`。

---

## GLM 技能生态

智谱构建了围绕 GLM 的 Skill 生态（OpenClaw 兼容）：

| Skill | 功能 |
|-------|------|
| `glmocr` | OCR 文字提取 |
| `glmocr-table` | 表格提取 |
| `glmocr-formula` | 公式提取 |
| `glm-image-gen` | 文生图 |
| `glmv-caption` | 图像/视频描述 |
| `glmv-grounding` | 目标定位与框选 |
| `glmv-prd-to-app` | PRD 文档生成全栈应用 |
| `glmv-web-replication` | 网站前端复刻 |

安装方式：
```bash
npx clawhub@latest install glmocr glmv-caption glm-image-gen
```

---

## 关键洞察与判断

### 1. 开源策略激进

MIT 许可证 + 完整权重 + 详细部署文档，智谱在开源力度上对标 Meta（Llama）甚至更进一步。这与国内其他厂商（如 Qwen 的 Apache 2.0）形成竞争。

### 2. DSA 是核心竞争力

DeepSeek Sparse Attention 让 GLM-5.2 在 1M 上下文下仍保持高效，这是其实现 "Agentic Engineering" 的架构基础。但 DSA 的硬件门槛（sm_90+）也是部署的主要障碍。

### 3. 社区移植意义重大

renning22 将 DSA kernel 移植到 RTX 4090 (sm_89) 是一个重要信号：大模型的 "消费级化" 正在加速。虽然 24 张 4090 仍非普通用户能负担，但相比 H100 集群已大幅降低门槛。

### 4. 与 DeepSWE 的关联

GLM-5.2 在 DeepSWE 上得分 46.2，相比 Claude Opus 4.8 (58) 和 GPT-5.5 (70) 仍有差距。这说明：
- 长周期软件工程仍是闭源模型的护城河
- 开源模型在单轮/短周期任务上已接近甚至超越闭源
- Agent 能力的差距可能比纯代码能力更大

### 5. 推理成本分析

| 配置 | 显存需求 | 估算成本 |
|------|----------|----------|
| 4× RTX PRO 6000 | 384GB | ~$20,000 硬件 |
| 24× RTX 4090-48GB | 1,152GB | ~$24,000 硬件 |
| 8× H100-80GB | 640GB | ~$200,000+ 硬件 |
| API (Z.ai) | 按需 | $0.01-0.02 / 1K tokens |

对于大多数应用，API 调用仍是最经济的选择；本地部署适合数据隐私要求高或需要大规模批处理的场景。

---

## 参考资源

- **GitHub**: https://github.com/zai-org/GLM-5
- **技术报告**: https://arxiv.org/abs/2602.15763
- **Hugging Face**: https://huggingface.co/zai-org/GLM-5.2-FP8
- **API 平台**: https://z.ai
- **API 文档**: https://docs.bigmodel.cn/
- **slime (训练框架)**: https://github.com/THUDM/slime
- **RTX PRO 6000 部署**: https://github.com/0xSero/glm-5.2-sm120
- **RTX 4090 社区移植**: https://github.com/renning22/glm-5.2-4090
