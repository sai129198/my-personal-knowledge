# GLM-5.2 深度研究报告

> 研究时间: 2026-06-22（初版），2026-07-06（增补：API 定价、MTP 5-token、NVFP4、IndexCache、DevPack 生态）
> 来源: zai-org/GLM-5 GitHub, Hugging Face, arXiv, 社区部署实践, Z.ai API 文档, vLLM/SGLang Recipes
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
| HLE (w/ Tools) | **54.7** | 52.3 | 53.5 | **57.9** | 52.2 |
| CritPt | 20.9 | 4.6 | 13.4 | 20.9 | **27.1** |
| AIME 2026 | **99.2** | 95.3 | 97 | 95.7 | 98.3 |
| HMMT Nov. 2025 | 94.4 | 94 | 95 | **96.5** | **96.5** |
| HMMT Feb. 2026 | 92.5 | 82.6 | **97.1** | **96.7** | **96.7** |
| IMOAnswerBench | **91.0** | 83.8 | 90 | 83.5 | - |
| GPQA-Diamond | 91.2 | 86.2 | 90 | **93.6** | **93.6** |

### 代码能力

| 基准 | GLM-5.2 | GLM-5.1 | Claude Opus 4.8 | GPT-5.5 |
|------|---------|---------|-----------------|---------|
| SWE-bench Pro | **62.1** | 58.4 | **69.2** | 58.6 |
| NL2Repo | **48.9** | 42.7 | **69.7** | 50.7 |
| DeepSWE | **46.2** | 18 | 58 | **70** |
| ProgramBench | **63.7** | 50.9 | **71.9** | **70.8** |
| Terminal Bench 2.1 | **81.0** | 63.5 | **85.0** | 84.0 |
| Terminal Bench (Best Harness) | **82.7** | 69 | 78.9 | **83.4** |
| FrontierSWE | **74.4** | 30.5 | **75.1** | 72.6 |
| PostTrainBench | **34.3** | 20.1 | **37.2** | 28.4 |
| SWE-Marathon | **13.0** | 1.0 | **26.0** | 12.0 |

### Agent 能力

| 基准 | GLM-5.2 | GLM-5.1 | Claude Opus 4.8 | DeepSeek-V4-Pro |
|------|---------|---------|-----------------|-----------------|
| MCP-Atlas | **76.8** | 71.8 | **77.8** | 73.6 |
| Tool-Decathlon | 48.2 | 40.7 | **59.9** | **52.8** |

> **我的判断**：GLM-5.2 在开源模型中全面领先，在代码和 Agent 任务上已接近 Claude Opus 4.8，但推理能力仍有差距。详细分析见 [GLM-5.2 基准测试深度解析](/docs/glm-5.2-benchmarks-deep-dive.md)。
>
> **核心洞察**：
> - 🥇 **绝对优势**：AIME 2026（99.2）、IMOAnswerBench（91.0）——竞赛数学世界第1
> - 🥈 **开源最强**：SWE-bench Pro（62.1）、Terminal Bench（81.0）、FrontierSWE（74.4）——代码能力开源第一
> - 🥉 **接近前沿**：MCP-Atlas（76.8 vs Claude 77.8）——Agent 能力接近闭源
> - ⚠️ **仍有差距**：HLE（40.5 vs Claude 49.8）、DeepSWE（46.2 vs GPT-5.5 70）、SWE-Marathon（13.0 vs Claude 26.0）——极限推理和超长周期工程

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

### 4. 与 DeepSeek-V4-Pro 的对比

同为 2026 年 6 月发布的开源旗舰模型，两者定位不同：

| 维度 | GLM-5.2 | DeepSeek-V4-Pro |
|------|---------|-----------------|
| 总参数量 | 744B | **1.6T** |
| 激活参数量 | 40B | **49B** |
| 竞赛数学 (AIME) | **99.2** 🥇 | 95.2 |
| 代码修复 (SWE-bench Pro) | **62.1** 🥇 | 55.4 |
| 终端操作 (Terminal Bench) | **81.0** 🥇 | 67.9 |
| Agent (MCP-Atlas) | **76.8** | 73.6 |
| 工具使用 (Tool-Decathlon) | 48.2 | **51.8** |
| 知识问答 (MMLU-Pro) | 未报告 | **87.5** |

**结论**：GLM-5.2 是"代码专家"，DeepSeek-V4-Pro 是"知识通才"。详细对比见 [GLM-5.2 vs DeepSeek-V4-Pro 深度对比](/docs/glm-5.2-vs-deepseek-v4-pro.md)。

### 5. 与 DeepSWE 的关联

GLM-5.2 在 DeepSWE 上得分 46.2，相比 Claude Opus 4.8 (58) 和 GPT-5.5 (70) 仍有差距。这说明：
- 长周期软件工程仍是闭源模型的护城河
- 开源模型在单轮/短周期任务上已接近甚至超越闭源
- Agent 能力的差距可能比纯代码能力更大

### 5. 推理成本分析

| 配置 | 显存需求 | 估算成本 |
|------|----------|----------|
| 4× RTX PRO 6000 | 384GB | ~$20,000 硬件 |
| 32× RTX 4090-48GB (最新社区实测) | 1,536GB | ~$32,000 硬件 |
| 8× H100-80GB | 640GB | ~$200,000+ 硬件 |
| API (Z.ai) | 按需 | $1.4 input / $4.4 output per 1M tokens |

> **⚠️ 价格校正（2026-07-06）**：之前估算的 $0.01-0.02/1K tokens 偏差较大。实际 Z.ai 官方定价为 **$1.4/M 输入 + $4.4/M 输出**。按输入输出比 4:1 计算，实际成本约 ~$2.0/1M tokens。Cached Input 仅 $0.26/M，对长对话场景友好。

对于大多数应用，API 调用仍是最经济的选择；本地部署适合数据隐私要求高或需要大规模批处理的场景。

### 6. 与 GLM-5.2 相关的定价梯度

Z.ai 的阶梯定价值得关注：

| 模型 | Input/1M | Output/1M | 性价比判断 |
|------|----------|-----------|-----------|
| GLM-5.2 | $1.4 | $4.4 | 旗舰，代码/Agent |
| GLM-5-Turbo | $1.2 | $4.0 | 略低定价的 5 代 |
| GLM-4.7 | $0.6 | $2.2 | 便宜 2.3×，日常可用 |
| GLM-4.7-Flash | 免费 | 免费 | 轻量场景零成本 |

**策略建议**：GLM-4.7 在日常对话和普通编程任务上性价比更高，仅在需要 1M 上下文或最优代码能力时使用 GLM-5.2。

---

## 🆕 2026-07-06 增补：新发现与深度洞察

### A. MTP 推测解码：从 3 token 到 5 token

> 这是我之前遗漏的关键细节。

GLM-5.2 的 Multi-Token Prediction 在实际部署中默认使用 **5 个 draft tokens**（vLLM/SGLang 均配置为 `--speculative-config.num_speculative_tokens 5`），而之前的技术报告描述的是 3 token。实际效果：

- 接受长度提升 20%（相比 GLM-5/5.1）
- 在短上下文解码速度测试中，5-token MTP 配合 FP8 KV Cache，8×H200 单节点可达到高吞吐
- **重要提示**：纯吞吐量基准测试（如 random 数据集）会低报 MTP 实际收益，因为 MTP 在高信息密度推理任务上的接受率更高

vLLM #45895 PR 专门修复了 MTP 接受率问题，部署时应使用最新分支或官方 Docker 镜像 `vllm/vllm-openai:glm52`。

### B. IndexCache / IndexShare 学术支撑

> 这是我本次留学最大的技术收获。

IndexShare 的本质在 arXiv:2603.12201 论文《Accelerating Sparse Attention via Cross-Layer Index Reuse》中得到了完整阐述：

**核心发现**：连续层的 top-k 选择高度相似。

**两种方案**：
1. **Training-free IndexCache**：贪心搜索哪些层保留 indexer，直接最小化 calibration 集合上的 LM loss，不需要权重更新
2. **Training-aware IndexCache**：多层级蒸馏损失，让保留的 indexer 学习它服务的所有层的平均注意力分布

**实验数据（30B DSA 模型）**：
- 移除 **75%** 的 indexer 计算，质量几乎无损
- Prefill 加速 **1.82×**，Decode 加速 **1.48×**

**GLM-5.2 中的应用**：78 层中 21 层 Full（自己跑 indexer）+ 57 层 Shared/Skip（复用最近 Full 层的 top-k），在 1M 上下文下每 token FLOPs 降低 **2.9×**。

> **我的判断**：这不仅仅是性能优化，而是稀疏注意力在工程化道路上的一个范式转变——"没必要每层都重新选择"是一个很工程化的洞察，简单但有效。

### C. NVFP4 + REAP 剪枝：469B 消费级变体

社区（0xSero）基于 NVIDIA ModelOpt 工具链，对 GLM-5.2 做了两项优化：
1. **REAP 剪枝**：将专家网络冗余参数剪掉
2. **NVFP4 量化**：仅 MoE 专家矩阵降到 4-bit（shared experts、attention、embedding 保持 FP8/BF16）

结果是 **GLM-5.2-NVFP4-REAP-469B**（约 313GB 磁盘空间），极大降低了部署门槛：

| 指标 | 值 |
|------|-----|
| 总参数 | 469B（pruned） |
| 磁盘占用 | ~313 GB（NVFP4） |
| 每 GPU 占用 | ~78.6 GB/GPU |
| 推荐 GPU | 4× RTX PRO 6000 Blackwell (SM120, 96GB) |
| 上下文 | 250K（DCP=4） |
| 解码速度 | ~60 tok/s（warm, MTP） |

> **关键约束**：DSA 的 fp4 indexer cache 硬要求 SM100（B200/GB200），所以 NVFP4 变体在 SM120 消费级卡上只能用 fp8 KV cache，不能进一步压到 fp4。

NVIDIA 也官方发布了 `nvidia/GLM-5.2-NVFP4`，自动检测量化格式，vLLM 零配置支持。

### D. AMD MI300X/MI355X 支持

GLM-5.2 已经通过 ROCm + AITER 后端支持 AMD GPU：

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
vllm serve zai-org/GLM-5.2-FP8 \
 --kv-cache-dtype fp8_e4m3 \
 --tensor-parallel-size 8 \
 --speculative-config.method mtp \
 --speculative-config.num_speculative_tokens 5 \
 --linear-backend aiter --moe-backend aiter
```

AITER 融合了 shared expert 的 MoE 路径，是 ROCm 生态的重要信号。

### E. Z.ai Developer Pack (DevPack) — 完整的 Coding Agent 生态

Z.ai 不只是提供模型 API，而是围绕 GLM-5.2 构建了一整套 Coding Agent 基础设施：

| 组件 | 功能 |
|------|------|
| **Coding Tool Helper** | Claude Code / Kilo Code / Cline / OpenCode / OpenClaw 一键接入 |
| **Web Search MCP Server** | 内置搜索工具 |
| **Web Reader MCP Server** | URL 内容抓取 |
| **Vision MCP Server** | 截图理解 |
| **Zread MCP Server** | 文档阅读 |
| **Coding Plan 订阅** | $18/月起，包含快速、可靠的代码生成和工具使用 |
| **Memory Mechanism** | 跨会话记忆 |
| **Context Caching** | 智能缓存长对话 |

> **产品洞察**：智谱的策略是"让 GLM-5.2 成为 Coding Agent 的默认后端"。$18/月的 Coding Plan 对标 Claude Code / GitHub Copilot 的价格带，但提供开源模型的透明度和灵活性。

### F. 社区部署进展更新

#### renning22/glm-5.2-4090（重大更新）

从初版的 24× RTX 4090 升级到了 **32× RTX 4090-48GB**（4 节点 × 8 卡）：

- 解码速度从 ~10 tok/s 提升到 **~24 tok/s**（CUDA Graph 单流）
- 聚合吞吐 **~725 tok/s**（128 并发流，CUDA Graph）
- 已验证驱动真实的 Claude Code session（tool calls、multi-turn loops、streaming）
- 所有 kernel 验证精度 ~1e-6，包括在实际模型 tensor 上的 cos 相似度 0.999999

#### 0xSero/glm-5.2-sm120（全新方案）

- 使用专用 vLLM Docker 镜像（voipmonitor/vllm:black-benediction 系列）
- 捆绑了 SM120 原生稀疏 MLA kernel (`B12X_MLA_SPARSE`)
- DCP=4 支持 250K 上下文
- 一键启动脚本（launch.sh），含自动健康检测

#### RTX 5090 兼容性

RTX 5090 虽然是 sm_120（consumer Blackwell），但官方 DSA kernel 只覆盖 sm_90/sm_100，不覆盖 sm_120。renning22 的 Ada 移植同样需要适配 RTX 5090（扩展 capability guard），估算 32× RTX 5090（32GB 版）可部署完整 FP8 模型。

### G. vLLM/SGLang 部署生态对比

| 框架 | 版本 | 特点 | 推荐场景 |
|------|------|------|----------|
| **vLLM** | ≥0.23.0 | Docker 镜像 `vllm/vllm-openai:glm52`、专用 tool/reasoning parser (`glm47`/`glm45`)、DeepGEMM 集成 | 生产部署、稳定版本 |
| **SGLang** | ≥0.5.13.post1 | 性能最佳、交互式 playground（自动生成配置）、DSA 原生优化 | 追求极致性能 |
| **KTransformers** | ≥0.5.12 | 消费级 GPU 友好、CPU/GPU 混合推理 | 消费级硬件 |
| **Unsloth** | ≥0.1.47-beta | 微调优化 | 微调场景 |

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
- **GLM-5.2-NVFP4-REAP-469B (消费级)**: https://huggingface.co/0xSero/GLM-5.2-NVFP4-REAP-469B
- **sm120 部署方案**: https://github.com/0xSero/glm-5.2-sm120
- **NVIDIA NVFP4 官方量化**: https://huggingface.co/nvidia/GLM-5.2-NVFP4
- **IndexCache/IndexShare 论文**: https://arxiv.org/abs/2603.12201
- **Z.ai 定价**: https://docs.z.ai/guides/overview/pricing
- **Z.ai DevPack**: https://docs.z.ai/devpack/overview
- **vLLM Recipe**: https://recipes.vllm.ai/zai-org/GLM-5.2
- **SGLang Cookbook**: https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2
