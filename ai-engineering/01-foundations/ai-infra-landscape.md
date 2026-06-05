#topic/ai-infra #year/2026 #status/draft

# AI Infra 知识全景图

> AI Infra（人工智能基础设施）是支撑 AI 模型训练、部署和应用的底层技术体系。本文梳理 Infra 层的核心知识模块。

---

## 1. 计算层（Compute）

### 1.1 AI 芯片与硬件

| 类型 | 代表产品 | 特点 | 适用场景 |
|------|----------|------|----------|
| **GPU** | NVIDIA H100/A100, RTX 4090 | 并行计算能力强，生态成熟 | 训练、推理 |
| **TPU** | Google TPU v5 | 专为 TensorFlow 优化 | Google Cloud 训练 |
| **NPU** | Apple Neural Engine, 华为昇腾 | 低功耗，端侧推理 | 手机、边缘设备 |
| **专用芯片** | Groq, Cerebras, SambaNova | 极致优化特定 workload | 高性能推理 |

### 1.2 训练基础设施

- **分布式训练**：数据并行、模型并行、流水线并行
- **训练框架**：PyTorch、TensorFlow、JAX、DeepSpeed
- **集群调度**：Kubernetes + GPU Operator、Slurm、Ray
- **存储**：高性能并行文件系统（Lustre、GPFS）

### 1.3 推理优化

| 技术 | 原理 | 收益 |
|------|------|------|
| **KV Cache** | 缓存注意力机制的 key/value | 减少重复计算，加速自回归生成 |
| **Quantization** | 低精度表示（INT8/INT4/FP8） | 减少显存占用，提升吞吐 |
| **Speculative Decoding** | 小模型草稿 + 大模型验证 | 2-3 倍加速 |
| **Continuous Batching** | 动态批处理请求 | 提升 GPU 利用率 |
| **PageAttention** | 分页管理 KV Cache | 减少显存碎片，支持更长上下文 |

---

## 2. 数据层（Data）

### 2.1 数据 pipeline

```
原始数据 → [采集] → [清洗] → [标注] → [增强] → [向量化] → [存储]
              ↓        ↓        ↓        ↓         ↓
           爬虫/API  去重/过滤  人工/自动  合成数据  Embedding  向量DB
```

### 2.2 数据存储

| 类型 | 工具 | 用途 |
|------|------|------|
| **对象存储** | S3, GCS, MinIO | 原始数据、模型 checkpoint |
| **向量数据库** | Pinecone, Chroma, Qdrant, Milvus | 语义检索 |
| **特征存储** | Feast, Tecton | 特征复用 |
| **数据湖** | Delta Lake, Iceberg, Hudi | 大规模数据管理 |

### 2.3 数据质量

- **数据清洗**：去重、过滤低质量内容、PII 检测
- **数据标注**：主动学习、人机协同标注
- **数据版本**：DVC、LakeFS
- **数据血缘**：追踪数据来源和转换过程

---

## 3. 模型层（Model）

### 3.1 模型训练

- **预训练**：自监督学习、大规模语料
- **微调（Fine-tuning）**：
  - Full Fine-tuning：更新全部参数
  - LoRA/QLoRA：低秩适配，只训练少量参数
  - Prefix Tuning：训练前缀嵌入
- **对齐**：RLHF、DPO、Constitutional AI

### 3.2 模型管理

| 环节 | 工具/实践 |
|------|-----------|
| **版本控制** | Git LFS、DVC、MLflow |
| **模型注册** | MLflow Model Registry、Hugging Face Hub |
| **模型压缩** | 剪枝、量化、知识蒸馏 |
| **模型评估** | 自动评估 pipeline、A/B 测试 |

### 3.3 模型服务（Model Serving）

```
[负载均衡] → [API Gateway] → [推理服务] → [模型实例]
                               ↓
                         [Auto Scaling]
```

- **推理框架**：vLLM、TensorRT-LLM、TGI、Triton
- **服务架构**：
  - 同步推理：实时响应（聊天、搜索）
  - 异步推理：批处理（文档分析、生成任务）
  - 流式推理：SSE/Websocket 逐 token 返回

---

## 4. 平台层（Platform）

### 4.1 MLOps / LLMOps

| 阶段 | 工具 | 功能 |
|------|------|------|
| **实验跟踪** | W&B, MLflow, Neptune | 记录超参数、指标、artifact |
| **工作流编排** | Airflow, Prefect, Kubeflow | 自动化 pipeline |
| **监控告警** | Prometheus + Grafana, Evidently | 模型漂移、数据漂移检测 |
| **A/B 测试** | Statsig, LaunchDarkly | 模型效果对比 |

### 4.2 模型网关（Model Gateway）

统一接入层，处理：
- **路由**：根据请求特征选择模型
- **限流**：防止过载和成本失控
- **缓存**：重复查询直接返回
- **Fallback**：主模型失败时切换备用
- **计费**：按 token/请求计费

### 4.3 可观测性（Observability）

- **日志**：结构化日志，追踪请求链路
- **指标**：延迟、吞吐量、错误率、成本
- **追踪**：OpenTelemetry，端到端请求追踪
- **评估**：RAGAS、TruLens、自定义评估

---

## 5. 安全与治理层（Security & Governance）

### 5.1 模型安全

- **Prompt 注入防护**：输入过滤、输出检测
- **数据泄露防护**：训练数据去标识化
- **模型窃取防护**：API 速率限制、水印
- **对抗攻击防御**：对抗训练、输入净化

### 5.2 AI 治理

- **模型卡（Model Card）**：记录模型能力、限制、偏见
- **数据卡（Data Card）**：记录数据来源、处理方式
- **审计日志**：追踪模型决策过程
- **合规**：GDPR、AI Act、算法备案

---

## 6. 工具链生态

### 6.1 开发工具

| 类别 | 工具 |
|------|------|
| **Notebook** | Jupyter, Google Colab, Deepnote |
| **IDE** | VS Code + AI 插件, Cursor, Windsurf |
| **调试** | LangSmith, Phoenix, Langfuse |
| **测试** | pytest, Great Expectations |

### 6.2 部署工具

| 类别 | 工具 |
|------|------|
| **容器** | Docker, Kubernetes |
| **Serverless** | AWS Lambda, Cloud Functions |
| **边缘部署** | ONNX Runtime, TensorFlow Lite, Core ML |
| **私有化** | vLLM, Ollama, llama.cpp |

---

## 💡 我的思考

1. **Infra 是 AI 应用的护城河**：模型能力越来越同质化，Infra 优化（延迟、成本、稳定性）成为竞争关键。

2. **推理优化是 2026 年的主战场**：训练需求相对稳定，但推理成本随用户量线性增长。KV Cache、Quantization、Speculative Decoding 等技术将快速普及。

3. **向量数据库是新的基础设施**：随着 RAG 成为标配，向量数据库从" nice to have"变为"must have"。

4. **LLMOps 正在快速成熟**：从 MLOps 演进而来，但更关注 prompt 版本管理、RAG 评估、Agent 追踪等特有需求。

5. **安全不能事后补救**：AI 安全需要在设计阶段就考虑，Prompt 注入、数据泄露等风险需要系统性防护。

---

## 参考来源

- [Huyen Chip: Building A Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html) — 访问日期：2026-06-05
- [vLLM Documentation](https://docs.vllm.ai/) — 访问日期：2026-06-05
- [Anyscale Blog](https://www.anyscale.com/blog) — 访问日期：2026-06-05
- [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — 访问日期：2026-06-05
