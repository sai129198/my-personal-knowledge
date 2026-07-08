# Cognee 深度研究报告

> 研究时间: 2026-07-08
> 来源: GitHub (topoteretes/cognee), Cognee 官方文档 (docs.cognee.ai), arXiv (2505.24478), BEAM benchmark
> 标签: #status/canonical #topic/ai-infrastructure #topic/agent-memory #topic/rag #topic/knowledge-graph #year/2026

---

## 一句话总结

**Cognee 是一个开源的 AI Agent 记忆平台**，用知识图谱 + 向量搜索为 AI Agent 提供跨会话的持久化长期记忆。它将文档自动转化为实体-关系知识图谱，支持自托管部署，提供 `remember` / `recall` / `improve` / `forget` 四个核心操作，并已集成到 Claude Code 和 OpenClaw 等主流 Coding Agent 中。

---

## 基本信息

| 属性 | 详情 |
|------|------|
| **项目名称** | Cognee (认知引擎) |
| **开发者** | Topoteretes (Vasilije Markovic, Lazar Obradovic 等) |
| **GitHub** | https://github.com/topoteretes/cognee |
| **许可证** | Apache License 2.0 |
| **主要语言** | Python 3.10-3.14 |
| **Stars** | 活跃开源项目（README 中鼓励 star 加速社区增长） |
| **论文** | [Optimizing the Interface Between Knowledge Graphs and LLMs for Complex Reasoning](https://arxiv.org/abs/2505.24478) (May 2025) |
| **文档** | https://docs.cognee.ai |
| **云端服务** | https://www.cognee.ai (Cognee Cloud) |
| **OpenClaw 插件** | `@cognee/cognee-openclaw` |
| **Claude Code 插件** | `cognee-memory@cognee` |

---

## 核心能力与架构

### 四大操作 (v1.0 API)

Cognee 将复杂的记忆管理抽象为四个操作：

```
remember  →  recall  →  improve  →  forget
  (存)       (查)       (优化)       (删)
```

| 操作 | 功能 | 层级 |
|------|------|------|
| `remember()` | 一行代码将文本/文件/URL 存入记忆。自动完成：分块 → 实体提取 → 知识图谱构建 → 向量嵌入 | 高级 |
| `recall()` | 自然语言查询记忆。自动路由选择最佳检索策略（图补全/摘要/时间/代码规则/词法），支持 session 感知 | 高级 |
| `improve()` | 丰富已有记忆，跨会话将 session 记忆桥接到永久图谱，支持反馈加权 | 高级 |
| `forget()` | 删除单个数据项、整个数据集或当前用户所有存储 | 高级 |
| `add()` / `cognify()` / `search()` / `memify()` | 底层操作，逐步控制每个 pipeline 阶段 | 低级 |

### 三层存储架构

Cognee 的核心设计哲学是 **"没有任何单一数据库能处理记忆的所有方面"**：

```
        ┌─────────────────────────────┐
        │     Relational Store        │  文档元数据、分块、溯源
        │  (Postgres / SQLite)        │  "每段数据从哪来"
        ├─────────────────────────────┤
        │     Vector Store            │  语义相似度搜索
        │  (pgvector / LanceDB /      │  "找概念相近的内容"
        │   Qdrant / Milvus / ...)    │
        ├─────────────────────────────┤
        │     Graph Store             │  实体和关系的知识图谱
        │  (Postgres Graph / Neo4j /  │  "概念之间的结构连接"
        │   Neptune / KuzuDB)         │
        └─────────────────────────────┘
```

**关键创新**：从 1.0 开始，Cognee 支持纯 Postgres 运行整个记忆层（关系 + 向量 + 图 + 会话缓存，全部在一个 Postgres 实例中）。这在 CI 基准测试中比分离的图+向量方案**快约 10%**。

### 处理管线

```
原始文档 → Chunking(分块) → Entity Extraction(实体提取)
    → Relationship Extraction(关系提取)
    → Ontology Induction(本体归纳)
    → Knowledge Graph(知识图谱)
    → Vector Embedding(向量嵌入)
    → Searchable Memory(可搜索记忆)
```

基于认知科学的本体生成使文档既可语义搜索，又可结构化推理。

---

## 检索策略矩阵

Cognee 内置了丰富的检索策略，通过 `query_type` 路由选择：

| 检索策略 | 说明 | 适用场景 |
|----------|------|----------|
| `GRAPH_COMPLETION` | 默认图补全检索 | 通用问答 |
| `GRAPH_COMPLETION_COT` | 带思维链的图检索 | 复杂推理问题 |
| `GRAPH_COMPLETION_DECOMPOSITION` | 问题分解 + 图检索 | 多跳推理 |
| `GRAPH_COMPLETION_CONTEXT_EXTENSION` | 图上下文扩展 | 关系探索 |
| `GRAPH_SUMMARY_COMPLETION` | 图表摘要 | 概览类问题 |
| `RAG_COMPLETION` | 基于嵌入的 RAG | 语义相似度搜索 |
| `HYBRID_COMPLETION` | 向量+图谱混合 | 兼顾语义和结构 |
| `TRIPLET_COMPLETION` | 三元组检索 | 精确事实查找 |
| `TEMPORAL` | 时间感知检索 | "什么时候"/"之前/之后" |
| `CHUNKS` | 语义分块搜索 | 段落级检索 |
| `CHUNKS_LEXICAL` | 词法分块搜索 | 精确短语匹配 |
| `SUMMARIES` | 摘要检索 | 概览/大纲 |
| `CODING_RULES` | 代码规则检索 | 编程规则/约定 |
| `CYPHER` | 直接 Cypher 查询 | 数据库级查询 |
| `AGENTIC_COMPLETION` | Agentic 检索 | 单数据集聚焦 |
| `FEELING_LUCKY` | 路由器自动选择 | 不确定时 |

**自动路由**：当不指定 `query_type` 时，Cognee 分析查询模式自动选择策略：
- "概览/总结/关键要点" → 摘要检索
- "X 和 Y 的关系" → 图上下文扩展
- "什么时候/之前/之后" → 时间检索
- `async def` / 代码规则 → 代码规则检索
- 精确带引号短语 → 词法分块搜索

---

## 部署架构

### 三层部署模式

| 模式 | 组件 | 适用 |
|------|------|------|
| **本地开发** | SQLite + LanceDB + KuzuDB | 零额外服务，开发测试 |
| **Postgres 单实例** | Postgres + pgvector + Postgres Graph | 生产默认推荐 |
| **专用后端** | Neo4j/Neptune + Redis + Pgvector/LanceDB/Milvus/Qdrant | 大规模 / 特殊需求 |

### Docker 化部署

```bash
# 完整服务栈
docker compose up                    # API server (8000)
docker compose --profile ui up       # + 前端 UI (3000)
docker compose --profile mcp up      # + MCP 服务器 (8001)
docker compose --profile postgres up # + Postgres + pgvector
docker compose --profile neo4j up    # + Neo4j
```

Docker Hub 提供预构建镜像：`cognee/cognee` (API) 和 `cognee/cognee-mcp` (MCP)。

### 一键部署平台

| 平台 | 特点 |
|------|------|
| **Cognee Cloud** | 全托管，无需维护基础设施 |
| **Modal** | 无服务器，自动扩缩，GPU 工作负载 |
| **Railway** | 最简单的 PaaS，原生 Postgres |
| **Fly.io** | 边缘部署，持久卷 |
| **Render** | 简单 PaaS，托管 Postgres |
| **Daytona** | 云端沙箱 (SDK/CLI) |

---

## 多语言客户端支持

Cognee 不只是 Python 库，还有官方维护的多语言客户端：

| 客户端 | 安装 | 说明 |
|--------|------|------|
| Python | `pip install cognee` | 主力客户端（完整功能） |
| Rust | `cargo add cognee` | Rust 原生绑定 |
| TypeScript/Node.js | `npm install @cognee/cognee-ts` | Node.js / 浏览器 |

---

## 与 Agent 框架的集成

### Claude Code 插件

Cognee 为 Claude Code 提供了最完整的集成：

```
Claude Code 启动
    → SessionStart: 选择模式，设置身份
    → UserPromptSubmit: 注入 dataset 范围内的上下文
    → PostToolUse: 捕获工具调用痕迹
    → Stop: 记录当前回答
    → PreCompact: 在上下文压缩时保留记忆
    → SessionEnd: 同步 session 记忆到永久图谱
```

安装方式：
```bash
claude plugin marketplace add topoteretes/cognee-integrations
claude plugin install cognee-memory@cognee

# 本地模式（自动引导本地 Cognee API）
export LLM_API_KEY="sk-..."
claude

# 云端模式
export COGNEE_BASE_URL="https://your-instance.cognee.ai"
export COGNEE_API_KEY="ck_..."
claude
```

### OpenClaw 插件

`@cognee/cognee-openclaw` — 在 OpenClaw Coding Agent 中提供持久记忆。

---

## 性能基准：BEAM 长上下文测试

Cognee 在 [BEAM](https://github.com/topoteretes/cognee) 基准测试（测试系统能否在长对话中跟踪变化）上表现突出：

| 配置 | Cognee | 此前 SOTA | RAG 基线 |
|------|--------|-----------|----------|
| **BEAM 100K tokens** | **0.79** (>0.8 带路由) | 0.735 | ~0.33 |
| **BEAM 10M tokens** | **0.67** | 0.641 | ~0.33 |

> 关键：这些结果使用 Cognee 的**默认设置**和**标准开源特征**实现——没有定制模型，没有 BEAM 专用管线。相比传统 RAG（~0.33），Cognee 的知识图谱方案带来了翻倍以上的提升。

---

## 核心优势

### 1. 知识图谱带来结构化推理能力

传统 RAG 只能做"相似度匹配"，Cognee 通过知识图谱实现了：
- **多跳推理**：A → 关系 → B → 关系 → C，而非仅靠向量相似度
- **关系感知**：识别"属于/导致/位于/依赖"等语义关系
- **结构化导航**：用 Cypher 沿图谱边遍历，而不是平面搜索

### 2. Session 记忆与永久图谱的分层

Cognee 的 session 记忆设计非常巧妙地解决了 Agent 记忆的两个层次：
- **Session 缓存**：快速键值查找，存对话 context
- **永久图谱**：经过 `improve()` 后从 session 蒸馏到图谱，形成可长期复用的知识

### 3. 纯 Postgres 简化运维

绝大多数 AI memory 方案需要运行图数据库 (Neo4j) + 向量数据库 (Pinecone/Qdrant) + 缓存 (Redis) + 关系数据库。Cognee 1.0 让 Postgres 承载全部，且性能反而更好（CI 测试快 10%）。这对个人开发者和中小团队是巨大的运维简化。

### 4. 认知科学根基

Cognee 不只是工程实现，还发表了同行评议论文（arXiv:2505.24478），研究知识图谱与 LLM 之间的接口优化。它的本体归纳 (Ontology Induction) 能力基于认知科学理论，而不仅仅是工程直觉。

### 5. Agent 原生

从一开始就是为 Agent 设计的：
- Agent memory decorator（装饰器模式给函数加记忆）
- 跨 agent 知识共享
- Agent 级别的用户/租户隔离
- 可观察性（OTEL collector）
- 审计追踪

---

## 局限与风险

### 1. 依赖 LLM 进行图构建

实体提取、关系提取、本体归纳等核心步骤都需要调用 LLM。这意味着：
- **需要持续的 API 成本**（每次 `remember()` 都会调用 LLM）
- **冷启动时间**（首次建图需要多轮 LLM 调用）
- **质量依赖于底层 LLM**（如果用便宜的模型，图谱质量可能下降）

### 2. 生态成熟度

相比 Mem0（更成熟、有更多集成）和 Letta（前身是 MemGPT），Cognee 相对较新。GitHub 仓库的星星数和社区贡献者规模还需要时间积累。

### 3. 图构建的通用性局限

虽然 Cognee 有 ontology 机制支持领域知识，但默认的实体提取/关系提取可能在某些高度专业化的领域（医疗、法律、金融术语）表现不佳，需要自定义 Task/Pipeline。

### 4. 闭源的 Max 模型不开放

Cognee Cloud 的商业化路径和 Qwen3.7 类似——开源客户端 + 闭源云端服务。如果 Cognee Cloud 在某个时候成为核心功能的唯一载体，可能会影响自托管用户的体验。

---

## 与竞品对比

| 维度 | **Cognee** | Mem0 | Letta (MemGPT) | Zep |
|------|-----------|------|---------------|-----|
| **核心方法** | 知识图谱 + 向量 + 认知科学 | 向量 + 元记忆层 | 操作系统的记忆层次 (Working/Archival) | 时序 + 向量 |
| **知识图谱** | ✅ 原生一流 | ⚠️ 有限 | ✅ 自管理 | ⚠️ 基础 |
| **开源** | ✅ Apache 2.0 | ✅ Apache 2.0 | ✅ Apache 2.0 | ✅ 部分开源 |
| **语言** | Python + Rust + TS | Python | Python | Python + Go |
| **Postgres 单实例** | ✅ 1.0 默认 | ❌ 需要多实例 | ❌ 需要多实例 | ❌ 需要多实例 |
| **Claude Code 集成** | ✅ 一流 | ⚠️ 社区 | ❌ | ❌ |
| **OpenClaw 集成** | ✅ 官方插件 | ❌ | ❌ | ❌ |
| **多语言客户端** | ✅ Rust + TS 官方 | ❌ Python only | ❌ Python only | ❌ Python + Go |
| **学术论文** | ✅ arXiv 2025 | ❌ | ✅ 论文 (MemGPT) | ❌ |
| **BEAM 基准** | ✅ SOTA (0.79) | 未公开 | 未公开 | 未公开 |
| **成熟度** | 🟡 较新但快速迭代 | 🟢 较成熟 | 🟡 转型中 | 🟢 企业级 |

> **我的判断**：Cognee 在"知识图谱 + 多语言客户端 + Agent 原生集成"这个组合上目前没有直接竞争对手。Mem0 更成熟但缺乏深层图谱能力，Letta 概念先进但社区生态较慢，Zep 偏向企业级会话存储。

---

## 典型使用场景

### 场景 1：编程 Agent 的长期记忆

```python
import cognee

# Agent 记住了项目架构
await cognee.remember(
    "Our project uses Clean Architecture. The domain layer "
    "contains User, Order, Product aggregates. Infrastructure "
    "handles Postgres repositories and Redis caching.",
    dataset="project_knowledge"
)

# 三周后，Agent 仍然记得架构
results = await cognee.recall(
    "What database does this project use?",
    datasets=["project_knowledge"]
)
# → "Postgres via repository pattern in Infrastructure layer"
```

### 场景 2：客服 Agent 的跨会话追踪

```python
# Session 1: 用户投诉账单问题
await cognee.remember(
    "User reports invoice INV-2024-001 showing wrong amount. "
    "Root cause: payment-invoice sync delay. Fix applied.",
    session_id="support_ticket_1234",
    dataset="customer_support"
)

# 两周后，新 session 但同用户
results = await cognee.recall(
    "Has this user reported billing issues before?",
    datasets=["customer_support"]
)
# → 找到历史案例，避免重复排查
```

### 场景 3：SQL Copilot（专家知识沉淀）

```python
# 存储专家级 SQL 模式
await cognee.remember(
    "For retention calculation: use window functions over "
    "subscription_events partitioned by user_id, with recursive "
    "CTE for cohort analysis. See pattern: retention_v2.sql",
    dataset="sql_patterns"
)

# 初级分析师提问
results = await cognee.recall(
    "How to calculate customer retention for e-commerce?",
    datasets=["sql_patterns"]
)
# → 返回专家级别的实现模式
```

---

## 快速上手

```python
import cognee
import asyncio

async def main():
    # 设置 LLM（支持 OpenAI、Anthropic、Ollama 等）
    import os
    os.environ["LLM_API_KEY"] = "your-api-key"

    # 一行存储：分块 → 提取实体 → 建图 → 嵌入 → 可检索
    await cognee.remember("Cognee turns documents into AI memory.")

    # 自然语言查询（自动路由选择最佳策略）
    results = await cognee.recall("What does Cognee do?")
    for r in results:
        print(r.text)

    # Session 记忆（快速缓存，后台同步到图谱）
    await cognee.remember(
        "User prefers detailed explanations.",
        session_id="chat_1"
    )
    results = await cognee.recall(
        "What does the user prefer?",
        session_id="chat_1"
    )

    # 删除
    await cognee.forget(dataset="main_dataset")

asyncio.run(main())
```

也可以用 CLI：
```bash
cognee-cli remember "Cognee turns documents into AI memory."
cognee-cli recall "What does Cognee do?"
cognee-cli -ui  # 打开本地 UI + MCP server
```

---

## 关键洞察与判断

### 1. 知识图谱是 Agent 记忆的正确答案

Cognee 用 BEAM 基准证明了：知识图谱相比纯 RAG 在长上下文理解上有 **翻倍** 的提升（0.79 vs 0.33）。这不是小改进，是范式优势。向量嵌入可以告诉你"这段话和那段话意思相近"，但知识图谱能告诉你"用户 John 购买了 Product X，Product X 属于 Category Y，Y 类产品有退货率问题"——这是 RAG 做不到的结构化推理。

### 2. Cognee 的"多 Agent"时代定位很准

随着 Claude Code、OpenClaw、Qwen Code 等 Coding Agent 的普及，Agent 需要"记住上一次对话我们做到哪里了"、"记住这个项目的架构"、"记住上次你是怎么调试这个 bug 的"。Cognee 恰好在解决这个问题。

### 3. 纯 Postgres 运行是降维打击

对于一个 side project 或小团队来说，"装个 Neo4j + 配置 Pinecone API key + 起 Redis 实例" 这三层运维是让人放弃自托管 AI memory 的主要原因。Cognee 让一切都跑在 Postgres 上，甚至不需要改 SQL——用 pgvector 扩展就行，和你的应用数据库共存。

### 4. 和 Agent 框架不是竞争，是互补

Cognee 的定位很聪明：它不和 LangChain/LlamaIndex 竞争（那些是 Agent 编排框架），而是作为"记忆层"嵌入到任何框架中。Claude Code 插件已经证明了这一点——Agent 不知道 Cognee 的存在，但 Cognee 在后台默默地注入上下文、记录对话、同步知识。

### 5. 与 OpenClaw 的结合潜力

Cognee 已经提供 `@cognee/cognee-openclaw` 插件。如果集成到 OpenClaw Coding Agent 中，可以实现：
- 项目级代码知识图谱（自动记住文件结构、模块依赖、架构决策）
- 跨 session 的对话记忆（"上次你建议用 Redis 缓存这个查询"）
- Agent 自我改进（通过 `improve()` 从反馈中学习）

---

## 待研究问题

- [ ] Cognee 的实体提取/关系提取在中文文档上的表现如何？
- [ ] 大规模数据集（10M+ 实体）下的图谱查询性能
- [ ] Cognee Cloud 的定价模型和 SLA
- [ ] 与 RAG（如 LangChain/LlamaIndex）+ Neo4j 手动集成的详细对比
- [ ] 在不同 LLM 提供商（OpenAI vs Anthropic vs Qwen）下的图谱构建质量差异

---

## 参考资源

- **GitHub**: https://github.com/topoteretes/cognee
- **官方文档**: https://docs.cognee.ai
- **论文**: [Optimizing the Interface Between Knowledge Graphs and LLMs for Complex Reasoning](https://arxiv.org/abs/2505.24478) — Markovic et al., 2025
- **Cognee Cloud**: https://www.cognee.ai
- **OpenClaw 插件**: https://www.npmjs.com/package/@cognee/cognee-openclaw
- **Rust 客户端**: https://github.com/topoteretes/cognee-rs
- **TypeScript 客户端**: https://www.npmjs.com/package/@cognee/cognee-ts
- **Claude Code 插件**: https://github.com/topoteretes/cognee-integrations/tree/main/integrations/claude-code
- **Colab 教程**: https://colab.research.google.com/drive/12Vi9zID-M3fpKpKiaqDBvkk98ElkRPWy
- **Docker Hub**: https://hub.docker.com/r/cognee/cognee
