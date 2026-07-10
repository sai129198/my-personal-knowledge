# Cognee 深度解析：面向 Agent 的开源记忆平台

> **一句话定位**：Cognee 是一个开源 AI Memory 平台，将非结构化数据自动构建为可查询的知识图谱，为 Agent 提供跨会话的持久化长期记忆。
>
> #status/canonical #topic/memory #topic/knowledge-graph #topic/agent-infra #year/2026 #tool/cognee

---

## 1. 核心定位与设计哲学

### 1.1 它解决什么问题？

LLM 每次调用是**无状态**的。Agent 会话结束，一切归零。Cognee 在 LLM 之上加了一层**持久记忆层**：

- 把文档、对话、代码、URL 等原始数据**自动转化为知识图谱**
- 支持跨会话的 **remember / recall / improve / forget** 四大操作
- 图谱中的实体和关系随使用不断**自我优化**

### 1.2 设计哲学

cognee 1.0 的核心信念：

1. **记忆不是 Agent 的一个 feature，而是整个 Agent 栈缺失的一层**
2. **开发者不需要为记忆再去运维一套分布式系统** — 一个 Postgres 就能跑完整套
3. **记忆数据必须可控、可迁移** — 开源 + COGX 开放格式，不锁定

> 🧠 *我的判断：cognee 把"记忆即基础设施"这个理念产品化了。它不是又一个 embedding + vector DB 的包装，而是在做"如何让记忆自我进化"这个更难的题。*

---

## 2. 技术架构

### 2.1 三层存储架构

Cognee 使用**三种互补存储**，各司其职：

```
┌─────────────────────────────────────────┐
│              Cognee 架构                 │
├─────────────┬─────────────┬─────────────┤
│ 关系存储     │ 向量存储     │ 图存储       │
│ (Relational)│ (Vector)    │ (Graph)     │
├─────────────┼─────────────┼─────────────┤
│ 文档/块元数据│ 语义嵌入     │ 实体+关系    │
│ 数据溯源     │ 相似性检索   │ 结构推理     │
│             │             │ 图遍历       │
└─────────────┴─────────────┴─────────────┘
```

- **关系存储**：追踪文档→chunk 的来龙去脉（provenance）
- **向量存储**：语义级别的相似匹配
- **图存储**：实体（节点）+ 关系（边）的结构化表达

三者有重叠——同一份信息可能被多个存储索引以提升效率。

### 2.2 v1.0 的关键简化：Just Postgres

**之前**：Neo4j（图）+ pgvector（向量）+ Redis（会话）+ SQL（元数据）= 四套系统

**现在**：一个 Postgres 实例承载全部：

| 记忆层 | 旧方案 | Cognee on Postgres |
|--------|--------|-------------------|
| 关系（图） | Neo4j 等图数据库 | Postgres 图后端 |
| 嵌入向量 | 独立向量数据库 | pgvector |
| 会话缓存 | Redis | SQL session-cache 后端 |
| 元数据 | 关系数据库 | 同一个 Postgres |

> 🧠 *我的判断：这是 cognee 最务实的设计决策。降低运维门槛比技术炫酷重要得多。企业内部推图数据库需要走安全审批、采购、运维培训——Postgres 大家都已经有。*

CI 基准测试中，单 Postgres 方案比分离架构快 ~10%，因为消除了跨服务调用的开销。

本地开发模式：SQLite + LanceDB + KuzuDB，零外部依赖即可启动。

---

## 3. API 设计：Memory-native 四动词

### 3.1 `remember()` — 存入记忆

```python
import cognee

# 存入永久图谱记忆
await cognee.remember("Cognee turns documents into AI memory.")

# 存入会话级记忆（快速缓存，后台同步到图谱）
await cognee.remember(
    "User prefers detailed explanations.",
    session_id="chat_1"
)

# 从其他工具迁移
from cognee.migration import Mem0Source, ZepSource, LettaSource
await cognee.remember(Mem0Source("mem0_export.json"))
```

**底层流程**（自动完成）：
1. 数据摄入（支持文本、文件、URL、数据库）
2. 分块（chunking）
3. 实体提取 & 关系建模
4. 嵌入生成
5. 图谱构建 + 向量索引

### 3.2 `recall()` — 检索记忆

```python
# 自动路由：cognee 自动选择最佳检索策略
results = await cognee.recall("What does Cognee do?")

# 会话感知：先查会话缓存，找不到再走图谱
results = await cognee.recall(
    "What does the user prefer?",
    session_id="chat_1"
)

# 指定检索类型
from cognee import SearchType
await cognee.recall(
    query_text="List coding guidelines",
    query_type=SearchType.CODING_RULES
)
```

**检索策略**（SearchType）：
- `CHUNKS` — 向量相似度直接匹配
- `SUMMARIES` — 基于文档摘要检索
- `GRAPH_COMPLETION` — 图遍历 + LLM 补全
- `GRAPH_COMPLETION_COT` — 带思维链的图推理
- `GRAPH_COMPLETION_CONTEXT_EXTENSION` — 扩展上下文获取
- `GRAPH_SUMMARY_COMPLETION` — 图谱摘要补全
- `RAG_COMPLETION` — 传统 RAG 路径
- `CYPHER` — 直接 Cypher 图查询
- `TEMPORAL` — 时间感知查询
- `NATURAL_LANGUAGE` — 自然语言查询

### 3.3 `improve()` — 记忆自我优化

```python
# 在图谱上运行增强 pass
await cognee.improve(dataset="main_dataset")

# remember 时自动触发
await cognee.remember(data, self_improvement=True)
```

**优化信号三来源**：
1. **反馈**（Feedback）：答案被确认/纠正/否决 → 对应节点和边的权重调整
2. **重要性**（Importance）：摄入时设定优先级，操作手册 > 闲聊记录
3. **频率**（Frequency）：高频使用的事实自然权重上升

> 🧠 *我的判断：improve 是 cognee 和大部分 memory 工具拉开差距的地方。其他工具是"存得越来越多"，cognee 是"用得越来越好"。反馈循环意味着系统不会重复犯错。*

### 3.4 `forget()` — 删除记忆

```python
# 按数据集删除
await cognee.forget(dataset="main_dataset")

# 按用户范围删除
await cognee.forget(user=user_id)

# 删除全部
await cognee.forget(all=True)
```

支持 GDPR/"被遗忘权"等合规场景。

---

## 4. 编程模型与扩展

### 4.1 构建块：DataPoints、Tasks、Pipelines

cognee 内部处理系统由三个基本组件构成：

- **DataPoints**：图节点的结构化数据单元，携带内容和元数据
- **Tasks**：单个数据处理单元（文本分析、关系提取等）
- **Pipelines**：Tasks 的编排工作流，类似数据加工流水线

```python
# 自定义 DataPoint
from cognee.modules.data.models import DataPoint

class CodeFile(DataPoint):
    language: str
    repository: str
    file_path: str

# 自定义 Task
from cognee.tasks import Task

@Task
async def extract_code_patterns(context):
    # 自定义提取逻辑
    pass
```

### 4.2 Agent Memory Decorator

最简洁的集成方式——一个装饰器搞定记忆注入：

```python
from cognee import agent_memory

@agent_memory()
async def my_agent(prompt: str) -> str:
    # cognee 在函数执行前自动注入相关记忆
    # 执行后自动持久化 trace
    return await run_llm(prompt)
```

### 4.3 多用户与权限系统

cognee 有完善的多租户架构：
- **Dataset**：数据隔离的基本单位
- **Tenant** → **User** → **Role** → **ACL**：企业级权限链
- 支持跨数据集共享和细粒度访问控制

---

## 5. 生态系统与集成

### 5.1 多语言 SDK

| 语言 | 包 | 特点 |
|------|-----|------|
| **Python** | `pip install cognee` | 最完整的功能 |
| **Rust** | `cognee-rs` | 边缘/设备端部署 |
| **TypeScript** | `@cognee/cognee-ts` | Node.js / 浏览器 |

### 5.2 Agent 集成

Cognee 提供开箱即用的集成：
- **Claude Code** (plugin)
- **OpenClaw** (`@cognee/cognee-openclaw`)
- **Cursor / Windsurf / Gemini CLI / Cline**（通过 MCP）
- **LangGraph**（SDK 集成）
- **MCP Server**（任何 MCP 兼容的 Agent）

### 5.3 MCP Server

```bash
# 启动 MCP Server（Docker 容器内运行）
cognee-cli -ui

# 直接 Docker
docker pull cognee/cognee-mcp:main
docker run -e TRANSPORT_MODE=http \
  --env-file ./.env -p 8000:8000 \
  --rm -it cognee/cognee-mcp:main
```

---

## 6. 部署模式

### 6.1 开发 → 生产的渐进路径

```
本地开发                  生产轻量                 企业生产
SQLite + LanceDB    →    单 Postgres        →    Neo4j + PGVector + Redis
(零外部依赖)              (一套数据库承载全部)       (独立扩展各组件)
```

### 6.2 部署选项

| 平台 | 适用场景 |
|------|---------|
| **Cognee Cloud** | 托管服务，零运维 |
| **Docker Compose** | 自托管，可选 UI/MCP/Neo4j profiles |
| **Modal** | Serverless，自动扩缩 |
| **Railway / Render** | 轻量 PaaS |
| **Kubernetes (Helm)** | 企业生产级部署 |
| **Fly.io** | 边缘部署 |

---

## 7. 研究背景与基准

### 7.1 学术论文

团队于 2025 年 5 月发表 arXiv 论文：
> *"Optimizing the Interface Between Knowledge Graphs and LLMs for Complex Reasoning"*
> — Markovic, Obradovic, Hajdu, Pavlovic

该论文在 HotPotQA、TwoWikiMultiHop、MuSiQue 三个多跳 QA 基准上，系统研究了分块策略、图构建、检索和提示词等超参数对性能的影响。

**核心发现**：
- 超参数调优确实能带来**一致的性能增益**
- 但增益**不均匀**，跨数据集和指标表现各异
- 未来进步不仅依赖架构改进，还需要更清晰的优化和评估框架

### 7.2 BEAM 基准

| 场景 | cognee | 此前 SOTA | RAG 基线 |
|------|--------|-----------|---------|
| 100K tokens | **0.79** (>0.8 带查询路由) | 0.735 | ~0.33 |
| 10M tokens | **0.67** | 0.641 | ~0.33 |

BEAM 测试的是系统能否跟踪长对话中的变化——比"大海捞针"更有实际意义。

> 🧠 *我的判断：这组数字不要当一个绝对分数看。BEAM 的设计目标与真实 Agent 记忆场景高度吻合——跟踪信息的变化比找到"针"更难也更有价值。*

---

## 8. 与竞品的差异

| 维度 | Cognee | Mem0 | Letta (MemGPT) | Zep |
|------|--------|------|---------------|-----|
| **核心模型** | 知识图谱 + 向量 | 向量记忆 | 操作系统级记忆管理 | 对话记忆图 |
| **自我优化** | ✅ improve 反馈循环 | ❌ 主要为追加 | 部分（OS 模型） | ❌ |
| **数据迁移** | COGX 开放格式 + 竞品导入 | 有限 | 有限 | 有限 |
| **部署复杂度** | 单 Postgres 搞定 | 中等 | 中等 | 中等 |
| **多租户** | ✅ 完善 | 有限 | 有限 | ✅ |
| **语言 SDK** | Python + Rust + TS | Python | Python | Python + TS |
| **Agent 集成** | MCP + Claude/OpenClaw/Cursor 等 | 有限 | 有限 | 有限 |

---

## 9. 实战用例

### 场景一：客服 Agent
- 统一客户在各渠道的交互数据（财务、支持、产品）
- 重建交互时间线，匹配到类似已解决案例
- 执行后更新记忆，Agent 不自重复

### 场景二：专家知识蒸馏（SQL Copilot）
- 提取专家 SQL 查询和工作流模式
- 匹配当前 schema 到已知结构
- 将专家推理适配到新人场景
- 更新成功模式 → 初级分析师接近专家水平

### 场景三：企业内部知识库（Bayer 案例）
- Bayer 用 cognee 支撑科学研究工作流
- 帮助研究员处理连接的知识、从复杂信息中生成假设
- 月均 600 万条记忆创建，100+ 公司生产使用

---

## 10. 我的判断与建议

### Cognee 适合的场景
- ✅ 需要跨会话持久记忆的 Agent 系统
- ✅ 企业内部多源数据统一为知识库
- ✅ 需要记忆自我优化（反馈驱动）的场景
- ✅ 已有 Postgres 基础设施、不想引入图数据库的团队
- ✅ 多 Agent 共享同一记忆层
- ✅ 需要数据可迁移（不被锁定）的场景

### Cognee 不太适合的场景
- ❌ 纯实时、低延迟的检索场景（图构建需要 LLM 调用，耗时较长）
- ❌ 数据量极小、不需要结构化关系的场景
- ❌ 对非 OpenAI 生态支持不如 OpenAI 完善

### 值得关注的点
1. **v1.0 还很年轻**：API 可能变化，生产落地有风险
2. **LLM 成本**：每次 remember 都需要 LLM 调用来提取实体/关系
3. **中文支持**：目前的文档和社区以英文为主，中文场景效果未知
4. **记忆质量**：依赖 LLM 提取实体和关系的能力，LLM 弱则图谱质量差

---

## 参考资源

| 资源 | URL |
|------|-----|
| GitHub | https://github.com/topoteretes/cognee |
| 官方文档 | https://docs.cognee.ai |
| 研究论文 | https://arxiv.org/abs/2505.24478 |
| Cognee Cloud | https://www.cognee.ai |
| BEAM Benchmark | https://agentmemorybenchmark.ai |
| v1.0 发布公告 | https://www.cognee.ai/blog/cognee-news/cognee-1-0-announcement |
| 内部架构详解 | https://www.cognee.ai/blog/deep-dives/inside-cognee-1-0 |
| Just Postgres 设计 | https://www.cognee.ai/blog/deep-dives/just-postgres |
