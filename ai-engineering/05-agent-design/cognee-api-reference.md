# Cognee 接口 API 参考

> **一句话定位**：cognee 为 Agent 提供 Python SDK（4 核心动词 + 7 辅助接口）、MCP 14 工具、REST API 三层接口体系。
>
> #status/canonical #topic/memory #tool/cognee #year/2026

---

## 0. 接口体系总览

```
┌─────────────────────────────────────────────┐
│              你的 Agent 代码                 │
├───────────────┬───────────────┬─────────────┤
│  Python SDK   │   MCP 协议    │  REST API   │
│  (import      │   (14 tools)  │  (HTTP)     │
│   cognee)     │               │             │
├───────────────┴───────────────┴─────────────┤
│           Cognee Memory Layer               │
└─────────────────────────────────────────────┘
```

---

## 一、Python SDK：四大核心动词

### 1.1 `remember()` — 存入记忆

```python
async def remember(
    data: str | list[str] | DataItem | list[DataItem] | BinaryIO,
    dataset_name: str = "main_dataset",
    *,
    session_id: str | None = None,       # None=永久图谱, 传了=会话缓存
    chunk_size: int | None = None,       # 分块大小
    chunker: Any = None,                 # 自定义分块
    custom_prompt: str | None = None,    # 自定义提取提示词
    run_in_background: bool = False,     # 异步后台执行
    self_improvement: bool = True,       # 自动触发 improve()
    session_ids: list[str] | None = None,
    # --- keyword extras ---
    graph_model: Any = None,             # 自定义图谱 schema
    node_set: list[str] = None,          # 节点集标签
    dataset_id: UUID = None,
    importance_weight: float = None,     # 检索排序权重
    incremental_loading: bool = None,
    temporal_cognify: bool = False,      # 时间感知
    user: User = None,                   # 多租户
    llm_config: LLMConfig = None,
    embedding_config: EmbeddingConfig = None,
) -> RememberResult
```

**支持的输入格式**：

| 输入类型 | 示例 | 说明 |
|---------|------|------|
| 纯文本 | `"Einstein was born in Ulm."` | 直接作为记忆数据 |
| 文件路径 | `"/path/to/doc.pdf"` | 支持 .txt .md .pdf .csv .json .png .mp3 等 |
| HTTP URL | `"https://example.com/doc.pdf"` | 自动 fetch 后解析（仅 permanent 模式） |
| S3 路径 | `"s3://bucket/key"` | 需 `s3fs` |
| DataItem | `DataItem(data=..., label=...)` | 带元数据标签 |
| HTTP Upload | FastAPI `UploadFile` | REST API 场景 |

**返回结构 `RememberResult`**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | str | `"running"` / `"completed"` |
| `dataset_name` | str | 写入的数据集 |
| `session_ids` | list[str] | 关联会话 ID（仅 session 模式） |
| `elapsed_seconds` | float | 处理耗时 |
| `raw_result` | Any | 底层 pipeline 原始结果 |

**底层处理流程（permanent 模式）**：
1. 数据归一化 → 挂载到数据集
2. 文档分块（chunking）
3. **实体提取** + **关系建模** → 图谱节点和边
4. 嵌入生成 → 向量索引
5. 默认触发 improve 增强 pass

**底层处理流程（session 模式）**：
1. 写入会话缓存 → 立即返回
2. 如 `self_improvement=True`（默认），后台异步桥接到永久图谱

**代码示例**：

```python
import cognee

# 永久图谱记忆
await cognee.remember(
    "Alice is a software engineer based in Paris. She uses Python daily.",
    dataset_name="team_knowledge",
)

# 会话级快速缓存
await cognee.remember(
    "User prefers dark mode and compact layout.",
    session_id="chat_ui_prefs",
)

# 带元数据
from cognee.tasks.ingestion.data_item import DataItem
await cognee.remember(
    DataItem(data="...", label="onboarding_note", external_metadata={"source": "slack"}),
)

# 后台异步——不阻塞 Agent 主流程
result = await cognee.remember(large_document, run_in_background=True)
print(result.status)  # → "running"
await result            # 等待完成
print(result.status)  # → "completed"
```

---

### 1.2 `recall()` — 检索记忆

```python
async def recall(
    query_text: str,
    query_type: SearchType | None = None,  # None=自动路由
    *,
    datasets: list[str] | None = None,
    dataset_ids: list[UUID] | None = None,
    top_k: int = 15,
    auto_route: bool = True,
    scope: str | list[str] | None = None,  # "session" / "graph"
    # --- keyword extras ---
    system_prompt: str = None,
    system_prompt_path: str = None,
    node_name: list[str] = None,
    node_name_filter_operator: str = "OR",
    only_context: bool = False,
    session_id: str = None,
    wide_search_top_k: int = 100,
    triplet_distance_penalty: float = 6.5,
    feedback_influence: float = None,
    verbose: bool = False,
    retriever_specific_config: dict = None,
    include_references: bool = False,
    user: User = None,
    llm_config: LLMConfig = None,
    embedding_config: EmbeddingConfig = None,
) -> list[RecallResponse]
```

**返回结构 `RecallResponse`**：根据 `source` 字段判别具体类型（Pydantic 对象，用属性访问）：

| `source` 值 | 具体 Pydantic 类型 | 关键属性 |
|------------|-------------------|---------|
| `"graph"` | `ResponseGraphEntry` | `text`（答案/上下文）, `kind`, `search_type`, `score`, `dataset_id`, `dataset_name`, `metadata`（含 data_id / chunk_id / chunk_index / document_name）, `raw`（标准化载荷）, `structured` |
| `"session"` | `ResponseQAEntry` | `time`, `qa_id`, `question`, `context`, `answer`, `feedback_text`, `feedback_score` |
| `"trace"` | `ResponseAgentTraceEntry` | session 中的 agent trace 字段 |
| `"graph_context"` | `ResponseGraphContextEntry` | `content` |

**Source Provenance（来源追溯）**：

```python
# metadata 中携带稳定来源标识
{
    "data_id": "uuid-of-ingested-data",
    "chunk_id": "uuid-of-chunk",
    "chunk_index": 3,              # 0-based 在文档中的位置
    "document_name": "report.pdf"
}

# include_references=True 时答案末尾自动拼 Evidence:
# - chunk 3 of document report.pdf (data_id: …, chunk_id: …): "cited text..."
```

**会话感知检索逻辑**：
1. 传 `session_id` 不传 `datasets` → 先查会话缓存 → null → 回退图谱
2. 传 `session_id` + `datasets` → 强制图谱检索，跳过期缓存
3. 不传 `session_id` → 纯图谱检索

**代码示例**：

```python
# 最简单：自动路由
results = await cognee.recall("Where does Alice work?")
for r in results:
    if r.source == "graph":
        print(f"Answer: {r.text}, Score: {r.score}")
        
# 带引用的检索
results = await cognee.recall(
    "What is the deployment process?",
    include_references=True,  # 答案末尾自动附 Evidence
)

# only_context：只返回上下文，Agent 自己拼接 prompt
context_parts = await cognee.recall(
    "How to reset user password?",
    only_context=True,
    top_k=3,
)
# → ["chunk text 1", "chunk text 2", "chunk text 3"]

# 会话感知
results = await cognee.recall(
    "What did the user prefer?",
    session_id="chat_42",  # 先查会话缓存
)

# 按数据集限定
results = await cognee.recall(
    "security vulnerabilities",
    datasets=["code_review", "pentest_reports"],
)

# Node Set 过滤
results = await cognee.recall(
    "procurement rules",
    node_name=["policies", "compliance"],
)

# verbose 模式：获取更多细节
results = await cognee.recall(
    "project status",
    verbose=True,
)
# r.text_result, r.context_result, r.objects_result
```

---

### 16 种 SearchType（检索策略）

| 枚举值 | LLM 调用 | 速度 | 适用场景 |
|--------|---------|------|---------|
| `CHUNKS` | 0 | ⚡️最快 | 只要原始片段，不做生成 |
| `CHUNKS_LEXICAL` | 0 | ⚡️最快 | BM25 词法检索，精确关键词 |
| `SUMMARIES` | 0 | ⚡️最快 | 预计算摘要 |
| `CYPHER` | 0 | ⚡️最快 | 手写图查询 |
| `RAG_COMPLETION` | 1 | 🟢快 | 传统 RAG |
| `GRAPH_COMPLETION`（**默认**） | 1 | 🟢快 | 图谱+LLM，最佳平衡 |
| `HYBRID_COMPLETION` | 1 | 🟢快 | 词法+语义+图谱融合 |
| `TRIPLET_COMPLETION` | 1 | 🟢快 | 三元组级检索 |
| `NATURAL_LANGUAGE` | 1~3 | 🟡中 | NL→Cypher 翻译 |
| `TEMPORAL` | 2 | 🟡中 | 时间感知查询 |
| `GRAPH_SUMMARY_COMPLETION` | 2 | 🟡中 | 大图谱摘要压缩 |
| `GRAPH_COMPLETION_DECOMPOSITION` | 若干 | 🟠慢 | 多实体/多面向复合问题，先拆分成子问题再分别检索 |
| `GRAPH_COMPLETION_CONTEXT_EXTENSION` | ≤4轮 | 🔴慢 | 开放式探索，逐轮扩展上下文 |
| `GRAPH_COMPLETION_COT` | ≤4轮×多调用 | 🔴最慢 | 多跳推理 A→B→C→D |
| `AGENTIC_COMPLETION` | 可变 | 🔴最慢 | 带 tool/skill 的 agentic 检索 |
| `FEELING_LUCKY` | 1+ | 可变 | 不确定用哪个，自动选 |

**选型建议**：日常用 `GRAPH_COMPLETION`（默认），复杂多跳问题用 `GRAPH_COMPLETION_COT`，速度优先用 `CHUNKS`，自动路由不传 `query_type` 即可。

---

### 1.3 `improve()` — 记忆自我优化

```python
async def improve(
    dataset: str | UUID = "main_dataset",
    *,
    run_in_background: bool = False,
    node_name: list[str] | None = None,
    session_ids: list[str] | None = None,       # 将指定会话的反馈/QA桥接到图谱
    build_global_context_index: bool = False,   # 构建数据集级摘要索引
    build_truth_subspace: bool = False,         # 构建真实性子空间（需session_ids）
    # --- keyword extras ---
    extraction_tasks: list = None,
    enrichment_tasks: list = None,
    feedback_alpha: float = None,               # 反馈对图谱权重的控制力
    user: User = None,
) -> PipelineResult
```

**三个优化信号**：

| 信号 | 来源 | 机制 |
|------|------|------|
| **反馈** | 答案被确认/纠正/否决 | 节点和边的权重上调或下调 |
| **重要性** | 摄入时 `importance_weight` 参数 | 操作手册 > 闲聊记录 |
| **频率** | 被检索和使用的次数 | 高频路径自动强化 |

**代码示例**：

```python
# 基本用法：运行图谱增强
await cognee.improve(dataset="team_knowledge")

# 桥接会话反馈：用户纠正了一个错误后
await cognee.remember("Actually Bob is in Berlin, not Paris.", session_id="chat_42")
await cognee.improve(
    dataset="team_knowledge",
    session_ids=["chat_42"],
    run_in_background=True,
)
# → 反馈信号融入图谱，下次检索不会再重复错误

# 构建全局上下文索引（加速大规模检索）
await cognee.improve(
    dataset="large_docs",
    build_global_context_index=True,
)

# 构建真实性子空间（需要先有 session_learnings）
await cognee.improve(
    dataset="team_knowledge",
    session_ids=["chat_42"],
    build_truth_subspace=True,
)
```

---

### 1.4 `forget()` — 删除记忆

```python
async def forget(
    *,
    data_id: UUID | None = None,         # 删除单条数据（需同时传 dataset）
    dataset: str | UUID | None = None,   # 删除整个数据集
    everything: bool = False,            # 删除当前用户所有记忆
    memory_only: bool = False,           # 只删图+向量，保留原始数据可重新cognify
    user: User = None,
) -> dict
```

**返回**：`{"status": "deleted", "dataset_id": ..., "data_id": ...}`

**代码示例**：

```python
# 按数据集删除
await cognee.forget(dataset="old_project")

# 保留原始数据，只清图+向量（可重新cognify）
await cognee.forget(dataset="test", memory_only=True)

# 用户级别的全量清空
await cognee.forget(everything=True)
```

---

## 二、Python SDK：辅助接口

### 2.1 `serve()` — 连接远程实例

```python
await cognee.serve(
    url="https://my-instance.cognee.ai",
    api_key="ck_...",
)
# 之后所有 remember/recall/improve/forget 自动路由到云端
await cognee.disconnect()
```

### 2.2 `push()` — 本地图谱上传到 Cloud

```python
await cognee.push(
    dataset="my_local_graph",
    url="https://my-instance.cognee.ai",
    api_key="ck_...",
)
# 导出为 COGX 格式 → 上传到 Cloud，无需重新提取
```

### 2.3 `export()` — 导出图谱数据

```python
# 导出为开放格式
await cognee.export("my_dataset", format="cogx")    # COGX 交换格式
await cognee.export("my_dataset", format="graphml") # GraphML（可导入 Gephi/Cytoscape）
await cognee.export("my_dataset", format="cypher")  # Cypher 语句
await cognee.export("my_dataset", format="json")    # JSON
```

### 2.4 `datasets` — 数据集管理

```python
from cognee import datasets
await datasets.list()           # 列出所有数据集
await datasets.create("my_db")  # 创建
await datasets.delete("my_db")  # 删除
```

### 2.5 `users` — 用户管理

```python
from cognee.modules.users.methods import get_user, create_user, user_exists
user = await get_user(user_id)        # 获取/创建默认用户
exists = await user_exists(email="a@b.com")
new_user = await create_user(name="Bob", email="bob@corp.com")
```

### 2.6 `get_schema_inventory()` — 图谱结构概览

```python
from cognee import get_schema_inventory
inventory = await get_schema_inventory()
# → {"Person": 42, "City": 18, "Organization": 7, ...}
# 带每类型样本名称和关系统计
```

### 2.7 `visualize_memory_provenance()` — 生成可视化 HTML

```python
from cognee import visualize_memory_provenance
await visualize_memory_provenance()
# → 生成交互式 HTML：展示 用户→Agent→数据集→文档 的归属关系图
```

### 2.8 遗留接口（底层控制）

| 接口 | 用途 |
|------|------|
| `add()` | 仅摄入数据，不做图谱构建 |
| `cognify()` | 对已摄入数据运行图谱构建 |
| `search()` | 底层检索（比 recall 更细粒度控制） |
| `memify()` | 对已有图谱运行增强 pass |
| `run_custom_pipeline()` | 执行自定义 task pipeline |

---

## 三、Agent Memory Decorator（一键集成）

```python
from cognee import agent_memory

@agent_memory()
async def my_agent(prompt: str) -> str:
    # 调用前：cognee 自动检索相关记忆，注入到 prompt
    # 调用后：cognee 自动持久化 trace
    return await run_llm(prompt)
```

---

## 四、MCP 协议接口（14 个工具）

### 4.1 v1.0 工具（推荐）

| 工具名 | 参数 | 说明 |
|--------|------|------|
| `remember` | `data`(必填), `dataset_name`, `session_id`, `custom_prompt` | 存文本 |
| `recall` | `query`(必填), `search_type`, `datasets`, `session_id`, `top_k`(1-100) | 检索（datasets 为逗号分隔字符串） |
| `forget` | `dataset` / `everything` | 删除数据集或全量 |
| `improve` | `dataset_name`, `session_ids`(逗号分隔) | 图谱增强 + 会话桥接 |

### 4.2 辅助工具

| 工具 | 参数 | 说明 |
|------|------|------|
| `get_document` | `document_id`, `include_metadata`, `max_chunks` | 拉完整文档 |
| `get_chunk_neighbors` | `chunk_id`, `neighbor_count`(默认2), `direction` | 获取相邻 chunk |
| `save_interaction` | `data` | 存储交互记录 |
| `list_data` | `dataset_id`(可选) | 列出数据集及数据项 |
| `delete` | `data_id`, `dataset_id`, `mode`(soft/hard) | 删除单条 |
| `delete_dataset` | `dataset_name` | 删除数据集 |
| `cognify_status` | `dataset_name`, `pipelines` | 查 pipeline 状态 |
| `cognify` | `data`, `dataset_name`, `graph_model_file`, `custom_prompt` | 遗留：构建图谱 |
| `search` | `search_query`, `search_type`, `top_k`, `datasets` | 遗留：底层检索 |
| `prune` | 无 | 重置本地 store |

### 4.3 MCP Client 配置示例

```json
{
  "mcpServers": {
    "cognee": {
      "command": "docker",
      "args": ["run", "-i", "--rm",
               "--env-file", ".env",
               "cognee/cognee-mcp:main"]
    }
  }
}
```

---

## 五、Agent 集成实战模式

### 模式 A：Decorator（最简单）

```python
@agent_memory()
async def handler(prompt): ...
```

### 模式 B：手动控制（灵活）

```python
async def handle(user_input: str, session_id: str):
    # 1. 查记忆
    memories = await cognee.recall(user_input, session_id=session_id)
    # 2. 拼 prompt
    ctx = "\n".join(r.text for r in memories if r.source == "graph")
    augmented = f"Context:\n{ctx}\n\nQ: {user_input}"
    # 3. 调 LLM
    answer = await llm.generate(augmented)
    # 4. 存记忆
    await cognee.remember(f"Q: {user_input}\nA: {answer}", session_id=session_id)
    return answer
```

### 模式 C：带反馈闭环

```python
# 用户纠正后
async def on_correction(session_id: str, correction: str):
    await cognee.remember(correction, session_id=session_id)
    await cognee.improve(session_ids=[session_id], run_in_background=True)
```

### 模式 D：MCP 跨 Agent 共享

任何 MCP 兼容的 Agent（Claude、Cursor、OpenClaw 等）共用同一个记忆层，配置好 MCP Server 即可。

---

## 相关文档

- [Cognee 深度解析](./cognee-deep-dive.md)
- [Cognee 图谱结构与前端渲染](./cognee-graph-visualization.md)
- [Agent 记忆系统设计](./agent-memory.md)
- [官方 Python API](https://docs.cognee.ai/python-api.md)
- [官方 MCP 工具参考](https://docs.cognee.ai/cognee-mcp/mcp-tools)
