#topic/rag #topic/embedding #topic/vector-database #year/2026 #status/draft

# RAG 核心概念

> 检索增强生成（Retrieval-Augmented Generation）的基础原理与实现。

---

## 1. 为什么需要 RAG？

### 1.1 LLM 的局限性

| 问题 | 说明 | 影响 |
|------|------|------|
| **知识截止** | 训练数据有截止日期 | 无法回答最新事件 |
| **幻觉** | 生成看似合理但错误的内容 | 误导用户 |
| **领域知识缺失** | 对专业领域了解不足 | 回答不准确 |
| **无法引用来源** | 不知道信息来自哪里 | 不可验证 |

### 1.2 RAG 如何解决

```
用户提问
    │
    ▼
┌─────────────┐
│  检索模块   │ ──→ 从知识库中找到相关文档
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  生成模块   │ ──→ 基于检索结果生成回答
└──────┬──────┘
       │
       ▼
   带引用的回答
```

**核心价值**：
- ✅ 提供最新知识
- ✅ 减少幻觉
- ✅ 可溯源验证
- ✅ 保护数据隐私（无需训练）

---

## 2. RAG 核心组件

### 2.1 文档处理（Ingestion）

#### 文档加载

| 来源 | 工具 | 注意事项 |
|------|------|----------|
| 网页 | BeautifulSoup, Firecrawl | 处理动态内容、反爬 |
| PDF | PyPDF, pdfplumber | 表格、图片提取 |
| Word | python-docx | 样式保留 |
| 数据库 | SQLAlchemy | 增量同步 |
| API | Requests | 速率限制 |

#### 文本分块（Chunking）

**分块策略对比**：

| 策略 | 方法 | 优点 | 缺点 |
|------|------|------|------|
| **固定长度** | 每 N 个 token/字符 | 简单 | 可能切断语义 |
| **递归分割** | 先按段落，再按句子 | 保持结构 | 块大小不均 |
| **语义分块** | 按语义边界分割 | 语义完整 | 计算成本高 |
| **Agentic 分块** | LLM 判断分割点 | 最智能 | 成本高 |

**分块大小选择**：

```
小 chunk（100-200 tokens）：
- 优点：检索精度高
- 缺点：可能丢失上下文
- 适用：事实查询

大 chunk（500-1000 tokens）：
- 优点：上下文完整
- 缺点：噪声多
- 适用：总结、分析

混合策略：
- 小 chunk 用于检索
- 检索后扩展上下文（前后各取一段）
```

#### 元数据提取

```python
chunk_metadata = {
    "source": "document.pdf",      # 来源文档
    "page": 42,                    # 页码
    "section": "3.2",              # 章节
    "title": "RAG Architecture",   # 标题
    "timestamp": "2024-01-15",     # 创建时间
    "doc_type": "technical",       # 文档类型
    "author": "John Doe",          # 作者
}
```

---

### 2.2 Embedding 模型

#### 什么是 Embedding？

将文本转换为高维向量（通常是 384-4096 维），使得语义相似的文本在向量空间中距离相近。

```
"猫" → [0.1, -0.3, 0.8, ..., 0.2]  (768 维)
" kitten" → [0.1, -0.3, 0.8, ..., 0.2]  (相似向量)
"汽车" → [-0.5, 0.2, -0.1, ..., 0.7]  (距离较远)
```

#### 主流 Embedding 模型

| 模型 | 维度 | 语言 | 特点 | 适用场景 |
|------|------|------|------|----------|
| **text-embedding-3-large** | 3072 | 多语言 | OpenAI 最新，性能强 | 通用 |
| **text-embedding-3-small** | 1536 | 多语言 | 性价比高 | 成本敏感 |
| **BGE-M3** | 1024 | 多语言 | 开源 SOTA | 私有化部署 |
| **E5-Mistral** | 4096 | 多语言 | 高质量 | 高精度需求 |
| **GTE-Qwen2** | 3584 | 中英 | 中文优化 | 中文场景 |
| **Jina-Embeddings** | 768 | 多语言 | 轻量快速 | 实时应用 |

#### Embedding 选择指南

```
预算充足 + 通用场景 → OpenAI text-embedding-3-large
预算敏感 + 通用场景 → OpenAI text-embedding-3-small
私有化部署 + 中文 → BGE-M3 / GTE-Qwen2
私有化部署 + 英文 → E5-Mistral / BGE-large
实时应用 + 低延迟 → Jina-Embeddings
```

---

### 2.3 向量数据库

#### 核心概念

**向量相似度计算**：

| 度量 | 公式 | 适用场景 |
|------|------|----------|
| **余弦相似度** | cos(θ) = A·B / (\|A\| \|B\|) | 方向相似性 |
| **欧氏距离** | d = √Σ(Ai - Bi)² | 绝对距离 |
| **点积** | A·B | 考虑向量长度 |

**近似最近邻（ANN）算法**：

| 算法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **HNSW** | 分层导航小世界图 | 搜索快、精度高 | 内存占用大 |
| **IVF** | 倒排文件索引 | 内存友好 | 精度略低 |
| **Flat** | 暴力搜索 | 100% 精度 | 慢 |

#### 主流向量数据库对比

| 数据库 | 类型 | 特点 | 适用场景 |
|--------|------|------|----------|
| **Pinecone** | 托管 | 全托管、易用 | 快速启动 |
| **Weaviate** | 开源/托管 | GraphQL 接口、模块化 | 复杂查询 |
| **Qdrant** | 开源/托管 | Rust 实现、高性能 | 高性能需求 |
| **Milvus/Zilliz** | 开源/托管 | 分布式、企业级 | 大规模数据 |
| **Chroma** | 开源 | 轻量、易集成 | 原型开发 |
| **pgvector** | PostgreSQL 扩展 | SQL 接口、事务支持 | 已有 PG 基础设施 |
| **Redis** | 内存数据库 | 低延迟 | 实时应用 |

#### 选型建议

```
快速原型 → Chroma
生产环境 + 小规模 → Qdrant / Weaviate
生产环境 + 大规模 → Milvus / Pinecone
已有 PostgreSQL → pgvector
需要事务支持 → pgvector / Weaviate
超低延迟 → Redis
```

---

### 2.4 检索策略

#### 基础检索：向量相似度搜索

```python
# 伪代码
query_embedding = embed(query)
results = vector_db.search(
    query_embedding,
    top_k=5,
    filter={"doc_type": "technical"}  # 元数据过滤
)
```

#### 高级检索策略

**1. MMR (Maximal Marginal Relevance)**

平衡相关性和多样性：

```python
# 选择既相关又多样的结果
results = mmr_search(
    query_embedding,
    lambda_param=0.5,  # 平衡因子：0=最相关，1=最多样
    top_k=5
)
```

**2. 多查询检索**

生成多个查询变体，合并结果：

```python
queries = generate_variations(original_query, n=3)
# "RAG 是什么" → ["RAG 技术介绍", "检索增强生成原理", "RAG 应用场景"]

all_results = []
for q in queries:
    all_results.extend(search(q))

final_results = deduplicate_and_rerank(all_results)
```

**3. 混合检索（Hybrid Search）**

结合向量检索和关键词检索：

```python
vector_results = vector_search(query_embedding, top_k=10)
keyword_results = bm25_search(query_text, top_k=10)

# 融合排序
final_results = reciprocal_rank_fusion(
    [vector_results, keyword_results],
    weights=[0.7, 0.3]
)
```

---

### 2.5 生成增强

#### 基础 RAG Prompt

```
基于以下检索到的信息回答问题。如果信息不足以回答，请明确说明。

检索信息：
{retrieved_documents}

用户问题：{question}

请提供：
1. 直接回答
2. 关键引用（标注来源）
3. 置信度评估（高/中/低）
```

#### 高级生成策略

**1. 上下文压缩**

```python
# 原始上下文可能很长，需要压缩
compressed_context = llm.compress(
    documents=retrieved_docs,
    query=query,
    max_tokens=4000
)
```

**2. 多文档融合**

```
请综合以下多个来源的信息，给出一个连贯的回答。
注意处理不同来源之间的冲突信息。

来源 1：[文档 1]
来源 2：[文档 2]
来源 3：[文档 3]

用户问题：{question}

要求：
- 优先采用高可信度来源
- 冲突时说明不同观点
- 标注每个观点的来源
```

---

## 3. RAG 架构模式

### 3.1 基础 RAG

```
用户查询 → Embedding → 向量检索 → Top-K 文档 → LLM 生成 → 回答
```

### 3.2 Advanced RAG

```
用户查询 → Query 理解 → Query 重写 → 多路检索 → Reranking → 
上下文构建 → LLM 生成 → 后处理 → 回答
```

### 3.3 Modular RAG

```
用户查询 → Router → 
    ├─ 简单查询 → 直接回答
    ├─ 事实查询 → RAG Pipeline
    ├─ 分析查询 → Multi-hop RAG
    └─ 总结查询 → 全文档扫描
```

---

## 4. 评估指标

### 4.1 检索质量

| 指标 | 说明 | 计算 |
|------|------|------|
| **Hit Rate** | 正确答案是否在 Top-K 中 | 是/否 |
| **MRR** | 平均倒数排名 | 1/rank |
| **NDCG** | 归一化折损累积增益 | 考虑排序质量 |
| **Precision@K** | Top-K 中相关文档比例 | TP / K |

### 4.2 生成质量

| 指标 | 说明 | 方法 |
|------|------|------|
| **Faithfulness** | 回答是否忠实于检索内容 | 人工/模型评估 |
| **Answer Relevance** | 回答是否相关 | 人工/模型评估 |
| **Context Precision** | 使用的上下文是否精准 | 人工评估 |
| **Context Recall** | 是否使用了所有相关上下文 | 人工评估 |

### 4.3 端到端评估

**RAGAS 框架**：

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
```

---

## 💡 我的思考

1. **RAG 不是银弹**：对于需要深度推理的任务，RAG 只能提供事实支撑，推理能力仍取决于 LLM。

2. **检索质量是瓶颈**：再强的 LLM 也救不了糟糕的检索。投资在 Embedding 模型和分块策略上的时间最值得。

3. **评估驱动优化**：没有评估就无法改进。建议尽早建立评估数据集和自动化评估流程。

4. **生产环境要考虑成本**：向量数据库、Embedding API、LLM API 都有成本，需要设计高效的缓存和降级策略。

5. **多模态是趋势**：文本 RAG 已经成熟，图像、视频、音频的 RAG 正在快速发展。

---

## 参考来源

- **RAG Survey**: "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al., 2023) — [arxiv:2312.10997](https://arxiv.org/abs/2312.10997)
- **LangChain RAG**: [python.langchain.com/docs/tutorials/rag](https://python.langchain.com/docs/tutorials/rag/) — 访问日期：2026-06-07
- **LlamaIndex**: [docs.llamaindex.ai](https://docs.llamaindex.ai/) — 访问日期：2026-06-07
- **RAGAS**: [docs.ragas.io](https://docs.ragas.io/) — 访问日期：2026-06-07
- **Hugging Face Embedding Guide**: [huggingface.co/blog/getting-started-with-embeddings](https://huggingface.co/blog/getting-started-with-embeddings) — 访问日期：2026-06-07

---

*访问日期: 2026-06-07*
