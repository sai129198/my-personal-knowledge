#topic/rag #topic/advanced-techniques #topic/hybrid-search #year/2026 #status/draft

# RAG 高级技术

> 超越基础向量检索：Hybrid Search、Reranking、Query 重写、多跳检索等进阶技术。

---

## 1. Hybrid Search（混合检索）

### 1.1 为什么需要混合检索？

**向量检索的局限**：
- 对精确匹配（如产品 ID、人名）不敏感
- 对罕见词、专业术语效果差
- 无法利用倒排索引的布尔过滤

**关键词检索的局限**：
- 无法理解语义相似性
- 对同义词、近义词不敏感
- 受分词质量影响大

**混合检索 = 向量检索的语义理解 + 关键词检索的精确匹配**

### 1.2 实现方式

#### Reciprocal Rank Fusion (RRF)

```python
def reciprocal_rank_fusion(results_lists, k=60):
    """
    融合多个检索结果列表
    k: 常数，通常取 60
    """
    scores = {}
    
    for results in results_lists:
        for rank, doc_id in enumerate(results):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1 / (k + rank + 1)
    
    # 按分数排序
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# 使用示例
vector_results = ["doc_1", "doc_3", "doc_5"]
keyword_results = ["doc_2", "doc_1", "doc_4"]

fused = reciprocal_rank_fusion([vector_results, keyword_results])
# 结果: [("doc_1", 0.032), ("doc_3", 0.016), ("doc_2", 0.016), ...]
```

#### 加权分数融合

```python
def weighted_fusion(vector_results, keyword_results, alpha=0.7):
    """
    alpha: 向量检索权重 (0-1)
    1-alpha: 关键词检索权重
    """
    scores = {}
    
    # 归一化向量检索分数
    max_v_score = max(r.score for r in vector_results)
    for r in vector_results:
        scores[r.doc_id] = alpha * (r.score / max_v_score)
    
    # 归一化关键词检索分数
    max_k_score = max(r.score for r in keyword_results)
    for r in keyword_results:
        if r.doc_id in scores:
            scores[r.doc_id] += (1 - alpha) * (r.score / max_k_score)
        else:
            scores[r.doc_id] = (1 - alpha) * (r.score / max_k_score)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 1.3 数据库支持

| 数据库 | Hybrid Search 支持 | 特点 |
|--------|-------------------|------|
| **Weaviate** | 原生支持 | 自动融合向量+BM25 |
| **Elasticsearch** | 原生支持 | 强大的文本搜索 + 向量插件 |
| **Qdrant** | 支持 | 可配置融合策略 |
| **Pinecone** | 部分支持 | 需手动实现关键词部分 |
| **pgvector** | 需手动 | 结合 PostgreSQL 全文搜索 |

---

## 2. Reranking（重排序）

### 2.1 为什么需要 Reranking？

**第一阶段检索的问题**：
- 向量检索速度快但精度有限
- Top-K 中可能包含噪声
- 粗排后的精排可以提升质量

**Reranking 的作用**：
- 用更精确的模型对候选文档重新打分
- 考虑查询和文档的细粒度交互
- 通常比直接全库检索更高效

### 2.2 Reranker 模型

| 模型 | 类型 | 特点 | 延迟 |
|------|------|------|------|
| **Cohere Rerank** | API | 效果好，易用 | ~100ms |
| **BGE-Reranker** | 开源 | 可私有化部署 | ~50ms |
| **Jina Reranker** | 开源 | 轻量快速 | ~30ms |
| **Cross-Encoder** | 通用 | 高精度，高延迟 | ~200ms |

### 2.3 实现架构

```
用户查询
    │
    ▼
┌─────────────┐
│  第一阶段   │  ← 向量检索 / BM25，召回 Top-100
│  (召回)     │     追求高召回率，允许低精度
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  第二阶段   │  ← Reranker，精排 Top-100 → Top-10
│  (精排)     │     追求高精度，可接受高延迟
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  第三阶段   │  ← LLM 生成最终回答
│  (生成)     │
└─────────────┘
```

### 2.4 代码示例

```python
from sentence_transformers import CrossEncoder

# 加载 reranker
reranker = CrossEncoder('BAAI/bge-reranker-large')

def retrieve_and_rerank(query, top_k=10):
    # 第一阶段：召回
    candidates = vector_db.search(query, top_k=100)
    
    # 第二阶段：精排
    pairs = [[query, doc.text] for doc in candidates]
    scores = reranker.predict(pairs)
    
    # 按分数排序
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    return ranked[:top_k]
```

---

## 3. Query 重写与扩展

### 3.1 Query 重写

**问题**：用户查询可能模糊、不完整或包含噪声。

**重写策略**：

| 策略 | 说明 | 示例 |
|------|------|------|
| **扩展** | 添加同义词、相关词 | "AI" → "人工智能、机器学习、深度学习" |
| **消歧** | 明确多义词含义 | "苹果" → "苹果公司" / "苹果水果" |
| **纠错** | 修正拼写错误 | "RAG arhitecture" → "RAG architecture" |
| **分解** | 复杂查询拆分为子查询 | "比较 RAG 和 Fine-tuning" → ["RAG 优缺点", "Fine-tuning 优缺点"] |
| **结构化** | 提取实体和关系 | "OpenAI 的 CEO 是谁" → {"entity": "OpenAI", "relation": "CEO"} |

### 3.2 HyDE (Hypothetical Document Embeddings)

**核心思想**：让 LLM 先生成一个假设的理想回答，然后用这个回答去检索。

```python
def hyde_retrieve(query, top_k=5):
    # Step 1: 生成假设文档
    hypothetical_doc = llm.generate(
        f"请回答以下问题，提供详细的技术解释：\n{query}"
    )
    
    # Step 2: 用假设文档做向量检索
    query_embedding = embed(hypothetical_doc)
    results = vector_db.search(query_embedding, top_k=top_k)
    
    return results
```

**优势**：
- 弥合查询和文档之间的词汇差异
- 生成更丰富的语义表示
- 对短查询特别有效

**劣势**：
- 增加一次 LLM 调用成本
- 假设文档可能偏离主题

### 3.3 Step-Back Prompting

**核心思想**：先问一个更通用的"退一步"问题，再基于结果回答具体问题。

```
用户问题："Transformer 中的注意力机制时间复杂度是多少？"

Step-Back 问题："Transformer 架构的核心组件和计算特性是什么？"

检索 Step-Back 问题的相关信息 → 获得 Transformer 整体背景
再检索具体问题 → 获得精确答案
```

---

## 4. 多跳检索（Multi-hop Retrieval）

### 4.1 什么是多跳检索？

**单跳检索**：一次查询，直接找到答案。
**多跳检索**：需要多次查询，逐步收集信息，最终综合得出答案。

**示例**：
```
问题："发明 Transformer 架构的研究者中，谁后来加入了 Google DeepMind？"

跳 1: 检索 "Transformer 架构发明者" → ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", ...]
跳 2: 检索 "Noam Shazeer Google DeepMind" → ["Noam Shazeer 离开 Google 后创立了 Character.AI"]
跳 3: 检索 "Niki Parmar Google DeepMind" → ["Niki Parmar 后来加入了 Google DeepMind"]

答案：Niki Parmar
```

### 4.2 实现方式

#### Iterative Retrieval（迭代检索）

```python
def iterative_retrieve(query, max_hops=3):
    context = []
    current_query = query
    
    for hop in range(max_hops):
        # 检索
        results = retrieve(current_query)
        context.extend(results)
        
        # 判断是否已有足够信息
        if has_sufficient_info(context, query):
            break
        
        # 生成下一跳查询
        current_query = generate_next_query(context, query)
    
    return context
```

#### Tree-based Retrieval

```
初始查询
    │
    ├── 子查询 1 → 检索结果 1
    │       └── 子查询 1.1 → 检索结果 1.1
    │
    ├── 子查询 2 → 检索结果 2
    │       └── 子查询 2.1 → 检索结果 2.1
    │
    └── 子查询 3 → 检索结果 3
```

### 4.3 框架支持

- **LangChain**: `MultiQueryRetriever`, `ParentDocumentRetriever`
- **LlamaIndex**: `MultiStepQueryEngine`, `RouterQueryEngine`
- **DSPy**: 模块化多跳检索 pipeline

---

## 5. 上下文压缩与选择

### 5.1 问题

检索到的文档可能很长，超出 LLM 上下文限制，或包含噪声。

### 5.2 压缩策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| **Map-Reduce** | 分块处理，再综合 | 长文档总结 |
| **Refine** | 逐步精炼答案 | 迭代改进 |
| **Stuff** | 直接填入所有上下文 | 上下文较短 |
| **Contextual Compression** | 提取与查询相关的片段 | 噪声较多 |

### 5.3 Contextual Compression 实现

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 检索时自动压缩
compressed_docs = compression_retriever.get_relevant_documents(query)
```

---

## 6. 自查询（Self-Query）

### 6.1 概念

让 LLM 将自然语言查询转换为结构化的过滤条件。

```
用户查询："2024 年发布的关于 RAG 的中文论文"

LLM 解析：
{
  "query": "RAG",
  "filter": {
    "year": {"$eq": 2024},
    "language": {"$eq": "zh"},
    "type": {"$eq": "paper"}
  }
}
```

### 6.2 实现

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="year",
        description="发布年份",
        type="integer",
    ),
    AttributeInfo(
        name="language",
        description="语言",
        type="string",
    ),
    AttributeInfo(
        name="type",
        description="文档类型",
        type="string",
    ),
]

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_contents="技术文档",
    metadata_field_info=metadata_field_info,
)
```

---

## 💡 我的思考

1. **技术组合 > 单一技术**：实际系统中，Hybrid Search + Reranking + Query 重写往往同时使用，效果最佳。

2. **延迟与精度的权衡**：每增加一个环节（Reranking、多跳检索），都会增加延迟。需要根据场景做取舍。

3. **评估是核心**：高级技术增加了系统复杂度，必须通过端到端评估验证是否真的有提升。

4. **不要过度工程**：对于简单场景，基础 RAG 可能已经足够。从简单开始，根据评估结果逐步添加复杂度。

5. **多跳检索的挑战**：如何生成有效的中间查询、何时停止检索、如何处理检索失败，都是需要仔细设计的环节。

---

## 参考来源

- **Hybrid Search**: [Weaviate Hybrid Search](https://weaviate.io/developers/weaviate/search/hybrid) — 访问日期：2026-06-07
- **Reranking**: [Cohere Rerank](https://docs.cohere.com/docs/reranking) — 访问日期：2026-06-07
- **HyDE**: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022) — [arxiv:2212.10496](https://arxiv.org/abs/2212.10496)
- **Multi-hop**: "Multi-Hop Reasoning with LLMs" (Yao et al., 2023) — [arxiv:2305.18323](https://arxiv.org/abs/2305.18323)
- **Self-Query**: [LangChain Self-Query](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/) — 访问日期：2026-06-07

---

*访问日期: 2026-06-07*
