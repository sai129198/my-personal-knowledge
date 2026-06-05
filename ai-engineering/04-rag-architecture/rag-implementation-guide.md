#topic/rag #product/langchain #product/llamaindex #year/2026 #status/draft

# RAG 系统实现指南

> 从文档加载到生产部署的完整 RAG 实现流程，基于 Huyen Chip 的生成式 AI 平台架构分析。

---

## 1. RAG 核心概念

### 1.1 什么是 RAG

RAG（Retrieval-Augmented Generation，检索增强生成）是一种将外部知识检索与 LLM 生成相结合的架构。核心思想：**给模型提供相关上下文，减少幻觉，提升回答准确性**。

### 1.2 为什么需要 RAG

| 问题 | 传统 LLM | RAG 增强 |
|------|----------|----------|
| 知识时效性 | 训练数据有截止日期 | 可实时更新知识库 |
| 幻觉 | 容易编造信息 | 基于检索到的真实内容 |
| 领域知识 | 通用知识，不深入 | 可注入专业领域文档 |
| 可解释性 | 黑盒 | 可追溯引用来源 |

---

## 2. RAG 架构组件

### 2.1 整体流程

```
用户查询 → [查询理解] → [检索] → [重排序] → [上下文构建] → [LLM 生成] → [输出]
                ↑           ↑          ↑              ↑
           Query改写    向量/关键词   相关性排序    Prompt 组装
```

### 2.2 核心组件详解

#### 2.2.1 文档加载与处理

```python
# 使用 LangChain 加载多种格式文档
from langchain.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
)

# PDF
pdf_loader = PyPDFLoader("document.pdf")
pdf_docs = pdf_loader.load()

# Markdown
md_loader = UnstructuredMarkdownLoader("notes.md")
md_docs = md_loader.load()
```

#### 2.2.2 文档分块（Chunking）

分块策略直接影响检索质量：

| 策略 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| **固定大小** | 通用场景 | 简单、均匀 | 可能切断语义 |
| **递归字符** | 文本文档 | 保持段落完整 | 块大小不均 |
| **语义分块** | 高质量需求 | 语义完整 | 计算成本高 |
| **Agentic 分块** | 复杂文档 | 智能判断边界 | 需要 LLM 参与 |

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 每块大小
    chunk_overlap=200,    # 重叠部分，保持上下文连贯
    separators=["\n\n", "\n", "。", " "]
)
chunks = splitter.split_documents(docs)
```

**最佳实践**：
- 块大小：通常 500-1000 tokens
- 重叠：10-20% 的块大小
- 根据内容类型调整（代码块可以更大，FAQ 可以更小）

#### 2.2.3 嵌入（Embedding）

将文本转换为向量表示：

| 模型 | 维度 | 特点 | 适用场景 |
|------|------|------|----------|
| **text-embedding-3-small** | 1536 | 便宜、快速 | 通用场景 |
| **text-embedding-3-large** | 3072 | 高质量 | 精度要求高的场景 |
| **BGE-large-zh** | 1024 | 中文优化 | 中文内容 |
| **E5-mistral-7b** | 4096 | 开源、高质量 | 本地部署 |

```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("查询文本")
```

#### 2.2.4 向量存储

| 数据库 | 类型 | 特点 | 适用场景 |
|--------|------|------|----------|
| **Pinecone** | 托管 | 易用、自动扩展 | 快速启动、生产环境 |
| **Chroma** | 本地/托管 | 开源、轻量 | 开发、本地部署 |
| **Qdrant** | 自托管 | 高性能、Rust 实现 | 大规模生产 |
| **FAISS** | 库 | Meta 开源、本地 | 研究、原型 |
| **Weaviate** | 自托管 | 模块化、GraphQL | 复杂查询场景 |

```python
from langchain.vectorstores import Chroma

# 创建向量存储
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 相似性搜索
results = vectorstore.similarity_search("查询内容", k=5)
```

#### 2.2.5 检索策略

**基础检索**：
```python
# 相似性搜索
results = vectorstore.similarity_search(query, k=5)

# MMR（最大边际相关性）— 平衡相关性和多样性
results = vectorstore.max_marginal_relevance_search(query, k=5, fetch_k=20)
```

**高级检索**：

| 策略 | 原理 | 适用场景 |
|------|------|----------|
| **Hybrid Search** | 向量 + 关键词结合 | 需要精确匹配的场景 |
| **Query Expansion** | 扩展查询词 | 用户查询简短模糊 |
| **Reranking** | 初筛后用重排序模型精排 | 高质量需求 |
| **Multi-Query** | 生成多个查询变体 | 覆盖不同表述 |

```python
# Hybrid Search 示例
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# 创建两种检索器
bm25_retriever = BM25Retriever.from_documents(docs)
vector_retriever = vectorstore.as_retriever()

# 组合
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
```

#### 2.2.6 上下文构建

将检索结果组装成 Prompt：

```python
from langchain.prompts import PromptTemplate

template = """基于以下上下文回答问题。如果上下文不包含答案，请说"我不知道"。

上下文：
{context}

问题：{question}

答案："""

prompt = PromptTemplate.from_template(template)
```

**上下文优化技巧**：
- 按相关性排序
- 去除重复内容
- 控制总长度（不超过模型上下文限制）
- 添加来源标注

---

## 3. 完整实现示例

### 3.1 基础 RAG 链

```python
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 加载向量存储
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

# 2. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. 创建 LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 4. 创建 RAG 链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 将所有文档放入一个 Prompt
    retriever=retriever,
    return_source_documents=True
)

# 5. 查询
result = qa_chain.invoke({"query": "什么是 RAG？"})
print(result["result"])
print("来源：", [doc.metadata for doc in result["source_documents"]])
```

### 3.2 高级 RAG（带查询重写）

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 查询重写
rewrite_template = """将用户的问题改写成更适合向量检索的形式。
保持原意，但使用更具体、更完整的表述。

用户问题：{question}

改写后的问题："""

rewrite_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(rewrite_template)
)

# 使用改写后的查询
rewritten = rewrite_chain.invoke({"question": user_query})
results = retriever.get_relevant_documents(rewritten["text"])
```

---

## 4. 评估与优化

### 4.1 评估指标

| 指标 | 含义 | 计算方法 |
|------|------|----------|
| **Context Precision** | 检索到的文档中有多少是相关的 | 相关文档数 / 总检索文档数 |
| **Context Recall** | 相关文档中有多少被检索到 | 检索到的相关文档 / 总相关文档 |
| **Faithfulness** | 答案是否忠实于上下文 | 人工或模型判断 |
| **Answer Relevance** | 答案是否回答了问题 | 人工或模型判断 |

### 4.2 评估工具

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# 准备评估数据
eval_data = {
    "question": ["问题1", "问题2"],
    "answer": ["答案1", "答案2"],
    "contexts": [["上下文1"], ["上下文2"]],
    "ground_truth": ["标准答案1", "标准答案2"]
}

# 运行评估
results = evaluate(
    eval_data,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
```

### 4.3 常见问题与优化

| 问题 | 诊断 | 解决方案 |
|------|------|----------|
| 检索不到相关内容 | 嵌入质量差 / 分块不当 | 换更好的嵌入模型 / 调整分块策略 |
| 答案包含幻觉 | 上下文不足 / LLM 过度发挥 | 增加检索数量 / 加约束提示 |
| 答案太泛 | 查询太宽泛 | 查询重写 / 添加过滤条件 |
| 速度慢 | 检索耗时 / LLM 调用慢 | 缓存 / 换更快的模型 / 优化索引 |

---

## 5. 生产部署建议

### 5.1 架构模式

```
[负载均衡] → [API Gateway] → [RAG Service] → [Vector DB]
                              ↓
                         [Cache Layer]
                              ↓
                         [LLM API]
```

### 5.2 关键考虑

1. **缓存**：对常见查询缓存检索结果和生成结果
2. **限流**：防止 LLM API 成本失控
3. **监控**：跟踪检索质量、生成质量、延迟、成本
4. **回退**：LLM 不可用时返回检索结果摘要
5. **安全**：过滤敏感查询，防止 Prompt 注入

---

## 💡 我的思考

1. **RAG 是 80% 场景的最优解**：对于大多数企业应用，RAG 比微调更实用。它解决了知识更新问题，且成本可控。只有在需要改变模型行为风格时才考虑微调。

2. **分块策略是隐形杀手**：很多 RAG 效果差的原因是分块不当。代码和文档需要不同的分块策略，值得花时间优化。

3. **检索质量比生成质量更重要**：如果检索不到相关内容，再好的 LLM 也救不了。建议投入 60% 精力优化检索，40% 优化生成。

4. **Hybrid Search 是必选项**：纯向量检索在处理专有名词、ID、代码时效果差，必须结合关键词检索。

5. **评估是持续过程**：RAG 系统上线后需要持续监控和优化。建议建立评估数据集，定期跑回归测试。

---

## 参考来源

- [Huyen Chip: Building A Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html) — 访问日期：2026-06-05
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) — 访问日期：2026-06-05
- [LlamaIndex RAG Guide](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/) — 访问日期：2026-06-05
- [RAGAS Documentation](https://docs.ragas.io/) — 访问日期：2026-06-05
- [Pinecone Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/) — 访问日期：2026-06-05
