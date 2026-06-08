# 检索算法详解

> **一句话定位**：从 BM25 到向量检索再到混合检索，选择最适合场景的检索策略。
>
> #status/canonical #topic/rag #topic/retrieval #year/2026

---

## 1. 检索基础

### 1.1 检索在 RAG 中的作用

```
用户查询 → 检索相关文档 → 增强 Prompt → LLM 生成答案
                ↑
           【检索算法】
```

**检索质量直接影响生成质量**：
- 检索准确 → 生成准确
- 检索遗漏 → 生成幻觉
- 检索噪声 → 生成偏离

### 1.2 检索算法分类

| 类型 | 代表算法 | 核心思想 | 适用场景 |
|------|----------|----------|----------|
| **稀疏检索** | BM25、TF-IDF | 基于词频和文档频率 | 精确匹配、关键词搜索 |
| **稠密检索** | DPR、Contriever | 基于语义向量相似度 | 语义搜索、概念匹配 |
| **混合检索** | BM25 + 向量 | 结合稀疏和稠密优势 | 通用场景 |

---

## 2. 稀疏检索：BM25

### 2.1 BM25 原理

BM25（Best Match 25）是基于概率检索框架的算法，计算查询词与文档的相关性得分。

**公式**：

```
Score(D, Q) = Σ IDF(q_i) * [f(q_i, D) * (k1 + 1)] / [f(q_i, D) + k1 * (1 - b + b * |D| / avgdl)]

其中：
- D: 文档
- Q: 查询
- q_i: 查询中的第 i 个词
- f(q_i, D): 词 q_i 在文档 D 中的频率
- |D|: 文档长度
- avgdl: 平均文档长度
- k1, b: 调节参数
```

### 2.2 参数调优

| 参数 | 作用 | 建议值 |
|------|------|--------|
| **k1** | 控制词频饱和度 | 1.2 - 2.0 |
| **b** | 控制文档长度归一化 | 0.75 |

```python
from rank_bm25 import BM25Okapi

# 准备文档
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outpaces a swift fox",
    "The lazy dog sleeps all day"
]

# 分词
tokenized_corpus = [doc.split(" ") for doc in corpus]

# 创建 BM25 索引
bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

# 查询
query = "quick fox"
tokenized_query = query.split(" ")

# 获取得分
doc_scores = bm25.get_scores(tokenized_query)
print(doc_scores)  # [0.937, 0.832, 0.0]

# 获取 Top-K
top_k = bm25.get_top_n(tokenized_query, corpus, n=2)
```

### 2.3 BM25 优缺点

| 优点 | 缺点 |
|------|------|
| 计算速度快 | 无法理解语义 |
| 可解释性强 | 需要精确匹配关键词 |
| 对短查询效果好 | 无法处理同义词 |
| 无需训练 | 对长文档效果下降 |

---

## 3. 稠密检索：向量检索

### 3.1 向量检索原理

```
文档 → Embedding 模型 → 向量（768/1024/1536 维）
查询 → Embedding 模型 → 向量（768/1024/1536 维）

相似度 = cosine_similarity(文档向量, 查询向量)
```

### 3.2 Embedding 模型选择

| 模型 | 维度 | 特点 | 适用场景 |
|------|------|------|----------|
| **text-embedding-3-small** | 1536 | OpenAI，性价比高 | 通用语义搜索 |
| **text-embedding-3-large** | 3072 | OpenAI，高精度 | 高质量要求 |
| **bge-large-zh** | 1024 | 中文优化 | 中文场景 |
| **e5-large-v2** | 1024 | 微软，多语言 | 多语言场景 |
| **gte-large** | 1024 | 阿里，中文优化 | 中文 RAG |
| **contriever** | 768 | 无监督训练 | 无标注数据 |

```python
from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer('BAAI/bge-large-zh')

# 编码文档
documents = [
    "机器学习是人工智能的一个分支",
    "深度学习使用神经网络进行学习",
    "自然语言处理让计算机理解人类语言"
]
doc_embeddings = model.encode(documents, normalize_embeddings=True)

# 编码查询
query = "什么是神经网络"
query_embedding = model.encode(query, normalize_embeddings=True)

# 计算相似度
import numpy as np
similarities = np.dot(doc_embeddings, query_embedding)
print(similarities)  # [0.234, 0.891, 0.456]
```

### 3.3 向量相似度计算

| 度量方式 | 公式 | 特点 |
|----------|------|------|
| **余弦相似度** | cos(θ) = A·B / (\|A\| \|B\|) | 忽略向量长度，关注方向 |
| **欧氏距离** | d = √(Σ(A_i - B_i)²) | 考虑向量长度 |
| **点积** | A·B = Σ(A_i * B_i) | 简单快速，考虑长度 |

```python
# 余弦相似度（推荐）
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings)

# 欧氏距离
from scipy.spatial.distance import euclidean

distances = [euclidean(query_embedding, doc_emb) for doc_emb in doc_embeddings]
```

### 3.4 近似最近邻（ANN）算法

| 算法 | 原理 | 特点 | 适用场景 |
|------|------|------|----------|
| **HNSW** | 分层可导航小世界图 | 高召回、快速 | 通用场景 |
| **IVF** | 倒排文件索引 | 内存友好 | 大规模数据 |
| **PQ** | 乘积量化 | 极省内存 | 超大规模 |
| **Flat** | 暴力搜索 | 100% 召回 | 小规模数据 |

```python
import faiss

# 创建 HNSW 索引
dimension = 768
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 邻居
index.hnsw.efConstruction = 200  # 构建时搜索深度

# 添加向量
index.add(doc_embeddings)

# 搜索
k = 5  # Top-5
index.hnsw.efSearch = 128  # 搜索时深度
distances, indices = index.search(query_embedding.reshape(1, -1), k)
```

---

## 4. 混合检索

### 4.1 为什么需要混合检索

```
BM25 优势：精确匹配、关键词敏感
向量检索优势：语义理解、同义词处理

混合检索 = BM25 得分 + 向量相似度得分
```

### 4.2 混合检索实现

```python
class HybridRetriever:
    def __init__(self, bm25_weight=0.5, vector_weight=0.5):
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.bm25 = None
        self.vector_index = None
        self.documents = []
    
    def index_documents(self, documents):
        self.documents = documents
        
        # 1. 构建 BM25 索引
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # 2. 构建向量索引
        self.doc_embeddings = model.encode(documents)
        dimension = self.doc_embeddings.shape[1]
        self.vector_index = faiss.IndexFlatIP(dimension)
        self.vector_index.add(self.doc_embeddings)
    
    def search(self, query, top_k=5):
        # 1. BM25 检索
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 2. 向量检索
        query_embedding = model.encode([query])
        vector_scores, _ = self.vector_index.search(query_embedding, len(self.documents))
        vector_scores = vector_scores[0]
        
        # 3. 归一化
        bm25_scores = self._normalize(bm25_scores)
        vector_scores = self._normalize(vector_scores)
        
        # 4. 融合得分
        hybrid_scores = (
            self.bm25_weight * bm25_scores +
            self.vector_weight * vector_scores
        )
        
        # 5. 返回 Top-K
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return [(self.documents[i], hybrid_scores[i]) for i in top_indices]
    
    def _normalize(self, scores):
        """Min-Max 归一化"""
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
```

### 4.3 权重调优

```python
def tune_weights(retriever, val_queries, ground_truth):
    """
    通过验证集调优混合权重
    """
    best_weights = None
    best_score = 0
    
    for bm25_w in np.arange(0, 1.1, 0.1):
        vector_w = 1 - bm25_w
        retriever.bm25_weight = bm25_w
        retriever.vector_weight = vector_w
        
        score = evaluate(retriever, val_queries, ground_truth)
        
        if score > best_score:
            best_score = score
            best_weights = (bm25_w, vector_w)
    
    return best_weights, best_score
```

---

## 5. 重排序（Reranking）

### 5.1 为什么需要重排序

```
第一阶段（召回）：快速检索 Top-100
第二阶段（精排）：精确模型排序 Top-5
```

### 5.2 重排序模型

| 模型 | 类型 | 特点 |
|------|------|------|
| **Cross-Encoder** | 交互式 | 高精度，慢 |
| **ColBERT** | 晚期交互 | 平衡精度速度 |
| **MonoT5** | 生成式 | 可解释性强 |
| **Cohere Rerank** | API | 即用即得 |

```python
from sentence_transformers import CrossEncoder

# 加载重排序模型
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 第一阶段：召回 Top-20
candidates = retriever.search(query, top_k=20)

# 第二阶段：重排序
pairs = [(query, doc) for doc, _ in candidates]
scores = reranker.predict(pairs)

# 获取最终 Top-5
final_results = sorted(
    zip(candidates, scores),
    key=lambda x: x[1],
    reverse=True
)[:5]
```

### 5.3 两阶段检索架构

```python
class TwoStageRetriever:
    def __init__(self):
        self.retriever = HybridRetriever()  # 第一阶段
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # 第二阶段
    
    def search(self, query, final_k=5, recall_k=100):
        # 第一阶段：快速召回
        candidates = self.retriever.search(query, top_k=recall_k)
        
        # 第二阶段：精确重排
        pairs = [(query, doc) for doc, _ in candidates]
        rerank_scores = self.reranker.predict(pairs)
        
        # 返回最终结果
        results = list(zip(candidates, rerank_scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:final_k]
```

---

## 6. 查询优化

### 6.1 查询扩展

```python
class QueryExpander:
    def __init__(self, model):
        self.model = model
    
    def expand(self, query, num_expansions=3):
        """
        使用 LLM 扩展查询
        """
        prompt = f"""
        为以下查询生成 {num_expansions} 个语义相似的变体，
        帮助检索更多相关文档。
        
        查询：{query}
        
        变体：
        1.
        """
        
        response = self.model.generate(prompt)
        expansions = parse_numbered_list(response)
        
        return [query] + expansions
    
    def hyde(self, query):
        """
        HyDE: Hypothetical Document Embeddings
        生成假设文档，用假设文档做检索
        """
        prompt = f"""
        为以下查询生成一个假设的文档段落，
        这个段落应该包含查询的答案。
        
        查询：{query}
        
        假设文档：
        """
        
        hypothetical_doc = self.model.generate(prompt)
        return hypothetical_doc
```

### 6.2 查询重写

```python
class QueryRewriter:
    def rewrite(self, query, conversation_history=None):
        """
        基于对话历史重写查询
        """
        if not conversation_history:
            return query
        
        prompt = f"""
        基于以下对话历史，重写最后一个查询，
        使其包含必要的上下文信息。
        
        对话历史：
        {format_history(conversation_history)}
        
        当前查询：{query}
        
        重写后的查询（包含上下文）：
        """
        
        return self.model.generate(prompt)
```

---

## 7. 评估方法

### 7.1 检索评估指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **Recall@K** | Top-K 中包含相关文档的比例 | relevant_in_k / total_relevant |
| **Precision@K** | Top-K 中相关文档的比例 | relevant_in_k / k |
| **MRR** | 第一个相关文档排名的倒数 | 1 / rank_of_first_relevant |
| **NDCG** | 考虑排序位置的加权指标 | 加权折扣累积增益 |

```python
def evaluate_retrieval(retriever, test_queries, ground_truth):
    """
    评估检索性能
    """
    metrics = {
        'recall@5': [],
        'recall@10': [],
        'mrr': [],
    }
    
    for query, relevant_docs in zip(test_queries, ground_truth):
        results = retriever.search(query, top_k=10)
        retrieved_docs = [doc for doc, _ in results]
        
        # Recall@K
        for k in [5, 10]:
            retrieved_k = set(retrieved_docs[:k])
            relevant = set(relevant_docs)
            recall = len(retrieved_k & relevant) / len(relevant)
            metrics[f'recall@{k}'].append(recall)
        
        # MRR
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                metrics['mrr'].append(1 / i)
                break
        else:
            metrics['mrr'].append(0)
    
    # 计算平均值
    return {k: np.mean(v) for k, v in metrics.items()}
```

---

## 参考资源

- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) - Gao et al., 2023
- [BM25: The Next Generation of Lucene's Classic Scoring Model](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)
- [Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) - Karpukhin et al., 2020
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction](https://arxiv.org/abs/2004.12832) - Khattab & Zaharia, 2020
- [Pinecone RAG Guide](https://www.pinecone.io/learn/series/rag/)
