# 向量数据库选型指南

> **一句话定位**：从 HNSW 索引到分布式架构，系统梳理向量数据库的核心原理与生产选型策略。
>
> #status/draft #topic/vector-db #topic/rag #topic/data-engineering #year/2026

---

## 一、核心概念

### 1.1 什么是向量数据库？

向量数据库是专门用于**存储、索引和查询高维向量**的数据库系统。

**与传统数据库的区别**：

| 特性 | 传统数据库 | 向量数据库 |
|------|------------|------------|
| 查询方式 | 精确匹配（=, LIKE） | 相似度搜索（ANN） |
| 数据类型 | 标量（数字、字符串） | 向量（数百至数千维） |
| 索引结构 | B+树、哈希表 | HNSW、IVF、PQ |
| 返回结果 | 完全匹配 | 最相似的 Top-K |

### 1.2 向量嵌入 (Embeddings)

将非结构化数据（文本、图像、音频）转换为数值向量的过程：

```
文本: "机器学习是人工智能的一个分支"
  ↓ Embedding Model (如 text-embedding-3-large)
向量: [0.023, -0.156, 0.089, ..., 0.034]  (1536 维)
```

**关键特性**：
- 语义相似的文本 → 向量距离近
- 不同模型生成的向量**不兼容**
- 维度固定（由模型决定）

### 1.3 相似度度量

| 度量方式 | 公式 | 适用场景 |
|----------|------|----------|
| **余弦相似度** | $\frac{A \cdot B}{\|A\| \|B\|}$ | 方向比大小重要（文本语义） |
| **欧氏距离 (L2)** | $\sqrt{\sum(A_i - B_i)^2}$ | 绝对距离重要（图像特征） |
| **点积 (Dot Product)** | $A \cdot B$ | 需要保留幅度信息 |
| **汉明距离** | 不同位数的数量 | 二进制向量 |

**实践建议**：
- 文本语义搜索 → **余弦相似度**
- 图像相似搜索 → **欧氏距离**
- 需要区分重要程度 → **点积**

---

## 二、近似最近邻 (ANN) 算法

### 2.1 暴力搜索 (Flat/Exact Search)

**原理**：计算查询向量与所有文档向量的距离，返回 Top-K。

**优点**：100% 召回率
**缺点**：O(N) 复杂度，数据量大时极慢

**适用**：数据量 < 10K，或需要精确结果的场景

### 2.2 HNSW (Hierarchical Navigable Small World)

**当前最流行的 ANN 算法**。

**原理**：构建多层图结构，上层稀疏（快速定位），下层密集（精确搜索）。

```
Layer 2 (稀疏):    o → o → o          (快速跳转)
                      ↓
Layer 1 (中等):    o → o → o → o      (中等精度)
                      ↓
Layer 0 (密集):    o-o-o-o-o-o-o-o    (精确搜索)
```

**关键参数**：
- `M`：每个节点的最大连接数（默认 16）
  - M 越大 → 图越稠密 → 搜索越慢但越准
- `ef_construction`：构建时的搜索深度（默认 200）
- `ef_search`：查询时的搜索深度（默认 50）

**优点**：
- 查询速度快（毫秒级）
- 召回率高（>95%）
- 支持增量添加

**缺点**：
- 内存占用大（需存储完整图结构）
- 构建索引慢

### 2.3 IVF (Inverted File Index)

**原理**：
1. 用 K-Means 将向量空间划分为 $nlist$ 个簇
2. 查询时只搜索最近的 $nprobe$ 个簇

```
向量空间：
[簇 1]  [簇 2]  [簇 3]  [簇 4]
  *       *       *       *
   \       \     /       /
    \       \   /       /
     [查询向量] → 只搜最近的 2 个簇
```

**关键参数**：
- `nlist`：簇的数量（通常 $4 \times \sqrt{N}$）
- `nprobe`：查询时搜索的簇数（权衡速度/精度）

**优点**：
- 内存效率高
- 适合十亿级数据

**缺点**：
- 需要训练（K-Means）
- 增量添加困难

### 2.4 PQ (Product Quantization)

**原理**：将高维向量压缩为短码，用码本近似距离计算。

```
原始向量 (128维): [0.1, 0.3, 0.2, ...]
  ↓ 分成 8 个子向量，每段 16 维
  ↓ 每段用 K-Means 量化为 8-bit 码
压缩后: [23, 156, 89, 34, 201, 45, 12, 78]  (仅 8 字节！)
```

**优点**：
- 极致压缩（内存减少 10-20x）
- 适合超大规模数据

**缺点**：
- 精度损失较大
- 通常与 IVF 结合使用（IVFPQ）

### 2.5 算法选型决策树

```
数据规模？
├── < 100K → Flat（精确搜索，无需调参）
├── 100K - 10M → HNSW（平衡速度和精度）
├── 10M - 1B → IVF + HNSW 或 IVFPQ
└── > 1B → IVFPQ + 分布式分片

对召回率要求极高？
├── 是 → HNSW（ef_search 调大）
└── 否 → IVF（nprobe 调小，速度更快）
```

---

## 三、主流向量数据库对比

### 3.1 综合对比表

| 数据库 | 类型 | 索引算法 | 最大规模 | 部署方式 | 最佳场景 |
|--------|------|----------|----------|----------|----------|
| **Pinecone** | 托管 SaaS | 自研 | 百亿级 | 全托管 | 快速启动、无运维 |
| **Weaviate** | 开源+云 | HNSW | 十亿级 | 本地/云/K8s | 多模态、GraphQL |
| **Milvus/Zilliz** | 开源+云 | IVF、HNSW、GPU | 千亿级 | 本地/分布式/云 | 超大规模、企业级 |
| **Qdrant** | 开源+云 | HNSW | 十亿级 | 本地/云 | 过滤查询、混合搜索 |
| **Chroma** | 开源 | HNSW | 百万级 | 本地/嵌入式 | 原型开发、轻量级 |
| **pgvector** | PostgreSQL 扩展 | IVF、HNSW | 千万级 | 现有 Postgres | 已有 PG 基础设施 |
| **Elasticsearch** | 搜索引擎 | HNSW | 十亿级 | 本地/云 | 混合搜索（文本+向量） |
| **Redis** | 内存数据库 | Flat、HNSW | 百万级 | 本地/云 | 缓存+向量混合 |
| **FAISS** | 算法库 | 多种 | 百亿级 | 嵌入式 | 研究、自定义系统 |

### 3.2 详细分析

#### Pinecone

**特点**：
- 完全托管，零运维
- 自动扩缩容
- 元数据过滤 + 向量搜索原生支持

**适用**：
- 快速启动 MVP
- 不想运维基础设施
- 预算充足

**缺点**：
-  vendor lock-in
-  自定义能力有限
-  成本较高

#### Weaviate

**特点**：
- 模块化架构（可插拔嵌入模型、生成模型）
- 原生 GraphQL 接口
- 多模态支持（文本、图像、视频）

**适用**：
- 需要多模态搜索
- 喜欢 GraphQL
- 需要向量+语义混合查询

#### Milvus / Zilliz Cloud

**特点**：
- 云原生架构，存储计算分离
- 支持 GPU 索引（CAGRA）
- 企业级功能（RBAC、多租户）

**适用**：
- 超大规模（十亿级以上）
- 需要分布式部署
- 企业级需求

#### Qdrant

**特点**：
- Rust 编写，性能优秀
- 强大的过滤查询能力
- 混合搜索（稀疏向量 + 稠密向量）

**适用**：
- 需要复杂过滤条件
- 混合搜索场景
- 自托管但要求高性能

#### Chroma

**特点**：
- 极简 API，5 分钟上手
- 嵌入式（in-process）
- Python 原生

**适用**：
- 原型开发
- 本地实验
- 小型项目

**缺点**：
- 不适合生产环境（无分布式）
- 性能有限

#### pgvector

**特点**：
- PostgreSQL 扩展
- 支持 IVF 和 HNSW
- 与 SQL 生态无缝集成

**适用**：
- 已有 PostgreSQL 基础设施
- 需要事务支持
- 向量+关系数据混合查询

### 3.3 选型决策矩阵

```
首要考虑因素？
├── 快速启动、无运维 → Pinecone / Chroma
├── 超大规模 (>1B) → Milvus / Zilliz
├── 已有 PG 基础设施 → pgvector
├── 多模态搜索 → Weaviate
├── 复杂过滤查询 → Qdrant
├── 极致性能（自研）→ FAISS
└── 混合搜索（文本+向量）→ Elasticsearch / Qdrant

预算敏感？
├── 是 → 开源方案（Milvus、Qdrant、pgvector）
└── 否 → 托管服务（Pinecone、Zilliz Cloud）
```

---

## 四、生产实践要点

### 4.1 索引构建最佳实践

1. **选择合适的索引类型**
   - 默认推荐 HNSW（召回率 > 95%）
   - 超大规模考虑 IVF 或 IVFPQ

2. **调参建议**
   ```python
   # HNSW 参数
   M = 16  # 数据量 < 1M
   M = 32  # 数据量 > 1M
   ef_construction = 200  # 构建质量
   ef_search = 64  # 查询精度（可动态调整）
   ```

3. **批量插入**
   - 避免逐条插入（触发频繁索引重建）
   - 批量大小：10K-100K 向量

### 4.2 查询优化

1. **预热索引**
   - 新构建的索引首次查询较慢
   - 建议预热后再上线

2. **混合搜索策略**
   ```
   第一阶段：向量搜索召回 Top-100
   第二阶段：用 BM25/TF-IDF 重排序
   第三阶段：业务规则过滤
   ```

3. **结果重排序 (Reranking)**
   - 粗排：ANN 快速召回 Top-K（K=100-1000）
   - 精排：用更精确的模型重排序 Top-N（N=10-50）

### 4.3 监控指标

| 指标 | 说明 | 告警阈值 |
|------|------|----------|
| 查询延迟 (P99) | 99% 查询的响应时间 | > 100ms |
| 召回率 | ANN 结果 vs 暴力搜索 | < 95% |
| 索引构建时间 | 新增数据的索引延迟 | > 1h |
| 内存使用率 | 向量索引内存占用 | > 80% |
| QPS | 每秒查询数 | 根据容量规划 |

---

## 五、代码示例

### 5.1 Chroma 快速开始

```python
import chromadb

# 创建客户端
client = chromadb.Client()

# 创建集合（相当于表）
collection = client.create_collection("my_docs")

# 添加数据
collection.add(
    documents=["机器学习是AI的分支", "深度学习是ML的子集"],
    metadatas=[{"source": "wiki"}, {"source": "book"}],
    ids=["doc1", "doc2"]
)

# 查询
results = collection.query(
    query_texts=["什么是人工智能？"],
    n_results=2
)
```

### 5.2 Qdrant 混合搜索

```python
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

# 搜索 + 过滤
results = client.search(
    collection_name="products",
    query_vector=[0.1, 0.2, ...],
    query_filter={
        "must": [
            {"key": "category", "match": {"value": "electronics"}},
            {"key": "price", "range": {"lte": 1000}}
        ]
    },
    limit=10
)
```

### 5.3 pgvector 在 Postgres 中

```sql
-- 启用扩展
CREATE EXTENSION vector;

-- 创建带向量列的表
CREATE TABLE items (
    id bigserial PRIMARY KEY,
    embedding vector(1536),
    content text
);

-- 创建 HNSW 索引
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops);

-- 向量搜索
SELECT * FROM items
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

---

## 💡 我的思考

### 关键洞察

1. **没有最好的向量数据库，只有最适合的**：Chroma 适合原型，Milvus 适合超大规模，pgvector 适合已有 PG 的团队
2. **HNSW 是默认选择**：除非数据量 > 10M 或内存极度受限，否则用 HNSW
3. **混合搜索是趋势**：纯向量搜索往往不够，结合关键词搜索（BM25）效果更好
4. **索引参数调优很重要**：ef_search 从 50 调到 200，召回率可能从 90% 提升到 98%

### 常见陷阱

- ❌ 用 Flat 索引处理百万级数据（性能极差）
- ❌ 忽视元数据过滤能力（很多场景需要）
- ❌ 不同模型的向量混用（不兼容！）
- ❌ 忘记监控召回率（ANN 可能漏掉正确答案）

### 下一步实践

- [ ] 用同一数据集测试 Chroma / Qdrant / pgvector 的性能
- [ ] 测量 HNSW 在不同 ef_search 下的召回率-延迟曲线
- [ ] 实现向量搜索 + BM25 的混合搜索 pipeline

---

## 参考来源

1. Pinecone - What is a Vector Database (2023)
2. Weaviate - What Is a Vector Database? (2024)
3. Chroma Documentation - Introduction (2024)
4. Qdrant Documentation - Overview (2024)
5. Elastic - What is a Vector Database? (2024)
6. FAISS GitHub - Facebook AI Similarity Search
7. Pinecone - Nearest Neighbor Indexes for Similarity Search (2023)

---

*最后更新：2026-06-08*
