# 向量数据库选型与优化

> **一句话定位**：对比主流向量数据库，根据场景选择最优方案并调优性能。
>
> #status/canonical #topic/vector-database #topic/rag #year/2026

---

## 1. 向量数据库基础

### 1.1 什么是向量数据库

向量数据库是专门存储和检索高维向量数据的数据库，通过近似最近邻（ANN）算法实现高效的相似度搜索。

```
传统数据库：存储结构化数据（文本、数字、日期）
向量数据库：存储向量（768维、1024维、1536维）

查询方式：
传统：SELECT * WHERE id = 123
向量：FIND top_k_similar(vector_query)
```

### 1.2 核心能力

| 能力 | 说明 |
|------|------|
| **向量存储** | 高效存储百万/十亿级向量 |
| **相似度搜索** | ANN 算法实现毫秒级检索 |
| **元数据过滤** | 向量 + 标量混合查询 |
| **分布式扩展** | 水平扩展支持海量数据 |

---

## 2. 主流向量数据库对比

### 2.1 选型矩阵

| 数据库 | 类型 | 特点 | 适用场景 | 部署方式 |
|--------|------|------|----------|----------|
| **Pinecone** | 托管服务 | 全托管、自动扩缩容 | 快速上线、无运维 | SaaS |
| **Milvus/Zilliz** | 开源/托管 | 功能丰富、企业级 | 大规模生产环境 | 自托管/SaaS |
| **Weaviate** | 开源/托管 | GraphQL 接口、模块化 | 需要灵活查询 | 自托管/SaaS |
| **Qdrant** | 开源/托管 | Rust 实现、高性能 | 高性能要求 | 自托管/SaaS |
| **Chroma** | 开源 | 轻量、易用 | 原型开发、小项目 | 嵌入式 |
| **pgvector** | PostgreSQL 扩展 | SQL 接口、事务支持 | 已有 PG 基础设施 | 自托管 |
| **Redis** | 内存数据库 | 低延迟、缓存友好 | 实时推荐 | 自托管 |
| **Elasticsearch** | 搜索引擎 | 混合搜索（文本+向量） | 需要全文检索 | 自托管/SaaS |

### 2.2 性能对比

| 数据库 | 十亿级数据 | QPS | 延迟 | 元数据过滤 |
|--------|-----------|-----|------|-----------|
| Pinecone | ✅ | 高 | <100ms | ✅ |
| Milvus | ✅ | 很高 | <50ms | ✅ |
| Weaviate | ✅ | 高 | <100ms | ✅ |
| Qdrant | ✅ | 很高 | <50ms | ✅ |
| Chroma | ❌ | 中 | <200ms | 有限 |
| pgvector | ⚠️ | 中 | <500ms | ✅ |
| Redis | ❌ | 很高 | <10ms | 有限 |

---

## 3. 详细分析

### 3.1 Pinecone

**特点**：
- 全托管，零运维
- 自动扩缩容
- 支持元数据过滤和混合搜索

```python
from pinecone import Pinecone, ServerlessSpec

# 初始化
pc = Pinecone(api_key="your-api-key")

# 创建索引
pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# 连接索引
index = pc.Index("my-index")

# 插入向量
index.upsert(
    vectors=[
        {
            "id": "vec1",
            "values": [0.1, 0.2, ...],  # 1536 维
            "metadata": {"category": "tech", "year": 2024}
        }
    ]
)

# 查询
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    filter={"category": {"$eq": "tech"}}
)
```

**定价**：
- Starter: 免费（1 个索引，10万向量）
- Standard: 按使用量付费
- Enterprise: 定制

### 3.2 Milvus

**特点**：
- 功能最丰富的开源向量数据库
- 支持 GPU 加速
- 分布式架构

```python
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 连接
connections.connect("default", host="localhost", port="19530")

# 定义集合
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)
]
schema = CollectionSchema(fields, "my_collection")
collection = Collection("my_collection", schema)

# 创建索引
index_params = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 500}
}
collection.create_index("embedding", index_params)

# 插入数据
collection.insert([[1], [[0.1, 0.2, ...]], ["text"]])

# 搜索
collection.load()
results = collection.search(
    data=[[0.1, 0.2, ...]],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"ef": 128}},
    limit=5
)
```

### 3.3 pgvector

**特点**：
- PostgreSQL 扩展
- 支持事务和 ACID
- 混合查询（向量 + SQL）

```sql
-- 安装扩展
CREATE EXTENSION vector;

-- 创建表
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(768)
);

-- 创建向量索引
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 插入数据
INSERT INTO documents (content, embedding)
VALUES ('text', '[0.1, 0.2, ...]');

-- 向量搜索
SELECT id, content, 1 - (embedding <=> '[0.1, 0.2, ...]') AS similarity
FROM documents
WHERE content LIKE '%keyword%'
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 5;
```

### 3.4 Chroma

**特点**：
- 最轻量、最易用
- 嵌入式部署
- 适合原型开发

```python
import chromadb

# 创建客户端（内存模式）
client = chromadb.Client()

# 或持久化模式
client = chromadb.PersistentClient(path="./chroma_db")

# 创建集合
collection = client.create_collection("my_collection")

# 添加文档
collection.add(
    documents=["doc1", "doc2"],
    embeddings=[[0.1, ...], [0.2, ...]],
    metadatas=[{"source": "web"}, {"source": "pdf"}],
    ids=["id1", "id2"]
)

# 查询
results = collection.query(
    query_embeddings=[[0.1, ...]],
    n_results=5,
    where={"source": "web"}
)
```

---

## 4. 选型决策树

```
开始
├── 数据量 < 10万?
│   ├── 是 → Chroma（简单）或 pgvector（需要 SQL）
│   └── 否 → 继续
├── 需要全托管?
│   ├── 是 → Pinecone 或 Zilliz Cloud
│   └── 否 → 继续
├── 已有 PostgreSQL?
│   ├── 是 → pgvector
│   └── 否 → 继续
├── 需要分布式?
│   ├── 是 → Milvus
│   └── 否 → 继续
├── 需要 GraphQL?
│   ├── 是 → Weaviate
│   └── 否 → Qdrant（性能）或 Milvus（功能）
```

---

## 5. 性能优化

### 5.1 索引选择

| 索引类型 | 特点 | 适用场景 |
|----------|------|----------|
| **HNSW** | 高召回、快速构建 | 通用场景，推荐 |
| **IVF** | 内存友好、构建快 | 大规模数据 |
| **IVF_PQ** | 极省内存 | 超大规模 |
| **FLAT** | 100% 召回 | 小规模数据 |

### 5.2 参数调优

**HNSW 参数**：

| 参数 | 说明 | 建议值 |
|------|------|--------|
| **M** | 邻居数 | 16-64 |
| **efConstruction** | 构建时搜索深度 | 100-500 |
| **ef** | 查询时搜索深度 | 100-500 |

```python
# HNSW 调优示例
index_params = {
    "index_type": "HNSW",
    "params": {
        "M": 32,              # 邻居数，越大精度越高
        "efConstruction": 400, # 构建深度
    }
}

search_params = {
    "params": {
        "ef": 200  # 查询深度，越大精度越高但越慢
    }
}
```

### 5.3 数据分区

```python
# 按类别分区存储
class PartitionedVectorStore:
    def __init__(self):
        self.partitions = {}
    
    def add(self, vectors, metadata):
        # 按类别分区
        category = metadata['category']
        if category not in self.partitions:
            self.partitions[category] = create_index()
        
        self.partitions[category].add(vectors)
    
    def search(self, query_vector, category=None, top_k=5):
        if category:
            # 只在特定分区搜索
            return self.partitions[category].search(query_vector, top_k)
        else:
            # 全局搜索
            results = []
            for partition in self.partitions.values():
                results.extend(partition.search(query_vector, top_k))
            return sorted(results, key=lambda x: x.score)[:top_k]
```

---

## 6. 生产部署建议

### 6.1 高可用架构

```
┌─────────────────────────────────────┐
│           Load Balancer              │
└─────────────┬───────────────────────┘
              │
    ┌─────────┴─────────┐
    │   API Gateway     │
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┐
    │  Vector DB Cluster │
    │  ┌─────┐ ┌─────┐  │
    │  │Node1│ │Node2│  │
    │  └─────┘ └─────┘  │
    └───────────────────┘
```

### 6.2 监控指标

| 指标 | 告警阈值 | 说明 |
|------|----------|------|
| **查询延迟 P99** | > 200ms | 搜索性能 |
| **索引构建时间** | > 1小时 | 数据更新 |
| **内存使用率** | > 80% | 资源使用 |
| **QPS** | < 目标值 50% | 吞吐量 |
| **召回率** | < 90% | 搜索质量 |

---

## 参考资源

- [Vector Database Comparison](https://thedataquarry.com/posts/vector-db-1/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Milvus Documentation](https://milvus.io/docs)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
