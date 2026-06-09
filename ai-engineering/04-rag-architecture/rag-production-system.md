#topic/rag #topic/production #topic/system-design #year/2026 #status/draft

# RAG 生产系统设计

> 从原型到生产：构建可扩展、可维护、可观测的 RAG 系统。

---

## 1. 架构设计

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                         接入层                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ Web App │  │ Mobile  │  │ API     │  │ Slack   │       │
│  │         │  │ App     │  │ Client  │  │ Bot     │       │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
└───────┼────────────┼────────────┼────────────┼─────────────┘
        │            │            │            │
        └────────────┴──────┬─────┴────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      API Gateway                             │
│  - 认证鉴权  - 速率限制  - 请求路由  - 负载均衡              │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼───────┐  ┌───────▼───────┐  ┌───────▼───────┐
│   Query       │  │   Ingestion   │  │   Admin       │
│   Service     │  │   Pipeline    │  │   Service     │
│               │  │               │  │               │
│ - Query理解   │  │ - 文档加载    │  │ - 配置管理    │
│ - 检索编排    │  │ - 文本分块    │  │ - 监控面板    │
│ - 生成增强    │  │ - Embedding   │  │ - 评估工具    │
│ - 后处理      │  │ - 向量存储    │  │ - 日志查询    │
└───────┬───────┘  └───────────────┘  └───────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                      核心服务层                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │  Embedding  │  │  Vector DB  │  │  LLM        │       │
│  │  Service    │  │  Cluster    │  │  Gateway    │       │
│  │             │  │             │  │             │       │
│  │ - 文本向量化│  │ - 索引管理  │  │ - 多模型路由│       │
│  │ - 缓存      │  │ - 近似搜索  │  │ - 降级策略  │       │
│  │ - 批处理    │  │ - 分片存储  │  │ - 流式输出  │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                      数据层                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │  Object     │  │  Metadata   │  │  Cache      │       │
│  │  Storage    │  │  Store      │  │  Layer      │       │
│  │  (S3/MinIO) │  │  (PG/MySQL) │  │  (Redis)    │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 服务拆分

| 服务 | 职责 | 技术选型 |
|------|------|----------|
| **Query Service** | 查询理解、检索编排、生成 | Python/FastAPI |
| **Embedding Service** | 文本向量化 | Python/ONNX |
| **Ingestion Service** | 文档处理、索引更新 | Python/Celery |
| **LLM Gateway** | 模型路由、流式处理 | Go/Envoy |
| **Vector DB** | 向量存储与检索 | Qdrant/Milvus |

---

## 2. 性能优化

### 2.1 检索优化

| 优化手段 | 方法 | 效果 |
|----------|------|------|
| **索引分区** | 按租户/类型分片 | 减少搜索空间 |
| **量化压缩** | PQ/SQ 降维 | 降低内存占用 |
| **缓存** | 热点查询缓存 | 减少重复计算 |
| **预过滤** | 元数据先过滤 | 减少向量计算 |
| **并行检索** | 多分片并行查询 | 降低延迟 |

### 2.2 生成优化

| 优化手段 | 方法 | 效果 |
|----------|------|------|
| **Prompt 缓存** | 相同上下文复用 | 减少 token 消耗 |
| **流式输出** | SSE/WebSocket | 提升用户体验 |
| **模型路由** | 简单任务用小模型 | 降低成本 |
| **批量生成** | 合并相似请求 | 提高吞吐 |

### 2.3 缓存策略

```python
class RAGCache:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def get_embedding(self, text_hash):
        """Embedding 缓存"""
        return self.redis.get(f"emb:{text_hash}")
    
    def get_retrieval(self, query_hash, top_k):
        """检索结果缓存"""
        return self.redis.get(f"ret:{query_hash}:{top_k}")
    
    def get_generation(self, query_hash, context_hash):
        """生成结果缓存（短 TTL）"""
        return self.redis.get(f"gen:{query_hash}:{context_hash}")
    
    def cache_with_ttl(self, key, value, ttl_seconds):
        self.redis.setex(key, ttl_seconds, value)
```

---

## 3. 评估体系

### 3.1 离线评估

**数据集构建**：
```python
eval_dataset = [
    {
        "query": "什么是 RAG？",
        "expected_answer": "检索增强生成是一种...",
        "expected_sources": ["doc_1", "doc_2"],
        "difficulty": "easy",
        "category": "concept"
    },
    # ...
]
```

**评估指标**：

| 维度 | 指标 | 目标值 |
|------|------|--------|
| 检索 | Recall@K | > 0.9 |
| 检索 | MRR | > 0.8 |
| 生成 | Faithfulness | > 0.85 |
| 生成 | Answer Relevance | > 0.9 |
| 端到端 | 用户满意度 | > 4.0/5 |

### 3.2 在线评估

**A/B 测试框架**：
```
流量分配
    │
    ├── 50% → 对照组（当前版本）
    │
    └── 50% → 实验组（新版本）
            │
            ├── 指标对比：准确率、延迟、用户满意度
            └── 统计显著性检验
```

**实时监控指标**：
- 查询延迟（P50/P95/P99）
- 检索召回率
- 生成 token 数
- 错误率
- 用户反馈（点赞/点踩）

---

## 4. 监控告警

### 4.1 监控维度

| 层级 | 指标 | 告警阈值 |
|------|------|----------|
| **基础设施** | CPU/内存/磁盘 | > 80% |
| **服务** | QPS/延迟/错误率 | P99 > 2s / 错误率 > 1% |
| **业务** | 检索质量/生成质量 | 连续下降 > 10% |
| **成本** | Token 消耗/调用次数 | 日环比增长 > 50% |

### 4.2 日志追踪

```python
import structlog

logger = structlog.get_logger()

def handle_query(query):
    with tracer.start_as_current_span("rag_query") as span:
        span.set_attribute("query", query)
        
        # 检索
        docs = retrieve(query)
        span.set_attribute("retrieval_count", len(docs))
        
        # 生成
        answer = generate(query, docs)
        span.set_attribute("answer_length", len(answer))
        span.set_attribute("token_count", answer.tokens)
        
        logger.info(
            "query_processed",
            query=query,
            latency=span.duration,
            retrieval_count=len(docs),
        )
        
        return answer
```

---

## 5. 安全与合规

### 5.1 数据安全

| 措施 | 说明 |
|------|------|
| **数据隔离** | 多租户向量隔离 |
| **访问控制** | RBAC 权限管理 |
| **加密传输** | TLS 1.3 |
| **加密存储** | 向量数据加密 |
| **审计日志** | 所有查询可审计 |

### 5.2 内容安全

| 措施 | 说明 |
|------|------|
| **输入过滤** | 敏感词、注入检测 |
| **输出审核** | 有害内容过滤 |
| **引用验证** | 确保回答有来源 |
| **置信度阈值** | 低置信度时拒绝回答 |

---

## 6. 部署与运维

### 6.1 容器化部署

```yaml
# docker-compose.yml
version: '3.8'
services:
  query-service:
    image: rag-query:latest
    replicas: 3
    environment:
      - VECTOR_DB_URL=qdrant:6333
      - LLM_API_KEY=${LLM_API_KEY}
    depends_on:
      - qdrant
      - redis
  
  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
```

### 6.2 索引更新策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| **全量重建** | 定期全量重建索引 | 数据量小 |
| **增量更新** | 实时/准实时添加 | 数据量大 |
| **双写切换** | 新索引构建完成后切换 | 零停机 |
| **版本控制** | 保留多版本索引 | 可回滚 |

---

## 💡 我的思考

1. **生产 RAG 的核心挑战不是技术，是工程**：检索和生成算法只是基础，可观测性、容错、成本控制才是决定成败的关键。

2. **评估驱动迭代**：没有评估体系，就无法判断优化是否有效。建议先建立评估 pipeline，再逐步优化。

3. **缓存是性能的关键**：Embedding 缓存、检索缓存、生成缓存，层层缓存可以大幅降低延迟和成本。

4. **多模型策略**：不同任务用不同模型（简单任务用小模型，复杂任务用大模型），可以显著降低成本。

5. **RAG 系统的持续运营**：文档会过期、知识会更新，需要建立定期刷新机制和过期检测。

---

## 参考来源

- **Qdrant Production**: [qdrant.tech/documentation/guides/installation/](https://qdrant.tech/documentation/guides/installation/) — 访问日期：2026-06-07
- **Milvus Sizing Tool**: [milvus.io/tools/sizing](https://milvus.io/tools/sizing) — 访问日期：2026-06-07
- **LangChain Production**: [python.langchain.com/docs/guides/productionization/](https://python.langchain.com/docs/guides/productionization/) — 访问日期：2026-06-07
- **OpenAI Production Best Practices**: [platform.openai.com/docs/guides/production-best-practices](https://platform.openai.com/docs/guides/production-best-practices) — 访问日期：2026-06-07

---

*访问日期: 2026-06-07*
