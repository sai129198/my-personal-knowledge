# RAG 生产部署与优化

> **一句话定位**：从原型到生产，构建高可用、高性能、可扩展的 RAG 服务。
>
> #status/canonical #topic/rag #topic/production #topic/deployment #year/2026

---

## 1. 生产架构设计

### 1.1 典型架构

```
┌─────────────────────────────────────────────────────┐
│                    Client Layer                      │
│         (Web App / Mobile / API Consumer)           │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────┐
│                   API Gateway                        │
│    (Rate Limiting / Auth / Load Balancing)          │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────┐
│                  RAG Service                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Query     │  │  Retrieval  │  │ Generation  │ │
│  │  Processor  │→ │   Engine    │→ │   Engine    │ │
│  └─────────────┘  └──────┬──────┘  └─────────────┘ │
│                          │                         │
│                   ┌──────┴──────┐                  │
│                   │ Vector Store │                  │
│                   │  (Milvus/    │                  │
│                   │   Pinecone)  │                  │
│                   └─────────────┘                  │
└─────────────────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────┐
│              Document Processing                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ Ingest  │→ │ Parse   │→ │ Embed   │            │
│  │ API     │  │ & Chunk │  │ & Index │            │
│  └─────────┘  └─────────┘  └─────────┘            │
└─────────────────────────────────────────────────────┘
```

### 1.2 组件职责

| 组件 | 职责 | 技术选型 |
|------|------|----------|
| **API Gateway** | 限流、认证、路由 | Kong, Nginx, AWS API Gateway |
| **Query Processor** | 查询重写、缓存、预处理 | Python/FastAPI, Redis |
| **Retrieval Engine** | 向量检索、重排序 | Milvus, Pinecone, Faiss |
| **Generation Engine** | LLM 调用、流式输出 | vLLM, TGI, OpenAI API |
| **Document Processor** | 文档解析、分块、索引 | LangChain, LlamaIndex |

---

## 2. 性能优化

### 2.1 检索优化

```python
# 1. 索引预热
class WarmedIndex:
    def __init__(self, index):
        self.index = index
        self._warm_up()
    
    def _warm_up(self):
        """预加载热点数据到内存"""
        hot_vectors = self._get_hot_vectors()
        self.index.preload(hot_vectors)

# 2. 查询缓存
class CachedRetriever:
    def __init__(self, retriever, cache):
        self.retriever = retriever
        self.cache = cache
    
    def search(self, query, top_k=5):
        cache_key = f"search:{hash(query)}:{top_k}"
        
        # 检查缓存
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # 执行检索
        results = self.retriever.search(query, top_k)
        
        # 缓存结果
        self.cache.set(cache_key, results, ttl=300)
        return results

# 3. 批量检索
class BatchRetriever:
    def __init__(self, retriever, batch_size=10):
        self.retriever = retriever
        self.batch_size = batch_size
    
    async def batch_search(self, queries):
        """批量处理查询"""
        results = []
        
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            batch_results = await self.retriever.asearch_batch(batch)
            results.extend(batch_results)
        
        return results
```

### 2.2 生成优化

```python
# 1. 流式输出
async def stream_generate(query, contexts):
    """流式生成，减少首 token 延迟"""
    prompt = build_prompt(query, contexts)
    
    async for token in llm.astream_generate(prompt):
        yield token

# 2. 并发控制
class ConcurrencyLimiter:
    def __init__(self, max_concurrent=10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate(self, query):
        async with self.semaphore:
            return await llm.generate(query)

# 3. 模型缓存
class ModelCache:
    def __init__(self):
        self.response_cache = {}
    
    def get_or_generate(self, query, contexts):
        """缓存相似查询的结果"""
        cache_key = self._compute_cache_key(query, contexts)
        
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        response = llm.generate(query, contexts)
        self.response_cache[cache_key] = response
        return response
```

---

## 3. 高可用设计

### 3.1 容错机制

```python
class FaultTolerantRAG:
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
    
    async def generate(self, query):
        try:
            # 尝试主服务
            return await self.primary.generate(query)
        except Exception as e:
            logger.error(f"Primary failed: {e}")
            
            try:
                # 降级到备用服务
                return await self.fallback.generate(query)
            except Exception as e2:
                logger.error(f"Fallback failed: {e2}")
                
                # 最终降级：直接回答
                return self._graceful_degradation(query)
    
    def _graceful_degradation(self, query):
        """优雅降级"""
        return {
            "answer": "服务暂时不可用，请稍后重试",
            "status": "degraded",
            "query": query
        }
```

### 3.2 健康检查

```python
class HealthChecker:
    def __init__(self, components):
        self.components = components
    
    async def check_health(self):
        """检查所有组件健康状态"""
        status = {}
        
        for name, component in self.components.items():
            try:
                await component.health_check()
                status[name] = "healthy"
            except Exception as e:
                status[name] = f"unhealthy: {e}"
        
        return status
    
    def is_healthy(self):
        """判断整体健康状态"""
        status = self.check_health()
        return all(s == "healthy" for s in status.values())
```

### 3.3 熔断器

```python
from circuitbreaker import circuit

class CircuitBreakerRAG:
    @circuit(failure_threshold=5, recovery_timeout=60)
    async def retrieve(self, query):
        """带熔断的检索"""
        return await self.retriever.search(query)
    
    @circuit(failure_threshold=3, recovery_timeout=30)
    async def generate(self, query, contexts):
        """带熔断的生成"""
        return await self.generator.generate(query, contexts)
```

---

## 4. 监控与告警

### 4.1 关键指标

| 指标 | 说明 | 告警阈值 |
|------|------|----------|
| **P50/P99 延迟** | 响应时间 | P99 > 2s |
| **检索召回率** | 检索质量 | < 80% |
| **答案忠实度** | 生成质量 | < 70% |
| **错误率** | 服务稳定性 | > 1% |
| **QPS** | 吞吐量 | < 目标 50% |
| **缓存命中率** | 缓存效果 | < 60% |

### 4.2 监控实现

```python
from prometheus_client import Counter, Histogram, Gauge

# 定义指标
rag_latency = Histogram('rag_latency_seconds', 'RAG latency')
rag_errors = Counter('rag_errors_total', 'RAG errors', ['type'])
rag_cache_hit = Gauge('rag_cache_hit_rate', 'Cache hit rate')

class MonitoredRAG:
    def __init__(self, rag_service):
        self.rag = rag_service
    
    @rag_latency.time()
    async def generate(self, query):
        try:
            result = await self.rag.generate(query)
            return result
        except RetrievalError as e:
            rag_errors.labels(type='retrieval').inc()
            raise
        except GenerationError as e:
            rag_errors.labels(type='generation').inc()
            raise
```

### 4.3 日志追踪

```python
import structlog

logger = structlog.get_logger()

class TracedRAG:
    async def generate(self, query, trace_id=None):
        trace_id = trace_id or generate_trace_id()
        
        logger.info(
            "rag_request_started",
            trace_id=trace_id,
            query=query
        )
        
        try:
            # 检索
            with timer("retrieval"):
                docs = await self.retriever.search(query)
                logger.info(
                    "retrieval_completed",
                    trace_id=trace_id,
                    num_docs=len(docs)
                )
            
            # 生成
            with timer("generation"):
                answer = await self.generator.generate(query, docs)
                logger.info(
                    "generation_completed",
                    trace_id=trace_id,
                    answer_length=len(answer)
                )
            
            return answer
            
        except Exception as e:
            logger.error(
                "rag_request_failed",
                trace_id=trace_id,
                error=str(e)
            )
            raise
```

---

## 5. 安全与合规

### 5.1 输入过滤

```python
class InputSanitizer:
    def __init__(self):
        self.blocked_patterns = [
            r"ignore previous instructions",
            r"system override",
            r"developer mode",
        ]
    
    def sanitize(self, query):
        """清理用户输入"""
        # 1. 长度限制
        if len(query) > 10000:
            raise ValueError("Query too long")
        
        # 2. 模式检测
        for pattern in self.blocked_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                raise SecurityError("Potentially malicious query detected")
        
        # 3. 内容过滤
        query = self._filter_sensitive_content(query)
        
        return query
```

### 5.2 输出审核

```python
class OutputModerator:
    def __init__(self, moderation_model):
        self.moderator = moderation_model
    
    async def moderate(self, answer):
        """审核生成内容"""
        result = await self.moderator.check(answer)
        
        if result.flagged:
            logger.warning(f"Content flagged: {result.categories}")
            return {
                "answer": "[内容被过滤]",
                "flagged": True,
                "categories": result.categories
            }
        
        return {"answer": answer, "flagged": False}
```

### 5.3 数据隐私

```python
class PrivacyPreservingRAG:
    def __init__(self, pii_detector):
        self.pii_detector = pii_detector
    
    def redact_pii(self, text):
        """脱敏处理"""
        pii_entities = self.pii_detector.detect(text)
        
        for entity in pii_entities:
            text = text.replace(
                entity.text,
                f"[{entity.type}]"
            )
        
        return text
    
    async def generate(self, query):
        # 1. 查询脱敏
        redacted_query = self.redact_pii(query)
        
        # 2. 检索和生成
        answer = await self.rag.generate(redacted_query)
        
        # 3. 输出脱敏
        return self.redact_pii(answer)
```

---

## 6. 部署实践

### 6.1 Docker 部署

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - VECTOR_DB_URL=milvus:19530
      - LLM_API_KEY=${LLM_API_KEY}
    depends_on:
      - milvus
      - redis
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '4'
          memory: 8G

  milvus:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
    volumes:
      - milvus_data:/var/lib/milvus

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

volumes:
  milvus_data:
```

### 6.2 Kubernetes 部署

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-service
  template:
    metadata:
      labels:
        app: rag-service
    spec:
      containers:
      - name: rag
        image: rag-service:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: VECTOR_DB_URL
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: vector_db_url
---
apiVersion: v1
kind: Service
metadata:
  name: rag-service
spec:
  selector:
    app: rag-service
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## 7. 成本控制

### 7.1 成本分析

| 组件 | 成本项 | 优化策略 |
|------|--------|----------|
| **Embedding** | API 调用费 | 批量处理、缓存 |
| **向量存储** | 存储 + 查询 | 数据生命周期管理 |
| **LLM 生成** | Token 费用 | 提示压缩、缓存 |
| **基础设施** | 计算资源 | 自动扩缩容 |

### 7.2 成本优化代码

```python
class CostOptimizedRAG:
    def __init__(self):
        self.embedding_cache = {}
        self.response_cache = {}
    
    async def generate(self, query):
        # 1. 检查响应缓存
        cache_key = hash(query)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # 2. 检查 Embedding 缓存
        if query not in self.embedding_cache:
            self.embedding_cache[query] = await embed(query)
        
        query_embedding = self.embedding_cache[query]
        
        # 3. 检索（通常成本较低）
        docs = await self.retriever.search(query_embedding)
        
        # 4. 生成（成本较高，优化提示）
        compressed_prompt = self._compress_prompt(query, docs)
        answer = await self.llm.generate(compressed_prompt)
        
        # 5. 缓存结果
        self.response_cache[cache_key] = answer
        
        return answer
    
    def _compress_prompt(self, query, docs):
        """压缩提示以减少 token"""
        # 只保留最相关的文档片段
        relevant_chunks = self._extract_relevant_chunks(query, docs)
        
        # 使用更简洁的格式
        return f"Q:{query}\nC:{relevant_chunks}\nA:"
```

---

## 参考资源

- [Building Production-Ready RAG Applications](https://www.pinecone.io/learn/build-production-rag/)
- [LangChain Production Guide](https://python.langchain.com/docs/guides/productionization/)
- [LLM System Design](https://github.com/ray-project/llm-applications)
- [RAG Flow Patterns](https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/rag)
- [Vector DB Performance Benchmark](https://benchmark.vectorview.ai/)
