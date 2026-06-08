# Agent 生产部署与优化

> **一句话定位**：将 Agent 从原型部署到生产环境，确保高可用、高性能和可观测性。
>
> #status/canonical #topic/agent #topic/production #topic/deployment #year/2026

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
│                  Agent Service                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Agent     │  │   Agent     │  │   Agent     │ │
│  │   Instance  │  │   Instance  │  │   Instance  │ │
│  │   (Worker)  │  │   (Worker)  │  │   (Worker)  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Tool      │  │   Memory    │  │   LLM       │ │
│  │   Registry  │  │   Store     │  │   Client    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────┐
│              Infrastructure Layer                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │  Redis  │  │  Vector │  │  LLM    │            │
│  │  Cache  │  │  Store  │  │  Service│            │
│  └─────────┘  └─────────┘  └─────────┘            │
└─────────────────────────────────────────────────────┘
```

### 1.2 组件职责

| 组件 | 职责 | 技术选型 |
|------|------|----------|
| **API Gateway** | 限流、认证、路由 | Kong, Nginx, AWS API Gateway |
| **Agent Service** | Agent 实例管理 | FastAPI, Celery |
| **Tool Registry** | 工具注册和发现 | etcd, Consul |
| **Memory Store** | 记忆持久化 | Redis, PostgreSQL |
| **LLM Service** | 模型推理服务 | vLLM, TGI, OpenAI API |

---

## 2. Agent 服务实现

### 2.1 FastAPI 服务

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import uuid

app = FastAPI(title="Agent Service")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class AgentRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    tools: Optional[List[str]] = None

class AgentResponse(BaseModel):
    session_id: str
    response: str
    actions: List[Dict[str, Any]]
    latency_ms: float

# Agent 实例管理
class AgentManager:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.sessions: Dict[str, str] = {}  # session_id -> agent_id
    
    async def get_or_create_agent(self, session_id: str = None) -> Agent:
        """获取或创建 Agent"""
        if session_id and session_id in self.sessions:
            agent_id = self.sessions[session_id]
            return self.agents[agent_id]
        
        # 创建新 Agent
        agent_id = str(uuid.uuid4())
        agent = Agent(id=agent_id)
        self.agents[agent_id] = agent
        
        if session_id:
            self.sessions[session_id] = agent_id
        
        return agent
    
    async def cleanup_inactive(self, max_inactive_minutes: int = 30):
        """清理不活跃的 Agent"""
        current_time = datetime.now()
        to_remove = []
        
        for agent_id, agent in self.agents.items():
            if (current_time - agent.last_active).minutes > max_inactive_minutes:
                to_remove.append(agent_id)
        
        for agent_id in to_remove:
            del self.agents[agent_id]
            # 清理 session 映射
            sessions_to_remove = [
                sid for sid, aid in self.sessions.items() if aid == agent_id
            ]
            for sid in sessions_to_remove:
                del self.sessions[sid]

agent_manager = AgentManager()

@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """
    运行 Agent
    """
    start_time = time.time()
    
    try:
        # 获取或创建 Agent
        agent = await agent_manager.get_or_create_agent(request.session_id)
        
        # 设置工具
        if request.tools:
            agent.set_tools(request.tools)
        
        # 执行
        result = await agent.run(
            query=request.query,
            context=request.context
        )
        
        latency = (time.time() - start_time) * 1000
        
        return AgentResponse(
            session_id=agent.id,
            response=result["response"],
            actions=result["actions"],
            latency_ms=latency
        )
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/session/{session_id}/reset")
async def reset_session(session_id: str):
    """
    重置会话
    """
    if session_id in agent_manager.sessions:
        agent_id = agent_manager.sessions[session_id]
        if agent_id in agent_manager.agents:
            await agent_manager.agents[agent_id].reset()
        return {"status": "success", "message": "Session reset"}
    
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/agent/health")
async def health_check():
    """
    健康检查
    """
    return {
        "status": "healthy",
        "active_agents": len(agent_manager.agents),
        "active_sessions": len(agent_manager.sessions)
    }
```

### 2.2 异步任务队列

```python
from celery import Celery
from celery.result import AsyncResult

# Celery 配置
celery_app = Celery(
    "agent_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

@celery_app.task(bind=True, max_retries=3)
def execute_agent_task(self, query: str, session_id: str, context: Dict = None):
    """
    异步执行 Agent 任务
    """
    try:
        agent = Agent()
        result = agent.run(query, context)
        return {
            "status": "success",
            "result": result,
            "session_id": session_id
        }
    except Exception as exc:
        # 重试
        self.retry(exc=exc, countdown=60)

# API 端点
@app.post("/agent/run/async")
async def run_agent_async(request: AgentRequest):
    """
    异步运行 Agent
    """
    task = execute_agent_task.delay(
        query=request.query,
        session_id=request.session_id or str(uuid.uuid4()),
        context=request.context
    )
    
    return {
        "task_id": task.id,
        "status": "pending",
        "session_id": request.session_id
    }

@app.get("/agent/task/{task_id}")
async def get_task_status(task_id: str):
    """
    获取任务状态
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    return {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else None
    }
```

---

## 3. 性能优化

### 3.1 连接池

```python
class LLMConnectionPool:
    """
    LLM 连接池
    """
    
    def __init__(self, pool_size: int = 10):
        self.pool_size = pool_size
        self.clients: List[OpenAI] = []
        self.available: asyncio.Queue = asyncio.Queue()
        
        # 初始化连接
        for _ in range(pool_size):
            client = OpenAI()
            self.clients.append(client)
            self.available.put_nowait(client)
    
    async def acquire(self) -> OpenAI:
        """获取连接"""
        return await self.available.get()
    
    async def release(self, client: OpenAI):
        """释放连接"""
        await self.available.put(client)
    
    async def generate(self, **kwargs) -> str:
        """使用连接池生成"""
        client = await self.acquire()
        try:
            response = await client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        finally:
            await self.release(client)
```

### 3.2 缓存策略

```python
class AgentCache:
    """
    Agent 结果缓存
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 3600  # 1 小时
    
    async def get(self, key: str) -> Optional[Dict]:
        """获取缓存"""
        data = await self.redis.get(f"agent:cache:{key}")
        if data:
            return json.loads(data)
        return None
    
    async def set(self, key: str, value: Dict):
        """设置缓存"""
        await self.redis.setex(
            f"agent:cache:{key}",
            self.ttl,
            json.dumps(value)
        )
    
    def _generate_key(self, query: str, tools: List[str]) -> str:
        """生成缓存键"""
        key_data = {
            "query": query,
            "tools": sorted(tools) if tools else []
        }
        return hashlib.md5(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()

# 使用缓存
@app.post("/agent/run")
async def run_agent_with_cache(request: AgentRequest):
    cache = AgentCache(redis_client)
    
    # 生成缓存键
    cache_key = cache._generate_key(
        request.query,
        request.tools or []
    )
    
    # 检查缓存
    cached = await cache.get(cache_key)
    if cached:
        return AgentResponse(**cached)
    
    # 执行
    result = await run_agent(request)
    
    # 缓存结果
    await cache.set(cache_key, result.dict())
    
    return result
```

### 3.3 流式响应

```python
from fastapi.responses import StreamingResponse

@app.post("/agent/run/stream")
async def run_agent_stream(request: AgentRequest):
    """
    流式运行 Agent
    """
    async def event_generator():
        agent = await agent_manager.get_or_create_agent(request.session_id)
        
        # 流式执行
        async for event in agent.run_stream(request.query):
            yield f"data: {json.dumps(event)}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

class StreamingAgent:
    async def run_stream(self, query: str):
        """
        流式执行
        """
        # 发送开始事件
        yield {"type": "start", "query": query}
        
        # 规划阶段
        plan = await self.plan(query)
        yield {"type": "plan", "steps": plan.steps}
        
        # 执行阶段
        for step in plan.steps:
            yield {"type": "step_start", "step": step.id}
            
            result = await self.execute_step(step)
            
            yield {
                "type": "step_complete",
                "step": step.id,
                "result": result
            }
        
        # 生成最终响应
        response = await self.generate_response(plan)
        yield {"type": "response", "content": response}
```

---

## 4. 监控与可观测性

### 4.1 指标收集

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# 定义指标
agent_requests = Counter('agent_requests_total', 'Total requests', ['status'])
agent_latency = Histogram('agent_latency_seconds', 'Request latency')
agent_active_sessions = Gauge('agent_active_sessions', 'Active sessions')
agent_tool_calls = Counter('agent_tool_calls_total', 'Tool calls', ['tool_name'])

@app.post("/agent/run")
async def run_agent_with_metrics(request: AgentRequest):
    with agent_latency.time():
        try:
            result = await run_agent(request)
            agent_requests.labels(status='success').inc()
            return result
        except Exception as e:
            agent_requests.labels(status='error').inc()
            raise

@app.get("/metrics")
async def metrics():
    """Prometheus 指标端点"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

### 4.2 分布式追踪

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# 配置追踪
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

@app.post("/agent/run")
async def run_agent_traced(request: AgentRequest):
    with tracer.start_as_current_span("agent_run") as span:
        # 设置属性
        span.set_attribute("session_id", request.session_id)
        span.set_attribute("query_length", len(request.query))
        
        # 规划阶段
        with tracer.start_span("planning"):
            plan = await agent.plan(request.query)
        
        # 执行阶段
        with tracer.start_span("execution"):
            result = await agent.execute(plan)
        
        span.set_attribute("steps_count", len(plan.steps))
        
        return result
```

### 4.3 日志记录

```python
import structlog

logger = structlog.get_logger()

class LoggingAgent:
    async def run(self, query: str):
        logger.info(
            "agent_execution_started",
            query=query,
            session_id=self.session_id
        )
        
        try:
            # 规划
            logger.debug("planning_started")
            plan = await self.plan(query)
            logger.debug("planning_completed", steps=len(plan.steps))
            
            # 执行
            for step in plan.steps:
                logger.debug(
                    "step_execution_started",
                    step_id=step.id,
                    tool=step.tool
                )
                
                result = await self.execute_step(step)
                
                logger.debug(
                    "step_execution_completed",
                    step_id=step.id,
                    success=result.success
                )
            
            logger.info(
                "agent_execution_completed",
                session_id=self.session_id,
                steps_executed=len(plan.steps)
            )
            
        except Exception as e:
            logger.error(
                "agent_execution_failed",
                error=str(e),
                session_id=self.session_id
            )
            raise
```

---

## 5. 安全与隔离

### 5.1 沙箱执行

```python
import docker
from docker.types import Resources

class SandboxExecutor:
    """
    沙箱执行器
    """
    
    def __init__(self):
        self.docker_client = docker.from_env()
    
    async def execute_tool(self, tool_name: str, params: Dict) -> str:
        """
        在沙箱中执行工具
        """
        # 创建容器
        container = self.docker_client.containers.run(
            image="agent-tool-sandbox",
            command=f"python run_tool.py {tool_name} '{json.dumps(params)}'",
            detach=True,
            mem_limit="128m",
            cpu_quota=50000,  # 50% CPU
            network_mode="none",  # 禁用网络
            read_only=True,  # 只读文件系统
            volumes={
                "/tmp/output": {"bind": "/output", "mode": "rw"}
            }
        )
        
        # 等待执行完成
        result = container.wait(timeout=30)
        
        # 获取输出
        logs = container.logs().decode("utf-8")
        
        # 清理
        container.remove(force=True)
        
        if result["StatusCode"] != 0:
            raise ToolExecutionError(f"Tool failed: {logs}")
        
        return logs
```

### 5.2 权限控制

```python
from functools import wraps

class PermissionChecker:
    def __init__(self):
        self.role_permissions = {
            "admin": ["*"],
            "user": ["search", "calculate", "summarize"],
            "guest": ["search"]
        }
    
    def check_permission(self, user_role: str, tool_name: str) -> bool:
        """检查权限"""
        permissions = self.role_permissions.get(user_role, [])
        return "*" in permissions or tool_name in permissions

def require_permission(tool_name: str):
    """权限检查装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 从请求中获取用户角色
            user_role = kwargs.get("user_role", "guest")
            
            checker = PermissionChecker()
            if not checker.check_permission(user_role, tool_name):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied for tool: {tool_name}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@app.post("/agent/run")
@require_permission("search")
async def run_agent_secure(request: AgentRequest, user_role: str = "guest"):
    """带权限控制的 Agent 执行"""
    return await run_agent(request)
```

---

## 6. 部署配置

### 6.1 Docker 部署

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 非 root 用户运行
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/agent/health || exit 1

# 启动
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis
      - vector-store
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/agent/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  vector-store:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
    volumes:
      - milvus_data:/var/lib/milvus

  celery-worker:
    build: .
    command: celery -A tasks worker --loglevel=info --concurrency=4
    environment:
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 4G

volumes:
  redis_data:
  milvus_data:
```

### 6.2 Kubernetes 部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-service
  labels:
    app: agent-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-service
  template:
    metadata:
      labels:
        app: agent-service
    spec:
      containers:
      - name: agent
        image: agent-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: agent-config
              key: redis_url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: openai_api_key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /agent/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /agent/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: agent-service
spec:
  selector:
    app: agent-service
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 参考资源

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Celery Distributed Task Queue](https://docs.celeryq.dev/)
- [Prometheus Monitoring](https://prometheus.io/docs/introduction/overview/)
- [OpenTelemetry Tracing](https://opentelemetry.io/docs/)
- [Docker Security Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
