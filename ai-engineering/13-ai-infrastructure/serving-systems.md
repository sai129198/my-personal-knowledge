# AI 模型服务系统

> **一句话定位**：从单机部署到分布式集群，构建高可用、高性能的 AI 模型服务。
>
> #status/canonical #topic/serving #topic/deployment #year/2026

---

## 1. 服务架构模式

### 1.1 单体服务架构

```
┌─────────────────────────────────────┐
│           Load Balancer             │
└─────────────┬───────────────────────┘
              │
    ┌─────────┴─────────┐
    │   API Gateway     │
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┐
    │  Model Server     │
    │  (vLLM/TGI/...)   │
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┐
    │   GPU Node(s)     │
    └───────────────────┘
```

**适用场景**：单模型、中小规模、快速上线

### 1.2 微服务架构

```
┌─────────────────────────────────────────────┐
│              API Gateway                     │
│  (Kong/AWS API Gateway/Azure Front Door)     │
└──────┬──────────┬──────────┬────────────────┘
       │          │          │
   ┌───┴───┐  ┌──┴───┐  ┌───┴────┐
   │Router │  │Router│  │ Router │
   └───┬───┘  └──┬───┘  └───┬────┘
       │         │          │
   ┌───┴───┐ ┌───┴───┐ ┌───┴────┐
   │Model A│ │Model B│ │Model C │
   │(LLM)  │ │(Embed)│ │(Vision)│
   └───┬───┘ └───┬───┘ └───┬────┘
       │         │         │
   ┌───┴───┐ ┌───┴───┐ ┌───┴────┐
   │GPU Pool│ │GPU Pool│ │GPU Pool│
   └───────┘ └───────┘ └────────┘
```

**适用场景**：多模型、多租户、复杂路由

### 1.3 分布式服务架构

```
┌──────────────────────────────────────────────┐
│              Global Load Balancer             │
└──────┬──────────────┬──────────────┬─────────┘
       │              │              │
   ┌───┴───┐     ┌───┴───┐     ┌───┴────┐
   │Region │     │Region │     │Region  │
   │  A    │     │   B   │     │   C    │
   └───┬───┘     └───┬───┘     └───┬────┘
       │             │             │
   ┌───┴───┐     ┌───┴───┐     ┌───┴────┐
   │Cluster│     │Cluster│     │Cluster │
   │(K8s)  │     │(K8s)  │     │(K8s)   │
   └───────┘     └───────┘     └────────┘
```

**适用场景**：全球化部署、低延迟、高可用

---

## 2. 主流服务框架

### 2.1 vLLM

**特点**：
- PagedAttention 实现高吞吐
- 兼容 OpenAI API 格式
- 支持多种量化格式
- 活跃社区

**部署方式**：

```bash
# Docker 部署
docker run --gpus all \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-2-7b \
    --tensor-parallel-size 2

# Kubernetes 部署
kubectl apply -f vllm-deployment.yaml
```

```yaml
# vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - --model
        - meta-llama/Llama-2-7b
        - --tensor-parallel-size
        - "2"
        resources:
          limits:
            nvidia.com/gpu: "2"
        ports:
        - containerPort: 8000
```

### 2.2 TGI (Text Generation Inference)

**特点**：
- HuggingFace 官方出品
- 生产级特性（监控、追踪）
- 支持 Safetensors
- 维护模式（推荐迁移到 vLLM/SGLang）

```bash
# 启动 TGI
docker run --gpus all \
    -p 8080:80 \
    -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:2.0 \
    --model-id meta-llama/Llama-2-7b \
    --num-shard 2
```

### 2.3 Triton Inference Server

**特点**：
- NVIDIA 官方
- 多框架支持（PyTorch/TensorRT/ONNX）
- 动态批处理
- 模型编排

```python
# Triton 模型配置 config.pbtxt
name: "llama_model"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [-1]
  }
]
output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [-1, 32000]
  }
]
dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 100
}
```

### 2.4 Ray Serve

**特点**：
- 与 Ray 生态集成
- 模型组合（Model Composition）
- 自动扩缩容
- 多框架支持

```python
# Ray Serve 部署
from ray import serve
from starlette.requests import Request

@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class LLMDeployment:
    def __init__(self):
        self.model = load_model()
    
    async def __call__(self, request: Request):
        prompt = await request.json()
        return self.model.generate(prompt)

# 部署
serve.run(LLMDeployment.bind(), route_prefix="/generate")
```

### 2.5 SGLang

**特点**：
- 结构化生成优化
- RadixAttention（前缀缓存）
- 高吞吐量
- 适合复杂推理任务

```python
# SGLang 使用
import sglang as sgl

@sgl.function
def multi_turn_question(s, question):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=256))

# 运行
state = multi_turn_question.run(
    question="What is AI?",
    temperature=0.7
)
print(state["answer"])
```

---

## 3. 服务编排与网关

### 3.1 API 网关

| 网关 | 特点 | 适用场景 |
|------|------|----------|
| **Kong** | 插件丰富、Lua 扩展 | 企业级 |
| **Envoy** | 云原生、高性能 | K8s 环境 |
| **Nginx** | 成熟稳定、配置简单 | 传统部署 |
| **AWS API Gateway** | 托管、集成好 | AWS 生态 |
| **Azure API Management** | 托管、企业特性 | Azure 生态 |

### 3.2 模型路由

```python
# 智能路由示例
class ModelRouter:
    def __init__(self):
        self.models = {
            "light": LightModel(),      # 小模型，低延迟
            "heavy": HeavyModel(),      # 大模型，高质量
        }
    
    def route(self, request):
        # 基于复杂度路由
        complexity = self.estimate_complexity(request)
        
        if complexity < 0.3 and request.latency_sla < 100:
            return self.models["light"]
        else:
            return self.models["heavy"]
    
    def estimate_complexity(self, request):
        # 基于 prompt 长度、历史轮数等
        score = len(request.prompt) / 1000
        score += len(request.history) * 0.1
        return min(score, 1.0)
```

### 3.3 A/B 测试与金丝雀发布

```yaml
# Istio 金丝雀发布
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: model-routing
spec:
  hosts:
  - model-service
  http:
  - match:
    - headers:
        x-model-version:
          exact: v2
    route:
    - destination:
        host: model-v2
      weight: 100
  - route:
    - destination:
        host: model-v1
      weight: 90
    - destination:
        host: model-v2
      weight: 10  # 10% 流量到新版本
```

---

## 4. 负载均衡与扩缩容

### 4.1 负载均衡策略

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| **轮询 (Round Robin)** | 均匀分配 | 同构服务 |
| **最少连接** | 分配到连接最少 | 长连接 |
| **一致性哈希** | 相同请求路由到相同实例 | 缓存友好 |
| **加权轮询** | 按权重分配 | 异构实例 |
| **自定义策略** | 基于 GPU 利用率等 | GPU 服务 |

### 4.2 自动扩缩容

```yaml
# K8s HPA (Horizontal Pod Autoscaler)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "70"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
```

### 4.3 GPU 集群调度

```yaml
# K8s GPU 调度
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: model-server
    resources:
      limits:
        nvidia.com/gpu: "2"  # 请求 2 个 GPU
      requests:
        nvidia.com/gpu: "2"
  nodeSelector:
    node-type: gpu-a100
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
```

---

## 5. 多租户与隔离

### 5.1 资源隔离

| 隔离级别 | 实现方式 | 粒度 |
|----------|----------|------|
| **Namespace** | K8s Namespace | 团队/项目 |
| **Pod** | 独立 Pod | 应用 |
| **Container** | 容器限制 | 服务 |
| **GPU 时间片** | MPS/MIG | 进程 |

### 5.2 NVIDIA MIG (Multi-Instance GPU)

```bash
# 启用 MIG
nvidia-smi mig -cgi 19,19,19 -C  # 创建 3 个 MIG 实例

# K8s 中使用 MIG
resources:
  limits:
    nvidia.com/mig-3g.40gb: "1"
```

**MIG 配置**：

| Profile | 显存 | 计算单元 | 适用模型 |
|---------|------|----------|----------|
| mig-1g.10gb | 10GB | 14 SM | 7B 量化 |
| mig-2g.20gb | 20GB | 28 SM | 7B FP16 |
| mig-3g.40gb | 40GB | 42 SM | 13B FP16 |
| mig-7g.80gb | 80GB | 98 SM | 70B 量化 |

### 5.3 配额与限流

```python
# 令牌桶限流
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=100, period=60)  # 每分钟 100 次
def call_model_api(request):
    return model.generate(request)

# 基于用户配额
class QuotaManager:
    def check_quota(self, user_id, requested_tokens):
        quota = self.get_user_quota(user_id)
        used = self.get_usage(user_id)
        
        if used + requested_tokens > quota:
            raise QuotaExceededError()
        
        self.record_usage(user_id, requested_tokens)
```

---

## 6. 缓存策略

### 6.1 多级缓存架构

```
┌─────────────────────────────────────────┐
│           Client Request                 │
└─────────────┬───────────────────────────┘
              │
    ┌─────────┴─────────┐
    │   L1: 本地缓存     │  (Prompt Hash -> Response)
    └─────────┬─────────┘
              │ miss
    ┌─────────┴─────────┐
    │   L2: Redis 缓存   │  (Embedding Cache)
    └─────────┬─────────┘
              │ miss
    ┌─────────┴─────────┐
    │   L3: 模型推理     │
    └───────────────────┘
```

### 6.2 前缀缓存 (Prefix Caching)

```python
# SGLang / vLLM 前缀缓存
class PrefixCache:
    def __init__(self):
        self.cache = {}  # prefix_hash -> kv_cache
    
    def get(self, prompt):
        # 查找最长匹配前缀
        for length in range(len(prompt), 0, -1):
            prefix = prompt[:length]
            hash_key = hash(prefix)
            if hash_key in self.cache:
                return self.cache[hash_key], length
        return None, 0
    
    def put(self, prompt, kv_cache):
        hash_key = hash(prompt)
        self.cache[hash_key] = kv_cache
```

### 6.3 Embedding 缓存

```python
# 语义缓存
import hashlib
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache = {}  # embedding_hash -> response
    
    def get(self, query, threshold=0.95):
        query_emb = self.encoder.encode(query)
        
        for cached_emb, response in self.cache.items():
            similarity = cosine_similarity(query_emb, cached_emb)
            if similarity > threshold:
                return response
        
        return None
    
    def put(self, query, response):
        emb = self.encoder.encode(query)
        self.cache[emb.tobytes()] = response
```

---

## 7. 监控与可观测性

### 7.1 关键指标

| 类别 | 指标 | 告警阈值 |
|------|------|----------|
| **性能** | P50/P99 延迟 | P99 > 500ms |
| **性能** | 吞吐量 (tokens/s) | < 目标值 80% |
| **资源** | GPU 利用率 | < 50% 或 > 95% |
| **资源** | 显存使用 | > 90% |
| **质量** | 错误率 | > 1% |
| **业务** | 请求数/秒 | 异常波动 |

### 7.2 监控栈

```yaml
# Prometheus + Grafana 监控
prometheus:
  scrape_configs:
  - job_name: 'vllm'
    static_configs:
    - targets: ['vllm:8000']
    metrics_path: /metrics

# Grafana Dashboard
# - 请求速率
# - 延迟分布 (heatmap)
# - GPU 利用率
# - 显存使用
# - 队列长度
```

### 7.3 分布式追踪

```python
# OpenTelemetry 追踪
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("model_inference")
def inference(request):
    with tracer.start_as_current_span("tokenize"):
        tokens = tokenizer.encode(request.prompt)
    
    with tracer.start_as_current_span("model_forward"):
        output = model.generate(tokens)
    
    with tracer.start_as_current_span("decode"):
        text = tokenizer.decode(output)
    
    return text
```

---

## 8. 安全与合规

### 8.1 安全最佳实践

| 层面 | 措施 |
|------|------|
| **认证** | API Key、OAuth2、JWT |
| **授权** | RBAC、细粒度权限 |
| **加密** | TLS 1.3、模型加密存储 |
| **审计** | 请求日志、访问记录 |
| **防护** | 输入过滤、输出审查、速率限制 |

### 8.2 内容安全

```python
# 输入过滤
class InputFilter:
    def __init__(self):
        self.blocked_patterns = [
            r"ignore previous instructions",
            r"DAN mode",
            # ...
        ]
    
    def check(self, prompt):
        for pattern in self.blocked_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                raise SecurityException("Blocked input detected")
        
        # 毒性检测
        toxicity_score = self.toxicity_classifier(prompt)
        if toxicity_score > 0.8:
            raise SecurityException("Toxic content detected")

# 输出审查
class OutputFilter:
    def check(self, response):
        # PII 检测
        pii = self.detect_pii(response)
        if pii:
            response = self.redact_pii(response, pii)
        
        return response
```

---

## 9. 高可用设计

### 9.1 故障转移

```
┌─────────────────────────────────────────┐
│              Load Balancer               │
└──────┬────────────────────┬─────────────┘
       │                    │
   ┌───┴───┐           ┌───┴───┐
   │Primary│◄─────────►│Standby│
   │ (AZ1) │  健康检查   │ (AZ2) │
   └───┬───┘           └───────┘
       │
   ┌───┴───┐
   │ GPU   │
   │ Pool  │
   └───────┘
```

### 9.2 健康检查

```python
# 多层次健康检查
class HealthChecker:
    def check(self):
        checks = {
            "api": self.check_api(),
            "model": self.check_model(),
            "gpu": self.check_gpu(),
            "disk": self.check_disk(),
        }
        
        # 只要关键检查通过就视为健康
        critical = ["api", "model"]
        return all(checks[c] for c in critical)
    
    def check_model(self):
        try:
            # 发送测试请求
            result = self.model.generate("test", max_tokens=10)
            return len(result) > 0
        except:
            return False
```

### 9.3 优雅关闭

```python
# 优雅关闭处理
import signal

class GracefulShutdown:
    def __init__(self, server):
        self.server = server
        self.shutdown_event = asyncio.Event()
    
    def handle_signal(self, sig):
        print(f"Received {sig}, starting graceful shutdown...")
        
        # 1. 停止接受新请求
        self.server.stop_accepting()
        
        # 2. 等待现有请求完成
        await self.server.wait_for_requests(timeout=30)
        
        # 3. 清理资源
        self.server.cleanup()
        
        self.shutdown_event.set()

# 注册信号处理
signal.signal(signal.SIGTERM, handler.handle_signal)
signal.signal(signal.SIGINT, handler.handle_signal)
```

---

## 10. 成本优化

### 10.1 成本构成

| 成本项 | 占比 | 优化方向 |
|--------|------|----------|
| GPU 实例 | 60-80% | 自动扩缩容、Spot 实例 |
| 存储 | 10-15% | 模型压缩、分层存储 |
| 网络 | 5-10% | 缓存、CDN |
| 人力运维 | 5-10% | 自动化、托管服务 |

### 10.2 成本优化策略

```markdown
1. **自动扩缩容**
   - 基于负载自动调整实例数
   - 设置最小/最大副本数

2. **使用 Spot/Preemptible 实例**
   - 可节省 60-90% 成本
   - 适合容错性好的工作负载

3. **模型量化**
   - 减少显存占用
   - 使用更小实例

4. **请求合并**
   - 批量处理请求
   - 提高 GPU 利用率

5. **缓存优化**
   - 减少重复计算
   - 降低模型调用次数
```

---

## 参考资源

- [vLLM Documentation](https://docs.vllm.ai/)
- [TGI Documentation](https://huggingface.co/docs/text-generation-inference/index)
- [Triton Inference Server Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)
- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/index.html)
- [SGLang Documentation](https://sgl-project.github.io/)
- [Kubernetes GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [NVIDIA MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
