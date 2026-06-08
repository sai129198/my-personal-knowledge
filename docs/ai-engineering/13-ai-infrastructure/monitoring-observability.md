# AI 系统监控与可观测性

> **一句话定位**：构建全面的 AI 系统可观测性体系，确保生产环境稳定运行。
>
> #status/canonical #topic/monitoring #topic/observability #year/2026

---

## 1. 可观测性三大支柱

### 1.1 Metrics（指标）

**系统级指标**：

| 指标类别 | 关键指标 | 采集频率 |
|----------|----------|----------|
| **GPU** | 利用率、显存使用、温度、功耗 | 15s |
| **CPU** | 使用率、负载、上下文切换 | 15s |
| **内存** | 使用率、交换、OOM 事件 | 15s |
| **网络** | 带宽、延迟、丢包率 | 15s |
| **磁盘** | I/O、使用率、inode | 60s |

**应用级指标**：

| 指标类别 | 关键指标 | 告警阈值 |
|----------|----------|----------|
| **延迟** | P50/P95/P99 | P99 > 500ms |
| **吞吐量** | QPS、tokens/s | < 目标 80% |
| **错误率** | 5xx 比例 | > 1% |
| **队列** | 等待请求数 | > 100 |
| **缓存** | 命中率 | < 80% |

### 1.2 Logs（日志）

**日志分类**：

```
┌─────────────────────────────────────────────┐
│  应用日志 (Application Logs)                  │
│  - 请求/响应详情                              │
│  - 模型推理输入输出                           │
│  - 业务逻辑记录                               │
├─────────────────────────────────────────────┤
│  系统日志 (System Logs)                       │
│  - GPU 驱动日志                               │
│  - CUDA 错误                                  │
│  - 容器运行时日志                             │
├─────────────────────────────────────────────┤
│  审计日志 (Audit Logs)                        │
│  - 用户访问记录                               │
│  - API 调用记录                               │
│  - 权限变更记录                               │
└─────────────────────────────────────────────┘
```

**日志规范**：

```json
{
  "timestamp": "2026-01-15T10:30:00Z",
  "level": "INFO",
  "service": "vllm-server",
  "trace_id": "abc123",
  "span_id": "def456",
  "message": "Request completed",
  "fields": {
    "model": "llama-2-70b",
    "prompt_tokens": 150,
    "completion_tokens": 500,
    "total_tokens": 650,
    "latency_ms": 1200,
    "gpu_id": "0",
    "batch_size": 4
  }
}
```

### 1.3 Traces（追踪）

**分布式追踪流程**：

```
Client Request
    │
    ▼
┌─────────────┐
│ API Gateway │ ◄── trace_id: abc123
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Auth Check │ ◄── span_id: auth-001
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Rate Limiter│ ◄── span_id: rate-001
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Model Server │ ◄── span_id: model-001
│  ├─ Tokenize│ ◄── span_id: tok-001 (5ms)
│  ├─ Forward │ ◄── span_id: fwd-001 (1000ms)
│  └─ Decode  │ ◄── span_id: dec-001 (10ms)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Response  │
└─────────────┘
```

---

## 2. 监控体系架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────┐
│                    数据采集层                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │Prometheus│ │ FluentBit│ │ OTel Col │            │
│  │ (Metrics)│ │  (Logs)  │ │ (Traces) │            │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘            │
└───────┼────────────┼────────────┼────────────────────┘
        │            │            │
┌───────┼────────────┼────────────┼────────────────────┐
│       ▼            ▼            ▼                    │
│              数据存储层                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │Prometheus│ │OpenSearch│ │  Jaeger  │            │
│  │  (TSDB)  │ │  (Logs)  │ │ (Traces) │            │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘            │
└───────┼────────────┼────────────┼────────────────────┘
        │            │            │
┌───────┼────────────┼────────────┼────────────────────┐
│       ▼            ▼            ▼                    │
│              展示告警层                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Grafana  │ │Kibana/OS │ │Jaeger UI │            │
│  │(Dashboard)│ │ (Logs)   │ │ (Traces) │            │
│  └──────────┘ └──────────┘ └──────────┘            │
│  ┌──────────┐ ┌──────────┐                          │
│  │ AlertMgr │ │ PagerDuty│                          │
│  │ (Alerts) │ │ (OnCall) │                          │
│  └──────────┘ └──────────┘                          │
└─────────────────────────────────────────────────────┘
```

### 2.2 组件选型

| 组件 | 功能 | 替代方案 |
|------|------|----------|
| **Prometheus** | 指标采集存储 | VictoriaMetrics, Thanos |
| **Grafana** | 可视化 | Datadog, New Relic |
| **Fluent Bit** | 日志采集 | Fluentd, Logstash |
| **OpenSearch** | 日志存储 | Elasticsearch, Loki |
| **Jaeger** | 分布式追踪 | Zipkin, Tempo |
| **Alertmanager** | 告警管理 | PagerDuty, OpsGenie |

---

## 3. GPU 监控详解

### 3.1 DCGM (Data Center GPU Manager)

**关键指标**：

| 指标名 | 说明 | 告警条件 |
|--------|------|----------|
| `DCGM_FI_DEV_GPU_UTIL` | GPU 利用率 | > 95% 持续 5min |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | 显存带宽利用率 | > 90% |
| `DCGM_FI_DEV_FB_FREE` | 空闲显存 | < 10% |
| `DCGM_FI_DEV_GPU_TEMP` | GPU 温度 | > 85°C |
| `DCGM_FI_DEV_POWER_USAGE` | 功耗 | > TDP 90% |
| `DCGM_FI_DEV_XID_ERRORS` | XID 错误 | > 0 |
| `DCGM_FI_PROF_PCIE_TX_BYTES` | PCIe 发送带宽 | 监控趋势 |
| `DCGM_FI_PROF_PCIE_RX_BYTES` | PCIe 接收带宽 | 监控趋势 |

**Prometheus 配置**：

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'dcgm'
    static_configs:
      - targets: ['dcgm-exporter:9400']
    scrape_interval: 15s
    metrics_path: /metrics
```

### 3.2 GPU 监控 Dashboard

```json
// Grafana Dashboard 关键面板
{
  "panels": [
    {
      "title": "GPU Utilization",
      "type": "timeseries",
      "targets": [
        {
          "expr": "DCGM_FI_DEV_GPU_UTIL{gpu=~\"$gpu\"}",
          "legendFormat": "GPU {{gpu}}"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "unit": "percent",
          "min": 0,
          "max": 100
        }
      }
    },
    {
      "title": "GPU Memory Usage",
      "type": "timeseries",
      "targets": [
        {
          "expr": "DCGM_FI_DEV_FB_USED{gpu=~\"$gpu\"} / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE) * 100",
          "legendFormat": "GPU {{gpu}}"
        }
      ]
    },
    {
      "title": "GPU Temperature",
      "type": "gauge",
      "targets": [
        {
          "expr": "DCGM_FI_DEV_GPU_TEMP{gpu=~\"$gpu\"}"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "thresholds": {
            "steps": [
              {"color": "green", "value": 0},
              {"color": "yellow", "value": 70},
              {"color": "red", "value": 85}
            ]
          }
        }
      }
    }
  ]
}
```

---

## 4. 模型服务监控

### 4.1 自定义指标

```python
# vLLM 自定义指标示例
from prometheus_client import Counter, Histogram, Gauge

# 定义指标
REQUEST_COUNT = Counter(
    'llm_requests_total',
    'Total requests',
    ['model', 'status']
)

REQUEST_LATENCY = Histogram(
    'llm_request_duration_seconds',
    'Request latency',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

TOKENS_GENERATED = Counter(
    'llm_tokens_generated_total',
    'Total tokens generated',
    ['model']
)

QUEUE_SIZE = Gauge(
    'llm_queue_size',
    'Current queue size',
    ['model']
)

# 使用
class ModelServer:
    def generate(self, request):
        start = time.time()
        
        try:
            # 处理请求
            output = self.model.generate(request.prompt)
            
            REQUEST_COUNT.labels(
                model=self.model_name,
                status='success'
            ).inc()
            
            TOKENS_GENERATED.labels(
                model=self.model_name
            ).inc(len(output.tokens))
            
        except Exception as e:
            REQUEST_COUNT.labels(
                model=self.model_name,
                status='error'
            ).inc()
            raise
        
        finally:
            REQUEST_LATENCY.labels(
                model=self.model_name
            ).observe(time.time() - start)
```

### 4.2 业务指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **TTFT (Time To First Token)** | 首 token 时间 | 请求到达 → 首个 token |
| **TPOT (Time Per Output Token)** | 每 token 生成时间 | 生成时间 / token 数 |
| **Throughput** | 吞吐量 | tokens/s 或 requests/s |
| **Batch Efficiency** | 批处理效率 | 实际 batch size / 最大 |
| **Cache Hit Rate** | 缓存命中率 | 命中数 / 总请求数 |

---

## 5. 日志管理

### 5.1 结构化日志

```python
import structlog
import logging

# 配置 structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# 使用
logger.info(
    "request_completed",
    model="llama-2-70b",
    prompt_tokens=150,
    completion_tokens=500,
    latency_ms=1200,
    trace_id="abc123"
)
```

### 5.2 日志收集架构

```yaml
# Fluent Bit 配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush         1
        Log_Level     info
        Daemon        off
        Parsers_File  parsers.conf
    
    [INPUT]
        Name              tail
        Tag               vllm.*
        Path              /var/log/containers/vllm*.log
        Parser            json
        Refresh_Interval  5
        Mem_Buf_Limit     5MB
        Skip_Long_Lines   On
    
    [FILTER]
        Name              kubernetes
        Match             vllm.*
        Kube_URL          https://kubernetes.default.svc:443
        Kube_CA_File      /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        Kube_Token_File   /var/run/secrets/kubernetes.io/serviceaccount/token
        Merge_Log         On
        Keep_Log          Off
    
    [FILTER]
        Name              nest
        Match             vllm.*
        Operation         lift
        Nested_under      fields
        Add_prefix        field_
    
    [OUTPUT]
        Name              opensearch
        Match             vllm.*
        Host              opensearch-cluster
        Port              9200
        Index             vllm-logs-%Y.%m.%d
        Suppress_Type_Name On
        tls               On
        tls.verify        Off
        HTTP_User         admin
        HTTP_Passwd       ${OPENSEARCH_PASSWORD}
```

---

## 6. 分布式追踪

### 6.1 OpenTelemetry 集成

```python
# OpenTelemetry 配置
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# 配置 Provider
provider = TracerProvider()
processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint="otel-collector:4317")
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

# 自定义追踪
@tracer.start_as_current_span("model_inference")
def inference(request):
    with tracer.start_as_current_span("tokenize") as span:
        span.set_attribute("prompt.length", len(request.prompt))
        tokens = tokenizer.encode(request.prompt)
        span.set_attribute("token.count", len(tokens))
    
    with tracer.start_as_current_span("generate") as span:
        span.set_attribute("model.name", "llama-2-70b")
        span.set_attribute("max_tokens", request.max_tokens)
        
        start = time.time()
        output = model.generate(tokens)
        latency = time.time() - start
        
        span.set_attribute("latency_ms", latency * 1000)
        span.set_attribute("output.tokens", len(output))
    
    return output
```

### 6.2 追踪上下文传播

```python
# 跨服务追踪
import requests
from opentelemetry.propagate import inject

def call_downstream_service(data, context):
    headers = {}
    inject(headers)  # 注入追踪上下文
    
    response = requests.post(
        "http://embedding-service/embed",
        json=data,
        headers=headers
    )
    return response.json()
```

---

## 7. 告警体系

### 7.1 Prometheus 告警规则

```yaml
# alert-rules.yml
groups:
  - name: gpu-alerts
    rules:
      - alert: HighGPUUtilization
        expr: DCGM_FI_DEV_GPU_UTIL > 95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU {{ $labels.gpu }} utilization high"
          description: "GPU {{ $labels.gpu }} on {{ $labels.instance }} has been above 95% for 5 minutes"
      
      - alert: GPUOutOfMemory
        expr: (DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE)) > 0.95
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "GPU {{ $labels.gpu }} running out of memory"
      
      - alert: GPUTemperatureHigh
        expr: DCGM_FI_DEV_GPU_TEMP > 85
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "GPU {{ $labels.gpu }} temperature critical"
      
      - alert: ModelHighLatency
        expr: histogram_quantile(0.99, rate(llm_request_duration_seconds_bucket[5m])) > 5
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "Model {{ $labels.model }} P99 latency high"
      
      - alert: ModelErrorRateHigh
        expr: rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Model {{ $labels.model }} error rate high"

  - name: system-alerts
    rules:
      - alert: NodeDiskPressure
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1
        for: 5m
        labels:
          severity: warning
      
      - alert: NodeMemoryPressure
        expr: (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) < 0.1
        for: 5m
        labels:
          severity: critical
```

### 7.2 告警路由

```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'smtp.example.com:587'
  smtp_from: 'alerts@example.com'

route:
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
      continue: true
    - match:
        severity: warning
      receiver: 'slack-warnings'

receivers:
  - name: 'default'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/...'
        channel: '#alerts'

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: '<pagerduty-key>'
        severity: critical

  - name: 'slack-warnings'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/...'
        channel: '#warnings'
        title: 'Warning: {{ .GroupLabels.alertname }}'
```

---

## 8. 模型性能分析

### 8.1 Profiling 工具

| 工具 | 用途 | 关键功能 |
|------|------|----------|
| **PyTorch Profiler** | 算子级分析 | kernel 时间、内存 |
| **Nsight Systems** | 系统级分析 | GPU/CPU 时间线 |
| **Nsight Compute** | Kernel 级分析 | 占用率、带宽 |
| **vLLM Profiler** | 服务级分析 | 吞吐量、延迟分布 |

### 8.2 性能分析示例

```python
# PyTorch Profiler
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("model_inference"):
        output = model.generate(input_ids)

# 导出结果
prof.export_chrome_trace("trace.json")
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 8.3 性能基准测试

```bash
# vLLM 基准测试
python benchmarks/benchmark_throughput.py \
    --model meta-llama/Llama-2-7b \
    --dataset ShareGPT \
    --num-prompts 1000 \
    --max-model-len 4096 \
    --output-json results.json

# 分析结果
{
  "total_tokens": 500000,
  "total_time": 120.5,
  "throughput": 4149.4,  # tokens/s
  "avg_latency": 250.3,   # ms
  "p99_latency": 850.2    # ms
}
```

---

## 9. 最佳实践

### 9.1 监控 checklist

```markdown
基础设施：
□ GPU 利用率、温度、显存
□ CPU、内存、磁盘
□ 网络带宽、延迟

应用服务：
□ 请求速率、延迟分布
□ 错误率、超时率
□ 队列长度、批处理效率

业务指标：
□ TTFT、TPOT
□ 吞吐量 (tokens/s)
□ 缓存命中率

日志：
□ 结构化日志格式
□ 关键字段完整
□ 日志级别合理

追踪：
□ 关键路径覆盖
□ 上下文传播正确
□ 采样率合理
```

### 9.2 告警原则

```markdown
1. **分层告警**
   - P0 (Critical): 服务不可用，立即处理
   - P1 (High): 功能受损，1 小时内处理
   - P2 (Medium): 性能下降，4 小时内处理
   - P3 (Low): 预警信息，24 小时内处理

2. **告警降噪**
   - 避免重复告警
   - 设置合理的 for 持续时间
   - 使用 group_by 聚合

3. **可操作的告警**
   - 包含明确的处理建议
   - 链接到 runbook
   - 提供相关 dashboard
```

---

## 参考资源

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [NVIDIA DCGM Documentation](https://docs.nvidia.com/datacenter/dcgm/latest/)
- [Fluent Bit Documentation](https://docs.fluentbit.io/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [OpenSearch Documentation](https://opensearch.org/docs/)
