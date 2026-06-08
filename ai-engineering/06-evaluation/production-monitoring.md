# 生产环境监控与评估

> **一句话定位**：在生产环境中持续监控模型性能，及时发现并解决问题。
>
> #status/canonical #topic/evaluation #topic/monitoring #topic/production #year/2026

---

## 1. 监控体系设计

### 1.1 监控维度

```
生产监控体系
├── 性能监控
│   ├── 延迟 (Latency)
│   ├── 吞吐量 (Throughput)
│   ├── 错误率 (Error Rate)
│   └── 资源使用 (CPU/Memory/GPU)
├── 质量监控
│   ├── 输出质量 (Output Quality)
│   ├── 相关性 (Relevance)
│   ├── 安全性 (Safety)
│   └── 用户满意度 (User Satisfaction)
├── 业务监控
│   ├── 用户活跃度 (DAU/MAU)
│   ├── 功能使用率 (Feature Usage)
│   ├── 转化率 (Conversion Rate)
│   └── 留存率 (Retention Rate)
└── 成本监控
    ├── Token 消耗 (Token Usage)
    ├── API 调用成本 (API Cost)
    ├── 基础设施成本 (Infrastructure Cost)
    └── 单位请求成本 (Cost Per Request)
```

### 1.2 关键指标

| 指标 | 说明 | 告警阈值 | 监控频率 |
|------|------|----------|----------|
| **P50/P99 延迟** | 响应时间 | P99 > 2s | 实时 |
| **错误率** | 请求失败比例 | > 1% | 实时 |
| **输出质量分** | 生成质量评分 | < 3.5/5 | 每小时 |
| **安全违规率** | 有害内容比例 | > 0.1% | 实时 |
| **Token 消耗** | 每请求 Token 数 | > 平均 150% | 每小时 |
| **用户满意度** | 用户反馈评分 | < 4.0/5 | 每日 |

---

## 2. 实时监控

### 2.1 指标收集

```python
from prometheus_client import Counter, Histogram, Gauge, Info
import time

# 定义指标
REQUEST_COUNT = Counter('llm_requests_total', 'Total requests', ['model', 'status'])
REQUEST_LATENCY = Histogram('llm_request_duration_seconds', 'Request latency', ['model'])
TOKEN_USAGE = Counter('llm_tokens_total', 'Token usage', ['model', 'type'])
ACTIVE_SESSIONS = Gauge('llm_active_sessions', 'Active sessions')
OUTPUT_QUALITY = Gauge('llm_output_quality_score', 'Output quality score', ['model'])
SAFETY_VIOLATIONS = Counter('llm_safety_violations_total', 'Safety violations', ['type'])

class MonitoredLLMService:
    def __init__(self, model):
        self.model = model
    
    async def generate(self, prompt: str, **kwargs) -> Dict:
        start_time = time.time()
        
        try:
            # 生成响应
            response = await self.model.generate(prompt, **kwargs)
            
            # 记录成功
            REQUEST_COUNT.labels(model=self.model.name, status='success').inc()
            
            # 记录 Token 使用
            input_tokens = len(prompt.split())  # 简化计算
            output_tokens = len(response.split())
            TOKEN_USAGE.labels(model=self.model.name, type='input').inc(input_tokens)
            TOKEN_USAGE.labels(model=self.model.name, type='output').inc(output_tokens)
            
            # 记录质量
            quality_score = self._evaluate_quality(prompt, response)
            OUTPUT_QUALITY.labels(model=self.model.name).set(quality_score)
            
            return {
                'response': response,
                'quality_score': quality_score,
                'tokens': {'input': input_tokens, 'output': output_tokens}
            }
            
        except Exception as e:
            # 记录失败
            REQUEST_COUNT.labels(model=self.model.name, status='error').inc()
            raise
            
        finally:
            # 记录延迟
            latency = time.time() - start_time
            REQUEST_LATENCY.labels(model=self.model.name).observe(latency)
    
    def _evaluate_quality(self, prompt: str, response: str) -> float:
        """评估输出质量"""
        # 使用轻量级质量评估
        # 实际应用中可以使用 LLM-as-Judge 或规则引擎
        return 4.0  # 简化示例
```

### 2.2 日志追踪

```python
import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# 配置日志
logger = structlog.get_logger()

# 配置追踪
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

class TracedLLMService:
    def __init__(self, model):
        self.model = model
    
    async def generate(self, request_id: str, prompt: str, **kwargs) -> Dict:
        with tracer.start_as_current_span("llm_generate") as span:
            # 设置追踪属性
            span.set_attribute("request.id", request_id)
            span.set_attribute("prompt.length", len(prompt))
            span.set_attribute("model.name", self.model.name)
            
            logger.info(
                "llm_request_started",
                request_id=request_id,
                model=self.model.name,
                prompt_length=len(prompt)
            )
            
            try:
                # 生成响应
                with tracer.start_span("model_inference"):
                    response = await self.model.generate(prompt, **kwargs)
                
                # 安全检测
                with tracer.start_span("safety_check"):
                    safety_result = self._check_safety(response)
                    if not safety_result.is_safe:
                        SAFETY_VIOLATIONS.labels(type=safety_result.violation_type).inc()
                        logger.warning(
                            "safety_violation_detected",
                            request_id=request_id,
                            violation_type=safety_result.violation_type
                        )
                
                span.set_attribute("response.length", len(response))
                span.set_attribute("safety.passed", safety_result.is_safe)
                
                logger.info(
                    "llm_request_completed",
                    request_id=request_id,
                    response_length=len(response),
                    safety_passed=safety_result.is_safe
                )
                
                return {
                    'response': response,
                    'safety_result': safety_result
                }
                
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                
                logger.error(
                    "llm_request_failed",
                    request_id=request_id,
                    error=str(e)
                )
                raise
```

---

## 3. 质量监控

### 3.1 自动质量评估

```python
class ProductionQualityMonitor:
    """
    生产环境质量监控
    """
    
    def __init__(self, evaluator, sampling_rate: float = 0.1):
        self.evaluator = evaluator
        self.sampling_rate = sampling_rate
        self.quality_history = []
    
    async def monitor_request(self, request: Dict, response: Dict) -> Dict:
        """
        监控单个请求
        """
        # 采样评估
        if random.random() > self.sampling_rate:
            return None
        
        # 评估质量
        quality_result = await self.evaluator.evaluate(
            input_text=request['prompt'],
            output=response['response']
        )
        
        # 记录历史
        self.quality_history.append({
            'timestamp': datetime.now(),
            'request_id': request['id'],
            'quality_score': quality_result['overall_score'],
            'dimensions': quality_result['dimensions']
        })
        
        # 检查是否需要告警
        if quality_result['overall_score'] < 3.0:
            await self._send_quality_alert(request, quality_result)
        
        return quality_result
    
    def get_quality_trends(self, window_hours: int = 24) -> Dict:
        """
        获取质量趋势
        """
        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent = [q for q in self.quality_history if q['timestamp'] > cutoff]
        
        if not recent:
            return {}
        
        scores = [q['quality_score'] for q in recent]
        
        return {
            'window_hours': window_hours,
            'sample_count': len(recent),
            'mean_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'trend': self._calculate_trend(scores)
        }
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """计算趋势"""
        if len(scores) < 10:
            return "insufficient_data"
        
        # 比较前半段和后半段
        mid = len(scores) // 2
        first_half = sum(scores[:mid]) / mid
        second_half = sum(scores[mid:]) / (len(scores) - mid)
        
        diff = second_half - first_half
        if diff > 0.2:
            return "improving"
        elif diff < -0.2:
            return "degrading"
        else:
            return "stable"
```

### 3.2 用户反馈收集

```python
class UserFeedbackCollector:
    """
    用户反馈收集器
    """
    
    def __init__(self, storage):
        self.storage = storage
    
    async def collect_feedback(self, request_id: str, feedback: Dict):
        """
        收集用户反馈
        
        Args:
            feedback: {
                "rating": 1-5,
                "category": "helpful" | "unhelpful" | "harmful" | "other",
                "comment": "optional text",
                "expected_response": "what user expected"
            }
        """
        feedback_record = {
            'request_id': request_id,
            'timestamp': datetime.now(),
            **feedback
        }
        
        await self.storage.store(feedback_record)
        
        # 实时分析
        if feedback['category'] == 'harmful':
            await self._handle_harmful_feedback(request_id, feedback)
        
        if feedback['rating'] <= 2:
            await self._handle_negative_feedback(request_id, feedback)
    
    async def get_feedback_stats(self, time_range: str = '7d') -> Dict:
        """
        获取反馈统计
        """
        feedbacks = await self.storage.query(time_range=time_range)
        
        if not feedbacks:
            return {}
        
        ratings = [f['rating'] for f in feedbacks if 'rating' in f]
        categories = [f['category'] for f in feedbacks if 'category' in f]
        
        from collections import Counter
        category_counts = Counter(categories)
        
        return {
            'total_feedback': len(feedbacks),
            'average_rating': sum(ratings) / len(ratings) if ratings else 0,
            'rating_distribution': {
                i: ratings.count(i) for i in range(1, 6)
            },
            'category_distribution': dict(category_counts),
            'negative_feedback_rate': category_counts.get('unhelpful', 0) / len(feedbacks)
        }
```

---

## 4. 告警系统

### 4.1 告警规则

```python
class AlertManager:
    """
    告警管理器
    """
    
    def __init__(self):
        self.rules = []
        self.alert_history = []
    
    def add_rule(self, name: str, condition: Callable, severity: str, channels: List[str]):
        """
        添加告警规则
        
        Args:
            name: 规则名称
            condition: 条件函数
            severity: 严重级别 (critical, warning, info)
            channels: 通知渠道
        """
        self.rules.append({
            'name': name,
            'condition': condition,
            'severity': severity,
            'channels': channels,
            'last_triggered': None,
            'cooldown_minutes': 30
        })
    
    async def check_alerts(self, metrics: Dict):
        """
        检查告警
        """
        for rule in self.rules:
            # 检查冷却期
            if rule['last_triggered']:
                elapsed = (datetime.now() - rule['last_triggered']).minutes
                if elapsed < rule['cooldown_minutes']:
                    continue
            
            # 检查条件
            if rule['condition'](metrics):
                alert = {
                    'rule': rule['name'],
                    'severity': rule['severity'],
                    'timestamp': datetime.now(),
                    'metrics': metrics
                }
                
                await self._send_alert(alert, rule['channels'])
                
                rule['last_triggered'] = datetime.now()
                self.alert_history.append(alert)
    
    async def _send_alert(self, alert: Dict, channels: List[str]):
        """发送告警"""
        for channel in channels:
            if channel == 'slack':
                await self._send_slack_alert(alert)
            elif channel == 'email':
                await self._send_email_alert(alert)
            elif channel == 'pagerduty':
                await self._send_pagerduty_alert(alert)
    
    def setup_default_rules(self):
        """设置默认告警规则"""
        # 延迟告警
        self.add_rule(
            name='high_latency',
            condition=lambda m: m.get('p99_latency', 0) > 5.0,
            severity='warning',
            channels=['slack']
        )
        
        # 错误率告警
        self.add_rule(
            name='high_error_rate',
            condition=lambda m: m.get('error_rate', 0) > 0.05,
            severity='critical',
            channels=['slack', 'pagerduty']
        )
        
        # 质量下降告警
        self.add_rule(
            name='quality_degradation',
            condition=lambda m: m.get('quality_score', 5) < 3.0,
            severity='warning',
            channels=['slack']
        )
        
        # 安全告警
        self.add_rule(
            name='safety_violation',
            condition=lambda m: m.get('safety_violation_rate', 0) > 0.01,
            severity='critical',
            channels=['slack', 'pagerduty', 'email']
        )
```

### 4.2 告警模板

```python
class AlertTemplates:
    """
    告警消息模板
    """
    
    SLACK_TEMPLATE = """
    🚨 *{severity} Alert: {rule_name}*
    
    • Time: {timestamp}
    • Details: {details}
    • Current Value: {current_value}
    • Threshold: {threshold}
    
    Please investigate immediately.
    """
    
    EMAIL_TEMPLATE = """
    Subject: [{severity}] LLM Service Alert: {rule_name}
    
    Alert Details:
    - Rule: {rule_name}
    - Severity: {severity}
    - Time: {timestamp}
    - Current Metrics:
      {metrics}
    
    Recommended Actions:
    {actions}
    """
    
    @staticmethod
    def format_alert(alert: Dict, template: str) -> str:
        """格式化告警消息"""
        return template.format(
            severity=alert['severity'].upper(),
            rule_name=alert['rule'],
            timestamp=alert['timestamp'],
            details=json.dumps(alert['metrics'], indent=2),
            current_value=alert['metrics'].get('current_value', 'N/A'),
            threshold=alert['metrics'].get('threshold', 'N/A'),
            metrics=json.dumps(alert['metrics'], indent=2),
            actions='- Check service logs\n- Review recent changes\n- Consider rollback if critical'
        )
```

---

## 5. 持续优化

### 5.1 A/B 测试

```python
class LLMABTest:
    """
    LLM A/B 测试
    """
    
    def __init__(self, model_a, model_b, traffic_split: float = 0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.results = {'a': [], 'b': []}
    
    async def route_request(self, request: Dict) -> Dict:
        """
        路由请求到不同模型
        """
        import random
        
        if random.random() < self.traffic_split:
            variant = 'a'
            model = self.model_a
        else:
            variant = 'b'
            model = self.model_b
        
        start_time = time.time()
        
        try:
            response = await model.generate(request['prompt'])
            status = 'success'
        except Exception as e:
            response = None
            status = 'error'
        
        latency = time.time() - start_time
        
        result = {
            'variant': variant,
            'request': request,
            'response': response,
            'status': status,
            'latency': latency
        }
        
        self.results[variant].append(result)
        
        return result
    
    def analyze_results(self) -> Dict:
        """
        分析 A/B 测试结果
        """
        import scipy.stats as stats
        
        # 提取指标
        latencies_a = [r['latency'] for r in self.results['a'] if r['status'] == 'success']
        latencies_b = [r['latency'] for r in self.results['b'] if r['status'] == 'success']
        
        error_rate_a = sum(1 for r in self.results['a'] if r['status'] == 'error') / len(self.results['a'])
        error_rate_b = sum(1 for r in self.results['b'] if r['status'] == 'error') / len(self.results['b'])
        
        # 统计检验
        if latencies_a and latencies_b:
            t_stat, p_value = stats.ttest_ind(latencies_a, latencies_b)
        else:
            t_stat, p_value = None, None
        
        return {
            'sample_size': {
                'a': len(self.results['a']),
                'b': len(self.results['b'])
            },
            'latency': {
                'a_mean': sum(latencies_a) / len(latencies_a) if latencies_a else 0,
                'b_mean': sum(latencies_b) / len(latencies_b) if latencies_b else 0,
                'p_value': p_value
            },
            'error_rate': {
                'a': error_rate_a,
                'b': error_rate_b
            },
            'recommendation': 'model_b' if (sum(latencies_b) / len(latencies_b) if latencies_b else float('inf')) < 
                                          (sum(latencies_a) / len(latencies_a) if latencies_a else float('inf')) else 'model_a'
        }
```

### 5.2 模型退化检测

```python
class ModelDegradationDetector:
    """
    模型退化检测
    """
    
    def __init__(self, baseline_metrics: Dict):
        self.baseline = baseline_metrics
        self.current_window = []
        self.window_size = 100
    
    async def add_sample(self, metrics: Dict):
        """
        添加样本
        """
        self.current_window.append(metrics)
        
        if len(self.current_window) >= self.window_size:
            await self._check_degradation()
            self.current_window = []
    
    async def _check_degradation(self):
        """
        检查是否退化
        """
        # 计算当前窗口指标
        current = {
            'quality_score': sum(m['quality_score'] for m in self.current_window) / len(self.current_window),
            'error_rate': sum(1 for m in self.current_window if m['status'] == 'error') / len(self.current_window),
            'latency': sum(m['latency'] for m in self.current_window) / len(self.current_window)
        }
        
        # 与基线比较
        alerts = []
        
        if current['quality_score'] < self.baseline['quality_score'] * 0.9:
            alerts.append({
                'type': 'quality_degradation',
                'current': current['quality_score'],
                'baseline': self.baseline['quality_score']
            })
        
        if current['error_rate'] > self.baseline['error_rate'] * 2:
            alerts.append({
                'type': 'error_rate_increase',
                'current': current['error_rate'],
                'baseline': self.baseline['error_rate']
            })
        
        if current['latency'] > self.baseline['latency'] * 1.5:
            alerts.append({
                'type': 'latency_increase',
                'current': current['latency'],
                'baseline': self.baseline['latency']
            })
        
        if alerts:
            await self._send_degradation_alert(alerts)
    
    async def _send_degradation_alert(self, alerts: List[Dict]):
        """发送退化告警"""
        logger.warning("Model degradation detected", alerts=alerts)
        # 发送通知
```

---

## 参考资源

- [ML Monitoring: A Comprehensive Guide](https://www.evidentlyai.com/ml-system-design)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/)
- [LLM Monitoring with LangSmith](https://docs.smith.langchain.com/)
