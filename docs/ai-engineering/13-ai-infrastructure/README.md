# 13 - AI 基础设施与部署

> **一句话定位**：从推理优化到生产部署，构建高可用、高性能的 AI 服务基础设施。
>
> #status/canonical #topic/infrastructure #topic/deployment #topic/observability #year/2026

---

## 📂 内容导航

| 文件 | 状态 | 说明 |
|------|------|------|
| `inference-optimization.md` | ✅ canonical | 推理优化技术（量化、缓存、批处理） |
| `serving-systems.md` | ✅ canonical | 模型服务系统（vLLM、TGI、Triton） |
| `cloud-platforms.md` | ✅ canonical | 云平台 AI 服务（AWS/Azure/GCP/阿里云） |
| `kubernetes-deployment.md` | ✅ canonical | K8s AI 工作负载部署 |
| `monitoring-observability.md` | ✅ canonical | 监控与可观测性 |

---

## 🏷️ 本板块标签

- `#topic/infrastructure`
- `#topic/deployment`
- `#topic/observability`
- `#topic/performance`
- `#topic/cloud`

---

## 💡 核心问题清单

1. 如何选择合适的推理引擎（vLLM vs TensorRT-LLM vs TGI）？
2. 量化对模型精度和性能的影响如何平衡？
3. K8s 上 GPU 工作负载的最佳实践是什么？
4. 如何设计高可用的 AI 服务架构？
5. 云平台 GPU 实例如何选型以优化成本？
6. 如何构建完整的 AI 系统可观测性体系？

---

## 🔗 关联板块

- [12-llm-training](../12-llm-training/) — 训练基础设施与分布式训练
- [11-product-practice](../11-product-practice/) — AI 产品实践与指标
- 14-ai-safety — AI 安全与对齐（即将推出）

---

## 📚 扩展阅读

- [vLLM Documentation](https://docs.vllm.ai/)
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server)
- [Kubernetes GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Prometheus + Grafana](https://prometheus.io/docs/visualization/grafana/)
