# 07 - AI 应用部署

> **一句话定位**：从实验室到生产——模型推理优化、服务化与成本控制的工程实践。

---

## 📂 内容导航

| 文件 | 状态 | 说明 |
|------|------|------|
| `inference-optimization.md` | ✅ canonical | 推理加速技术（量化、KV Cache、PagedAttention、服务框架） |
| `model-serving.md` | 🚧 draft | 模型服务化框架（vLLM、TGI、TensorRT-LLM） |
| `api-design.md` | 🚧 draft | LLM API 设计最佳实践 |
| `cost-optimization.md` | 🚧 draft | Token 成本分析、缓存策略、路由优化 |
| `scaling-strategies.md` | 🚧 draft | 水平扩展、负载均衡、弹性伸缩 |
| `edge-deployment.md` | 🚧 draft | 边缘部署与端侧推理（ONNX、Core ML） |

---

## 🏷️ 本板块标签

- `#topic/deployment`
- `#topic/inference`
- `#topic/optimization`
- `#topic/scaling`

---

## 💡 核心问题清单

1. 量化对模型效果的影响如何评估？
2. vLLM 的 PagedAttention 原理是什么？
3. 如何设计高可用的 LLM 服务架构？
4. Token 成本如何精确计算与优化？
