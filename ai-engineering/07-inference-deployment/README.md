# 07-inference-deployment

> AI 模型推理与部署：从量化压缩到生产级服务，构建高性能、低成本的模型推理系统。

---

## 目录

| 文件 | 摘要 | 状态 |
|------|------|------|
| [`model-quantization.md`](./model-quantization.md) | 模型量化：INT8、INT4、GPTQ、AWQ、GGUF 原理与实践 | #status/draft |
| [`inference-optimization.md`](./inference-optimization.md) | 推理优化：KV Cache、FlashAttention、投机解码、连续批处理 | #status/draft |
| [`model-serving.md`](./model-serving.md) | 模型服务化：vLLM、TensorRT-LLM、TGI、SGLang 对比与选型 | #status/draft |

## 规划内容

- [x] 模型量化技术（INT8/INT4/GPTQ/AWQ）
- [x] 推理优化技术（KV Cache、FlashAttention、投机解码）
- [x] 模型服务化框架对比
- [ ] 模型并行与分布式推理
- [ ] 边缘设备部署（移动端、嵌入式）
- [ ] 推理成本优化与自动扩缩容
