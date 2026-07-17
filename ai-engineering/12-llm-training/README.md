# 12 - LLM 训练与预训练

> **一句话定位**：从零开始训练大语言模型——数据准备、预训练、持续预训练、后训练的全流程指南。
>
> #status/canonical #topic/training #year/2026

---

## 📂 内容导航

| 文件 | 状态 | 说明 |
|------|------|------|
| `pretraining.md` | ✅ canonical | 预训练基础：数据、模型、训练流程 |
| `data-preparation.md` | ✅ canonical | 数据准备与清洗策略 |
| `training-infrastructure.md` | ✅ canonical | 训练基础设施与分布式训练 |
| `continual-pretraining.md` | ✅ canonical | 持续预训练与领域适配 |
| `scaling-laws.md` | ✅ canonical | 扩展定律与计算效率优化 |
| `post-training.md` | ✅ canonical | 后训练全流程：SFT → RLHF → DPO/ORPO/KTO → GRPO → 推理增强 |
| `pretraining-posttraining-pipeline.md` | ✅ canonical | 预训练与后训练全流程鸟瞰：概念对比、经济分析 |

---

## 🏷️ 本板块标签

- `#topic/training`
- `#topic/pretraining`
- `#topic/data`
- `#topic/distributed`
- `#topic/scaling`

---

## 💡 核心问题清单

1. 预训练需要多少数据？如何评估数据质量？
2. 分布式训练有哪些并行策略？如何选择？
3. 持续预训练与微调的区别是什么？
4. Scaling Laws 如何指导模型设计？
5. 如何优化训练效率和降低成本？

---

## 📚 参考资源

- [Hugging Face: Training LLMs](https://huggingface.co/docs/transformers/training)
- [DeepSpeed: Training Optimization](https://www.deepspeed.ai/)
- [Megatron-LM: NVIDIA's Training Framework](https://github.com/NVIDIA/Megatron-LM)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)

---

*最后更新：2026-07-17*
