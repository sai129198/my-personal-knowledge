# 08 - 数据工程

#topic/data-engineering #year/2026 #status/canonical

> **一句话定位**：AI 系统的数据基础——从原始数据到高质量训练/检索数据的全流程工程实践。

---

## 目录

| 文件 | 状态 | 一句话摘要 |
|------|------|-----------|
| [data-preprocessing.md](data-preprocessing.md) | #status/canonical | 数据清洗、去重、质量评估与敏感信息处理 |
| [embedding-models.md](embedding-models.md) | #status/canonical | 嵌入模型选型、评估与微调策略 |
| [vector-databases.md](vector-databases.md) | #status/canonical | 向量数据库选型、索引与性能优化 |
| [data-labeling.md](data-labeling.md) | #status/canonical | 标注流程设计、质量控制与主动学习 |
| [synthetic-data.md](synthetic-data.md) | #status/canonical | 合成数据生成、验证与混合训练策略 |
| [data-pipeline.md](data-pipeline.md) | #status/canonical | 数据流水线设计、编排工具与质量监控 |

---

## 核心概念

### 数据质量金字塔

```
        ┌─────────────┐
        │   价值密度   │  ← 数据能带来的业务价值
        ├─────────────┤
        │   可解释性   │  ← 数据来源、处理过程可追溯
        ├─────────────┤
        │   安全性    │  ← PII 脱敏、合规处理
        ├─────────────┤
        │   多样性    │  ← 覆盖不同场景、群体、语言
        ├─────────────┤
        │   准确性    │  ← 标注正确、内容真实
        ├─────────────┤
        │   完整性    │  ← 字段齐全、格式统一
        ├─────────────┤
        │   清洁度    │  ← 去重、去噪、去垃圾
        └─────────────┘
```

### 数据工程 Pipeline

```
数据采集 → 清洗 → 标注 → 增强 → 验证 → 版本管理 → 训练/检索
   ↑___________________________________________________↓
                    监控与反馈
```

---

## 关键决策矩阵

| 场景 | 推荐方案 | 注意事项 |
|------|----------|----------|
| 文本清洗 | 规则 + ML 混合 | 领域特定规则优先 |
| 嵌入模型 | BGE-M3 / E5 | 在业务数据上测试 |
| 向量数据库 | Milvus / Qdrant | 考虑规模与延迟要求 |
| 数据标注 | LLM 辅助 + 人工审核 | 质量 > 数量 |
| 合成数据 | LLM 生成 + 规则约束 | 验证真实性 |
| Pipeline 编排 | Prefect / Airflow | 从简单开始 |

---

## 与其他板块的关系

- **01-foundations** → 理解数据对模型性能的影响
- **04-rag-architecture** → Embedding 与向量数据库的应用
- **09-security** → 数据隐私保护与合规
- **12-llm-training** → 训练数据的准备与质量

---

## 待办

- [ ] 补充实时数据流处理（Kafka / Flink）
- [ ] 补充数据血缘追踪工具（OpenLineage）
- [ ] 补充多模态数据处理（图像、音频标注）

---

*最后更新：2026-06-12*
