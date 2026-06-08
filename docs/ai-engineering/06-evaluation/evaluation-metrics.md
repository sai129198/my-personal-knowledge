# 模型评估指标详解

> **一句话定位**：从困惑度到 LLM-as-a-Judge，系统梳理 AI 模型评估的核心指标与适用场景。
>
> #status/draft #topic/evaluation #topic/llm #year/2026

---

## 一、基础概念：信息论视角

### 1.1 熵 (Entropy)

熵衡量的是"平均信息量"，由香农提出。对于语言模型而言：

- **高熵** → 模型对下一个词预测很"不确定"
- **低熵** → 模型对下一个词预测很"自信"

人类语言的熵约为 **1.0-1.5 bits/字符**（英文），这是理论下限。

### 1.2 交叉熵 (Cross-Entropy)

衡量模型分布与真实分布的差异：

$$H(P, Q) = -\sum_{x} P(x) \log Q(x)$$

其中 $P$ 是真实分布，$Q$ 是模型预测分布。**交叉熵越低，模型越好**。

### 1.3 困惑度 (Perplexity, PPL)

**最经典的语言模型评估指标**。

$$PPL(X) = \exp\left(-\frac{1}{t}\sum_{i=1}^{t} \log p_\theta(x_i|x_{<i})\right)$$

**直观理解**：模型面临的一个"分支选择"的平均数量。

- PPL = 100 → 相当于每次面对100个等概率选择
- PPL = 2 → 相当于每次面对2个选择（接近确定）

**⚠️ 关键注意事项**：
1. **分词方式影响巨大**：Word-level vs Character-level vs Subword-level 不可直接比较
2. **仅适用于自回归模型**：BERT 等掩码模型不适用
3. **与 BPC 的关系**：$PPL = 2^{BPC}$（当使用字符级分词时）

---

## 二、生成任务指标

### 2.1 BLEU (Bilingual Evaluation Understudy)

**用途**：机器翻译、文本生成质量评估

**核心思想**：比较候选文本与参考文本的 n-gram 重叠度

$$BLEU = BP \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

其中：
- $p_n$：n-gram 精确率
- $BP$：简短惩罚因子（Brevity Penalty）
- $w_n$：权重（通常 $N=4$，均匀权重）

**优点**：
- 计算快速，可复现
- 与人工评估有一定相关性

**缺点**：
- ❌ 不考虑语义，只看重词重叠
- ❌ 对同义词不敏感（"good" vs "great" 算不同）
- ❌ 需要参考文本，且质量依赖参考文本数量

**适用场景**：机器翻译、摘要生成（有参考文本时）

### 2.2 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**用途**：文本摘要评估

**核心思想**：基于召回率，衡量候选摘要与参考摘要的重叠

| 变体 | 计算方式 | 特点 |
|------|----------|------|
| ROUGE-1 | 1-gram 重叠 | 最基础 |
| ROUGE-2 | 2-gram 重叠 | 考虑局部词序 |
| ROUGE-L | 最长公共子序列 (LCS) | 考虑词序，更灵活 |
| ROUGE-SU | Skip-bigram + Unigram | 允许词间有间隔 |

**与 BLEU 的区别**：
- BLEU 偏向精确率（Precision）
- ROUGE 偏向召回率（Recall）

### 2.3 METEOR

**改进点**：解决 BLEU 的不足

- 支持**同义词匹配**（WordNet）
- 支持**词干提取**（stemming）
- 引入召回率，F-score 平衡

$$METEOR = F_{mean} \times (1 - Penalty)$$

**适用场景**：机器翻译（比 BLEU 更接近人工判断）

---

## 三、检索任务指标

### 3.1 Precision@K

$$Precision@K = \frac{True\ Positives@K}{K}$$

**含义**：前 K 个结果中，有多少是相关的。

**适用场景**：搜索引擎、推荐系统（用户只看前 K 个结果）

### 3.2 Recall@K

$$Recall@K = \frac{True\ Positives@K}{Total\ Relevant\ Documents}$$

**含义**：所有相关文档中，有多少被召回在前 K 个结果里。

### 3.3 Mean Reciprocal Rank (MRR)

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

**含义**：第一个相关结果的排名的倒数平均值。

**适用场景**：问答系统、对话系统（用户通常只关心第一个正确答案）

### 3.4 Normalized Discounted Cumulative Gain (NDCG)

**核心思想**：
1. 相关文档有**分级相关性**（不是二元的）
2. 排名越靠前，权重越高（Discount）

$$DCG@K = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

$$NDCG@K = \frac{DCG@K}{IDCG@K}$$

**适用场景**：搜索引擎、推荐系统（结果有相关性分级时）

### 3.5 Mean Average Precision (MAP)

$$AP = \sum_{k} Precision@k \times \Delta Recall@k$$

$$MAP = \frac{1}{|Q|} \sum_{q} AP(q)$$

**适用场景**：信息检索（需要综合考虑 Precision 和 Recall）

---

## 四、RAG 系统评估指标

### 4.1 检索组件指标

| 指标 | 含义 | 使用场景 |
|------|------|----------|
| Context Precision | 检索到的上下文中有多少是相关的 | 评估检索器质量 |
| Context Recall | 相关文档有多少被检索到 | 评估召回能力 |
| Context Relevance | 上下文与问题的相关性 | 端到端评估 |

### 4.2 生成组件指标

| 指标 | 含义 | 使用场景 |
|------|------|----------|
| Faithfulness | 生成内容是否忠实于检索到的上下文 | 检测幻觉 |
| Answer Relevance | 答案是否回答了问题 | 评估回答质量 |
| Answer Correctness | 答案是否正确（有标准答案时） | 问答系统 |

### 4.3 RAGAS 框架

**RAGAS** 是专门为 RAG 设计的评估框架：

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall
)

result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_relevancy, context_recall]
)
```

**特点**：
- 无需人工标注参考答案
- 使用 LLM 作为评判者（LLM-as-a-Judge）
- 适合持续监控 RAG 系统性能

---

## 五、LLM-as-a-Judge：现代评估范式

### 5.1 为什么传统指标不够？

| 问题 | 说明 |
|------|------|
| 语义理解 | BLEU/ROUGE 无法捕捉语义相似性 |
| 开放性生成 | 没有唯一标准答案 |
| 多维度质量 | 需要同时评估正确性、流畅性、安全性等 |
| 成本 | 人工评估昂贵且不可扩展 |

### 5.2 LLM-as-a-Judge 原理

**核心思想**：用更强的 LLM（如 GPT-4）来评估较弱 LLM 的输出。

**常见方法**：

#### G-Eval (LLM Evaluation with GPT-4)

```python
# 定义评估维度
EVALUATION_PROMPT = """
You will be given one summary written for an article. 
Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions very carefully.

Evaluation Criteria:
Coherence (1-5) - the collective quality of all sentences.

Evaluation Steps:
1. Read the article carefully and identify the main topic and key points.
2. Read the summary and compare it to the article. 
3. Assign a score for coherence.

Article: {article}
Summary: {summary}

Score:"""
```

**优点**：
- 可评估任意维度（只需定义标准）
- 与人类判断相关性高（>0.8）

**缺点**：
- 需要调用 API，有成本
- 存在位置偏见（prefer 第一个选项）
- 可能过于宽容（对相似模型的输出）

### 5.3 减少偏见的技术

| 技术 | 说明 |
|------|------|
| **交换位置** | 交换候选答案顺序，取平均 |
| **多轮评估** | 多次采样，取多数投票 |
| **明确标准** | 提供详细的评分 rubric |
| **参考对比** | 与标准答案或人类答案对比 |

---

## 六、指标选择决策树

```
你有标准参考答案吗？
├── 有 → 是生成任务吗？
│   ├── 是 → 需要语义相似性吗？
│   │   ├── 是 → LLM-as-a-Judge / BERTScore
│   │   └── 否 → BLEU / ROUGE / METEOR
│   └── 否 → 检索任务？
│       ├── 是 → Precision@K / Recall@K / NDCG / MRR
│       └── 否 → 分类任务？
│           └── 是 → Accuracy / F1 / AUC-ROC
└── 无 → 是 RAG 系统吗？
    ├── 是 → RAGAS (Faithfulness, Relevance)
    └── 否 → 开放式生成？
        ├── 是 → LLM-as-a-Judge
        └── 否 → 人工评估
```

---

## 七、实践建议

### 7.1 不要依赖单一指标

**反例**：BLEU 高 ≠ 质量好

```
参考：The cat sat on the mat.
候选1：The cat sat on the mat. (BLEU=1.0, 完美)
候选2：On the mat sat the cat. (BLEU≈0.3, 但语义等价)
候选3：The the the the the. (BLEU可能很高, 但毫无意义)
```

### 7.2 建立评估基线

1. **人类评估分数**（黄金标准）
2. **简单基线模型**（如随机选择、TF-IDF）
3. **当前生产模型**（对比改进）

### 7.3 持续监控指标

| 监控项 | 方法 |
|--------|------|
| 数据漂移 | 输入分布变化检测 |
| 概念漂移 | 模型输出分布变化 |
| 性能退化 | 关键指标趋势监控 |
| 异常样本 | 离群点检测 |

---

## 八、工具推荐

| 工具 | 用途 | 链接 |
|------|------|------|
| **DeepEval** | LLM 评估框架（开源） | https://github.com/confident-ai/deepeval |
| **RAGAS** | RAG 评估 | https://docs.ragas.io/ |
| **OpenAI Evals** | 基准测试注册表 | https://github.com/openai/evals |
| **BERTScore** | 语义相似度 | https://github.com/Tiiiger/bert_score |
| **sacreBLEU** | 标准化 BLEU | https://github.com/mjpost/sacrebleu |

---

## 💡 我的思考

### 关键洞察

1. **指标是手段，不是目的**：选择指标要服务于业务目标，不要为了指标而优化
2. **LLM-as-a-Judge 是趋势**：但需要注意偏见和成本，建议与传统指标结合使用
3. **RAG 评估需要分层**：检索和生成要分开评估，才能定位问题

### 常见陷阱

- ❌ 用 BLEU 评估对话质量（对话没有标准答案）
- ❌ 用 PPL 比较不同分词器的模型
- ❌ 忽视指标之间的相关性（高 BLEU 不一定高人工分）

### 下一步学习

- [ ] 深入研究 RAGAS 源码实现
- [ ] 实践 LLM-as-a-Judge 的位置偏见实验
- [ ] 建立自己业务的评估指标体系

---

## 参考来源

1. Hugging Face - Perplexity of fixed-length models (2024)
2. The Gradient - Understanding Evaluation Metrics for Language Models (2019)
3. Pinecone - RAG Evaluation: Don't let customers tell you first (2024)
4. Confident AI - LLM Evaluation Metrics: The Ultimate Guide (2024)
5. Hugging Face Cookbook - RAG Evaluation (2024)
6. OpenAI Evals - Framework for evaluating LLMs (GitHub)

---

*最后更新：2026-06-08*
