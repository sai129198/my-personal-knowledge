# RAG 系统评估指标与方法

> **一句话定位**：建立全面的 RAG 评估体系，量化检索质量和生成质量。
>
> #status/canonical #topic/rag #topic/evaluation #year/2026

---

## 1. 评估框架

### 1.1 RAG 评估维度

```
RAG 评估
├── 检索质量 (Retrieval Quality)
│   ├── 相关性 (Relevance)
│   ├── 召回率 (Recall)
│   └── 多样性 (Diversity)
├── 生成质量 (Generation Quality)
│   ├── 忠实度 (Faithfulness)
│   ├── 答案相关性 (Answer Relevance)
│   └── 流畅度 (Fluency)
└── 端到端质量 (End-to-End)
    ├── 准确性 (Accuracy)
    ├── 有用性 (Helpfulness)
    └── 延迟 (Latency)
```

### 1.2 评估方法分类

| 方法 | 说明 | 成本 | 可靠性 |
|------|------|------|--------|
| **人工评估** | 人类标注员评分 | 高 | 最高 |
| **自动评估** | 基于规则的指标 | 低 | 中等 |
| **模型评估** | 用 LLM 评估 | 中 | 较高 |
| **混合评估** | 自动 + 人工抽样 | 中 | 高 |

---

## 2. 检索质量评估

### 2.1 经典指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **Precision@K** | relevant_k / k | Top-K 中相关文档比例 |
| **Recall@K** | relevant_k / total_relevant | 相关文档被找回的比例 |
| **F1@K** | 2 * P * R / (P + R) | Precision 和 Recall 调和平均 |
| **MRR** | 1 / rank_first_relevant | 第一个相关文档排名的倒数 |
| **MAP** | mean(AP) | 平均精度均值 |
| **NDCG@K** | DCG_k / IDCG_k | 考虑排序位置的加权指标 |

```python
def calculate_retrieval_metrics(retrieved_docs, relevant_docs, k=5):
    """
    计算检索指标
    """
    retrieved_k = set(retrieved_docs[:k])
    relevant = set(relevant_docs)
    
    # Precision@K
    precision = len(retrieved_k & relevant) / k
    
    # Recall@K
    recall = len(retrieved_k & relevant) / len(relevant) if relevant else 0
    
    # F1@K
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # MRR
    mrr = 0
    for i, doc in enumerate(retrieved_docs, 1):
        if doc in relevant:
            mrr = 1 / i
            break
    
    return {
        'precision@k': precision,
        'recall@k': recall,
        'f1@k': f1,
        'mrr': mrr
    }
```

### 2.2 上下文相关性

```python
def context_relevance(query, retrieved_contexts, model):
    """
    评估检索上下文与查询的相关性
    """
    query_emb = model.encode(query)
    
    scores = []
    for ctx in retrieved_contexts:
        ctx_emb = model.encode(ctx)
        similarity = cosine_similarity(query_emb, ctx_emb)
        scores.append(similarity)
    
    return {
        'mean_relevance': np.mean(scores),
        'min_relevance': np.min(scores),
        'max_relevance': np.max(scores)
    }
```

---

## 3. 生成质量评估

### 3.1 忠实度（Faithfulness）

评估生成答案是否忠实于检索到的上下文。

```python
def faithfulness_score(answer, contexts, model):
    """
    评估答案忠实度
    """
    prompt = f"""
    基于以下上下文，评估答案是否忠实于上下文信息。
    
    上下文：
    {contexts}
    
    答案：{answer}
    
    请检查：
    1. 答案中的每个事实是否都能在上下文中找到支持？
    2. 答案是否包含上下文中没有的信息（幻觉）？
    
    评分（1-5）：
    1 = 大量幻觉，完全不忠实
    3 = 部分忠实，有一些幻觉
    5 = 完全忠实，所有信息都有上下文支持
    
    评分：
    """
    
    score = model.generate(prompt)
    return parse_score(score)
```

### 3.2 答案相关性

```python
def answer_relevance(query, answer, model):
    """
    评估答案与问题的相关性
    """
    prompt = f"""
    评估以下答案是否直接回答了问题。
    
    问题：{query}
    答案：{answer}
    
    评分标准：
    1 = 完全无关
    2 = 部分相关但没有直接回答
    3 = 回答了但包含无关信息
    4 = 直接回答了问题
    5 = 完美回答，简洁准确
    
    评分：
    """
    
    score = model.generate(prompt)
    return parse_score(score)
```

### 3.3 RAGAS 框架

RAGAS 是一个专门用于 RAG 评估的框架。

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall,
    answer_similarity,
    answer_correctness
)

# 准备数据
dataset = Dataset.from_dict({
    'question': ['什么是 RAG?', ...],
    'answer': ['RAG 是...', ...],
    'contexts': [['RAG 是一种...'], ...],
    'ground_truth': ['RAG 是检索增强生成...', ...]
})

# 评估
results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
)

print(results)
```

### 3.4 自动评估指标

| 指标 | 说明 | 适用场景 |
|------|------|----------|
| **BLEU** | n-gram 重叠 | 机器翻译 |
| **ROUGE** | 召回率导向的 n-gram | 文本摘要 |
| **BERTScore** | 基于 BERT 嵌入的相似度 | 通用生成 |
| **MoverScore** | 基于词移距离的相似度 | 语义评估 |

```python
from bert_score import score

# BERTScore
P, R, F1 = score(
    [generated_answer],
    [reference_answer],
    lang="zh",
    model_type="bert-base-chinese"
)

print(f"Precision: {P.mean():.4f}")
print(f"Recall: {R.mean():.4f}")
print(f"F1: {F1.mean():.4f}")
```

---

## 4. 端到端评估

### 4.1 问答准确率

```python
def qa_accuracy(predictions, references):
    """
    计算问答准确率
    """
    correct = 0
    for pred, ref in zip(predictions, references):
        # 精确匹配
        if pred.strip() == ref.strip():
            correct += 1
        # 或语义匹配
        elif semantic_match(pred, ref):
            correct += 1
    
    return correct / len(predictions)

def semantic_match(pred, ref, threshold=0.85):
    """
    基于语义相似度的匹配
    """
    pred_emb = model.encode(pred)
    ref_emb = model.encode(ref)
    similarity = cosine_similarity(pred_emb, ref_emb)
    return similarity >= threshold
```

### 4.2 人工评估模板

```markdown
## RAG 系统人工评估表

### 查询信息
- 查询 ID: ___
- 查询内容: ___
- 查询类型: [事实性/推理性/比较性/开放式]

### 检索质量（1-5 分）
- [ ] 检索到的文档与查询相关（1-5）
- [ ] 检索结果覆盖了查询的所有方面（1-5）
- [ ] 检索结果没有冗余（1-5）

### 生成质量（1-5 分）
- [ ] 答案直接回答了问题（1-5）
- [ ] 答案忠实于检索到的文档（1-5）
- [ ] 答案完整且详细（1-5）
- [ ] 答案表达清晰流畅（1-5）

### 整体评价
- [ ] 这个回答有帮助吗？（是/否）
- [ ] 你会信任这个回答吗？（是/否）
- [ ] 改进建议：___
```

---

## 5. 评估数据集构建

### 5.1 数据收集

```python
def build_evaluation_dataset(documents, num_questions=100):
    """
    自动构建评估数据集
    """
    dataset = []
    
    for doc in documents:
        # 使用 LLM 生成问题
        prompt = f"""
        基于以下文档，生成 {num_questions} 个问答对。
        问题应该能够从文档中找到答案。
        
        文档：{doc}
        
        格式：
        Q: 问题
        A: 答案
        """
        
        qa_pairs = model.generate(prompt)
        dataset.extend(parse_qa_pairs(qa_pairs))
    
    return dataset
```

### 5.2 数据标注

```python
class AnnotationTool:
    def __init__(self):
        self.annotations = []
    
    def add_annotation(self, query, answer, relevance_score, faithfulness_score):
        """添加人工标注"""
        self.annotations.append({
            'query': query,
            'answer': answer,
            'relevance': relevance_score,
            'faithfulness': faithfulness_score,
            'annotator': 'human_1',
            'timestamp': datetime.now()
        })
    
    def calculate_agreement(self, annotator1, annotator2):
        """计算标注员一致性"""
        # Cohen's Kappa
        from sklearn.metrics import cohen_kappa_score
        
        scores1 = [a['relevance'] for a in self.annotations if a['annotator'] == annotator1]
        scores2 = [a['relevance'] for a in self.annotations if a['annotator'] == annotator2]
        
        return cohen_kappa_score(scores1, scores2)
```

---

## 6. 持续评估

### 6.1 A/B 测试

```python
class RAGABTest:
    def __init__(self, variant_a, variant_b):
        self.variant_a = variant_a  # 对照组
        self.variant_b = variant_b  # 实验组
        self.results = {'a': [], 'b': []}
    
    def run_test(self, queries, traffic_split=0.5):
        """
        运行 A/B 测试
        """
        for query in queries:
            if random.random() < traffic_split:
                # 使用 Variant A
                result = self.variant_a.query(query)
                self.results['a'].append({
                    'query': query,
                    'result': result,
                    'metrics': self.evaluate(result)
                })
            else:
                # 使用 Variant B
                result = self.variant_b.query(query)
                self.results['b'].append({
                    'query': query,
                    'result': result,
                    'metrics': self.evaluate(result)
                })
    
    def analyze_results(self):
        """
        分析 A/B 测试结果
        """
        import scipy.stats as stats
        
        # 提取指标
        metrics_a = [r['metrics']['faithfulness'] for r in self.results['a']]
        metrics_b = [r['metrics']['faithfulness'] for r in self.results['b']]
        
        # t 检验
        t_stat, p_value = stats.ttest_ind(metrics_a, metrics_b)
        
        return {
            'mean_a': np.mean(metrics_a),
            'mean_b': np.mean(metrics_b),
            'improvement': (np.mean(metrics_b) - np.mean(metrics_a)) / np.mean(metrics_a),
            'p_value': p_value,
            'significant': p_value < 0.05
        }
```

### 6.2 监控告警

```python
class RAGMonitor:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.metrics_history = []
    
    def log_query(self, query, result, metrics):
        """记录每次查询的指标"""
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'query': query,
            'metrics': metrics
        })
        
        # 检查阈值
        self._check_thresholds(metrics)
    
    def _check_thresholds(self, metrics):
        """检查是否超过阈值"""
        for metric, value in metrics.items():
            if metric in self.thresholds:
                threshold = self.thresholds[metric]
                if value < threshold:
                    self._send_alert(metric, value, threshold)
    
    def _send_alert(self, metric, value, threshold):
        """发送告警"""
        print(f"ALERT: {metric} = {value:.3f} (threshold: {threshold})")
    
    def get_dashboard_data(self, window='1h'):
        """获取监控面板数据"""
        # 按时间窗口聚合
        recent = [m for m in self.metrics_history 
                  if m['timestamp'] > datetime.now() - timedelta(hours=1)]
        
        return {
            'avg_faithfulness': np.mean([m['metrics']['faithfulness'] for m in recent]),
            'avg_relevance': np.mean([m['metrics']['relevance'] for m in recent]),
            'query_count': len(recent),
            'error_rate': sum(1 for m in recent if m['metrics']['faithfulness'] < 0.5) / len(recent)
        }
```

---

## 参考资源

- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) - Es et al., 2023
- [ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2311.09476) - Saad-Falcon et al., 2023
- [TruLens: Evaluating and Tracking LLM Experiments](https://www.trulens.org/)
- [LangChain Evaluation](https://python.langchain.com/docs/guides/evaluation/)
- [OpenAI Evals](https://github.com/openai/evals)
