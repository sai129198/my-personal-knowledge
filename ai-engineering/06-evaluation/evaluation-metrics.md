# 评估指标详解

> **一句话定位**：掌握 NLP 和 LLM 评估指标，选择最适合的指标衡量模型性能。
>
> #status/canonical #topic/evaluation #topic/metrics #year/2026

---

## 1. 基础指标

### 1.1 精确匹配（Exact Match）

最简单的评估方式，预测与参考答案完全一致才算正确。

```python
def exact_match(prediction: str, reference: str) -> bool:
    """
    精确匹配
    """
    return prediction.strip().lower() == reference.strip().lower()

def exact_match_score(predictions: List[str], references: List[str]) -> float:
    """
    计算精确匹配分数
    """
    matches = sum(
        1 for pred, ref in zip(predictions, references)
        if exact_match(pred, ref)
    )
    return matches / len(predictions)
```

**适用场景**：
- 问答系统（抽取式）
- 分类任务
- 简单推理任务

**局限性**：
- 对表述变化敏感（"北京" vs "北京市"）
- 无法评估生成质量

---

### 1.2 F1 Score

基于 token 级别的精确率和召回率。

```python
def token_f1(prediction: str, reference: str) -> float:
    """
    计算 Token-level F1
    """
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    common = pred_tokens & ref_tokens
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)
```

---

## 2. N-gram 重叠指标

### 2.1 BLEU

BLEU（Bilingual Evaluation Understudy）基于 n-gram 精确率。

```python
from collections import Counter
import math

def bleu_score(prediction: str, references: List[str], max_n: int = 4) -> float:
    """
    计算 BLEU 分数
    
    Args:
        prediction: 预测文本
        references: 参考文本列表
        max_n: 最大 n-gram 阶数
    """
    pred_tokens = prediction.split()
    ref_tokens_list = [ref.split() for ref in references]
    
    # 计算 n-gram 精确率
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(get_ngrams(pred_tokens, n))
        
        # 计算最大匹配数
        max_matches = Counter()
        for ref_tokens in ref_tokens_list:
            ref_ngrams = Counter(get_ngrams(ref_tokens, n))
            for ngram in pred_ngrams:
                max_matches[ngram] = max(max_matches[ngram], ref_ngrams[ngram])
        
        # 计算匹配数
        matches = sum(min(pred_ngrams[ngram], max_matches[ngram]) 
                     for ngram in pred_ngrams)
        total = sum(pred_ngrams.values())
        
        if total == 0:
            precisions.append(0)
        else:
            precisions.append(matches / total)
    
    # 几何平均
    if min(precisions) == 0:
        return 0.0
    
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    
    # 简短惩罚
    bp = brevity_penalty(len(pred_tokens), 
                         min(len(ref) for ref in ref_tokens_list))
    
    return bp * geo_mean

def get_ngrams(tokens: List[str], n: int) -> List[tuple]:
    """获取 n-grams"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def brevity_penalty(pred_len: int, ref_len: int) -> float:
    """简短惩罚"""
    if pred_len > ref_len:
        return 1.0
    return math.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0
```

**特点**：
- 优点：计算快速，与人工评估有一定相关性
- 缺点：不考虑语义，对同义词不敏感

**适用场景**：机器翻译、文本摘要

### 2.2 ROUGE

ROUGE（Recall-Oriented Understudy for Gisting Evaluation）基于召回率。

```python
def rouge_n(prediction: str, reference: str, n: int = 1) -> Dict[str, float]:
    """
    计算 ROUGE-N
    
    Returns:
        {'precision': float, 'recall': float, 'f1': float}
    """
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    
    pred_ngrams = Counter(get_ngrams(pred_tokens, n))
    ref_ngrams = Counter(get_ngrams(ref_tokens, n))
    
    matches = sum((pred_ngrams & ref_ngrams).values())
    
    precision = matches / sum(pred_ngrams.values()) if pred_ngrams else 0
    recall = matches / sum(ref_ngrams.values()) if ref_ngrams else 0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    """
    计算 ROUGE-L（最长公共子序列）
    """
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    
    lcs_length = longest_common_subsequence(pred_tokens, ref_tokens)
    
    precision = lcs_length / len(pred_tokens) if pred_tokens else 0
    recall = lcs_length / len(ref_tokens) if ref_tokens else 0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def longest_common_subsequence(seq1: List, seq2: List) -> int:
    """计算最长公共子序列长度"""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

**特点**：
- ROUGE-1：unigram 重叠
- ROUGE-2：bigram 重叠
- ROUGE-L：最长公共子序列

**适用场景**：文本摘要、文本生成

---

## 3. 语义相似度指标

### 3.1 BERTScore

基于 BERT 嵌入的相似度计算。

```python
from bert_score import score

# 计算 BERTScore
P, R, F1 = score(
    predictions,  # 预测文本列表
    references,   # 参考文本列表
    lang="zh",    # 语言
    model_type="bert-base-chinese",  # 模型
    verbose=True
)

print(f"Precision: {P.mean():.4f}")
print(f"Recall: {R.mean():.4f}")
print(f"F1: {F1.mean():.4f}")
```

**特点**：
- 优点：考虑语义，对同义词敏感
- 缺点：计算较慢，需要 GPU

**适用场景**：通用文本生成评估

### 3.2 MoverScore

基于 Word Mover's Distance 的相似度。

```python
from moverscore_v2 import get_idf_dict, word_mover_score

# 计算 IDF
idf_dict_hyp = get_idf_dict(predictions)
idf_dict_ref = get_idf_dict(references)

# 计算 MoverScore
scores = word_mover_score(
    references,
    predictions,
    idf_dict_ref,
    idf_dict_hyp,
    stop_words=[],
    n_gram=1,
    remove_subwords=True
)
```

### 3.3 Sentence-BERT 相似度

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 加载模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 编码
pred_embeddings = model.encode(predictions)
ref_embeddings = model.encode(references)

# 计算余弦相似度
similarities = cosine_similarity(pred_embeddings, ref_embeddings)
```

---

## 4. LLM 专用指标

### 4.1 困惑度（Perplexity）

衡量模型对文本的预测能力。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_perplexity(text: str, model, tokenizer) -> float:
    """
    计算困惑度
    """
    encodings = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings.input_ids)
        loss = outputs.loss
    
    perplexity = torch.exp(loss).item()
    return perplexity

# 使用
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

ppl = calculate_perplexity("This is a test sentence.", model, tokenizer)
print(f"Perplexity: {ppl:.2f}")
```

**解读**：
- 困惑度越低越好
- 与训练数据的分布有关
- 不能直接比较不同模型的困惑度

### 4.2 多样性指标

```python
def distinct_n(predictions: List[str], n: int = 2) -> float:
    """
    计算 Distinct-N（n-gram 多样性）
    """
    all_ngrams = set()
    total_ngrams = 0
    
    for text in predictions:
        tokens = text.split()
        ngrams = get_ngrams(tokens, n)
        all_ngrams.update(ngrams)
        total_ngrams += len(ngrams)
    
    return len(all_ngrams) / total_ngrams if total_ngrams > 0 else 0.0

def self_bleu(predictions: List[str]) -> float:
    """
    计算 Self-BLEU（衡量生成多样性）
    """
    scores = []
    for i, pred in enumerate(predictions):
        refs = predictions[:i] + predictions[i+1:]
        scores.append(bleu_score(pred, refs))
    
    return sum(scores) / len(scores)
```

---

## 5. 任务特定指标

### 5.1 问答任务

```python
def qa_f1_score(prediction: str, ground_truths: List[str]) -> float:
    """
    问答任务的 F1 分数
    """
    pred_tokens = set(prediction.lower().split())
    
    scores = []
    for ground_truth in ground_truths:
        gt_tokens = set(ground_truth.lower().split())
        
        common = pred_tokens & gt_tokens
        
        if not common:
            scores.append(0)
            continue
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        
        f1 = 2 * precision * recall / (precision + recall)
        scores.append(f1)
    
    return max(scores)
```

### 5.2 代码生成

```python
def code_bleu(predictions: List[str], references: List[str]) -> float:
    """
    CodeBLEU：针对代码生成的评估指标
    """
    from codebleu import calc_codebleu
    
    result = calc_codebleu(
        references,
        predictions,
        lang="python",
        weights=(0.25, 0.25, 0.25, 0.25),
        tokenizer=None
    )
    
    return result['codebleu']
```

### 5.3 事实准确性

```python
def fact_accuracy(prediction: str, facts: List[str]) -> float:
    """
    事实准确性评估
    """
    # 使用 NLI 模型判断事实是否被支持
    from transformers import pipeline
    
    nli = pipeline("text-classification", model="facebook/bart-large-mnli")
    
    correct = 0
    for fact in facts:
        # 判断预测是否包含事实
        result = nli(f"{prediction} [SEP] {fact}")
        if result[0]['label'] == 'ENTAILMENT':
            correct += 1
    
    return correct / len(facts)
```

---

## 6. 指标选择指南

### 6.1 选择矩阵

| 任务类型 | 推荐指标 | 备选指标 |
|----------|----------|----------|
| **机器翻译** | BLEU | TER, METEOR |
| **文本摘要** | ROUGE | BERTScore |
| **对话生成** | BERTScore | 人工评估 |
| **代码生成** | CodeBLEU | Pass@k |
| **问答系统** | EM / F1 | BERTScore |
| **事实核查** | 准确率 | F1 |

### 6.2 综合评估

```python
class ComprehensiveEvaluator:
    """
    综合评估器
    """
    
    def __init__(self):
        self.metrics = {
            'bleu': bleu_score,
            'rouge': rouge_l,
            'bertscore': self._bertscore,
            'distinct': distinct_n
        }
    
    def evaluate(self, predictions: List[str], references: List[str]) -> Dict:
        """
        综合评估
        """
        results = {}
        
        for name, metric_func in self.metrics.items():
            try:
                results[name] = metric_func(predictions, references)
            except Exception as e:
                results[name] = f"Error: {e}"
        
        return results
    
    def _bertscore(self, predictions, references):
        from bert_score import score
        P, R, F1 = score(predictions, references, lang="zh")
        return {'precision': P.mean().item(), 
                'recall': R.mean().item(), 
                'f1': F1.mean().item()}
```

---

## 参考资源

- [BLEU: A Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) - Papineni et al., 2002
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) - Lin, 2004
- [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675) - Zhang et al., 2019
- [MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance](https://arxiv.org/abs/1909.02622) - Zhao et al., 2019
- [CodeBLEU: a Method for Automatic Evaluation of Code Synthesis](https://arxiv.org/abs/2009.10297) - Ren et al., 2020
