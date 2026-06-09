#topic/llm #topic/evaluation #topic/metrics #year/2026 #status/draft

# 模型评估指标

> 大语言模型评估的指标体系：从传统 NLP 指标到现代 LLM 评估方法。

---

## 1. 传统 NLP 指标

### 1.1 Perplexity（困惑度）

**定义**：模型对测试集的"困惑"程度，越低越好。

```
PP(W) = exp(-1/N * Σ log P(w_i | w_1...w_{i-1}))
```

**解读**：
- PP = 100：相当于每次从 100 个等概率词中选择
- PP = 2：模型非常确定下一个词

**局限**：
- 只衡量概率建模能力
- 不直接反映生成质量
- 不同分词器不可比

### 1.2 BLEU（Bilingual Evaluation Understudy）

**定义**：比较生成文本与参考文本的 n-gram 重叠度。

```python
# 简化的 BLEU 计算
def bleu_score(candidate, references, n=4):
    scores = []
    for i in range(1, n+1):
        candidate_ngrams = get_ngrams(candidate, i)
        reference_ngrams = [get_ngrams(ref, i) for ref in references]
        
        clipped_count = sum(
            min(candidate_ngrams[ngram], max(ref.get(ngram, 0) for ref in reference_ngrams))
            for ngram in candidate_ngrams
        )
        total_count = sum(candidate_ngrams.values())
        
        scores.append(clipped_count / total_count)
    
    # 几何平均 + 简短惩罚
    return geometric_mean(scores) * brevity_penalty(candidate, references)
```

**特点**：
- 范围：0-1，越高越好
- 需要参考文本
- 对短文本不友好
- 无法捕捉语义

### 1.3 ROUGE（Recall-Oriented Understudy for Gisting Evaluation）

**定义**：面向召回率的评估指标，常用于摘要任务。

| 变体 | 计算方式 | 适用 |
|------|----------|------|
| ROUGE-N | n-gram 重叠 | 通用 |
| ROUGE-L | 最长公共子序列 | 考虑词序 |
| ROUGE-W | 加权 LCS | 连续匹配 |

### 1.4 METEOR

**改进点**：
- 考虑同义词、词干
- 调和精确率和召回率
- 对语义更敏感

---

## 2. 代码评估指标

### 2.1 HumanEval / MBPP

**HumanEval**：
- 164 个手写编程问题
- 测试函数正确性
- Pass@k 指标

```python
# Pass@k 计算
def pass_at_k(n, c, k):
    """
    n: 总生成数
    c: 通过测试的生成数
    k: 评估的 k 值
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)
```

**MBPP（Mostly Basic Python Programming）**：
- 974 个 Python 问题
- 更适合入门级评估

### 2.2 CodeBLEU

**结合**：
- 语法 AST 匹配
- 数据流分析
- n-gram 重叠

---

## 3. 现代 LLM 评估

### 3.1 MMLU（Massive Multitask Language Understanding）

**定义**：57 个学科的多选题测试。

| 类别 | 学科数 | 示例 |
|------|--------|------|
| STEM | 20 | 数学、物理、化学、生物 |
| Humanities | 15 | 历史、哲学、文学 |
| Social Sciences | 14 | 经济、心理、政治 |
| Other | 8 | 医学、法律、商业 |

**评分**：准确率

### 3.2 HellaSwag

**定义**：常识推理测试，选择最合理的续写。

**特点**：
- 对抗性设计（错误选项看似合理）
- 测试物理世界常识

### 3.3 TruthfulQA

**定义**：测试模型是否生成真实（而非模仿人类偏见）的回答。

**重点**：
- 对抗常见误解
- 测试事实准确性
- 识别模型幻觉

### 3.4 GSM8K

**定义**：小学数学应用题（8.5K 题）。

**测试能力**：
- 数学推理
- 多步计算
- 问题理解

### 3.5 BBH（Big-Bench Hard）

**定义**：23 个复杂推理任务的子集。

**特点**：
- 需要多步推理
- 超越简单模式匹配
- 当前模型仍表现不佳

---

## 4. 人类评估

### 4.1 Elo 评分系统

**LMSYS Chatbot Arena**：
- 人类对比两个模型的回答
- 使用 Elo 评分排名
- 最权威的模型排行榜之一

### 4.2 评估维度

| 维度 | 说明 | 评分 |
|------|------|------|
| **有用性** | 回答是否解决了问题 | 1-5 |
| **准确性** | 事实是否正确 | 1-5 |
| **连贯性** | 逻辑是否通顺 | 1-5 |
| **安全性** | 是否有害内容 | 1-5 |
| **创造性** | 是否有新意 | 1-5 |

---

## 5. 自动评估框架

### 5.1 LLM-as-a-Judge

**概念**：用更强的模型评估较弱模型的输出。

```python
judge_prompt = """请评估以下回答的质量。

问题：{question}
参考回答：{reference}
待评估回答：{candidate}

请从以下维度评分（1-5）：
1. 准确性
2. 完整性
3. 连贯性

输出格式：
准确性：x/5
完整性：x/5
连贯性：x/5
总体评价：[简要说明]
"""
```

**注意事项**：
- 评委模型可能存在偏见
- 需要校准和验证
- 成本较高

### 5.2 RAGAS

**RAG 专用评估**：

| 指标 | 说明 |
|------|------|
| Faithfulness | 回答是否忠实于上下文 |
| Answer Relevancy | 回答与问题的相关度 |
| Context Precision | 上下文中有用信息的比例 |
| Context Recall | 回答问题所需信息在上下文中的比例 |

---

## 6. 指标选择指南

```
评估目标
    │
    ├─ 通用能力？
    │   ├─ MMLU、HellaSwag、TruthfulQA
    │   └─ 综合基准（如 Open LLM Leaderboard）
    │
    ├─ 代码能力？
    │   ├─ HumanEval、MBPP
    │   └─ CodeBLEU、执行通过率
    │
    ├─ 数学推理？
    │   ├─ GSM8K、MATH
    │   └─ 自定义数学题库
    │
    ├─ RAG 系统？
    │   ├─ RAGAS
    │   └─ 自定义检索+生成指标
    │
    ├─ 对话质量？
    │   ├─ MT-Bench
    │   └─ 人类评估（Elo）
    │
    └─ 生产监控？
        ├─ 用户满意度
        ├─ 任务完成率
        └─ 错误率
```

---

## 💡 我的思考

1. **没有完美的自动指标**：自动指标只能捕捉部分质量维度，人类评估仍是金标准。

2. **指标组合使用**：单一指标容易过拟合，组合多个指标才能全面评估。

3. **任务相关**：选择与你实际应用场景最相关的评估指标。

4. **持续评估**：模型能力在进化，评估数据集也需要更新。

5. **LLM-as-a-Judge 是趋势**：虽然不完美，但可扩展性好，是实用的折中方案。

---

## 参考来源

- **Open LLM Leaderboard**: [huggingface.co/spaces/open-llm-leaderboard](https://huggingface.co/spaces/open-llm-leaderboard) — 访问日期：2026-06-07
- **LMSYS Arena**: [chat.lmsys.org](https://chat.lmsys.org/) — 访问日期：2026-06-07
- **RAGAS**: [docs.ragas.io](https://docs.ragas.io/) — 访问日期：2026-06-07
- **EleutherAI LM Evaluation**: [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 访问日期：2026-06-07

---

*访问日期: 2026-06-07*
