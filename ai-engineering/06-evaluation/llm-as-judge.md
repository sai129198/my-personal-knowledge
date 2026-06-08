# LLM 作为评估器

> **一句话定位**：使用 LLM 自动评估生成质量，平衡效率与准确性。
>
> #status/canonical #topic/evaluation #topic/llm-as-judge #year/2026

---

## 1. LLM-as-Judge 基础

### 1.1 为什么使用 LLM 评估

```
传统人工评估的问题：
- 成本高（时间 + 金钱）
- 速度慢
- 一致性差
- 难以扩展

LLM 评估的优势：
- 成本低
- 速度快
- 一致性高
- 可扩展
- 可复现

LLM 评估的局限：
- 可能存在偏见
- 对复杂推理评估困难
- 需要验证可靠性
```

### 1.2 评估模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **点对点** | 评估单个输出 | 质量评分 |
| **成对比较** | 比较两个输出 | A/B 测试 |
| **参考对比** | 与标准答案比较 | 准确性评估 |
| **多维度** | 多维度评分 | 全面评估 |

---

## 2. 实现方法

### 2.1 点对点评估

```python
class PointwiseEvaluator:
    """
    点对点评估器
    """
    
    def __init__(self, judge_model: str = "gpt-4"):
        self.judge = judge_model
    
    def evaluate(self, input_text: str, output: str, criteria: List[str]) -> Dict:
        """
        评估单个输出
        
        Args:
            input_text: 输入文本
            output: 模型输出
            criteria: 评估维度
        
        Returns:
            各维度评分
        """
        prompt = f"""
        You are an expert evaluator. Evaluate the following model output based on the given criteria.
        
        Input: {input_text}
        
        Model Output: {output}
        
        Please evaluate on the following criteria (1-5 scale):
        """
        
        for criterion in criteria:
            prompt += f"\n- {criterion}"
        
        prompt += """
        
        Provide your evaluation as JSON:
        {
            "scores": {
                "criterion1": score,
                "criterion2": score,
                ...
            },
            "reasoning": "Brief explanation of your ratings"
        }
        """
        
        response = self._call_llm(prompt)
        return self._parse_evaluation(response)
    
    def _call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        # 实现 LLM 调用
        pass
    
    def _parse_evaluation(self, response: str) -> Dict:
        """解析评估结果"""
        import json
        try:
            return json.loads(response)
        except:
            return {"error": "Failed to parse", "raw": response}
```

### 2.2 成对比较

```python
class PairwiseEvaluator:
    """
    成对比较评估器
    """
    
    def __init__(self, judge_model: str = "gpt-4"):
        self.judge = judge_model
    
    def compare(self, input_text: str, output_a: str, output_b: str, criteria: str = "overall") -> Dict:
        """
        比较两个输出
        
        Returns:
            {
                "winner": "A" | "B" | "tie",
                "reasoning": "解释",
                "confidence": 0-1
            }
        """
        prompt = f"""
        Compare the following two model outputs for the given input.
        
        Input: {input_text}
        
        Output A:
        {output_a}
        
        Output B:
        {output_b}
        
        Which output is better in terms of {criteria}?
        
        Respond with:
        - Winner: A, B, or Tie
        - Reasoning: Brief explanation
        - Confidence: High, Medium, or Low
        """
        
        response = self._call_llm(prompt)
        return self._parse_comparison(response)
    
    def evaluate_batch(self, comparisons: List[Dict]) -> Dict:
        """
        批量比较
        """
        results = []
        for comp in comparisons:
            result = self.compare(
                comp["input"],
                comp["output_a"],
                comp["output_b"],
                comp.get("criteria", "overall")
            )
            results.append(result)
        
        # 统计胜率
        wins_a = sum(1 for r in results if r["winner"] == "A")
        wins_b = sum(1 for r in results if r["winner"] == "B")
        ties = sum(1 for r in results if r["winner"] == "tie")
        
        return {
            "total": len(results),
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "win_rate_a": wins_a / len(results),
            "detailed_results": results
        }
```

### 2.3 参考对比评估

```python
class ReferenceBasedEvaluator:
    """
    基于参考的评估
    """
    
    def __init__(self, judge_model: str = "gpt-4"):
        self.judge = judge_model
    
    def evaluate_with_reference(self, input_text: str, output: str, reference: str) -> Dict:
        """
        与参考答案对比评估
        """
        prompt = f"""
        Evaluate the model output by comparing it with the reference answer.
        
        Input: {input_text}
        
        Model Output: {output}
        
        Reference Answer: {reference}
        
        Evaluate:
        1. Accuracy: Does the output contain the same information as the reference?
        2. Completeness: Does the output cover all points in the reference?
        3. Hallucination: Does the output contain information not in the reference?
        
        Provide scores (1-5) and brief explanations.
        """
        
        response = self._call_llm(prompt)
        return self._parse_evaluation(response)
```

---

## 3. 评估框架

### 3.1 RAGAS

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
    'question': ['What is X?', 'How does Y work?'],
    'answer': ['X is...', 'Y works by...'],
    'contexts': [['Context about X'], ['Context about Y']],
    'ground_truth': ['X is defined as...', 'Y operates through...']
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

### 3.2 TruLens

```python
from trulens_eval import TruChain, Feedback, Tru
from trulens_eval.feedback import OpenAI as fOpenAI

# 初始化
tru = Tru()
provider = fOpenAI()

# 定义反馈函数
f_groundedness = Feedback(
    provider.groundedness_measure_with_cot_reasons,
    name="Groundedness"
).on_input_output()

f_qa_relevance = Feedback(
    provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on_input_output()

f_context_relevance = Feedback(
    provider.qs_relevance_with_cot_reasons,
    name="Context Relevance"
).on_input()

# 包装应用
tru_chain = TruChain(
    chain,
    app_id="my_app",
    feedbacks=[f_groundedness, f_qa_relevance, f_context_relevance]
)

# 运行并评估
result = tru_chain.run("What is the capital of France?")
```

### 3.3 自定义评估框架

```python
class LLMJudgeFramework:
    """
    自定义 LLM 评估框架
    """
    
    def __init__(self, judge_model: str, criteria: List[Dict]):
        self.judge = judge_model
        self.criteria = criteria
    
    def evaluate_dataset(self, dataset: List[Dict]) -> Dict:
        """
        评估数据集
        """
        results = []
        
        for sample in dataset:
            evaluation = self._evaluate_sample(sample)
            results.append(evaluation)
        
        # 汇总统计
        summary = self._summarize_results(results)
        
        return {
            "summary": summary,
            "detailed_results": results
        }
    
    def _evaluate_sample(self, sample: Dict) -> Dict:
        """评估单个样本"""
        scores = {}
        
        for criterion in self.criteria:
            score = self._evaluate_criterion(
                sample["input"],
                sample["output"],
                criterion
            )
            scores[criterion["name"]] = score
        
        return {
            "sample_id": sample["id"],
            "scores": scores,
            "overall": sum(scores.values()) / len(scores)
        }
    
    def _evaluate_criterion(self, input_text: str, output: str, criterion: Dict) -> float:
        """评估单个维度"""
        prompt = f"""
        Evaluate the following output based on the criterion.
        
        Criterion: {criterion['name']}
        Description: {criterion['description']}
        
        Input: {input_text}
        Output: {output}
        
        Rate on a scale of 1-5. Provide only the numeric score.
        """
        
        response = self._call_llm(prompt)
        
        # 提取分数
        try:
            score = float(response.strip())
            return min(max(score, 1), 5)  # 限制在 1-5 范围内
        except:
            return 3.0  # 默认分数
    
    def _summarize_results(self, results: List[Dict]) -> Dict:
        """汇总结果"""
        summary = {}
        
        # 按维度汇总
        for criterion in self.criteria:
            name = criterion["name"]
            scores = [r["scores"][name] for r in results]
            summary[name] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "std": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5
            }
        
        # 总体分数
        overall_scores = [r["overall"] for r in results]
        summary["overall"] = {
            "mean": sum(overall_scores) / len(overall_scores),
            "distribution": self._calculate_distribution(overall_scores)
        }
        
        return summary
    
    def _calculate_distribution(self, scores: List[float]) -> Dict:
        """计算分数分布"""
        distribution = {i: 0 for i in range(1, 6)}
        for score in scores:
            bucket = int(score)
            distribution[bucket] = distribution.get(bucket, 0) + 1
        
        # 转换为百分比
        total = len(scores)
        return {k: v/total for k, v in distribution.items()}
```

---

## 4. 可靠性验证

### 4.1 与人类评估对比

```python
class ReliabilityValidator:
    """
    验证 LLM 评估的可靠性
    """
    
    def __init__(self, llm_evaluator, human_annotations: List[Dict]):
        self.llm_evaluator = llm_evaluator
        self.human_annotations = human_annotations
    
    def validate(self, test_samples: List[Dict]) -> Dict:
        """
        验证 LLM 评估与人类评估的一致性
        """
        llm_scores = []
        human_scores = []
        
        for sample in test_samples:
            # LLM 评估
            llm_result = self.llm_evaluator.evaluate(
                sample["input"],
                sample["output"]
            )
            
            # 人类评估（取平均）
            human_result = self._get_human_average(sample["id"])
            
            llm_scores.append(llm_result["overall"])
            human_scores.append(human_result)
        
        # 计算相关性
        correlation = self._calculate_correlation(llm_scores, human_scores)
        
        # 计算一致性
        agreement = self._calculate_agreement(llm_scores, human_scores)
        
        return {
            "correlation": correlation,
            "agreement": agreement,
            "llm_scores": llm_scores,
            "human_scores": human_scores
        }
    
    def _calculate_correlation(self, llm_scores: List[float], human_scores: List[float]) -> float:
        """计算 Pearson 相关系数"""
        from scipy.stats import pearsonr
        
        corr, p_value = pearsonr(llm_scores, human_scores)
        return {
            "pearson": corr,
            "p_value": p_value
        }
    
    def _calculate_agreement(self, llm_scores: List[float], human_scores: List[float], threshold: float = 0.5) -> float:
        """计算一致性比例"""
        agreements = 0
        for llm, human in zip(llm_scores, human_scores):
            if abs(llm - human) <= threshold:
                agreements += 1
        
        return agreements / len(llm_scores)
```

### 4.2 偏见检测

```python
class BiasDetector:
    """
    检测 LLM 评估中的偏见
    """
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
    
    def detect_position_bias(self, outputs: List[str], n_comparisons: int = 10) -> Dict:
        """
        检测位置偏见
        （LLM 是否倾向于选择第一个或第二个选项）
        """
        results = []
        
        for _ in range(n_comparisons):
            # 随机选择两个输出
            a, b = random.sample(outputs, 2)
            
            # 正向比较 (A vs B)
            result1 = self.evaluator.compare("test input", a, b)
            
            # 反向比较 (B vs A)
            result2 = self.evaluator.compare("test input", b, a)
            
            # 检查一致性
            consistent = (
                (result1["winner"] == "A" and result2["winner"] == "B") or
                (result1["winner"] == "B" and result2["winner"] == "A") or
                (result1["winner"] == "tie" and result2["winner"] == "tie")
            )
            
            results.append(consistent)
        
        consistency_rate = sum(results) / len(results)
        
        return {
            "consistency_rate": consistency_rate,
            "position_bias_detected": consistency_rate < 0.8
        }
    
    def detect_length_bias(self, short_outputs: List[str], long_outputs: List[str]) -> Dict:
        """
        检测长度偏见
        """
        short_wins = 0
        total = 0
        
        for short in short_outputs:
            for long in long_outputs:
                result = self.evaluator.compare("test input", short, long)
                if result["winner"] == "A":  # short wins
                    short_wins += 1
                total += 1
        
        short_win_rate = short_wins / total
        
        return {
            "short_win_rate": short_win_rate,
            "length_bias": "long" if short_win_rate < 0.4 else "short" if short_win_rate > 0.6 else "none"
        }
```

---

## 5. 最佳实践

### 5.1 提示工程

```python
class EvaluationPrompts:
    """
    评估提示模板
    """
    
    POINTWISE_TEMPLATE = """
    You are an expert evaluator tasked with assessing the quality of AI-generated responses.
    
    Evaluation Criteria:
    {criteria}
    
    Input: {input}
    
    Model Output: {output}
    
    Please provide your evaluation following these guidelines:
    1. Be objective and consistent
    2. Focus on the specific criteria
    3. Consider both strengths and weaknesses
    4. Provide specific examples from the output
    
    Format your response as:
    Scores: {{"criterion1": score, "criterion2": score}}
    Reasoning: Your detailed explanation
    """
    
    PAIRWISE_TEMPLATE = """
    Compare the following two responses to the same input.
    
    Input: {input}
    
    Response A:
    {output_a}
    
    Response B:
    {output_b}
    
    Evaluation Focus: {criteria}
    
    Which response is better? Consider:
    - Accuracy and correctness
    - Completeness
    - Clarity and coherence
    - Helpfulness
    
    Your verdict (choose one):
    - A is significantly better
    - A is slightly better
    - Tie
    - B is slightly better
    - B is significantly better
    
    Reasoning:
    """
```

### 5.2 多法官评估

```python
class MultiJudgeEvaluation:
    """
    多法官评估
    """
    
    def __init__(self, judges: List[str]):
        self.judges = judges
    
    def evaluate(self, input_text: str, output: str) -> Dict:
        """
        使用多个法官评估
        """
        evaluations = []
        
        for judge in self.judges:
            evaluator = PointwiseEvaluator(judge)
            result = evaluator.evaluate(input_text, output)
            evaluations.append(result)
        
        # 汇总结果
        aggregated = self._aggregate_evaluations(evaluations)
        
        return {
            "individual_evaluations": evaluations,
            "aggregated": aggregated,
            "agreement": self._calculate_judge_agreement(evaluations)
        }
    
    def _aggregate_evaluations(self, evaluations: List[Dict]) -> Dict:
        """聚合多个评估结果"""
        # 取平均
        scores = {}
        for eval in evaluations:
            for criterion, score in eval["scores"].items():
                if criterion not in scores:
                    scores[criterion] = []
                scores[criterion].append(score)
        
        aggregated = {}
        for criterion, score_list in scores.items():
            aggregated[criterion] = {
                "mean": sum(score_list) / len(score_list),
                "median": sorted(score_list)[len(score_list)//2],
                "std": (sum((s - sum(score_list)/len(score_list))**2 for s in score_list) / len(score_list))**0.5
            }
        
        return aggregated
    
    def _calculate_judge_agreement(self, evaluations: List[Dict]) -> float:
        """计算法官间一致性"""
        # 使用 Krippendorff's Alpha 或 Fleiss' Kappa
        pass
```

---

## 参考资源

- [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) - Liu et al., 2023
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) - Zheng et al., 2023
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) - Es et al., 2023
- [TruLens Documentation](https://www.trulens.org/)
- [LangChain Evaluation](https://python.langchain.com/docs/guides/evaluation/)
