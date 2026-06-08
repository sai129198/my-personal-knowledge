# 提示优化与自动调优方法

> **一句话定位**：通过系统化方法优化提示，提升模型输出质量，降低人工调试成本。
>
> #status/canonical #topic/prompt-engineering #topic/optimization #year/2026

---

## 1. 核心概念

### 1.1 为什么需要提示优化

| 问题 | 影响 | 优化目标 |
|------|------|----------|
| **输出不稳定** | 相同输入不同输出 | 提高一致性 |
| **质量不达标** | 准确率低于预期 | 提升准确性 |
| **成本过高** | 需要大量 token | 降低成本 |
| **调试困难** | 不知道哪里出问题 | 可解释性 |

### 1.2 优化维度

```
提示优化
├── 内容优化
│   ├── 指令清晰度
│   ├── 示例质量
│   └── 上下文完整性
├── 结构优化
│   ├── 格式规范
│   ├── 分隔符使用
│   └── 顺序安排
└── 参数优化
    ├── 温度（Temperature）
    ├── Top-p / Top-k
    └── 最大长度
```

---

## 2. 手动优化方法

### 2.1 迭代优化流程

```
初始提示 → 测试 → 分析错误 → 修改提示 → 再测试
    ↑___________________________________________|
```

**具体步骤**：

```python
def iterative_optimize(initial_prompt, test_cases, max_iterations=10):
    """
    迭代优化提示
    """
    prompt = initial_prompt
    history = []
    
    for i in range(max_iterations):
        # 1. 测试当前提示
        results = evaluate_prompt(prompt, test_cases)
        accuracy = calculate_accuracy(results)
        
        history.append({
            'iteration': i,
            'prompt': prompt,
            'accuracy': accuracy,
            'errors': analyze_errors(results)
        })
        
        # 2. 检查是否达标
        if accuracy >= target_accuracy:
            break
        
        # 3. 分析错误并修改
        error_analysis = analyze_errors(results)
        prompt = improve_prompt(prompt, error_analysis)
    
    return prompt, history
```

### 2.2 常见优化技巧

**1. 明确指令**：

```markdown
优化前：
总结这段文字。

优化后：
请用 3 句话总结以下文字的核心观点。
要求：
1. 包含主要结论
2. 使用简洁的语言
3. 不超过 100 字
```

**2. 添加约束**：

```markdown
优化前：
生成一个产品描述。

优化后：
为以下产品生成描述：
- 目标用户：25-35 岁职场人士
- 语气：专业但友好
- 长度：150-200 字
- 必须包含：产品名称、核心功能、使用场景
```

**3. 使用角色设定**：

```markdown
优化前：
解释量子计算。

优化后：
你是一位物理学教授，正在给本科生上课。
请用通俗易懂的语言解释量子计算，
使用类比帮助理解，避免复杂公式。
```

---

## 3. 自动优化方法

### 3.1 APE (Automatic Prompt Engineering)

**核心思想**：让模型自己生成和优化提示

```python
def ape_optimize(task_description, initial_prompt, eval_fn, num_candidates=10):
    """
    APE 自动提示优化
    """
    # 1. 生成候选提示
    candidate_prompts = generate_candidates(task_description, num_candidates)
    
    # 2. 评估每个候选
    scores = []
    for prompt in candidate_prompts:
        score = evaluate_prompt(prompt, eval_fn)
        scores.append((score, prompt))
    
    # 3. 选择最佳提示
    best_score, best_prompt = max(scores, key=lambda x: x[0])
    
    # 4. 迭代优化
    for iteration in range(5):
        # 基于最佳提示生成变体
        variants = generate_variants(best_prompt)
        
        for variant in variants:
            score = evaluate_prompt(variant, eval_fn)
            if score > best_score:
                best_score = score
                best_prompt = variant
    
    return best_prompt, best_score

def generate_candidates(task_description, n):
    """生成候选提示"""
    prompt = f"""
    任务：{task_description}
    
    请生成 {n} 个不同的提示，用于完成这个任务。
    每个提示应该有不同的方法或角度。
    
    输出格式：
    1. [提示 1]
    2. [提示 2]
    ...
    """
    
    response = model.generate(prompt)
    return parse_numbered_list(response)
```

### 3.2 OPRO (Optimization by PROmpting)

**核心思想**：将优化过程建模为自然语言优化问题

```python
def opro_optimize(task, initial_prompt, score_fn, steps=20):
    """
    OPRO 优化算法
    """
    # 历史记录：(提示, 分数)
    history = [(initial_prompt, score_fn(initial_prompt))]
    
    for step in range(steps):
        # 构建优化提示
        meta_prompt = build_meta_prompt(task, history)
        
        # 生成新提示
        new_prompt = model.generate(meta_prompt)
        
        # 评估新提示
        score = score_fn(new_prompt)
        
        history.append((new_prompt, score))
        
        # 保留 top-k
        history = sorted(history, key=lambda x: x[1], reverse=True)[:10]
    
    return history[0]

def build_meta_prompt(task, history):
    """构建元优化提示"""
    history_str = "\n".join([
        f"提示：{p}\n分数：{s}"
        for p, s in history
    ])
    
    return f"""
    任务：{task}
    
    以下是之前尝试的提示及其分数：
    {history_str}
    
    请生成一个新的提示，目标是获得比以上所有提示更高的分数。
    新提示应该：
    1. 学习高分提示的优点
    2. 避免低分提示的问题
    3. 尝试新的方法
    
    只输出新提示，不要其他文字：
    """
```

### 3.3 DSPy 框架

```python
import dspy

# 定义签名
class Summarize(dspy.Signature):
    """用一句话总结文档"""
    document = dspy.InputField()
    summary = dspy.OutputField()

# 定义模块
summarizer = dspy.ChainOfThought(Summarize)

# 编译优化
teleprompter = dspy.BootstrapFewShot(
    metric=dsp.evaluate.bleu,
    max_bootstrapped_demos=4
)

optimized_summarizer = teleprompter.compile(
    summarizer,
    trainset=train_data
)

# 使用优化后的模块
result = optimized_summarizer(document="...")
```

---

## 4. 评估方法

### 4.1 自动评估

```python
# 1. 规则匹配
def rule_based_eval(output, expected):
    """基于规则的评估"""
    return output.strip() == expected.strip()

# 2. 相似度评估
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def similarity_eval(output, expected, threshold=0.8):
    """语义相似度评估"""
    emb1 = model.encode(output)
    emb2 = model.encode(expected)
    similarity = cosine_similarity(emb1, emb2)
    return similarity >= threshold

# 3. LLM 评估
def llm_eval(output, expected, criteria):
    """使用 LLM 评估"""
    prompt = f"""
    评估以下输出是否符合预期。
    
    标准：{criteria}
    
    预期输出：{expected}
    实际输出：{output}
    
    请评分（1-10）并说明理由：
    """
    
    response = model.generate(prompt)
    return parse_score(response)
```

### 4.2 人工评估

| 维度 | 说明 | 评分标准 |
|------|------|----------|
| **准确性** | 内容是否正确 | 1-5 分 |
| **完整性** | 是否覆盖所有要求 | 1-5 分 |
| **清晰度** | 表达是否清晰 | 1-5 分 |
| **格式** | 是否符合格式要求 | 1-5 分 |
| **创造性** | 是否有价值的新观点 | 1-5 分 |

---

## 5. 高级技术

### 5.1 多目标优化

```python
def multi_objective_optimize(prompt, objectives):
    """
    多目标优化：平衡准确性、成本、延迟
    """
    # 帕累托前沿
    pareto_front = []
    
    for variant in generate_variants(prompt):
        scores = {
            'accuracy': evaluate_accuracy(variant),
            'cost': evaluate_cost(variant),
            'latency': evaluate_latency(variant)
        }
        
        # 检查是否被支配
        if not is_dominated(scores, pareto_front):
            pareto_front.append((variant, scores))
    
    return pareto_front
```

### 5.2 在线学习

```python
class OnlinePromptOptimizer:
    def __init__(self, initial_prompt):
        self.prompt = initial_prompt
        self.feedback_history = []
        self.performance_history = []
    
    def update(self, input_text, output, feedback):
        """
        根据用户反馈更新提示
        """
        self.feedback_history.append({
            'input': input_text,
            'output': output,
            'feedback': feedback
        })
        
        # 定期优化
        if len(self.feedback_history) % 10 == 0:
            self.prompt = self._optimize()
    
    def _optimize(self):
        """基于反馈历史优化"""
        # 分析负面反馈
        negative_feedback = [
            f for f in self.feedback_history
            if f['feedback'] == 'negative'
        ]
        
        # 生成改进提示
        optimize_prompt = f"""
        当前提示：{self.prompt}
        
        负面反馈案例：
        {format_feedback(negative_feedback)}
        
        请改进提示以解决这些问题：
        """
        
        return model.generate(optimize_prompt)
```

---

## 6. 工具与框架

| 工具 | 功能 | 适用场景 |
|------|------|----------|
| **DSPy** | 提示编译优化 | 复杂任务 |
| **PromptLayer** | 提示版本管理 | 团队协作 |
| **Weights & Biases** | 实验追踪 | A/B 测试 |
| **LangSmith** | 提示评估 | 生产环境 |
| **OpenAI Evals** | 基准测试 | 模型评估 |

---

## 参考资源

- [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409) - Yang et al., 2023
- [Automatic Prompt Engineering](https://arxiv.org/abs/2211.01910) - Zhou et al., 2022
- [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714) - Khattab et al., 2023
- [Prompt Engineering Guide - Optimization](https://www.promptingguide.ai/techniques/optimization)
