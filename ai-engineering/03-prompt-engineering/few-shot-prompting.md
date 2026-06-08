# Few-Shot 少样本提示技术

> **一句话定位**：通过提供少量示例，让模型快速理解任务模式并生成符合要求的输出。
>
> #status/canonical #topic/prompt-engineering #topic/few-shot #year/2026

---

## 1. 核心概念

### 1.1 什么是 Few-Shot Prompting

Few-Shot Prompting 是在提示中提供**少量（通常 1-10 个）输入-输出示例**，让模型理解任务模式并生成类似输出。

**示例对比**：

```
Zero-Shot（无示例）：
将以下中文翻译成英文：
"你好，世界"

Few-Shot（有示例）：
中文 → 英文
"猫" → "cat"
"狗" → "dog"
"你好，世界" →
```

### 1.2 示例数量与效果

| 示例数 | 适用场景 | 优缺点 |
|--------|----------|--------|
| **1-shot** | 简单任务 | 快速，但可能不够稳定 |
| **3-shot** | 标准任务 | 平衡了效果和成本 |
| **5-shot** | 复杂任务 | 更好的模式理解 |
| **10+ shot** | 特殊格式 | 上下文窗口限制 |

### 1.3 关键影响因素

```
示例质量 > 示例数量 > 示例顺序
```

---

## 2. 示例设计原则

### 2.1 示例选择策略

**多样性采样**：

```python
# 覆盖不同类型的输入
examples = [
    {"input": "简单问题", "output": "简单回答"},
    {"input": "复杂问题", "output": "复杂回答"},
    {"input": "边界情况", "output": "特殊处理"},
    {"input": "常见错误", "output": "正确做法"},
]
```

**相似性采样**（动态 Few-Shot）：

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def select_similar_examples(query, example_pool, k=3):
    query_emb = model.encode(query)
    
    similarities = []
    for ex in example_pool:
        ex_emb = model.encode(ex['input'])
        sim = cosine_similarity(query_emb, ex_emb)
        similarities.append((sim, ex))
    
    # 返回最相似的 k 个示例
    return sorted(similarities, reverse=True)[:k]
```

### 2.2 示例格式设计

**标准格式**：

```markdown
## 任务说明
{task_description}

## 示例

### 示例 1
输入：{input_1}
输出：{output_1}

### 示例 2
输入：{input_2}
输出：{output_2}

### 示例 3
输入：{input_3}
输出：{output_3}

## 待处理
输入：{test_input}
输出：
```

**简化格式**（节省 token）：

```markdown
{input_1} → {output_1}
{input_2} → {output_2}
{input_3} → {output_3}
{test_input} →
```

### 2.3 示例质量检查

| 检查项 | 标准 | 常见问题 |
|--------|------|----------|
| **正确性** | 输入输出对应正确 | 示例本身有误 |
| **一致性** | 格式统一 | 不同示例格式混乱 |
| **代表性** | 覆盖主要场景 | 遗漏常见情况 |
| **简洁性** | 不冗余 | 示例过于复杂 |

---

## 3. 高级技术

### 3.1 In-Context Learning (ICL)

**核心机制**：模型从示例中学习任务模式，无需参数更新。

```python
# ICL 示例：情感分析
prompt = """
判断以下评论的情感倾向（正面/负面/中性）：

评论：这部电影太精彩了，强烈推荐！
情感：正面

评论：完全浪费时间，剧情毫无逻辑。
情感：负面

评论：一般般吧，没什么特别的。
情感：中性

评论：{user_comment}
情感：
"""
```

**ICL 的关键因素**：

| 因素 | 影响 | 优化建议 |
|------|------|----------|
| **示例标签分布** | 影响输出偏差 | 保持标签平衡 |
| **示例顺序** | 影响最近示例权重 | 随机排序或按相似度排序 |
| **输入文本分布** | 影响泛化能力 | 覆盖目标分布 |

### 3.2 Exemplar Selection Methods

**基于相似度的选择**：

```python
# KNN 示例选择
def knn_exemplar_selection(query, examples, k=3):
    """
    选择与查询最相似的 k 个示例
    """
    query_emb = embed(query)
    
    scored_examples = []
    for ex in examples:
        ex_emb = embed(ex['input'])
        similarity = cosine_similarity(query_emb, ex_emb)
        scored_examples.append((similarity, ex))
    
    scored_examples.sort(reverse=True)
    return [ex for _, ex in scored_examples[:k]]
```

**基于多样性的选择**：

```python
def diverse_exemplar_selection(examples, k=3):
    """
    选择覆盖不同类别的示例
    """
    # 使用聚类确保多样性
    clusters = cluster_examples(examples, n_clusters=k)
    
    selected = []
    for cluster in clusters:
        # 从每个聚类选最中心的示例
        center = find_center(cluster)
        selected.append(center)
    
    return selected
```

### 3.3 示例增强技术

**输入扰动**：

```python
# 对示例输入进行轻微修改，增强鲁棒性
def augment_example(example):
    augmentations = [
        lambda x: x,  # 原始
        lambda x: x.lower(),  # 小写
        lambda x: x.replace("!", "."),  # 标点变化
        lambda x: "请问" + x,  # 添加前缀
    ]
    
    return [aug(example) for aug in augmentations]
```

**输出解释**：

```python
# 为示例添加解释，帮助模型理解
enhanced_example = """
Q: 2 + 3 × 4 = ?
A: 14

解释：根据运算优先级，先乘除后加减。
     3 × 4 = 12，然后 2 + 12 = 14。
"""
```

---

## 4. 实践应用

### 4.1 分类任务

```python
# 意图识别
prompt = """
将用户查询分类到以下类别：查询余额、转账、修改密码、其他

查询："我想看看账户里还有多少钱"
类别：查询余额

查询："给张三转 1000 元"
类别：转账

查询："我忘了密码怎么办"
类别：修改密码

查询：{user_query}
类别：
"""
```

### 4.2 生成任务

```python
# 代码生成
prompt = """
# 任务：编写 Python 函数

# 示例 1
# 需求：计算列表平均值
def average(numbers):
    return sum(numbers) / len(numbers)

# 示例 2
# 需求：检查字符串是否为回文
def is_palindrome(s):
    return s == s[::-1]

# 待完成
# 需求：{user_requirement}
def {function_name}({params}):
    # 请实现
"""
```

### 4.3 结构化输出

```python
# 信息抽取
prompt = """
从文本中提取人物信息，格式为 JSON：

示例 1：
文本："张三，30 岁，软件工程师，住在北京。"
输出：
{
  "name": "张三",
  "age": 30,
  "occupation": "软件工程师",
  "location": "北京"
}

示例 2：
文本："李四是一名医生，今年 45 岁，工作在上海。"
输出：
{
  "name": "李四",
  "age": 45,
  "occupation": "医生",
  "location": "上海"
}

待处理：
文本："{input_text}"
输出：
"""
```

---

## 5. 优化策略

### 5.1 示例数量优化

```python
def find_optimal_k(task, example_pool, val_set):
    """
    通过验证集找到最佳示例数量
    """
    results = {}
    
    for k in [1, 2, 3, 5, 8, 10]:
        correct = 0
        for query, expected in val_set:
            examples = select_examples(query, example_pool, k=k)
            prompt = build_prompt(examples, query)
            output = model.generate(prompt)
            
            if evaluate(output, expected):
                correct += 1
        
        accuracy = correct / len(val_set)
        results[k] = accuracy
        print(f"k={k}: accuracy={accuracy:.2%}")
    
    # 选择最佳 k
    best_k = max(results, key=results.get)
    return best_k
```

### 5.2 示例质量筛选

```python
def filter_low_quality_examples(examples, threshold=0.8):
    """
    过滤低质量示例
    """
    quality_scores = []
    
    for ex in examples:
        # 使用模型自评示例质量
        score_prompt = f"""
        评估以下示例的质量（0-1）：
        输入：{ex['input']}
        输出：{ex['output']}
        
        质量评分：
        """
        score = float(model.generate(score_prompt).strip())
        quality_scores.append((score, ex))
    
    # 保留高质量示例
    return [ex for score, ex in quality_scores if score >= threshold]
```

### 5.3 动态更新示例库

```python
class DynamicExamplePool:
    def __init__(self, max_size=100):
        self.examples = []
        self.max_size = max_size
        self.performance_history = {}
    
    def add_example(self, input_text, output_text, metadata=None):
        """添加新示例"""
        example = {
            'input': input_text,
            'output': output_text,
            'metadata': metadata,
            'added_time': datetime.now(),
            'use_count': 0,
            'success_count': 0
        }
        self.examples.append(example)
        
        # 如果超过上限，移除最差的示例
        if len(self.examples) > self.max_size:
            self._remove_worst_example()
    
    def update_performance(self, example_idx, success):
        """更新示例使用记录"""
        ex = self.examples[example_idx]
        ex['use_count'] += 1
        if success:
            ex['success_count'] += 1
    
    def _remove_worst_example(self):
        """移除成功率最低的示例"""
        worst_idx = min(
            range(len(self.examples)),
            key=lambda i: self.examples[i]['success_count'] / max(self.examples[i]['use_count'], 1)
        )
        self.examples.pop(worst_idx)
```

---

## 6. 常见陷阱

### 6.1 标签偏差

**问题**：示例中标签分布不均导致输出偏差

```python
# 错误：所有示例都是"正面"
examples = [
    ("好", "正面"),
    ("很好", "正面"),
    ("非常好", "正面"),
]

# 正确：保持标签平衡
examples = [
    ("好", "正面"),
    ("坏", "负面"),
    ("一般", "中性"),
]
```

### 6.2 顺序效应

**问题**：模型倾向于输出最后一个示例的标签

```python
# 解决方案：随机排序
import random

random.shuffle(examples)
prompt = build_prompt(examples, query)
```

### 6.3 示例过拟合

**问题**：模型过度学习示例表面模式

```python
# 错误：示例过于相似
examples = [
    ("我喜欢这个产品", "正面"),
    ("我喜欢这个服务", "正面"),
    ("我喜欢这个应用", "正面"),
]

# 正确：示例多样化
examples = [
    ("这个产品改变了我的生活", "正面"),
    ("客服态度太差了", "负面"),
    ("性价比一般，无功无过", "中性"),
]
```

---

## 7. 评估方法

### 7.1 评估指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **准确率** | 输出正确比例 | correct / total |
| **格式遵循率** | 输出格式正确比例 | format_correct / total |
| **一致性** | 相同输入多次输出一致 | consistent / total |

### 7.2 对比实验

```python
def compare_prompting_methods(test_set):
    """
    对比不同提示方法的效果
    """
    methods = {
        'zero-shot': zero_shot_prompt,
        'one-shot': one_shot_prompt,
        'few-shot-3': lambda q: few_shot_prompt(q, k=3),
        'few-shot-5': lambda q: few_shot_prompt(q, k=5),
        'dynamic-shot': dynamic_shot_prompt,
    }
    
    results = {}
    for name, prompt_fn in methods.items():
        correct = 0
        for query, expected in test_set:
            prompt = prompt_fn(query)
            output = model.generate(prompt)
            if evaluate(output, expected):
                correct += 1
        
        accuracy = correct / len(test_set)
        results[name] = accuracy
        print(f"{name}: {accuracy:.2%}")
    
    return results
```

---

## 参考资源

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - Brown et al., 2020
- [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/abs/2101.06804) - Liu et al., 2021
- [Learning to Retrieve Prompts for In-Context Learning](https://arxiv.org/abs/2112.08633) - Rubin et al., 2021
- [Unified Demonstration Retriever for In-Context Learning](https://arxiv.org/abs/2305.14128) - Li et al., 2023
- [Prompt Engineering Guide - Few-Shot](https://www.promptingguide.ai/techniques/fewshot)
