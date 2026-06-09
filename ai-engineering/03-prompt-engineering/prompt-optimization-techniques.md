#topic/prompt-engineering #topic/optimization #year/2026 #status/draft

# Prompt 优化技巧

> 从手动调优到自动优化，系统提升 Prompt 效果的工程方法。

---

## 1. 手动优化技巧

### 1.1 清晰度优化

**问题**：模糊的指令导致模型输出不稳定。

**优化前后对比**：

```
【优化前】
写一篇关于 AI 的文章。

【优化后】
写一篇 800 字的技术博客文章，主题："AI 在医疗影像诊断中的应用"。
要求：
- 面向有技术背景的医生
- 包含 2 个具体案例
- 讨论技术局限性和伦理考量
- 使用专业但易懂的语言
- 结尾给出未来 5 年的发展预测
```

**清晰度检查清单**：
- [ ] 指定输出格式（长度、结构、风格）
- [ ] 明确目标受众
- [ ] 列出必须包含的要素
- [ ] 说明约束条件（不要什么）
- [ ] 提供评价标准

---

### 1.2 结构化优化

**使用标记语言增强结构**：

```
请分析以下代码的复杂度：

```java
public class Example {
    // 代码...
}
```

请按以下格式输出：

## 时间复杂度
- 最坏情况：[分析]
- 最好情况：[分析]
- 平均情况：[分析]

## 空间复杂度
- [分析]

## 优化建议
1. [建议 1]
2. [建议 2]

## 评分
- 可读性：1-10 分
- 效率：1-10 分
- 可维护性：1-10 分
```

**为什么有效**：
- 结构化的输出更容易解析
- 模型遵循模板的能力很强
- 减少格式不一致问题

---

### 1.3 边界情况处理

**显式声明边界条件**：

```
请分类以下用户反馈的情感倾向（正面/负面/中性）。

特殊规则：
- 如果反馈同时包含正面和负面，判断主导倾向
- 如果反馈是疑问句且无明确情感，归类为中性
- 如果反馈包含讽刺，按字面意思分类并标注 [sarcasm]
- 如果反馈语言不是中文或英文，回复 "UNSUPPORTED_LANGUAGE"

示例：
"你们的 App 真‘好用’，每次都要重新登录" → 负面 [sarcasm]
"这个功能怎么用？" → 中性
"太棒了，终于解决了我的问题！" → 正面
```

---

## 2. 自动优化方法

### 2.1 APE (Automatic Prompt Engineering)

**核心思想**：让模型自己生成和优化 prompt。

```python
# APE 流程
# Step 1: 生成候选 prompts
prompt_candidates = model.generate(
    f"生成 10 个不同的 prompt 来完成以下任务：\n"
    f"任务：{task_description}\n"
    f"要求：prompt 应该清晰、具体、有效"
)

# Step 2: 评估每个 prompt
scores = []
for prompt in prompt_candidates:
    outputs = []
    for test_case in test_cases:
        output = model.generate(prompt + test_case.input)
        score = evaluate(output, test_case.expected)
        outputs.append(score)
    scores.append(mean(outputs))

# Step 3: 选择最佳 prompt
best_prompt = prompt_candidates[argmax(scores)]
```

**实现框架**：
- DSPy
- PromptBreeder
- OPRO (Optimization by PROmpting)

---

### 2.2 DSPy 框架

**定义**：声明式 LLM 编程框架，将 prompt 工程转化为优化问题。

```python
import dspy

# 定义签名
class SentimentAnalysis(dspy.Signature):
    """分析文本的情感倾向"""
    text = dspy.InputField()
    sentiment = dspy.OutputField(desc="正面、负面或中性")
    confidence = dspy.OutputField(desc="0-1 之间的置信度")

# 定义模块
class SentimentClassifier(dspy.Module):
    def __init__(self):
        self.classifier = dspy.ChainOfThought(SentimentAnalysis)
    
    def forward(self, text):
        return self.classifier(text=text)

# 编译（自动优化 prompt）
classifier = SentimentClassifier()
optimized = dspy.teleprompt.BootstrapFewShot(
    metric=accuracy_metric
).compile(classifier, trainset=train_data)

# 使用优化后的模块
result = optimized("这个产品太棒了！")
```

**DSPy 核心概念**：
- **Signature**：输入输出规范
- **Module**：可组合的 LLM 组件
- **Teleprompter**：自动优化器
- **Example**：训练示例

---

### 2.3 Multi-Prompt 集成

**Ensemble 策略**：

```python
# 使用多个不同 prompt 生成答案，然后集成
prompts = [
    "请直接回答：{question}",
    "请一步步思考：{question}",
    "请从多个角度分析：{question}",
    "请先列出相关事实，再回答：{question}"
]

answers = [model.generate(p.format(question=q)) for p in prompts]
final_answer = majority_vote(answers)  # 或加权平均
```

**变体：Prompt 路由**：

```python
def route_question(question):
    if is_factual(question):
        return "direct_prompt"
    elif is_mathematical(question):
        return "cot_prompt"
    elif is_creative(question):
        return "divergent_prompt"
    else:
        return "general_prompt"
```

---

## 3. 评估与迭代

### 3.1 Prompt 评估指标

| 指标 | 说明 | 测量方法 |
|------|------|----------|
| **准确性** | 输出是否正确 | 与标准答案对比 |
| **一致性** | 多次运行结果是否稳定 | 多次采样计算方差 |
| **完整性** | 是否覆盖所有要求 | 检查清单匹配 |
| **格式合规** | 是否符合指定格式 | 正则/规则验证 |
| **延迟** | 生成时间 | 时间测量 |
| **Token 效率** | 是否简洁 | 输出长度 |

### 3.2 迭代优化流程

```
1. 基线测试
   └── 记录当前 prompt 的各项指标

2. 问题诊断
   └── 分析失败案例，定位问题原因

3. 假设生成
   └── 提出改进假设（如：增加示例、调整措辞）

4. 实验验证
   └── A/B 测试新旧 prompt

5. 结果分析
   └── 统计显著性检验

6. 部署或回滚
   └── 选择表现更好的版本
```

---

## 💡 我的思考

1. **Prompt 优化是数据科学**：需要系统化的实验设计、指标定义和统计分析，而不是凭感觉的调参。

2. **自动优化是趋势**：手动调优 prompt 的效率太低。DSPy、OPRO 等框架代表了未来方向——让算法来优化算法。

3. **评估是最难的环节**：没有好的评估，就无法判断优化是否有效。建议尽早建立自动评估 pipeline。

4. **Prompt 版本管理很重要**：使用 Git 管理 prompt，记录每次变更和对应的效果，便于回溯和对比。

5. **过度优化有风险**：针对特定测试集过度优化可能导致过拟合。保持测试集的独立性和多样性。

---

## 参考来源

- **DSPy**: [dspy-docs.vercel.app](https://dspy-docs.vercel.app/) — 访问日期：2026-06-07
- **APE**: "Large Language Models Are Human-Level Prompt Engineers" (Zhou et al., 2022) — [arxiv:2211.01910](https://arxiv.org/abs/2211.01910)
- **OPRO**: "Large Language Models as Optimizers" (Yang et al., 2023) — [arxiv:2309.03409](https://arxiv.org/abs/2309.03409)
- **PromptBreeder**: "Self-Referential Self-Improvement Via Prompt Evolution" (Fernando et al., 2023) — [arxiv:2309.16797](https://arxiv.org/abs/2309.16797)

---

*访问日期: 2026-06-07*
