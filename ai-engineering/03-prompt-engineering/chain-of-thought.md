# Chain-of-Thought 思维链技术

> **一句话定位**：通过引导模型生成中间推理步骤，显著提升复杂推理任务的准确性。
>
> #status/canonical #topic/prompt-engineering #topic/reasoning #year/2026

---

## 1. 核心概念

### 1.1 什么是 Chain-of-Thought (CoT)

Chain-of-Thought 是一种提示工程技术，通过在提示中展示**中间推理步骤**，引导语言模型逐步思考，最终得出正确答案。

**标准提示 vs CoT 提示**：

```
标准提示：
Q: 一个农场有 5 只鸡，每只鸡每天下 2 个蛋。
   3 天后总共有多少个蛋？
A: 30

CoT 提示：
Q: 一个农场有 5 只鸡，每只鸡每天下 2 个蛋。
   3 天后总共有多少个蛋？
A: 让我们逐步思考：
   - 每只鸡每天下 2 个蛋
   - 5 只鸡每天下 5 × 2 = 10 个蛋
   - 3 天总共下 10 × 3 = 30 个蛋
   所以答案是 30
```

### 1.2 为什么 CoT 有效

| 机制 | 说明 |
|------|------|
| **分解复杂问题** | 将多步推理拆解为简单步骤 |
| **增加计算步骤** | 模型有更多 token 进行"思考" |
| **利用训练数据** | 大型模型在预训练中见过类似推理模式 |
| **错误定位** | 可以检查中间步骤找出错误 |

### 1.3 适用场景

✅ **适合使用 CoT**：
- 数学问题（算术、代数、几何）
- 逻辑推理（谜题、逻辑题）
- 多步骤任务（规划、调度）
- 符号推理（代码、公式推导）

❌ **不适合使用 CoT**：
- 简单事实问答（"法国首都是什么？"）
- 创意写作（故事、诗歌）
- 单步分类任务（情感分析）

---

## 2. CoT 变体技术

### 2.1 Zero-Shot CoT

**核心思想**：不加示例，仅通过指令触发推理

```python
prompt = """
Q: {question}
A: Let's think step by step.
"""
```

**效果**：
- 无需手动编写示例
- 在足够大的模型上（>100B）效果良好
- 比标准 Zero-Shot 提升显著

### 2.2 Few-Shot CoT

**核心思想**：提供几个带推理过程的示例

```python
prompt = """
Q: 一个农场有 3 只鸡，每只鸡每天下 2 个蛋。
   4 天后总共有多少个蛋？
A: 让我们逐步思考：
   - 每只鸡每天下 2 个蛋
   - 3 只鸡每天下 3 × 2 = 6 个蛋
   - 4 天总共下 6 × 4 = 24 个蛋
   所以答案是 24

Q: 一个果园有 4 棵树，每棵树结 5 个苹果。
   2 天后总共收获多少个苹果？
A: 让我们逐步思考：
   - 每棵树结 5 个苹果
   - 4 棵树结 4 × 5 = 20 个苹果
   - 注意：苹果数量不随天数增加
   所以答案是 20

Q: {question}
A: 让我们逐步思考：
"""
```

### 2.3 Self-Consistency CoT

**核心思想**：生成多个推理路径，选择最一致的答案

```python
# 1. 多次采样生成答案
answers = []
for _ in range(10):
    response = model.generate(prompt, temperature=0.7)
    answers.append(extract_answer(response))

# 2. 投票选择最常见答案
final_answer = most_common(answers)
```

**效果提升**：
- GSM8K 数据集：从 56.5% → 74.4%
- 需要更多推理成本，但准确性显著提升

### 2.4 Tree of Thoughts (ToT)

**核心思想**：维护多个推理路径，主动评估和选择

```
        [初始问题]
       /    |    \
    [思路1] [思路2] [思路3]
     /   \    |      |
  [评价] [评价] [评价] [评价]
    |      |     |      |
   继续   继续   放弃   继续
    ...
```

**适用场景**：
- 需要探索的复杂问题（24点游戏、创意写作）
- 有明确评估标准的任务

### 2.5 Automatic CoT (Auto-CoT)

**核心思想**：自动构建示例，无需人工编写

```python
# 1. 聚类问题
clusters = cluster_questions(questions, k=10)

# 2. 从每个聚类选择代表性问题
representatives = [cluster.center for cluster in clusters]

# 3. 使用 Zero-Shot CoT 生成推理过程
for q in representatives:
    cot = model.generate(f"Q: {q}\nA: Let's think step by step.")
    examples.append((q, cot))

# 4. 使用生成的示例进行 Few-Shot CoT
```

---

## 3. 实践技巧

### 3.1 提示模板设计

**基础模板**：

```markdown
## 角色设定
你是一位擅长逐步推理的 AI 助手。

## 任务
请回答以下问题，并展示完整的推理过程。

## 格式要求
1. 首先理解问题
2. 列出已知条件
3. 逐步推导
4. 给出最终答案

## 问题
{question}

## 回答
```

### 3.2 示例选择策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| **多样性采样** | 选择不同类型的问题 | 通用任务 |
| **相似性采样** | 选择与测试问题相似的示例 | 特定领域 |
| **复杂度递增** | 从简单到复杂排列示例 | 教学场景 |

### 3.3 常见错误与解决

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| **推理跳跃** | 模型跳过关键步骤 | 明确要求"不要跳过任何步骤" |
| **计算错误** | 中间步骤算术错误 | 要求"每步都验证计算" |
| **偏离问题** | 推理方向错误 | 要求"先理解问题再回答" |
| **过早结论** | 未完成推理就给出答案 | 要求"完成所有步骤后再给答案" |

---

## 4. 高级应用

### 4.1 CoT + 工具使用

```python
# ReAct 模式：Reasoning + Acting
prompt = """
你可以使用以下工具：
- Calculator: 进行数学计算
- Search: 搜索事实信息

Q: 2023 年诺贝尔奖物理学奖得主是谁？
   他/她的年龄加上 10 是多少？

思考 1: 我需要先找到 2023 年物理学奖得主
行动 1: Search["2023 Nobel Prize Physics winner"]
观察 1: Anne L'Huillier, Ferenc Krausz, Pierre Agostini

思考 2: 我需要知道 Anne L'Huillier 的年龄
行动 2: Search["Anne L'Huillier age"]
观察 2: Born in 1958

思考 3: 2023 - 1958 = 65，65 + 10 = 75
行动 3: Calculator["65 + 10"]
观察 3: 75

思考 4: 答案是 75
最终答案: 75
"""
```

### 4.2 CoT + 验证

```python
# 自我验证
prompt = """
Q: {question}

A: 让我们逐步思考：
{reasoning_steps}

验证：让我们检查每个步骤：
步骤 1: {step1} → 正确/错误，因为...
步骤 2: {step2} → 正确/错误，因为...
...

最终确认答案: {final_answer}
"""
```

### 4.3 多语言 CoT

```python
# 中文 CoT 示例
prompt = """
Q: 一个水池有进水管和出水管。
   进水管单独注满需要 6 小时，出水管单独排空需要 4 小时。
   同时打开两个水管，多久能注满水池？

A: 让我们逐步分析：
   第一步：理解问题
   - 进水管效率：1/6（每小时注满 1/6）
   - 出水管效率：1/4（每小时排空 1/4）
   
   第二步：计算净效率
   - 同时打开：1/6 - 1/4 = 2/12 - 3/12 = -1/12
   - 等等，结果是负数，说明排水比进水快
   
   第三步：重新理解
   - 这意味着水池永远不会被注满，反而会被排空
   
   最终答案：无法注满，水会被排空
"""
```

---

## 5. 评估与优化

### 5.1 评估指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **准确率** | 最终答案正确率 | > 80% |
| **步骤正确率** | 中间步骤正确比例 | > 90% |
| **推理完整性** | 是否包含所有必要步骤 | 100% |
| **一致性** | 多次采样答案一致率 | > 70% |

### 5.2 优化方法

```python
# 1. 提示调优
# 尝试不同措辞
prompts = [
    "Let's think step by step",
    "让我们逐步思考",
    "请详细展示推理过程",
    "First, understand the problem, then solve it"
]

# 2. 示例优化
# 使用 Auto-CoT 自动选择最佳示例

# 3. 后处理验证
# 使用代码执行验证数学计算

# 4. 集成方法
# 结合 Self-Consistency 和验证
```

---

## 6. 最佳实践

### 6.1 设计原则

```markdown
1. **明确性**：明确要求"逐步思考"
2. **结构化**：使用编号步骤
3. **完整性**：不跳过任何推理步骤
4. **可验证性**：每个步骤都可检查
```

### 6.2 检查清单

```markdown
□ 是否明确要求展示推理过程？
□ 示例是否覆盖目标问题类型？
□ 示例复杂度是否与测试问题匹配？
□ 是否包含验证步骤？
□ 是否处理了边界情况？
```

---

## 参考资源

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) - Wei et al., 2022
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) - Yao et al., 2023
- [Automatic Chain of Thought Prompting in Large Language Models](https://arxiv.org/abs/2210.03493) - Zhang et al., 2022
- [Prompt Engineering Guide - CoT](https://www.promptingguide.ai/techniques/cot)
