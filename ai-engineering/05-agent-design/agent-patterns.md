#topic/agent #topic/design-patterns #topic/multi-agent #year/2026 #status/draft

# Agent 设计模式

> 从单 Agent 到多 Agent 协作：ReAct、Plan-and-Execute、Multi-Agent 等核心模式详解。

---

## 1. ReAct 模式

### 1.1 核心思想

**ReAct = Reasoning（推理）+ Acting（行动）**

交替进行思考和行动，形成"思考-行动-观察"循环。

### 1.2 工作流程

```
用户输入："2024 年诺贝尔物理学奖得主是谁？"

Step 1 - Thought:
"用户询问 2024 年诺贝尔物理学奖得主。我需要搜索最新信息。"

Step 2 - Action:
search("2024 Nobel Prize Physics winner")

Step 3 - Observation:
"2024 年诺贝尔物理学奖授予 John J. Hopfield 和 Geoffrey E. Hinton..."

Step 4 - Thought:
"我已经找到了答案，可以直接回答用户。"

Step 5 - Action:
finish("2024 年诺贝尔物理学奖得主是 John J. Hopfield 和 Geoffrey E. Hinton...")
```

### 1.3 实现代码

```python
class ReActAgent:
    def __init__(self, llm, tools, max_iterations=10):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
    
    def run(self, query):
        context = f"Question: {query}\n"
        
        for i in range(self.max_iterations):
            # 生成 Thought + Action
            response = self.llm.generate(
                self._build_prompt(context),
                stop=["\nObservation:"]
            )
            
            thought, action = self._parse_response(response)
            context += f"Thought: {thought}\nAction: {action}\n"
            
            # 执行 Action
            if action.startswith("finish"):
                return action[7:]  # 提取答案
            
            observation = self._execute_action(action)
            context += f"Observation: {observation}\n"
        
        return "达到最大迭代次数，未能完成任务。"
    
    def _build_prompt(self, context):
        return f"""你是一位助手，通过思考和行动来解决问题。
可用工具：{self._tool_descriptions()}

格式：
Thought: [你的思考]
Action: [工具名]([参数])

{context}"""
```

### 1.4 适用场景

- ✅ 需要实时信息的问答
- ✅ 多步骤工具调用
- ✅ 探索性任务
- ❌ 确定性计算（浪费 token）
- ❌ 简单单步任务

---

## 2. Plan-and-Execute 模式

### 2.1 核心思想

**先制定完整计划，再按步骤执行。**

与 ReAct 的区别：
- ReAct：边做边想，动态调整
- Plan-and-Execute：先规划后执行，适合确定性任务

### 2.2 工作流程

```
用户输入："帮我分析 Q3 财报并生成 PPT"

Phase 1 - Planning:
1. 获取 Q3 财务数据
2. 计算关键指标（营收、利润、增长率）
3. 与 Q2 和去年同期对比
4. 识别趋势和异常
5. 生成 PPT 大纲
6. 创建 PPT 文件

Phase 2 - Execution:
Step 1: 调用财务 API 获取数据
Step 2: 使用计算器处理数据
Step 3: 使用分析工具对比
Step 4: LLM 生成洞察
Step 5: LLM 生成大纲
Step 6: 调用 PPT 生成工具
```

### 2.3 实现代码

```python
class PlanAndExecuteAgent:
    def __init__(self, llm, executor):
        self.llm = llm
        self.executor = executor
    
    def run(self, task):
        # Phase 1: 制定计划
        plan = self._create_plan(task)
        
        # Phase 2: 执行计划
        results = []
        for step in plan.steps:
            result = self.executor.execute(step)
            results.append(result)
            
            # 可选：执行中调整计划
            if self._need_replan(results):
                plan = self._replan(task, results)
        
        # 综合结果
        return self._synthesize(results)
    
    def _create_plan(self, task):
        prompt = f"""请为以下任务制定详细执行计划：
任务：{task}

要求：
1. 步骤具体可执行
2. 考虑依赖关系
3. 预估每步耗时

请按以下格式输出：
Plan:
1. [步骤 1]
2. [步骤 2]
..."""
        
        response = self.llm.generate(prompt)
        return self._parse_plan(response)
```

### 2.4 适用场景

- ✅ 复杂多步骤任务
- ✅ 需要资源协调
- ✅ 可预见执行路径
- ❌ 需要大量探索的任务
- ❌ 环境动态变化

---

## 3. Multi-Agent 模式

### 3.1 核心思想

**多个 Agent 协作，各自负责不同角色或子任务。**

### 3.2 常见拓扑

#### 层级式（Hierarchical）

```
        ┌─────────┐
        │ Manager │  ← 任务分配、结果汇总
        │  Agent  │
        └────┬────┘
             │
    ┌────────┼────────┐
    │        │        │
    ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐
│Worker│ │Worker│ │Worker│
│  A   │ │  B   │ │  C   │
└──────┘ └──────┘ └──────┘
```

#### 对等式（Peer-to-Peer）

```
┌──────┐      ┌──────┐
│Agent │◄────►│Agent │
│  A   │      │  B   │
└──┬───┘      └──┬───┘
   │             │
   └──────┬──────┘
          │
     ┌────┴────┐
     │  Shared │
     │  Memory │
     └─────────┘
```

#### 管道式（Pipeline）

```
Input → [Agent A] → [Agent B] → [Agent C] → Output
          提取        分析        生成
```

### 3.3 实现：CrewAI 风格

```python
from crewai import Agent, Task, Crew

# 定义 Agent
researcher = Agent(
    role='研究员',
    goal='收集和分析信息',
    backstory='你是一位资深行业研究员',
    tools=[search_tool, calculator],
    verbose=True
)

writer = Agent(
    role='写手',
    goal='撰写高质量报告',
    backstory='你是一位技术写作专家',
    tools=[doc_tool],
    verbose=True
)

# 定义任务
research_task = Task(
    description='研究 2024 年 AI 行业趋势',
    agent=researcher,
    expected_output='一份详细的研究摘要'
)

writing_task = Task(
    description='基于研究结果撰写报告',
    agent=writer,
    context=[research_task],
    expected_output='一份完整的行业报告'
)

# 组建团队
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process='sequential'  # sequential/hierarchical
)

result = crew.kickoff()
```

### 3.4 通信协议

| 协议 | 说明 | 适用场景 |
|------|------|----------|
| **消息队列** | Agent 通过队列异步通信 | 解耦、高并发 |
| **共享内存** | 共享状态/上下文 | 实时协作 |
| **函数调用** | 直接调用其他 Agent | 同步协作 |
| **事件驱动** | 发布-订阅模式 | 动态响应 |

---

## 4. 反思模式（Reflection）

### 4.1 核心思想

**生成初稿 → 自我批评 → 改进 → 重复**

### 4.2 实现

```python
class ReflectionAgent:
    def __init__(self, llm, max_reflections=3):
        self.llm = llm
        self.max_reflections = max_reflections
    
    def run(self, task):
        # 生成初稿
        draft = self.llm.generate(f"请完成以下任务：\n{task}")
        
        for i in range(self.max_reflections):
            # 反思
            critique = self.llm.generate(
                f"请 critique 以下输出，指出问题和改进点：\n{draft}"
            )
            
            # 检查是否满意
            if self._is_satisfactory(critique):
                break
            
            # 改进
            draft = self.llm.generate(
                f"请根据以下 critique 改进输出：\n"
                f"原输出：{draft}\n"
                f"Critique：{critique}"
            )
        
        return draft
```

### 4.3 Reflexion 变体

引入外部评估和记忆：

```python
class ReflexionAgent:
    def __init__(self, llm, evaluator):
        self.llm = llm
        self.evaluator = evaluator
        self.memory = []  # 存储失败经验
    
    def run(self, task):
        for attempt in range(max_attempts):
            # 基于记忆调整策略
            strategy = self._select_strategy(task)
            
            # 执行
            result = self._execute(task, strategy)
            
            # 评估
            score = self.evaluator.evaluate(result)
            
            if score >= threshold:
                return result
            
            # 记录失败经验
            self.memory.append({
                "task": task,
                "strategy": strategy,
                "result": result,
                "score": score,
                "lesson": self._extract_lesson(result, score)
            })
        
        return result  # 返回最佳尝试
```

---

## 5. 模式选择指南

```
任务分析
    │
    ├─ 需要多个角色协作？
    │   ├─ 是 → Multi-Agent
    │   └─ 否 → 继续
    │
    ├─ 执行路径可预见？
    │   ├─ 是 → Plan-and-Execute
    │   └─ 否 → 继续
    │
    ├─ 需要实时信息/工具？
    │   ├─ 是 → ReAct
    │   └─ 否 → 继续
    │
    ├─ 需要高质量输出？
    │   ├─ 是 → Reflection
    │   └─ 否 → 直接 LLM
    │
    └─ 简单任务 → 直接 LLM
```

---

## 💡 我的思考

1. **没有最好的模式，只有最合适的模式**：实际系统中往往需要组合多种模式。

2. **Multi-Agent 的通信开销**：Agent 之间的通信成本不容忽视，设计时要考虑效率。

3. **反思模式的质量取决于评估器**：没有好的评估标准，反思就是无的放矢。

4. **Plan-and-Execute 的脆弱性**：计划赶不上变化，需要设计好 replan 机制。

5. **模式正在快速演化**：从 ReAct 到 Multi-Agent 到 AutoGPT，Agent 架构还在快速迭代。

---

## 参考来源

- **ReAct**: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022) — [arxiv:2210.03629](https://arxiv.org/abs/2210.03629)
- **Plan-and-Solve**: "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning" (Wang et al., 2023) — [arxiv:2305.04091](https://arxiv.org/abs/2305.04091)
- **Reflexion**: "Reflexion: Self-Reflective Agents with Verbal Reinforcement Learning" (Shinn et al., 2023) — [arxiv:2303.11366](https://arxiv.org/abs/2303.11366)
- **CrewAI**: [docs.crewai.com](https://docs.crewai.com/) — 访问日期：2026-06-07
- **AutoGen**: [microsoft.github.io/autogen](https://microsoft.github.io/autogen/) — 访问日期：2026-06-07

---

*访问日期: 2026-06-07*
