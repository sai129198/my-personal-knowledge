#topic/agent #product/langchain #year/2026 #status/draft

# Agent 架构模式详解

> 基于 ReAct、Plan-and-Execute、Multi-Agent 等核心模式的深度分析，结合 Anthropic Research 最新进展。

---

## 1. Agent 核心概念

### 1.1 什么是 Agent

Agent（智能代理）是能够**自主感知环境、做出决策并执行动作**的 AI 系统。与简单的 LLM 调用不同，Agent 具备：

- **规划能力**：将复杂任务分解为子任务
- **记忆能力**：维护对话历史和知识
- **工具使用**：调用外部 API、执行代码、查询数据库
- **自主决策**：根据环境反馈调整行动

### 1.2 Agent vs. 传统应用

| 维度 | 传统 LLM 应用 | Agent |
|------|--------------|-------|
| 交互方式 | 单轮/多轮对话 | 自主执行多步骤任务 |
| 工具使用 | 无或简单调用 | 动态选择和使用工具 |
| 错误处理 | 用户发现 | 自我检测和修复 |
| 状态管理 | 简单上下文 | 复杂状态机 |
| 适用场景 | 问答、生成 | 复杂工作流、自动化 |

---

## 2. ReAct 模式（Reasoning + Acting）

### 2.1 核心思想

ReAct 由 Google 在 2022 年提出，核心思想：**推理（Reasoning）和行动（Acting）交替进行**。模型不仅输出最终答案，还显式输出思考过程。

### 2.2 执行循环

```
观察（Observation）→ 思考（Thought）→ 行动（Action）→ 观察（Observation）→ ...
```

### 2.3 示例

```
问题：北京今天的天气如何？适合穿什么衣服？

思考 1：我需要知道北京今天的天气信息。
行动 1：调用 weather_api(location="北京")
观察 1：{ "temperature": 25, "condition": "晴天", "humidity": 40% }

思考 2：25°C 晴天，湿度适中，适合穿短袖或薄长袖。
行动 2：输出最终答案

答案：北京今天晴天，25°C，建议穿短袖或薄长袖，可以带一件薄外套备用。
```

### 2.4 实现（LangChain）

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain import hub

# 定义工具
tools = [
    Tool(
        name="weather_api",
        func=get_weather,
        description="获取指定城市的天气信息"
    ),
    Tool(
        name="search",
        func=web_search,
        description="搜索网络信息"
    )
]

# 加载 ReAct Prompt
prompt = hub.pull("hwchase17/react")

# 创建 Agent
llm = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(llm, tools, prompt)

# 执行
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({"input": "北京今天天气如何？"})
```

### 2.5 优缺点

| 优点 ✅ | 缺点 ❌ |
|---------|---------|
| 思考过程透明 | 容易陷入循环 |
| 错误可追踪 | 对简单任务过于复杂 |
| 适合多步骤任务 | 工具调用失败时难以恢复 |

---

## 3. Plan-and-Execute 模式

### 3.1 核心思想

先制定完整计划，再逐步执行。适合**需要长期规划**的复杂任务。

### 3.2 执行流程

```
用户输入 → [Planner] 生成计划 → [Executor] 执行步骤 → [检查结果] → 完成/重规划
```

### 3.3 示例

```
任务：帮我规划一次 3 天 2 晚的上海旅行，预算 3000 元。

计划：
1. 搜索上海热门景点
2. 查询酒店价格（2晚，预算 800-1200）
3. 规划每日行程
4. 计算交通和餐饮费用
5. 汇总总预算

执行步骤 1：...
执行步骤 2：...
...
```

### 3.4 实现

```python
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

# 创建规划器和执行器
planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)

# 创建 Plan-and-Execute Agent
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

result = agent.invoke("帮我规划一次 3 天 2 晚的上海旅行，预算 3000 元")
```

### 3.5 与 ReAct 对比

| 维度 | ReAct | Plan-and-Execute |
|------|-------|------------------|
| 规划方式 | 边做边想 | 先想后做 |
| 适用任务 | 短步骤、需灵活调整 | 长步骤、需整体规划 |
| 错误恢复 | 即时调整 | 可能需要重规划 |
| 透明度 | 每步都可见 | 计划清晰，执行细节可选 |

---

## 4. Multi-Agent 模式

### 4.1 核心思想

多个 Agent 协作完成任务，模拟人类团队协作。每个 Agent 有特定角色和职责。

### 4.2 典型架构

```
[用户] → [协调者 Agent]
              ↓
    ┌─────────┼─────────┐
    ↓         ↓         ↓
[研究 Agent] [写作 Agent] [审核 Agent]
    └─────────┬─────────┘
              ↓
         [整合 Agent] → [输出]
```

### 4.3 角色设计示例

```python
# 研究 Agent
researcher = Agent(
    role="研究员",
    goal="收集全面、准确的信息",
    backstory="你是一位资深研究员，擅长快速收集和整理信息",
    tools=[search_tool, web_scraper],
    llm=llm
)

# 写作 Agent
writer = Agent(
    role="作家",
    goal="将信息转化为高质量内容",
    backstory="你是一位专业作家，擅长将复杂信息转化为易懂的文章",
    tools=[outline_tool],
    llm=llm
)

# 审核 Agent
reviewer = Agent(
    role="审核员",
    goal="确保内容准确、完整",
    backstory="你是一位严格的编辑，擅长发现错误和遗漏",
    tools=[fact_check_tool],
    llm=llm
)
```

### 4.4 协作模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **顺序协作** | Agent A → Agent B → Agent C | 流水线任务 |
| **并行协作** | 多个 Agent 同时工作，最后整合 | 独立子任务 |
| **讨论协作** | Agent 之间多轮讨论达成共识 | 需要多方意见的决策 |
| **层级协作** | 管理者分配任务，执行者完成 | 复杂项目管理 |

### 4.5 实现（CrewAI 示例）

```python
from crewai import Agent, Task, Crew

# 定义任务
task = Task(
    description="研究 AI Agent 最新进展并撰写报告",
    agent=researcher,
    expected_output="一份 2000 字的调研报告"
)

# 创建团队
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[task],
    verbose=True
)

# 执行
result = crew.kickoff()
```

---

## 5. 工具设计与注册

### 5.1 工具设计原则

1. **单一职责**：每个工具只做一件事
2. **清晰描述**：描述要准确，让 Agent 知道何时使用
3. **错误处理**：返回清晰的错误信息
4. **输入验证**：验证参数合法性

### 5.2 工具注册示例

```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    location: str = Field(description="城市名称，如'北京'")
    date: str = Field(description="日期，格式'YYYY-MM-DD'，默认为今天")

class WeatherTool(BaseTool):
    name = "weather"
    description = "获取指定城市的天气信息"
    args_schema = WeatherInput
    
    def _run(self, location: str, date: str = None):
        # 实现逻辑
        return get_weather_data(location, date)
    
    async def _arun(self, location: str, date: str = None):
        # 异步实现
        return await get_weather_data_async(location, date)
```

---

## 6. 记忆管理

### 6.1 记忆类型

| 类型 | 说明 | 实现方式 |
|------|------|----------|
| **短期记忆** | 当前对话历史 | 直接传入上下文 |
| **长期记忆** | 跨会话知识 | 向量数据库 |
| **实体记忆** | 关于用户/世界的信息 | 知识图谱 |
| **过程记忆** | 任务执行经验 | 日志 + 总结 |

### 6.2 实现示例

```python
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory

# 短期记忆
short_term_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 长期记忆
long_term_memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever()
)

# 组合使用
agent = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=short_term_memory,  # 当前对话
    verbose=True
)
```

---

## 7. 前沿进展

### 7.1 Anthropic 的 Computer Use

Anthropic 最新发布的 Computer Use 功能让 Claude 能够：
- 查看屏幕截图
- 移动鼠标、点击、输入
- 操作计算机完成复杂任务

这标志着 Agent 从"文本世界"走向"物理世界"。

### 7.2 OpenAI 的 Operator

OpenAI 的 Operator 能够：
- 自主浏览网页
- 填写表单
- 完成在线购物等任务

### 7.3 行业趋势

根据 DeepLearning.AI The Batch 分析：
- **Agent 框架成熟化**：LangChain、LlamaIndex、CrewAI 等框架快速迭代
- **多 Agent 协作**：从单 Agent 向 Multi-Agent 系统演进
- **工具生态爆发**：越来越多的专用工具被开发出来
- **评估标准建立**：如何评估 Agent 性能成为关键问题

---

## 💡 我的思考

1. **ReAct 是大多数场景的最佳起点**：简单、透明、易于调试。只有当任务确实需要长期规划时才考虑 Plan-and-Execute。

2. **Multi-Agent 是"高级玩法"**：虽然概念很酷，但协调多个 Agent 的复杂度很高。建议先从单 Agent 做起，确实有明确分工需求时再引入 Multi-Agent。

3. **工具设计是 Agent 质量的关键**：Agent 的能力上限取决于工具的质量。投入时间设计好工具接口和错误处理，比优化 Prompt 更重要。

4. **记忆管理是被忽视的难点**：短期记忆容易实现，但长期记忆（如何有效存储和检索跨会话信息）仍然是个挑战。

5. **Agent 的可靠性仍是最大问题**：当前 Agent 在复杂任务上的成功率还不够高（约 60-80%），错误累积是主要挑战。建议在生产环境中设置人工检查点。

6. **2026 年将是 Agent 落地年**：随着 Computer Use、Operator 等能力的成熟，Agent 将从演示走向实际应用。但可靠性和安全性仍是主要障碍。

---

## 参考来源

- [ReAct Paper: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 访问日期：2026-06-05
- [Anthropic: Building effective agents](https://www.anthropic.com/research/building-effective-agents) — 访问日期：2026-06-05
- [LangChain Agent Documentation](https://python.langchain.com/docs/modules/agents/) — 访问日期：2026-06-05
- [CrewAI Documentation](https://docs.crewai.com/) — 访问日期：2026-06-05
- [DeepLearning.AI The Batch: Agents Drive Online Traffic](https://www.deeplearning.ai/the-batch/issue-355) — 访问日期：2026-06-05
