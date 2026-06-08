# ReAct 模式实现详解

> **一句话定位**：通过推理（Reasoning）与行动（Acting）的交替循环，让 LLM 能够自主解决复杂任务。
>
> #status/canonical #topic/agent #topic/react #year/2026

---

## 1. ReAct 核心概念

### 1.1 什么是 ReAct

ReAct（Reasoning + Acting）是一种让 LLM 交替进行推理和行动的 Agent 架构模式。

```
传统 Chain-of-Thought：
问题 → 推理 → 推理 → 推理 → 答案

ReAct：
问题 → 推理 → 行动 → 观察 → 推理 → 行动 → 观察 → 答案
```

**核心洞察**：推理帮助模型规划行动，行动帮助模型获取外部信息，两者形成正反馈循环。

### 1.2 ReAct vs 其他模式

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| **CoT** | 纯推理，无外部交互 | 数学、逻辑题 |
| **Tool Use** | 直接调用工具，无显式推理 | 简单 API 调用 |
| **ReAct** | 推理+行动交替 | 复杂多步任务 |
| **Plan-and-Execute** | 先规划后执行 | 确定性任务 |

---

## 2. ReAct 架构详解

### 2.1 基本循环

```python
class ReActAgent:
    def __init__(self, llm, tools, max_iterations=10):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
    
    def run(self, query):
        """
        ReAct 主循环
        """
        # 初始化
        context = f"Question: {query}\n"
        
        for i in range(self.max_iterations):
            # 1. 思考（Thought）
            thought = self._think(context)
            context += f"Thought {i+1}: {thought}\n"
            
            # 2. 行动（Action）
            action = self._act(context)
            context += f"Action {i+1}: {action}\n"
            
            # 检查是否完成
            if action.type == "finish":
                return action.answer
            
            # 3. 观察（Observation）
            observation = self._observe(action)
            context += f"Observation {i+1}: {observation}\n"
        
        # 超过最大迭代次数
        return self._fallback_answer(context)
    
    def _think(self, context):
        """生成思考过程"""
        prompt = f"""
        {context}
        Based on the above, what should I do next? Think step by step.
        Thought:"""
        
        return self.llm.generate(prompt)
    
    def _act(self, context):
        """生成行动"""
        prompt = f"""
        {context}
        What action should I take? Choose one:
        - Search[query]: Search for information
        - Calculate[expression]: Calculate a value
        - Lookup[key]: Look up a fact
        - Finish[answer]: Provide the final answer
        
        Action:"""
        
        action_str = self.llm.generate(prompt)
        return self._parse_action(action_str)
    
    def _observe(self, action):
        """执行行动并观察结果"""
        if action.type == "search":
            return self.tools["search"].run(action.params)
        elif action.type == "calculate":
            return self.tools["calculator"].run(action.params)
        elif action.type == "lookup":
            return self.tools["lookup"].run(action.params)
        else:
            return "Invalid action"
```

### 2.2 完整示例

```python
class ReActExample:
    """
    示例：使用 ReAct 回答复杂问题
    """
    
    def __init__(self):
        self.tools = {
            "search": SearchTool(),
            "calculator": CalculatorTool(),
            "wikipedia": WikipediaTool()
        }
    
    def solve(self, question):
        """
        问题：2024年诺贝尔奖物理学奖得主是谁？他/她的主要贡献是什么？
        """
        
        # 初始上下文
        context = f"""Question: {question}
You have access to the following tools:
- search[query]: Search the web for information
- calculator[expression]: Calculate mathematical expressions
- wikipedia[topic]: Look up information on Wikipedia

Solve this step by step."""
        
        # 迭代 1
        thought1 = "I need to find out who won the 2024 Nobel Prize in Physics."
        action1 = "search[2024 Nobel Prize Physics winner]"
        observation1 = "The 2024 Nobel Prize in Physics was awarded to John J. Hopfield and Geoffrey E. Hinton."
        
        # 迭代 2
        thought2 = "Now I need to find their main contributions. Let me search for Hopfield's work."
        action2 = "wikipedia[John Hopfield physicist]"
        observation2 = "John Hopfield is known for the Hopfield network, a form of recurrent artificial neural network."
        
        # 迭代 3
        thought3 = "Now let me search for Geoffrey Hinton's contributions."
        action3 = "wikipedia[Geoffrey Hinton]"
        observation3 = "Geoffrey Hinton is known for his work on deep learning, backpropagation, and Boltzmann machines."
        
        # 迭代 4
        thought4 = "I now have all the information needed to answer the question."
        action4 = "finish[The 2024 Nobel Prize in Physics was awarded to John J. Hopfield and Geoffrey E. Hinton. Hopfield is known for the Hopfield network, while Hinton is known for his work on deep learning and backpropagation.]"
        
        return action4.answer
```

---

## 3. ReAct 实现模式

### 3.1 基础实现

```python
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Action:
    type: str
    params: str
    
    @property
    def is_finish(self):
        return self.type == "finish"

class Tool:
    def __init__(self, name: str, func):
        self.name = name
        self.func = func
    
    def run(self, params: str) -> str:
        return self.func(params)

class ReActAgent:
    def __init__(self, llm, tools: List[Tool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.tool_descriptions = self._build_tool_descriptions()
    
    def _build_tool_descriptions(self) -> str:
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.func.__doc__}")
        return "\n".join(descriptions)
    
    def run(self, query: str, max_iterations: int = 10) -> str:
        context = self._build_initial_prompt(query)
        
        for i in range(max_iterations):
            # 生成 Thought + Action
            response = self.llm.generate(context + "\nThought:")
            
            # 解析 Thought 和 Action
            thought, action = self._parse_response(response)
            
            if action.is_finish:
                return action.params
            
            # 执行 Action
            observation = self._execute_action(action)
            
            # 更新上下文
            context += f"\nThought: {thought}\nAction: {action.type}[{action.params}]\nObservation: {observation}"
        
        return "Max iterations reached without finding an answer."
    
    def _build_initial_prompt(self, query: str) -> str:
        return f"""Answer the following question by thinking step by step and using tools when needed.

Available tools:
{self.tool_descriptions}

Use the following format:
Thought: your reasoning about what to do
Action: tool_name[parameters]
Observation: result of the action

Question: {query}"""
    
    def _parse_response(self, response: str) -> tuple:
        """解析 LLM 响应，提取 Thought 和 Action"""
        lines = response.strip().split('\n')
        
        thought = ""
        action = None
        
        for line in lines:
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action_str = line.replace("Action:", "").strip()
                action = self._parse_action(action_str)
        
        return thought, action
    
    def _parse_action(self, action_str: str) -> Action:
        """解析 Action 字符串"""
        # 格式: tool_name[parameters]
        import re
        match = re.match(r'(\w+)\[(.*)\]', action_str)
        
        if match:
            return Action(type=match.group(1), params=match.group(2))
        elif "finish" in action_str.lower():
            return Action(type="finish", params=action_str)
        else:
            return Action(type="unknown", params=action_str)
    
    def _execute_action(self, action: Action) -> str:
        """执行 Action"""
        if action.type in self.tools:
            return self.tools[action.type].run(action.params)
        else:
            return f"Error: Unknown tool '{action.type}'"
```

### 3.2 使用 LangChain 实现

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain import hub

# 定义工具
tools = [
    Tool(
        name="search",
        func=search_engine.run,
        description="Search the internet for information"
    ),
    Tool(
        name="calculator",
        func=calculator.run,
        description="Calculate mathematical expressions"
    )
]

# 加载 ReAct Prompt
prompt = hub.pull("hwchase17/react")

# 创建 Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = create_react_agent(llm, tools, prompt)

# 执行
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({"input": "What is the population of France?"})
```

### 3.3 使用 LangGraph 实现（推荐）

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

class ReActState(TypedDict):
    messages: Annotated[list, "add_messages"]
    next_step: str

# 定义工具
tools = [search_tool, calculator_tool]
tool_node = ToolNode(tools)

# 定义 LLM 节点
def agent_node(state: ReActState):
    """Agent 决策节点"""
    messages = state["messages"]
    
    # 调用 LLM 决定下一步
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": [response],
        "next_step": "tools" if response.tool_calls else "end"
    }

# 构建图
workflow = StateGraph(ReActState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    lambda state: state["next_step"],
    {"tools": "tools", "end": END}
)
workflow.add_edge("tools", "agent")

# 编译
app = workflow.compile()

# 运行
result = app.invoke({
    "messages": [("human", "What is the weather in Paris?")]
})
```

---

## 4. ReAct 优化技巧

### 4.1 提示工程

```python
class OptimizedReActPrompt:
    def build_prompt(self, query, tools, examples=None):
        """
        构建优化的 ReAct Prompt
        """
        prompt = f"""You are a helpful AI assistant that can use tools to solve problems.

## Tools
{self._format_tools(tools)}

## Instructions
1. Think step by step about what you need to do
2. Use tools to gather information when needed
3. Always verify facts before making conclusions
4. If you're unsure, say so rather than guessing

## Format
Thought: your reasoning
Action: tool_name[parameters]
Observation: result
... (repeat as needed)
Thought: I have enough information
Action: finish[your answer]
"""
        
        # 添加示例（Few-Shot）
        if examples:
            prompt += f"\n## Examples\n{self._format_examples(examples)}\n"
        
        prompt += f"\n## Your Task\nQuestion: {query}\nThought:"
        
        return prompt
    
    def _format_tools(self, tools):
        """格式化工具描述"""
        return "\n".join([
            f"- {tool.name}: {tool.description}\n  Usage: {tool.name}[parameters]"
            for tool in tools
        ])
    
    def _format_examples(self, examples):
        """格式化示例"""
        formatted = []
        for ex in examples:
            formatted.append(f"""Question: {ex['question']}
{ex['trajectory']}""")
        return "\n\n".join(formatted)
```

### 4.2 错误处理

```python
class RobustReActAgent:
    def run(self, query, max_iterations=10):
        context = self._build_prompt(query)
        
        for i in range(max_iterations):
            try:
                # 生成响应
                response = self.llm.generate(context)
                
                # 解析
                thought, action = self._parse(response)
                
                # 验证 Action
                if not self._validate_action(action):
                    context += "\nInvalid action. Please try again."
                    continue
                
                # 执行
                observation = self._execute(action)
                
                # 验证 Observation
                if self._is_error(observation):
                    context += f"\nError: {observation}. Try a different approach."
                    continue
                
                # 更新上下文
                context += f"\nThought: {thought}\nAction: {action}\nObservation: {observation}"
                
                # 检查是否完成
                if action.type == "finish":
                    return action.params
                
            except Exception as e:
                logger.error(f"Error in iteration {i}: {e}")
                context += f"\nError occurred: {e}. Please try again."
        
        return "Failed to find answer after maximum iterations."
```

### 4.3 记忆管理

```python
class ReActWithMemory:
    def __init__(self, llm, tools, memory_store):
        self.llm = llm
        self.tools = tools
        self.memory = memory_store
    
    def run(self, query):
        # 检索相关记忆
        relevant_memories = self.memory.search(query, k=3)
        
        # 构建增强的 Prompt
        memory_context = "\n".join([
            f"Previous: {mem['query']} -> {mem['answer']}"
            for mem in relevant_memories
        ])
        
        prompt = f"""Previous relevant information:
{memory_context}

Current task: {query}"""
        
        # 执行 ReAct
        result = self._react_loop(prompt)
        
        # 存储新记忆
        self.memory.store({
            "query": query,
            "answer": result,
            "timestamp": datetime.now()
        })
        
        return result
```

---

## 5. ReAct 变体

### 5.1 ReWOO（Reasoning WithOut Observation）

```python
class ReWOOAgent:
    """
    ReWOO: 先规划所有步骤，再执行
    减少 LLM 调用次数，提高效率
    """
    
    def run(self, query):
        # 1. 规划阶段：生成所有步骤
        plan = self._plan(query)
        
        # 2. 执行阶段：按顺序执行
        results = []
        for step in plan.steps:
            result = self._execute_step(step, results)
            results.append(result)
        
        # 3. 总结
        return self._summarize(results)
    
    def _plan(self, query):
        """生成执行计划"""
        prompt = f"""
        Plan how to solve this problem step by step.
        For each step, specify the tool to use.
        
        Question: {query}
        
        Plan:
        Step 1: [tool] [parameters]
        Step 2: [tool] [parameters]
        ...
        """
        
        return self.llm.generate(prompt)
```

### 5.2 Reflexion

```python
class ReflexionAgent:
    """
    Reflexion: 带有自我反思的 ReAct
    """
    
    def run(self, query, max_trials=3):
        for trial in range(max_trials):
            # 执行 ReAct
            result = self._react(query)
            
            # 自我评估
            evaluation = self._evaluate(result)
            
            if evaluation.is_success:
                return result
            
            # 反思失败原因
            reflection = self._reflect(evaluation.feedback)
            
            # 更新策略
            query = f"{query}\nPrevious attempt failed: {reflection}"
        
        return result
    
    def _reflect(self, feedback):
        """生成反思"""
        prompt = f"""
        The previous attempt failed with this feedback:
        {feedback}
        
        What went wrong and how should I approach this differently?
        """
        
        return self.llm.generate(prompt)
```

---

## 6. 评估与调试

### 6.1 评估指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **成功率** | 完成任务的比例 | > 80% |
| **平均步数** | 完成任务所需的平均迭代次数 | < 5 |
| **工具使用准确率** | 正确选择工具的比例 | > 90% |
| **答案准确率** | 最终答案的正确性 | > 85% |

### 6.2 调试技巧

```python
class ReActDebugger:
    def trace(self, agent, query):
        """追踪 ReAct 执行过程"""
        trace = []
        
        # 劫持执行过程
        original_think = agent._think
        original_act = agent._act
        original_observe = agent._observe
        
        def debug_think(context):
            thought = original_think(context)
            trace.append({"type": "thought", "content": thought})
            return thought
        
        def debug_act(context):
            action = original_act(context)
            trace.append({"type": "action", "content": action})
            return action
        
        def debug_observe(action):
            observation = original_observe(action)
            trace.append({"type": "observation", "content": observation})
            return observation
        
        agent._think = debug_think
        agent._act = debug_act
        agent._observe = debug_observe
        
        # 执行
        result = agent.run(query)
        
        # 恢复
        agent._think = original_think
        agent._act = original_act
        agent._observe = original_observe
        
        return {
            "result": result,
            "trace": trace,
            "num_steps": len([t for t in trace if t["type"] == "action"])
        }
```

---

## 参考资源

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) - Yao et al., 2022
- [Reflexion: Self-Reflective Agents](https://arxiv.org/abs/2303.11366) - Shinn et al., 2023
- [ReWOO: Decoupling Reasoning from Observations](https://arxiv.org/abs/2305.18323) - Xu et al., 2023
- [LangChain ReAct Documentation](https://python.langchain.com/docs/modules/agents/agent_types/react)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
