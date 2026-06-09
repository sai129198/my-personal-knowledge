#topic/agent #topic/fundamentals #topic/tool-use #year/2026 #status/draft

# Agent 基础

> AI Agent 的核心概念：什么是 Agent、关键组件、工具调用与记忆系统。

---

## 1. 什么是 AI Agent？

### 1.1 定义

**Agent = LLM + 工具 + 记忆 + 规划能力**

```
传统 LLM：
用户输入 → LLM → 文本输出

Agent：
用户输入 → Agent → 思考 → 工具调用 → 观察 → 思考 → ... → 最终输出
              ↑___________________________________________↓
                            循环直到任务完成
```

**核心区别**：
- LLM：被动响应，单次交互
- Agent：主动行动，多步决策，与环境交互

### 1.2 Agent 的核心能力

| 能力 | 说明 | 示例 |
|------|------|------|
| **推理** | 分析问题、制定计划 | "要查天气，我需要调用 weather API" |
| **工具使用** | 调用外部工具获取信息或执行操作 | 搜索、计算、发邮件 |
| **记忆** | 保存和检索上下文信息 | 记住用户偏好、对话历史 |
| **规划** | 将复杂任务分解为子任务 | "先查资料，再写报告，最后发邮件" |
| **反思** | 评估行动结果并调整 | "上次搜索没找到，换关键词试试" |

---

## 2. Agent 架构

### 2.1 基础架构

```
┌─────────────────────────────────────────┐
│              Agent Core                 │
│                                         │
│  ┌─────────┐    ┌─────────┐           │
│  │  LLM    │◄──►│  Memory │           │
│  │ Engine  │    │  Store  │           │
│  └────┬────┘    └─────────┘           │
│       │                                 │
│       ▼                                 │
│  ┌─────────┐    ┌─────────┐           │
│  │ Planner │───►│ Executor│           │
│  │         │    │         │           │
│  └─────────┘    └────┬────┘           │
│                      │                  │
└──────────────────────┼──────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Tool Registry │
              │                 │
              │ ┌───┐ ┌───┐ ┌──┐│
              │ │🔍 │ │🧮 │ │📧││
              │ └───┘ └───┘ └──┘│
              └─────────────────┘
```

### 2.2 关键组件

#### LLM Engine

- **功能**：推理、决策、生成
- **选型**：
  - 通用任务：GPT-4o / Claude 4 Sonnet
  - 代码任务：Claude 4 Sonnet / GPT-4.1
  - 成本敏感：GPT-4o-mini / Claude 3 Haiku

#### Memory（记忆系统）

| 记忆类型 | 说明 | 实现 |
|----------|------|------|
| **短期记忆** | 当前对话上下文 | 滑动窗口、Token 限制 |
| **长期记忆** | 跨会话信息 | 向量数据库 |
| **工作记忆** | 当前任务状态 | 变量/数据结构 |
| ** episodic 记忆** | 具体事件经验 | 结构化存储 |

```python
class Memory:
    def __init__(self):
        self.short_term = []  # 短期：当前对话
        self.long_term = VectorDB()  # 长期：向量存储
        self.working = {}  # 工作：任务状态
    
    def add(self, message, memory_type="short_term"):
        if memory_type == "short_term":
            self.short_term.append(message)
        elif memory_type == "long_term":
            embedding = embed(message)
            self.long_term.add(embedding, message)
    
    def retrieve(self, query, memory_type="long_term", top_k=5):
        if memory_type == "long_term":
            query_emb = embed(query)
            return self.long_term.search(query_emb, top_k)
```

#### Tool Registry（工具注册表）

```python
class Tool:
    def __init__(self, name, description, parameters, function):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function

class ToolRegistry:
    def __init__(self):
        self.tools = {}
    
    def register(self, tool: Tool):
        self.tools[tool.name] = tool
    
    def get_schema(self):
        """生成 OpenAI Function Calling 格式的 schema"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self.tools.values()
        ]
    
    def execute(self, tool_name, arguments):
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        return tool.function(**arguments)
```

---

## 3. 工具调用（Tool Use）

### 3.1 Function Calling 机制

```python
# OpenAI Function Calling 示例
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    },
                    "date": {
                        "type": "string",
                        "description": "日期，格式 YYYY-MM-DD"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "北京明天天气怎么样？"}],
    tools=tools
)

# 模型决定调用工具
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    # 执行工具
    result = execute_tool(function_name, arguments)
    
    # 将结果返回给模型
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result)
    })
    
    # 获取最终回答
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
```

### 3.2 工具设计原则

| 原则 | 说明 | 示例 |
|------|------|------|
| **单一职责** | 每个工具只做一件事 | `search` 和 `calculate` 分开 |
| **自描述** | 描述清晰，模型能理解 | "获取天气" vs "weather_api" |
| **参数明确** | 类型、约束、必填项清楚 | 枚举值、正则验证 |
| **错误处理** | 返回结构化错误信息 | `{"error": "city not found"}` |
| **幂等性** | 多次调用结果一致 | 查询类工具天然幂等 |

### 3.3 常见工具类型

| 类别 | 工具 | 用途 |
|------|------|------|
| **搜索** | Web Search、内部搜索 | 获取信息 |
| **计算** | Calculator、Code Interpreter | 精确计算 |
| **数据** | Database Query、API Call | 数据操作 |
| **通信** | Send Email、Slack Message | 外部通信 |
| **文件** | Read File、Write File | 文件操作 |
| **代码** | Execute Code、Run Test | 代码执行 |

---

## 4. 规划能力

### 4.1 任务分解

```
复杂任务："帮我策划一次日本旅行"

分解：
1. 确定目的地和时间
   └── 工具：搜索日本旅游热门城市
   
2. 查询机票价格
   └── 工具：航班搜索 API
   
3. 查询酒店
   └── 工具：酒店预订 API
   
4. 制定行程
   └── 工具：景点搜索 + 地图 API
   
5. 预算计算
   └── 工具：计算器
   
6. 生成行程单
   └── LLM 生成
```

### 4.2 规划策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| **单步规划** | 每次只做一步，观察结果再决定下一步 | 探索性任务 |
| **全量规划** | 先制定完整计划，再执行 | 确定性任务 |
| **动态规划** | 先制定大纲，执行中调整细节 | 复杂任务 |
| **回滚规划** | 执行失败时回退并重新规划 | 高风险任务 |

---

## 5. Agent 状态管理

### 5.1 状态机模型

```
         ┌─────────┐
         │  Idle   │
         └────┬────┘
              │ 收到任务
              ▼
         ┌─────────┐
         │Planning │
         └────┬────┘
              │ 计划完成
              ▼
    ┌─────────────────┐
    │   Executing     │◄────────┐
    │  (工具调用中)    │         │
    └────┬─────┬──────┘         │
         │     │                │
    成功 │     │ 失败           │
         │     │                │
         ▼     ▼                │
    ┌────────┐ ┌────────┐      │
    │Observe │ │Reflect │──────┘
    └────┬───┘ └────────┘
         │
         ▼
    ┌─────────┐
    │ Complete│
    └─────────┘
```

### 5.2 状态持久化

```python
class AgentState:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.status = "idle"  # idle/planning/executing/complete/error
        self.plan = []
        self.current_step = 0
        self.results = []
        self.error = None
    
    def to_dict(self):
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "plan": self.plan,
            "current_step": self.current_step,
            "results": self.results,
            "error": self.error,
            "updated_at": datetime.now().isoformat()
        }
    
    def save(self, redis_client):
        redis_client.setex(
            f"agent:{self.agent_id}",
            ttl=3600,
            value=json.dumps(self.to_dict())
        )
```

---

## 💡 我的思考

1. **Agent 的核心是「决策」**：不是工具越多越好，而是模型能否在正确的时间选择正确的工具。

2. **记忆是 Agent 的瓶颈**：当前大多数 Agent 的"记忆"很简陋，真正的长期记忆、经验学习还有很大提升空间。

3. **工具设计决定上限**：再聪明的模型也救不了设计糟糕的工具。投入时间设计清晰、鲁棒的工具 API。

4. **规划能力仍是挑战**：复杂任务的多步规划容易出错，特别是在需要调整计划时。这是当前研究热点。

5. **安全不能事后考虑**：Agent 能调用工具意味着能执行操作，权限控制、沙箱隔离、人工确认是必须的。

---

## 参考来源

- **OpenAI Function Calling**: [platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling) — 访问日期：2026-06-07
- **Anthropic Tool Use**: [docs.anthropic.com/en/docs/build-with-claude/tool-use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — 访问日期：2026-06-07
- **LangChain Agents**: [python.langchain.com/docs/tutorials/agents](https://python.langchain.com/docs/tutorials/agents/) — 访问日期：2026-06-07
- **LLM Powered Autonomous Agents**: [lilianweng.github.io/posts/2023-06-23-llm-agent](https://lilianweng.github.io/posts/2023-06-23-llm-agent/) — 访问日期：2026-06-07

---

*访问日期: 2026-06-07*
