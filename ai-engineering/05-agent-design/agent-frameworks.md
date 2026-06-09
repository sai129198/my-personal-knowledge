#topic/agent #topic/frameworks #topic/comparison #year/2026 #status/draft

# Agent 框架对比

> 主流 Agent 开发框架对比：LangChain、AutoGen、CrewAI、Dify、LlamaIndex。

---

## 1. 框架概览

| 框架 | 开发方 | 定位 | 成熟度 | 适用场景 |
|------|--------|------|--------|----------|
| **LangChain** | LangChain Inc. | 通用 LLM 应用框架 | ⭐⭐⭐⭐⭐ | 全场景 |
| **LlamaIndex** | LlamaIndex Inc. | 数据增强 LLM 框架 | ⭐⭐⭐⭐⭐ | RAG、数据应用 |
| **AutoGen** | Microsoft | 多 Agent 对话框架 | ⭐⭐⭐⭐ | 研究、复杂协作 |
| **CrewAI** | CrewAI Inc. | 角色扮演多 Agent | ⭐⭐⭐⭐ | 工作流自动化 |
| **Dify** | LangGenius | 可视化 LLM 应用平台 | ⭐⭐⭐⭐ | 低代码构建 |
| **DSPy** | Stanford | 声明式 LLM 编程 | ⭐⭐⭐ | 算法优化 |

---

## 2. 详细对比

### 2.1 LangChain

**定位**：最全面的 LLM 应用开发框架

**核心特性**：
- 统一的模型接口（支持 100+ 模型）
- 丰富的链式组件（Chains）
- 内置 Agent 实现（ReAct、Plan-and-Execute）
- 工具集成生态
- 记忆系统

**代码示例**：
```python
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.agents import initialize_agent, Tool

# 定义工具
tools = [
    Tool(
        name="Search",
        func=search_engine.run,
        description="用于搜索信息"
    )
]

# 初始化 Agent
agent = initialize_agent(
    tools,
    OpenAI(temperature=0),
    agent="zero-shot-react-description",
    verbose=True
)

# 运行
agent.run("2024 年诺贝尔物理学奖得主是谁？")
```

**优点**：
- ✅ 生态最丰富
- ✅ 文档完善
- ✅ 社区活跃
- ✅ 企业支持

**缺点**：
- ❌ 抽象层较厚，调试困难
- ❌ 版本兼容性 issues
- ❌ 性能开销

**适用**：企业级应用、快速原型、复杂工作流

---

### 2.2 LlamaIndex

**定位**：数据增强 LLM 的首选框架

**核心特性**：
- 强大的数据加载和索引
- 多种检索策略
- 高级 RAG 技术（HyDE、多跳检索）
- 与 LangChain 互补

**代码示例**：
```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.retrievers import VectorIndexRetriever

# 加载数据
documents = SimpleDirectoryReader("data").load_data()

# 构建索引
index = VectorStoreIndex.from_documents(documents)

# 查询
query_engine = index.as_query_engine()
response = query_engine.query("什么是 RAG？")
```

**优点**：
- ✅ RAG 能力最强
- ✅ 数据连接器丰富
- ✅ 检索策略多样
- ✅ 与 LangChain 集成好

**缺点**：
- ❌ Agent 能力相对弱
- ❌ 学习曲线较陡

**适用**：RAG 应用、知识库、文档问答

---

### 2.3 AutoGen

**定位**：多 Agent 对话和协作框架

**核心特性**：
- 对话式 Agent 编程
- 人机协作（Human-in-the-loop）
- 代码执行环境
- 灵活的对话模式

**代码示例**：
```python
from autogen import AssistantAgent, UserProxyAgent

# 创建 Agent
assistant = AssistantAgent(
    name="assistant",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": "..."}]}
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding"}
)

# 启动对话
user_proxy.initiate_chat(
    assistant,
    message="写一个 Python 函数计算斐波那契数列"
)
```

**优点**：
- ✅ 多 Agent 协作原生支持
- ✅ 代码执行安全（Docker 沙箱）
- ✅ 人机协作设计
- ✅ 微软背书

**缺点**：
- ❌ 概念较抽象
- ❌ 调试复杂
- ❌ 文档不够完善

**适用**：研究实验、代码生成、多角色协作

---

### 2.4 CrewAI

**定位**：角色扮演式多 Agent 框架

**核心特性**：
- 基于角色的 Agent 定义
- 任务依赖管理
- 流程编排（顺序/层级）
- 工具共享

**代码示例**：
```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role='研究员',
    goal='收集全面信息',
    backstory='资深行业分析师',
    tools=[search_tool]
)

writer = Agent(
    role='写手',
    goal='撰写高质量内容',
    backstory='技术写作专家'
)

task1 = Task(description='研究主题', agent=researcher)
task2 = Task(description='撰写报告', agent=writer, context=[task1])

crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
result = crew.kickoff()
```

**优点**：
- ✅ 概念直观（角色、任务、团队）
- ✅ 适合工作流场景
- ✅ 快速上手

**缺点**：
- ❌ 灵活性不如 AutoGen
- ❌ 社区相对小

**适用**：内容创作、研究分析、工作流自动化

---

### 2.5 Dify

**定位**：可视化 LLM 应用开发平台

**核心特性**：
- 可视化编排（Orchestration）
- 提示词版本管理
- 多模型支持
- 内置 RAG
- 运营分析

**使用方式**：
- 可视化界面拖拽配置
- 也支持 API 调用

**优点**：
- ✅ 零代码/低代码
- ✅ 快速上线
- ✅ 运营工具完善
- ✅ 开源可自托管

**缺点**：
- ❌ 灵活性受限
- ❌ 复杂逻辑难实现
- ❌  vendor lock-in

**适用**：快速 POC、运营工具、非技术用户

---

### 2.6 DSPy

**定位**：声明式 LLM 编程框架

**核心特性**：
- 签名（Signature）定义输入输出
- 自动提示优化
- 模块化组件
- 编译时优化

**代码示例**：
```python
import dspy

class Summarize(dspy.Signature):
    """总结长文档"""
    document = dspy.InputField()
    summary = dspy.OutputField()

summarizer = dspy.ChainOfThought(Summarize)

# 编译优化
optimized = dspy.teleprompt.BootstrapFewShot(
    metric=rouge_metric
).compile(summarizer, trainset=train_data)

result = optimized(document="长文档内容...")
```

**优点**：
- ✅ 自动优化 prompt
- ✅ 系统化工程方法
- ✅ 学术背景强

**缺点**：
- ❌ 学习曲线陡峭
- ❌ 生态较小
- ❌ 调试困难

**适用**：算法研究、系统优化、大规模应用

---

## 3. 选型决策树

```
需求分析
    │
    ├─ 需要可视化/低代码？
    │   ├─ 是 → Dify
    │   └─ 否 → 继续
    │
    ├─ 主要是 RAG/数据应用？
    │   ├─ 是 → LlamaIndex
    │   └─ 否 → 继续
    │
    ├─ 需要多 Agent 协作？
    │   ├─ 是 → 继续
    │   └─ 否 → LangChain
    │
    ├─ 需要角色扮演/工作流？
    │   ├─ 是 → CrewAI
    │   └─ 否 → AutoGen
    │
    ├─ 需要自动优化 prompt？
    │   ├─ 是 → DSPy
    │   └─ 否 → LangChain
    │
    └─ 通用场景 → LangChain
```

---

## 4. 组合使用

**最佳实践：框架组合**

```
LlamaIndex（数据层）
    ↓
LangChain（编排层）
    ↓
AutoGen/CrewAI（Agent 层）
    ↓
Dify（运营层 - 可选）
```

**示例架构**：
```python
# 数据层：LlamaIndex
index = VectorStoreIndex.from_documents(docs)
retriever = index.as_retriever()

# 编排层：LangChain
tools = [
    Tool(name="search", func=retriever.retrieve),
    Tool(name="calculator", func=calculator)
]

# Agent 层：AutoGen
assistant = AssistantAgent(name="assistant", tools=tools)
```

---

## 💡 我的思考

1. **框架选择取决于团队背景**：Python 强 → LangChain/LlamaIndex；研究背景 → DSPy/AutoGen；业务背景 → Dify。

2. **不要过度依赖框架**：框架是加速器，不是替代品。理解底层原理才能用好框架。

3. **框架在快速迭代**：LLM 领域变化快，框架也在快速演进。保持关注，但不要频繁切换。

4. **生产环境要考虑稳定性**：LangChain 和 LlamaIndex 相对成熟，AutoGen 和 CrewAI 还在快速发展。

5. **自研 vs 框架**：简单场景可以直接用 SDK，复杂场景才需要框架。避免为了用框架而用框架。

---

## 参考来源

- **LangChain Docs**: [python.langchain.com](https://python.langchain.com/) — 访问日期：2026-06-07
- **LlamaIndex Docs**: [docs.llamaindex.ai](https://docs.llamaindex.ai/) — 访问日期：2026-06-07
- **AutoGen Docs**: [microsoft.github.io/autogen](https://microsoft.github.io/autogen/) — 访问日期：2026-06-07
- **CrewAI Docs**: [docs.crewai.com](https://docs.crewai.com/) — 访问日期：2026-06-07
- **Dify Docs**: [docs.dify.ai](https://docs.dify.ai/) — 访问日期：2026-06-07
- **DSPy Docs**: [dspy-docs.vercel.app](https://dspy-docs.vercel.app/) — 访问日期：2026-06-07

---

*访问日期: 2026-06-07*
