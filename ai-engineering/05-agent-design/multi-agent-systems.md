# 多 Agent 协作系统

> **一句话定位**：设计和实现多个 Agent 协作的架构，解决单 Agent 无法完成的复杂任务。
>
> #status/canonical #topic/agent #topic/multi-agent #topic/architecture #year/2026

---

## 1. 多 Agent 基础

### 1.1 为什么需要多 Agent

```
单 Agent 限制：
- 上下文窗口有限
- 专业领域知识不足
- 无法并行处理多个任务
- 单点故障风险

多 Agent 优势：
- 专业化分工
- 并行处理
- 容错能力
- 可扩展性
```

### 1.2 多 Agent 架构模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **层级式** | 主管 Agent + 工作 Agent | 任务分配和协调 |
| **对等式** | Agent 之间平等协作 | 协作编辑、讨论 |
| **流水线** | Agent 按顺序处理 | 数据处理流程 |
| **市场式** | Agent 竞标任务 | 资源分配优化 |

---

## 2. 层级式架构

### 2.1 主管-工作者模式

```python
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Task:
    id: str
    description: str
    requirements: Dict[str, Any]
    assigned_to: str = None
    status: str = "pending"
    result: Any = None

class SupervisorAgent:
    """
    主管 Agent：负责任务分解和分配
    """
    
    def __init__(self, llm, workers: Dict[str, 'WorkerAgent']):
        self.llm = llm
        self.workers = workers
    
    async def execute(self, goal: str) -> str:
        """
        执行复杂任务
        """
        # 1. 分析任务并分解
        tasks = await self._decompose_task(goal)
        
        # 2. 分配任务给工作者
        assignments = self._assign_tasks(tasks)
        
        # 3. 监控执行
        results = await self._monitor_execution(assignments)
        
        # 4. 汇总结果
        final_result = await self._synthesize_results(results, goal)
        
        return final_result
    
    async def _decompose_task(self, goal: str) -> List[Task]:
        """
        将目标分解为子任务
        """
        prompt = f"""
        Decompose the following goal into specific subtasks.
        Each subtask should be assignable to a specialized worker.
        
        Available workers:
        - researcher: Searches and gathers information
        - analyst: Analyzes data and identifies patterns
        - writer: Creates written content
        - coder: Writes and tests code
        
        Goal: {goal}
        
        Provide subtasks as JSON:
        [
          {{
            "id": "task1",
            "description": "...",
            "requirements": {{"skills": ["research"]}},
            "dependencies": []
          }}
        ]
        """
        
        response = self.llm.generate(prompt)
        tasks_data = json.loads(response)
        
        return [Task(**task) for task in tasks_data]
    
    def _assign_tasks(self, tasks: List[Task]) -> Dict[str, List[Task]]:
        """
        将任务分配给合适的工作者
        """
        assignments = {worker_id: [] for worker_id in self.workers.keys()}
        
        for task in tasks:
            # 选择最合适的工作者
            best_worker = self._select_worker(task)
            task.assigned_to = best_worker
            assignments[best_worker].append(task)
        
        return assignments
    
    def _select_worker(self, task: Task) -> str:
        """
        选择最合适的工作者
        """
        required_skills = set(task.requirements.get("skills", []))
        
        best_match = None
        best_score = 0
        
        for worker_id, worker in self.workers.items():
            worker_skills = set(worker.skills)
            score = len(required_skills & worker_skills)
            
            if score > best_score:
                best_score = score
                best_match = worker_id
        
        return best_match or list(self.workers.keys())[0]
    
    async def _monitor_execution(self, assignments: Dict[str, List[Task]]) -> Dict[str, Any]:
        """
        监控任务执行
        """
        import asyncio
        
        # 并行执行所有任务
        tasks = []
        for worker_id, worker_tasks in assignments.items():
            worker = self.workers[worker_id]
            for task in worker_tasks:
                tasks.append(self._execute_task(worker, task))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            task.id: result 
            for task, result in zip(
                [t for tasks in assignments.values() for t in tasks],
                results
            )
        }
    
    async def _execute_task(self, worker: 'WorkerAgent', task: Task) -> Any:
        """
        执行单个任务
        """
        try:
            result = await worker.execute(task.description)
            task.status = "completed"
            task.result = result
            return result
        except Exception as e:
            task.status = "failed"
            task.result = str(e)
            raise
    
    async def _synthesize_results(self, results: Dict[str, Any], goal: str) -> str:
        """
        汇总所有结果
        """
        prompt = f"""
        Synthesize the following task results into a coherent response.
        
        Original goal: {goal}
        
        Task results:
        {json.dumps(results, indent=2)}
        
        Provide a comprehensive answer:
        """
        
        return self.llm.generate(prompt)

class WorkerAgent:
    """
    工作者 Agent：执行具体任务
    """
    
    def __init__(self, name: str, skills: List[str], llm, tools: List[Any]):
        self.name = name
        self.skills = skills
        self.llm = llm
        self.tools = tools
    
    async def execute(self, task_description: str) -> str:
        """
        执行任务
        """
        prompt = f"""
        You are a {self.name} specialist.
        Your skills: {', '.join(self.skills)}
        
        Task: {task_description}
        
        Use your expertise to complete this task.
        If you need to use tools, specify them clearly.
        """
        
        return self.llm.generate(prompt)
```

### 2.2 使用示例

```python
async def multi_agent_example():
    """
    示例：使用多 Agent 完成复杂任务
    """
    # 创建工作者
    researcher = WorkerAgent(
        name="researcher",
        skills=["research", "search", "data_collection"],
        llm=llm,
        tools=[search_tool, web_scraper_tool]
    )
    
    analyst = WorkerAgent(
        name="analyst",
        skills=["analysis", "statistics", "visualization"],
        llm=llm,
        tools=[calculator_tool, chart_tool]
    )
    
    writer = WorkerAgent(
        name="writer",
        skills=["writing", "editing", "summarization"],
        llm=llm,
        tools=[grammar_tool, style_tool]
    )
    
    # 创建主管
    supervisor = SupervisorAgent(
        llm=llm,
        workers={
            "researcher": researcher,
            "analyst": analyst,
            "writer": writer
        }
    )
    
    # 执行任务
    goal = "研究 AI 在医疗领域的应用，分析市场趋势，撰写报告"
    result = await supervisor.execute(goal)
    
    return result
```

---

## 3. 对等式架构

### 3.1 讨论模式

```python
class DiscussionAgent:
    """
    参与讨论的 Agent
    """
    
    def __init__(self, name: str, role: str, perspective: str, llm):
        self.name = name
        self.role = role
        self.perspective = perspective
        self.llm = llm
        self.memory = []
    
    async def respond(self, topic: str, context: List[str]) -> str:
        """
        对话题发表观点
        """
        prompt = f"""
        You are {self.name}, a {self.role}.
        Your perspective: {self.perspective}
        
        Topic: {topic}
        
        Previous discussion:
        {'\n'.join(context)}
        
        Provide your perspective (be concise, 2-3 sentences):
        """
        
        response = self.llm.generate(prompt)
        self.memory.append({"topic": topic, "response": response})
        
        return response

class DiscussionModerator:
    """
    讨论主持人
    """
    
    def __init__(self, agents: List[DiscussionAgent], llm):
        self.agents = agents
        self.llm = llm
    
    async def facilitate_discussion(self, topic: str, rounds: int = 3) -> str:
        """
        主持多轮讨论
        """
        context = []
        
        for round_num in range(rounds):
            print(f"\n=== Round {round_num + 1} ===")
            
            round_responses = []
            for agent in self.agents:
                response = await agent.respond(topic, context)
                round_responses.append(f"{agent.name}: {response}")
                print(f"{agent.name}: {response}")
            
            context.extend(round_responses)
        
        # 总结讨论
        summary = await self._summarize_discussion(topic, context)
        return summary
    
    async def _summarize_discussion(self, topic: str, context: List[str]) -> str:
        """总结讨论结果"""
        prompt = f"""
        Summarize the following discussion on "{topic}".
        Identify key points of agreement and disagreement.
        
        Discussion:
        {'\n'.join(context)}
        
        Summary:
        """
        
        return self.llm.generate(prompt)

# 使用示例
async def discussion_example():
    """
    示例：多 Agent 讨论
    """
    # 创建不同观点的 Agent
    optimist = DiscussionAgent(
        name="Dr. Optimist",
        role="AI Researcher",
        perspective="Optimistic about AI's potential to solve global challenges",
        llm=llm
    )
    
    pessimist = DiscussionAgent(
        name="Dr. Pessimist",
        role="Ethicist",
        perspective="Concernerned about AI risks and unintended consequences",
        llm=llm
    )
    
    pragmatist = DiscussionAgent(
        name="Dr. Pragmatist",
        role="Industry Expert",
        perspective="Focused on practical implementation and regulation",
        llm=llm
    )
    
    # 主持讨论
    moderator = DiscussionModerator(
        agents=[optimist, pessimist, pragmatist],
        llm=llm
    )
    
    result = await moderator.facilitate_discussion(
        topic="The future of AI in healthcare",
        rounds=3
    )
    
    return result
```

---

## 4. 通信机制

### 4.1 消息传递

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class MessageType(Enum):
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    RESPONSE = "response"
    BROADCAST = "broadcast"

@dataclass
class Message:
    sender: str
    receiver: str  # "all" for broadcast
    type: MessageType
    content: Any
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class MessageBus:
    """
    Agent 间消息总线
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[Message] = []
    
    def subscribe(self, agent_id: str, handler: Callable):
        """订阅消息"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(handler)
    
    async def publish(self, message: Message):
        """发布消息"""
        self.message_history.append(message)
        
        if message.receiver == "all":
            # 广播
            for handlers in self.subscribers.values():
                for handler in handlers:
                    await handler(message)
        else:
            # 定向发送
            handlers = self.subscribers.get(message.receiver, [])
            for handler in handlers:
                await handler(message)
    
    def get_history(self, agent_id: str = None) -> List[Message]:
        """获取消息历史"""
        if agent_id:
            return [
                m for m in self.message_history
                if m.sender == agent_id or m.receiver in [agent_id, "all"]
            ]
        return self.message_history
```

### 4.2 共享内存

```python
class SharedMemory:
    """
    Agent 间共享内存
    """
    
    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._subscribers: Dict[str, List[Callable]] = {}
    
    async def write(self, key: str, value: Any, agent_id: str):
        """写入共享内存"""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        
        async with self._locks[key]:
            old_value = self._store.get(key)
            self._store[key] = value
            
            # 通知订阅者
            await self._notify(key, old_value, value, agent_id)
    
    async def read(self, key: str) -> Any:
        """读取共享内存"""
        return self._store.get(key)
    
    async def update(self, key: str, updater: Callable, agent_id: str):
        """原子更新"""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        
        async with self._locks[key]:
            old_value = self._store.get(key)
            new_value = updater(old_value)
            self._store[key] = new_value
            
            await self._notify(key, old_value, new_value, agent_id)
    
    def subscribe(self, key: str, callback: Callable):
        """订阅变更通知"""
        if key not in self._subscribers:
            self._subscribers[key] = []
        self._subscribers[key].append(callback)
    
    async def _notify(self, key: str, old_val: Any, new_val: Any, agent_id: str):
        """通知订阅者"""
        for callback in self._subscribers.get(key, []):
            await callback(key, old_val, new_val, agent_id)
```

---

## 5. 协调与冲突解决

### 5.1 任务调度

```python
class TaskScheduler:
    """
    任务调度器
    """
    
    def __init__(self):
        self.task_queue = asyncio.PriorityQueue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
    
    async def submit(self, task: Task, priority: int = 5) -> str:
        """
        提交任务
        
        Args:
            task: 任务对象
            priority: 优先级（1-10，1 最高）
        """
        await self.task_queue.put((priority, task.id, task))
        return task.id
    
    async def run_scheduler(self):
        """
        调度器主循环
        """
        while True:
            priority, task_id, task = await self.task_queue.get()
            
            # 检查依赖
            if task.dependencies:
                if not all(dep in self.completed_tasks for dep in task.dependencies):
                    # 依赖未完成，重新放入队列
                    await self.task_queue.put((priority, task_id, task))
                    await asyncio.sleep(1)
                    continue
            
            # 执行任务
            self.running_tasks[task_id] = asyncio.create_task(
                self._execute_task(task)
            )
    
    async def _execute_task(self, task: Task):
        """执行任务"""
        try:
            result = await task.execute()
            self.completed_tasks[task.id] = result
            return result
        except Exception as e:
            self.failed_tasks[task.id] = e
            raise
```

### 5.2 冲突解决

```python
class ConflictResolver:
    """
    Agent 间冲突解决器
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    async def resolve(self, conflict: Dict) -> str:
        """
        解决 Agent 间的冲突
        
        Args:
            conflict: {
                "type": "resource_contention" | "disagreement" | "dependency",
                "agents": ["agent1", "agent2"],
                "description": "..."
            }
        """
        if conflict["type"] == "resource_contention":
            return await self._resolve_resource_conflict(conflict)
        elif conflict["type"] == "disagreement":
            return await self._resolve_disagreement(conflict)
        else:
            return await self._resolve_generic(conflict)
    
    async def _resolve_resource_conflict(self, conflict: Dict) -> str:
        """解决资源竞争"""
        # 优先级策略
        agents = conflict["agents"]
        
        prompt = f"""
        Resolve this resource conflict between agents.
        
        Agents involved: {', '.join(agents)}
        Conflict: {conflict['description']}
        
        Suggest a fair resolution:
        """
        
        return self.llm.generate(prompt)
    
    async def _resolve_disagreement(self, conflict: Dict) -> str:
        """解决意见分歧"""
        prompt = f"""
        Agents have different opinions on this matter.
        
        {conflict['description']}
        
        Provide a balanced perspective that considers all viewpoints:
        """
        
        return self.llm.generate(prompt)
```

---

## 6. 监控与调试

### 6.1 执行追踪

```python
class MultiAgentTracer:
    def __init__(self):
        self.traces: List[Dict] = []
    
    def trace_agent_action(self, agent_id: str, action: str, input_data: Any, output_data: Any):
        """追踪 Agent 行为"""
        self.traces.append({
            "timestamp": datetime.now(),
            "agent": agent_id,
            "action": action,
            "input": input_data,
            "output": output_data
        })
    
    def trace_message(self, message: Message):
        """追踪消息传递"""
        self.traces.append({
            "timestamp": datetime.now(),
            "type": "message",
            "sender": message.sender,
            "receiver": message.receiver,
            "content": message.content
        })
    
    def generate_report(self) -> str:
        """生成执行报告"""
        # 统计信息
        agent_actions = {}
        message_count = 0
        
        for trace in self.traces:
            if trace.get("type") == "message":
                message_count += 1
            else:
                agent = trace["agent"]
                agent_actions[agent] = agent_actions.get(agent, 0) + 1
        
        return f"""
Execution Report:
- Total actions: {len(self.traces)}
- Messages exchanged: {message_count}
- Agent activity: {json.dumps(agent_actions, indent=2)}
"""
```

---

## 参考资源

- [AutoGen: Multi-Agent Conversation Framework](https://microsoft.github.io/autogen/) - Microsoft
- [CrewAI: Framework for Orchestrating Role-Playing Agents](https://docs.crewai.com/)
- [MetaGPT: Multi-Agent Collaborative Framework](https://github.com/geekan/MetaGPT)
- [Multi-Agent Reinforcement Learning](https://www.marl-book.com/)
- [Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924) - Qian et al., 2023
