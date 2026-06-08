# Plan-and-Execute 模式

> **一句话定位**：先规划后执行，将复杂任务分解为可管理的子任务，提高完成率和可控性。
>
> #status/canonical #topic/agent #topic/planning #year/2026

---

## 1. 核心概念

### 1.1 什么是 Plan-and-Execute

Plan-and-Execute 是一种将任务分解为规划阶段和执行阶段的 Agent 架构模式。

```
传统 ReAct：
问题 → 思考 → 行动 → 观察 → 思考 → 行动 → ... → 答案

Plan-and-Execute：
问题 → 生成完整计划 → 执行步骤1 → 执行步骤2 → ... → 汇总结果 → 答案
```

**核心优势**：
- 计划阶段可以全局优化，避免局部最优
- 执行阶段可以并行化，提高效率
- 更容易监控和控制执行过程

### 1.2 适用场景

| 场景 | 说明 | 示例 |
|------|------|------|
| **复杂多步任务** | 需要多个步骤才能完成的任务 | 数据分析报告生成 |
| **可并行任务** | 子任务之间没有依赖关系 | 同时查询多个数据源 |
| **需要全局优化** | 步骤顺序影响最终结果 | 旅行路线规划 |
| **确定性流程** | 任务步骤相对固定 | 订单处理流程 |

---

## 2. 架构设计

### 2.1 基本架构

```python
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class PlanStep:
    id: str
    description: str
    tool: str
    parameters: Dict[str, Any]
    dependencies: List[str]  # 依赖的其他步骤 ID
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str = None

@dataclass
class Plan:
    steps: List[PlanStep]
    
    def get_ready_steps(self) -> List[PlanStep]:
        """获取可以执行的步骤（依赖已完成）"""
        ready = []
        completed_ids = {
            step.id for step in self.steps 
            if step.status == StepStatus.COMPLETED
        }
        
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                if all(dep in completed_ids for dep in step.dependencies):
                    ready.append(step)
        
        return ready
    
    def is_complete(self) -> bool:
        """检查计划是否全部完成"""
        return all(
            step.status in [StepStatus.COMPLETED, StepStatus.FAILED]
            for step in self.steps
        )
    
    def get_failed_steps(self) -> List[PlanStep]:
        """获取失败的步骤"""
        return [step for step in self.steps if step.status == StepStatus.FAILED]

class PlanAndExecuteAgent:
    def __init__(self, llm, tool_executor):
        self.llm = llm
        self.tool_executor = tool_executor
    
    async def run(self, query: str) -> str:
        """
        Plan-and-Execute 主流程
        """
        # 1. 规划阶段
        plan = await self._plan(query)
        
        # 2. 执行阶段
        while not plan.is_complete():
            ready_steps = plan.get_ready_steps()
            
            if not ready_steps:
                # 没有可执行的步骤，但计划未完成
                # 说明有循环依赖或失败
                break
            
            # 并行执行就绪的步骤
            await self._execute_steps(ready_steps)
        
        # 3. 检查是否有失败
        failed = plan.get_failed_steps()
        if failed:
            # 重试或重新规划
            plan = await self._replenish_plan(plan, failed)
        
        # 4. 汇总结果
        result = await self._summarize(plan, query)
        
        return result
    
    async def _plan(self, query: str) -> Plan:
        """
        生成执行计划
        """
        prompt = f"""
        Create a step-by-step plan to solve the following task.
        Each step should be specific and use one tool.
        
        Available tools:
        - search: Search for information
        - calculate: Calculate mathematical expressions
        - summarize: Summarize text
        - write_file: Write content to a file
        
        Task: {query}
        
        Provide the plan as a JSON list of steps:
        [
          {{
            "id": "step1",
            "description": "...",
            "tool": "search",
            "parameters": {{"query": "..."}},
            "dependencies": []
          }},
          ...
        ]
        """
        
        response = self.llm.generate(prompt)
        steps_data = json.loads(response)
        
        steps = [PlanStep(**step) for step in steps_data]
        return Plan(steps=steps)
    
    async def _execute_steps(self, steps: List[PlanStep]):
        """
        并行执行多个步骤
        """
        import asyncio
        
        tasks = [self._execute_step(step) for step in steps]
        await asyncio.gather(*tasks)
    
    async def _execute_step(self, step: PlanStep):
        """
        执行单个步骤
        """
        step.status = StepStatus.RUNNING
        
        try:
            result = await self.tool_executor.execute({
                "name": step.tool,
                "arguments": step.parameters
            })
            
            step.result = result
            step.status = StepStatus.COMPLETED
            
        except Exception as e:
            step.error = str(e)
            step.status = StepStatus.FAILED
    
    async def _replenish_plan(self, plan: Plan, failed_steps: List[PlanStep]) -> Plan:
        """
        重新规划失败的步骤
        """
        # 生成修复计划
        prompt = f"""
        The following steps failed:
        {json.dumps([{"id": s.id, "error": s.error} for s in failed_steps])}
        
        Please create a new plan to handle these failures.
        """
        
        return await self._plan(prompt)
    
    async def _summarize(self, plan: Plan, query: str) -> str:
        """
        汇总执行结果
        """
        results = []
        for step in plan.steps:
            if step.status == StepStatus.COMPLETED:
                results.append(f"Step {step.id}: {step.result}")
        
        prompt = f"""
        Based on the following execution results, provide a final answer.
        
        Original task: {query}
        
        Execution results:
        {"\n".join(results)}
        
        Final answer:
        """
        
        return self.llm.generate(prompt)
```

### 2.2 可视化流程

```
用户查询: "分析过去一年的销售数据并生成报告"

规划阶段:
┌─────────────────────────────────────────┐
│ Step 1: 获取销售数据 (query_database)    │
│ Step 2: 数据清洗和预处理 (process_data)  │
│ Step 3: 计算关键指标 (calculate_metrics) │
│ Step 4: 生成可视化图表 (create_charts)   │
│ Step 5: 撰写分析报告 (write_report)      │
└─────────────────────────────────────────┘

执行阶段:
Step 1 ──→ Step 2 ──→ Step 3 ──→ Step 4 ──→ Step 5
 (数据)    (清洗)     (指标)     (图表)     (报告)
```

---

## 3. 高级特性

### 3.1 动态重规划

```python
class AdaptivePlanAndExecute(PlanAndExecuteAgent):
    async def run(self, query: str) -> str:
        plan = await self._plan(query)
        
        while not plan.is_complete():
            ready_steps = plan.get_ready_steps()
            
            if not ready_steps:
                break
            
            # 执行步骤
            await self._execute_steps(ready_steps)
            
            # 检查是否需要重规划
            if self._should_replan(plan):
                plan = await self._replan(plan, query)
        
        return await self._summarize(plan, query)
    
    def _should_replan(self, plan: Plan) -> bool:
        """
        判断是否需要重新规划
        """
        # 1. 有步骤失败
        if plan.get_failed_steps():
            return True
        
        # 2. 中间结果与预期不符
        for step in plan.steps:
            if step.status == StepStatus.COMPLETED:
                if not self._is_result_valid(step):
                    return True
        
        # 3. 执行过程中发现新信息
        # ...
        
        return False
    
    async def _replan(self, plan: Plan, query: str) -> Plan:
        """
        基于当前状态重新规划
        """
        # 收集已完成步骤的结果
        completed_results = []
        for step in plan.steps:
            if step.status == StepStatus.COMPLETED:
                completed_results.append({
                    "step": step.id,
                    "result": step.result
                })
        
        prompt = f"""
        Original task: {query}
        
        Completed steps:
        {json.dumps(completed_results, indent=2)}
        
        Failed steps:
        {json.dumps([{"id": s.id, "error": s.error} for s in plan.get_failed_steps()], indent=2)}
        
        Please create an updated plan considering the current state.
        """
        
        return await self._plan(prompt)
```

### 3.2 并行执行优化

```python
class ParallelPlanExecutor:
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
    
    async def execute(self, plan: Plan) -> Plan:
        """
        优化并行执行
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def execute_with_limit(step: PlanStep):
            async with semaphore:
                return await self._execute_step(step)
        
        while not plan.is_complete():
            ready_steps = plan.get_ready_steps()
            
            if not ready_steps:
                break
            
            # 限制并发数
            tasks = [execute_with_limit(step) for step in ready_steps]
            await asyncio.gather(*tasks)
        
        return plan
    
    def _group_parallel_steps(self, plan: Plan) -> List[List[PlanStep]]:
        """
        将步骤分组，每组内的步骤可以并行执行
        """
        groups = []
        remaining = set(plan.steps)
        
        while remaining:
            # 找到当前可以执行的步骤
            group = []
            completed_ids = {
                step.id for step in plan.steps 
                if step.status == StepStatus.COMPLETED
            }
            
            for step in remaining:
                if all(dep in completed_ids for dep in step.dependencies):
                    group.append(step)
            
            if not group:
                break
            
            groups.append(group)
            remaining -= set(group)
        
        return groups
```

### 3.3 步骤缓存

```python
class CachedStepExecutor:
    def __init__(self, cache_store):
        self.cache = cache_store
    
    async def execute_step(self, step: PlanStep) -> Any:
        """
        带缓存的步骤执行
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(step)
        
        # 检查缓存
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for step {step.id}")
            return cached_result
        
        # 执行步骤
        result = await self._execute(step)
        
        # 缓存结果
        await self.cache.set(cache_key, result, ttl=3600)
        
        return result
    
    def _generate_cache_key(self, step: PlanStep) -> str:
        """生成缓存键"""
        key_data = {
            "tool": step.tool,
            "parameters": step.parameters
        }
        return hashlib.md5(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()
```

---

## 4. 与 ReAct 的对比

### 4.1 对比表

| 特性 | Plan-and-Execute | ReAct |
|------|------------------|-------|
| **规划方式** | 预先规划完整步骤 | 逐步决策 |
| **灵活性** | 较低，计划固定 | 高，动态调整 |
| **可解释性** | 高，计划清晰可见 | 中等，推理过程可见 |
| **并行能力** | 强，可并行执行独立步骤 | 弱，串行执行 |
| **适用任务** | 结构化、确定性任务 | 开放式、探索性任务 |
| **错误恢复** | 需要重规划 | 自然融入循环 |
| **效率** | 高，减少 LLM 调用 | 较低，每步都需推理 |

### 4.2 混合模式

```python
class HybridAgent:
    """
    混合模式：先用 Plan-and-Execute，遇到问题时切换到 ReAct
    """
    
    def __init__(self, plan_agent, react_agent):
        self.plan_agent = plan_agent
        self.react_agent = react_agent
    
    async def run(self, query: str) -> str:
        # 1. 尝试 Plan-and-Execute
        try:
            result = await self.plan_agent.run(query)
            if self._is_success(result):
                return result
        except Exception as e:
            logger.warning(f"Plan-and-Execute failed: {e}")
        
        # 2. 降级到 ReAct
        logger.info("Falling back to ReAct")
        return await self.react_agent.run(query)
    
    def _is_success(self, result: str) -> bool:
        """判断结果是否成功"""
        # 检查是否包含错误信息
        error_indicators = ["error", "failed", "unable", "cannot"]
        return not any(indicator in result.lower() for indicator in error_indicators)
```

---

## 5. 实际应用示例

### 5.1 数据分析任务

```python
async def data_analysis_example():
    """
    示例：使用 Plan-and-Execute 完成数据分析任务
    """
    agent = PlanAndExecuteAgent(llm, tool_executor)
    
    query = """
    分析我们公司过去一个季度的销售数据：
    1. 从数据库获取 Q4 销售数据
    2. 计算各产品线的销售额和增长率
    3. 识别销售趋势和异常
    4. 生成可视化图表
    5. 撰写分析报告
    """
    
    result = await agent.run(query)
    print(result)
    
    # 预期执行流程：
    # Step 1: query_database (Q4 sales data)
    # Step 2: process_data (clean and structure)
    # Step 3: calculate_metrics (sales, growth rate)
    # Step 4: analyze_trends (identify patterns)
    # Step 5: create_visualizations (charts)
    # Step 6: generate_report (final report)
```

### 5.2 旅行规划

```python
async def travel_planning_example():
    """
    示例：旅行规划
    """
    agent = PlanAndExecuteAgent(llm, tool_executor)
    
    query = "帮我规划一个去日本的 7 天旅行，预算 2 万元"
    
    # 预期计划：
    # Step 1: search_flights (查找航班)
    # Step 2: search_hotels (查找酒店) - 可与 Step 1 并行
    # Step 3: plan_itinerary (规划行程)
    # Step 4: calculate_budget (计算预算)
    # Step 5: book_services (预订服务)
    
    result = await agent.run(query)
    return result
```

---

## 参考资源

- [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning](https://arxiv.org/abs/2305.04091) - Wang et al., 2023
- [LLM+P: Empowering Large Language Models with Optimal Planning Proficiency](https://arxiv.org/abs/2304.11477) - Liu et al., 2023
- [LangGraph Plan-and-Execute](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
