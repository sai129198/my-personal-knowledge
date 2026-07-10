# Agent 记忆系统设计

> **一句话定位**：设计高效、可靠的 Agent 记忆系统，支持短期上下文和长期知识积累。
>
> #status/canonical #topic/agent #topic/memory #year/2026

---

## 1. 记忆类型

### 1.1 记忆分类

```
Agent 记忆系统
├── 短期记忆（Short-term Memory）
│   ├── 对话历史
│   ├── 当前任务上下文
│   └── 工作记忆
├── 长期记忆（Long-term Memory）
│   ├── 事实知识（Semantic Memory）
│   ├── 经验记忆（Episodic Memory）
│   └── 程序记忆（Procedural Memory）
└── 外部记忆（External Memory）
    ├── 向量数据库
    ├── 知识图谱
    └── 文档存储
```

### 1.2 记忆特性对比

| 类型 | 容量 | 持久性 | 访问速度 | 实现方式 |
|------|------|--------|----------|----------|
| **短期记忆** | 有限（上下文窗口） | 临时 | 快 | 上下文窗口 |
| **长期记忆** | 大 | 持久 | 中等 | 数据库 |
| **外部记忆** | 无限 | 持久 | 较慢 | 向量存储 |

---

## 2. 短期记忆

### 2.1 对话历史管理

```python
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Message:
    role: str  # user, assistant, system, tool
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

class ConversationBuffer:
    """
    对话缓冲区
    """
    
    def __init__(self, max_messages: int = 10):
        self.messages: List[Message] = []
        self.max_messages = max_messages
    
    def add(self, message: Message):
        """添加消息"""
        self.messages.append(message)
        
        # 保持最近的消息
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_recent(self, n: int = None) -> List[Message]:
        """获取最近的消息"""
        if n is None:
            return self.messages
        return self.messages[-n:]
    
    def clear(self):
        """清空历史"""
        self.messages = []
    
    def to_llm_format(self) -> List[Dict]:
        """转换为 LLM 消息格式"""
        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in self.messages
        ]

class ConversationSummaryBuffer:
    """
    带摘要的对话缓冲区
    当历史太长时，自动摘要早期对话
    """
    
    def __init__(self, llm, max_token_limit: int = 2000):
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.messages: List[Message] = []
        self.summary: str = ""
    
    def add(self, message: Message):
        """添加消息"""
        self.messages.append(message)
        
        # 检查是否需要摘要
        if self._estimate_tokens() > self.max_token_limit:
            self._summarize_older_messages()
    
    def _estimate_tokens(self) -> int:
        """估算 token 数"""
        total = len(self.summary) // 4  # 粗略估计
        for msg in self.messages:
            total += len(msg.content) // 4
        return total
    
    def _summarize_older_messages(self):
        """摘要早期消息"""
        # 保留最近的消息
        messages_to_summarize = self.messages[:-5]
        self.messages = self.messages[-5:]
        
        # 生成摘要
        conversation_text = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in messages_to_summarize
        ])
        
        prompt = f"""
        Summarize the following conversation concisely:
        
        {conversation_text}
        
        Summary:
        """
        
        new_summary = self.llm.generate(prompt)
        
        # 合并摘要
        if self.summary:
            self.summary = f"{self.summary}\n{new_summary}"
        else:
            self.summary = new_summary
    
    def to_llm_format(self) -> List[Dict]:
        """转换为 LLM 消息格式"""
        messages = []
        
        # 添加系统摘要
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.summary}"
            })
        
        # 添加近期消息
        messages.extend([
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ])
        
        return messages
```

### 2.2 工作记忆

```python
class WorkingMemory:
    """
    工作记忆：维护当前任务的上下文信息
    """
    
    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.current_goal: str = None
        self.subtasks: List[str] = []
        self.focus: str = None
    
    def set_goal(self, goal: str):
        """设置当前目标"""
        self.current_goal = goal
    
    def add_variable(self, key: str, value: Any):
        """添加变量"""
        self.variables[key] = value
    
    def get_variable(self, key: str) -> Any:
        """获取变量"""
        return self.variables.get(key)
    
    def set_focus(self, focus: str):
        """设置当前关注点"""
        self.focus = focus
    
    def to_context(self) -> str:
        """转换为上下文字符串"""
        context = []
        
        if self.current_goal:
            context.append(f"Current Goal: {self.current_goal}")
        
        if self.focus:
            context.append(f"Current Focus: {self.focus}")
        
        if self.variables:
            context.append("Variables:")
            for key, value in self.variables.items():
                context.append(f"  {key}: {value}")
        
        return "\n".join(context)
```

---

## 3. 长期记忆

### 3.1 事实知识存储

```python
class SemanticMemory:
    """
    语义记忆：存储事实知识
    """
    
    def __init__(self, embedding_model, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    async def store_fact(self, fact: str, category: str = "general"):
        """
        存储事实
        
        Args:
            fact: 事实陈述
            category: 分类
        """
        # 生成嵌入
        embedding = self.embedding_model.encode(fact)
        
        # 存储到向量数据库
        await self.vector_store.upsert(
            vectors=[embedding],
            metadatas=[{"category": category, "fact": fact}],
            ids=[f"fact_{hash(fact)}"]
        )
    
    async def recall_facts(self, query: str, top_k: int = 5) -> List[str]:
        """
        检索相关事实
        """
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode(query)
        
        # 搜索
        results = await self.vector_store.search(
            vector=query_embedding,
            top_k=top_k
        )
        
        return [r.metadata["fact"] for r in results]
    
    async def update_fact(self, old_fact: str, new_fact: str):
        """更新事实"""
        # 删除旧事实
        await self.vector_store.delete([f"fact_{hash(old_fact)}"])
        
        # 添加新事实
        await self.store_fact(new_fact)
```

### 3.2 经验记忆

```python
class EpisodicMemory:
    """
    情景记忆：存储经验和事件
    """
    
    def __init__(self, storage):
        self.storage = storage
    
    async def record_episode(self, episode: Dict):
        """
        记录经验
        
        Args:
            episode: {
                "context": "任务背景",
                "action": "采取的行动",
                "result": "结果",
                "outcome": "success/failure",
                "lessons": "学到的经验"
            }
        """
        episode_record = {
            "timestamp": datetime.now(),
            **episode
        }
        
        await self.storage.store(episode_record)
    
    async def recall_similar_experiences(self, context: str, top_k: int = 3) -> List[Dict]:
        """
        检索相似经验
        """
        # 基于上下文检索
        experiences = await self.storage.search(
            query=context,
            filter={"outcome": "success"},
            top_k=top_k
        )
        
        return experiences
    
    async def get_lessons_learned(self, task_type: str) -> List[str]:
        """
        获取特定任务类型的经验教训
        """
        episodes = await self.storage.search(
            filter={"task_type": task_type}
        )
        
        lessons = []
        for ep in episodes:
            if ep.get("lessons"):
                lessons.append(ep["lessons"])
        
        return lessons
```

### 3.3 程序记忆

```python
class ProceduralMemory:
    """
    程序记忆：存储技能和流程
    """
    
    def __init__(self):
        self.procedures: Dict[str, Dict] = {}
    
    def store_procedure(self, name: str, steps: List[str], preconditions: List[str] = None):
        """
        存储程序
        
        Args:
            name: 程序名称
            steps: 步骤列表
            preconditions: 前置条件
        """
        self.procedures[name] = {
            "steps": steps,
            "preconditions": preconditions or [],
            "success_count": 0,
            "failure_count": 0
        }
    
    def get_procedure(self, name: str) -> Dict:
        """获取程序"""
        return self.procedures.get(name)
    
    def record_execution(self, name: str, success: bool):
        """记录执行结果"""
        if name in self.procedures:
            if success:
                self.procedures[name]["success_count"] += 1
            else:
                self.procedures[name]["failure_count"] += 1
    
    def get_success_rate(self, name: str) -> float:
        """获取成功率"""
        proc = self.procedures.get(name)
        if not proc:
            return 0.0
        
        total = proc["success_count"] + proc["failure_count"]
        if total == 0:
            return 0.0
        
        return proc["success_count"] / total
```

---

## 4. 记忆检索与整合

### 4.1 多路记忆检索

```python
class MemoryRetrieval:
    """
    多路记忆检索
    """
    
    def __init__(self, semantic_memory, episodic_memory, procedural_memory):
        self.semantic = semantic_memory
        self.episodic = episodic_memory
        self.procedural = procedural_memory
    
    async def retrieve(self, query: str, context: Dict = None) -> Dict:
        """
        综合检索记忆
        """
        results = {
            "facts": [],
            "experiences": [],
            "procedures": []
        }
        
        # 并行检索
        import asyncio
        
        tasks = [
            self._retrieve_facts(query),
            self._retrieve_experiences(query),
            self._retrieve_procedures(query)
        ]
        
        facts, experiences, procedures = await asyncio.gather(*tasks)
        
        results["facts"] = facts
        results["experiences"] = experiences
        results["procedures"] = procedures
        
        return results
    
    async def _retrieve_facts(self, query: str) -> List[str]:
        """检索事实"""
        return await self.semantic.recall_facts(query)
    
    async def _retrieve_experiences(self, query: str) -> List[Dict]:
        """检索经验"""
        return await self.episodic.recall_similar_experiences(query)
    
    async def _retrieve_procedures(self, query: str) -> List[Dict]:
        """检索程序"""
        # 基于关键词匹配
        relevant = []
        for name, proc in self.procedural.procedures.items():
            if any(keyword in name.lower() for keyword in query.lower().split()):
                relevant.append({"name": name, **proc})
        return relevant
```

### 4.2 记忆整合

```python
class MemoryIntegration:
    """
    记忆整合：将新信息整合到现有记忆中
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    async def integrate(self, new_information: str, existing_memories: List[str]) -> str:
        """
        整合新信息到记忆
        
        Args:
            new_information: 新信息
            existing_memories: 现有记忆
        
        Returns:
            整合后的记忆
        """
        prompt = f"""
        Integrate the new information into the existing knowledge.
        Resolve any conflicts and maintain consistency.
        
        Existing knowledge:
        {'\n'.join(existing_memories)}
        
        New information:
        {new_information}
        
        Integrated knowledge:
        """
        
        return self.llm.generate(prompt)
    
    async def consolidate(self, memories: List[str]) -> str:
        """
        合并多个记忆，去除冗余
        """
        prompt = f"""
        Consolidate the following memories into a coherent summary.
        Remove redundancies and resolve contradictions.
        
        Memories:
        {'\n'.join(memories)}
        
        Consolidated memory:
        """
        
        return self.llm.generate(prompt)
```

---

## 5. 记忆系统实现

### 5.1 完整记忆系统

```python
class AgentMemory:
    """
    Agent 完整记忆系统
    """
    
    def __init__(self, llm, embedding_model, vector_store):
        self.llm = llm
        
        # 短期记忆
        self.conversation = ConversationSummaryBuffer(llm)
        self.working = WorkingMemory()
        
        # 长期记忆
        self.semantic = SemanticMemory(embedding_model, vector_store)
        self.episodic = EpisodicMemory(vector_store)
        self.procedural = ProceduralMemory()
        
        # 检索和整合
        self.retrieval = MemoryRetrieval(self.semantic, self.episodic, self.procedural)
        self.integration = MemoryIntegration(llm)
    
    async def add_message(self, role: str, content: str):
        """添加对话消息"""
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now()
        )
        self.conversation.add(message)
    
    async def store_experience(self, episode: Dict):
        """存储经验"""
        await self.episodic.record_episode(episode)
        
        # 提取事实并存储
        facts = await self._extract_facts(episode)
        for fact in facts:
            await self.semantic.store_fact(fact)
    
    async def _extract_facts(self, episode: Dict) -> List[str]:
        """从经验中提取事实"""
        prompt = f"""
        Extract factual statements from the following experience:
        
        {json.dumps(episode)}
        
        Facts (one per line):
        """
        
        response = self.llm.generate(prompt)
        return [f.strip() for f in response.split('\n') if f.strip()]
    
    async def retrieve_relevant(self, query: str) -> str:
        """
        检索相关记忆并整合
        """
        # 检索各类记忆
        memories = await self.retrieval.retrieve(query)
        
        # 整合为上下文
        context_parts = []
        
        if memories["facts"]:
            context_parts.append("Relevant facts:\n" + "\n".join(memories["facts"]))
        
        if memories["experiences"]:
            exp_text = "\n".join([
                f"- {exp.get('context', '')}: {exp.get('lessons', '')}"
                for exp in memories["experiences"]
            ])
            context_parts.append("Relevant experiences:\n" + exp_text)
        
        if memories["procedures"]:
            proc_text = "\n".join([
                f"- {proc['name']}: {', '.join(proc['steps'])}"
                for proc in memories["procedures"]
            ])
            context_parts.append("Relevant procedures:\n" + proc_text)
        
        return "\n\n".join(context_parts)
    
    def get_conversation_context(self) -> List[Dict]:
        """获取对话上下文"""
        return self.conversation.to_llm_format()
    
    def get_working_context(self) -> str:
        """获取工作记忆上下文"""
        return self.working.to_context()
```

---

## 6. 记忆优化

### 6.1 记忆压缩

```python
class MemoryCompressor:
    """
    记忆压缩
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    async def compress(self, memories: List[str], target_length: int = 1000) -> str:
        """
        压缩记忆到目标长度
        """
        combined = "\n".join(memories)
        
        if len(combined) <= target_length:
            return combined
        
        prompt = f"""
        Compress the following information into {target_length} characters or less.
        Preserve key facts and relationships.
        
        Information:
        {combined}
        
        Compressed:
        """
        
        return self.llm.generate(prompt)
```

### 6.2 记忆衰减

```python
class MemoryDecay:
    """
    记忆衰减：模拟遗忘过程
    """
    
    def __init__(self, decay_rate: float = 0.01):
        self.decay_rate = decay_rate
    
    def apply_decay(self, memory: Dict) -> float:
        """
        应用记忆衰减
        
        Returns:
            衰减后的记忆强度（0-1）
        """
        age = (datetime.now() - memory["timestamp"]).days
        strength = memory.get("strength", 1.0)
        
        # 指数衰减
        decayed_strength = strength * (1 - self.decay_rate) ** age
        
        return max(0.1, decayed_strength)  # 最低保留 10%
    
    def should_forget(self, memory: Dict, threshold: float = 0.2) -> bool:
        """判断是否应该遗忘"""
        strength = self.apply_decay(memory)
        return strength < threshold
```

---

## 参考资源

- [Cognee 深度解析](./cognee-deep-dive.md) — 开源 AI Memory 平台，知识图谱 + 向量 + 自我优化
- [Memory in Large Language Models](https://arxiv.org/abs/2309.16528) - Wang et al., 2023
- [LangChain Memory Documentation](https://python.langchain.com/docs/modules/memory/)
- [Vector Memory for Conversational AI](https://www.pinecone.io/learn/vector-memory/)
- [Episodic Memory in AI Systems](https://arxiv.org/abs/2308.03869)
- [Long-term Memory in LLM-based Agents](https://arxiv.org/abs/2402.04620)
