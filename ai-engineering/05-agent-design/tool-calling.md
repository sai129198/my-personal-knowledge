# 工具调用与函数调用

> **一句话定位**：设计可靠的工具调用机制，让 LLM 能够安全、准确地使用外部工具。
>
> #status/canonical #topic/agent #topic/tool-calling #year/2026

---

## 1. 工具调用基础

### 1.1 什么是工具调用

工具调用（Tool Calling / Function Calling）是让 LLM 能够调用外部函数或 API 的能力。

```
用户查询 → LLM 理解意图 → 选择工具 → 生成参数 → 执行工具 → 返回结果 → LLM 生成答案
```

### 1.2 工具调用 vs ReAct

| 特性 | 工具调用 | ReAct |
|------|----------|-------|
| **推理过程** | 隐式 | 显式（Thought） |
| **适用场景** | 简单、确定性任务 | 复杂、多步任务 |
| **实现复杂度** | 低 | 高 |
| **可解释性** | 较低 | 高 |

---

## 2. 工具定义与注册

### 2.1 工具定义格式

```python
from typing import Dict, Any, Callable
from dataclasses import dataclass

@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable
    
    def to_openai_schema(self) -> Dict:
        """转换为 OpenAI 函数调用格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys())
                }
            }
        }
    
    def to_anthropic_schema(self) -> Dict:
        """转换为 Anthropic 工具格式"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys())
            }
        }

# 示例工具定义
calculator_tool = Tool(
    name="calculator",
    description="Calculate mathematical expressions",
    parameters={
        "expression": {
            "type": "string",
            "description": "The mathematical expression to calculate"
        }
    },
    func=lambda params: eval(params["expression"])
)

weather_tool = Tool(
    name="get_weather",
    description="Get weather information for a location",
    parameters={
        "location": {
            "type": "string",
            "description": "City name or coordinates"
        },
        "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature unit"
        }
    },
    func=get_weather_api
)
```

### 2.2 工具注册中心

```python
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """注册工具"""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
    
    def unregister(self, name: str):
        """注销工具"""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found")
        del self._tools[name]
    
    def get(self, name: str) -> Tool:
        """获取工具"""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """列出所有工具"""
        return list(self._tools.keys())
    
    def get_schemas(self, format: str = "openai") -> List[Dict]:
        """获取所有工具的 Schema"""
        if format == "openai":
            return [tool.to_openai_schema() for tool in self._tools.values()]
        elif format == "anthropic":
            return [tool.to_anthropic_schema() for tool in self._tools.values()]
        else:
            raise ValueError(f"Unknown format: {format}")

# 使用示例
registry = ToolRegistry()
registry.register(calculator_tool)
registry.register(weather_tool)
```

---

## 3. 工具调用实现

### 3.1 OpenAI 函数调用

```python
from openai import OpenAI

client = OpenAI()

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# 调用
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"}
    ],
    tools=tools,
    tool_choice="auto"
)

# 处理工具调用
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    # 执行工具
    result = execute_tool(function_name, arguments)
    
    # 继续对话
    second_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "What's the weather in Paris?"},
            response.choices[0].message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            }
        ]
    )
```

### 3.2 Anthropic 工具使用

```python
from anthropic import Anthropic

client = Anthropic()

# 定义工具
tools = [
    {
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
]

# 调用
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"}
    ]
)

# 处理工具使用
if response.stop_reason == "tool_use":
    tool_use = response.content[-1]
    tool_name = tool_use.name
    tool_input = tool_use.input
    
    # 执行工具
    result = execute_tool(tool_name, tool_input)
    
    # 继续对话
    second_response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "What's the weather in Paris?"},
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": str(result)
                    }
                ]
            }
        ]
    )
```

### 3.3 通用工具调用框架

```python
class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
    
    async def execute(self, tool_call: Dict) -> Any:
        """
        执行工具调用
        
        Args:
            tool_call: {
                "name": "tool_name",
                "arguments": {"param1": "value1"}
            }
        """
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]
        
        # 获取工具
        tool = self.registry.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        # 验证参数
        self._validate_arguments(tool, arguments)
        
        # 执行工具
        try:
            if asyncio.iscoroutinefunction(tool.func):
                result = await tool.func(arguments)
            else:
                result = tool.func(arguments)
            
            return {
                "status": "success",
                "result": result
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _validate_arguments(self, tool: Tool, arguments: Dict):
        """验证参数"""
        required = tool.parameters.get("required", [])
        for param in required:
            if param not in arguments:
                raise ValueError(f"Missing required parameter: {param}")
        
        # 类型检查
        for param, value in arguments.items():
            if param in tool.parameters["properties"]:
                expected_type = tool.parameters["properties"][param].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    raise TypeError(f"Parameter '{param}' should be {expected_type}")
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """类型检查"""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected = type_map.get(expected_type)
        if expected:
            return isinstance(value, expected)
        return True
```

---

## 4. 工具设计最佳实践

### 4.1 工具设计原则

| 原则 | 说明 | 示例 |
|------|------|------|
| **单一职责** | 每个工具只做一件事 | `get_weather` 不处理时间转换 |
| **清晰命名** | 名称体现功能 | `search_products` vs `query` |
| **明确参数** | 参数有清晰描述和类型 | `location: string (city name)` |
| **合理默认值** | 减少必填参数 | `unit: "celsius"` |
| **错误处理** | 返回清晰的错误信息 | `{"error": "City not found"}` |

### 4.2 工具描述优化

```python
# ❌ 不好的工具定义
bad_tool = {
    "name": "api",
    "description": "Call API",
    "parameters": {
        "type": "object",
        "properties": {
            "q": {"type": "string"}
        }
    }
}

# ✅ 好的工具定义
good_tool = {
    "name": "search_products",
    "description": "Search for products in the catalog. Returns product details including name, price, and availability.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (e.g., 'wireless headphones', 'laptop under $1000')"
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "home"],
                "description": "Product category to filter by"
            },
            "max_price": {
                "type": "number",
                "description": "Maximum price in USD (optional)"
            }
        },
        "required": ["query"]
    }
}
```

### 4.3 工具组合

```python
class ToolComposer:
    """组合多个工具完成复杂任务"""
    
    def __init__(self, executor: ToolExecutor):
        self.executor = executor
    
    async def execute_workflow(self, workflow: List[Dict]) -> List[Any]:
        """
        执行工具工作流
        
        Args:
            workflow: [
                {"tool": "search", "args": {"query": "..."}},
                {"tool": "analyze", "args": {"data": "${prev_result}"}}
            ]
        """
        results = []
        context = {}
        
        for step in workflow:
            # 解析参数（支持变量替换）
            args = self._resolve_args(step["args"], context)
            
            # 执行工具
            result = await self.executor.execute({
                "name": step["tool"],
                "arguments": args
            })
            
            results.append(result)
            context[f"step_{len(results)}"] = result
        
        return results
    
    def _resolve_args(self, args: Dict, context: Dict) -> Dict:
        """解析参数中的变量引用"""
        resolved = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("${"):
                var_name = value[2:-1]
                resolved[key] = context.get(var_name)
            else:
                resolved[key] = value
        return resolved
```

---

## 5. 安全与权限

### 5.1 工具权限控制

```python
class PermissionedToolRegistry(ToolRegistry):
    def __init__(self):
        super().__init__()
        self._permissions: Dict[str, List[str]] = {}
    
    def register(self, tool: Tool, required_roles: List[str] = None):
        """注册带权限的工具"""
        super().register(tool)
        self._permissions[tool.name] = required_roles or []
    
    def can_execute(self, tool_name: str, user_roles: List[str]) -> bool:
        """检查用户是否有权限执行工具"""
        required = self._permissions.get(tool_name, [])
        if not required:
            return True
        return any(role in required for role in user_roles)

# 使用示例
registry = PermissionedToolRegistry()
registry.register(delete_tool, required_roles=["admin"])
registry.register(read_tool, required_roles=["admin", "user"])

# 检查权限
if registry.can_execute("delete", user_roles=["user"]):
    # 拒绝执行
    pass
```

### 5.2 输入验证

```python
class SecureToolExecutor(ToolExecutor):
    def __init__(self, registry: ToolRegistry):
        super().__init__(registry)
        self._validators = {}
    
    def add_validator(self, tool_name: str, validator: Callable):
        """为工具添加自定义验证器"""
        self._validators[tool_name] = validator
    
    async def execute(self, tool_call: Dict) -> Any:
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]
        
        # 运行自定义验证
        if tool_name in self._validators:
            is_valid, error = self._validators[tool_name](arguments)
            if not is_valid:
                return {"status": "error", "error": error}
        
        # 执行工具
        return await super().execute(tool_call)

# 示例：SQL 查询验证器
def sql_validator(args: Dict) -> tuple:
    query = args.get("query", "")
    
    # 禁止危险操作
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT"]
    for keyword in forbidden:
        if keyword in query.upper():
            return False, f"Forbidden SQL operation: {keyword}"
    
    return True, None

executor.add_validator("query_database", sql_validator)
```

---

## 6. 工具调用监控

### 6.1 调用追踪

```python
class ToolMonitor:
    def __init__(self):
        self._calls: List[Dict] = []
    
    def record(self, tool_name: str, arguments: Dict, result: Any, duration: float):
        """记录工具调用"""
        self._calls.append({
            "timestamp": datetime.now(),
            "tool": tool_name,
            "arguments": arguments,
            "result": result,
            "duration": duration
        })
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        from collections import Counter
        
        tool_counts = Counter(call["tool"] for call in self._calls)
        avg_duration = sum(c["duration"] for c in self._calls) / len(self._calls)
        
        return {
            "total_calls": len(self._calls),
            "tool_distribution": dict(tool_counts),
            "average_duration": avg_duration
        }

# 装饰器方式使用
from functools import wraps

def monitored(monitor: ToolMonitor):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start
            
            monitor.record(
                tool_name=func.__name__,
                arguments=kwargs,
                result=result,
                duration=duration
            )
            
            return result
        return wrapper
    return decorator
```

---

## 参考资源

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use Documentation](https://docs.anthropic.com/claude/docs/tool-use)
- [LangChain Tools Documentation](https://python.langchain.com/docs/modules/agents/tools/)
- [Building LLM Systems with Tool Use](https://www.anthropic.com/research/building-effective-agents)
