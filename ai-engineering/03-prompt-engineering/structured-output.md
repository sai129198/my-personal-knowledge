# 结构化输出与 JSON 模式

> **一句话定位**：让大模型稳定输出结构化数据（JSON/XML/YAML），实现与程序的无缝对接。
>
> #status/canonical #topic/prompt-engineering #topic/structured-output #year/2026

---

## 1. 核心概念

### 1.1 为什么需要结构化输出

| 场景 | 问题 | 解决方案 |
|------|------|----------|
| **API 集成** | 模型输出文本，程序难以解析 | 输出 JSON |
| **数据抽取** | 信息混杂在段落中 | 输出结构化字段 |
| **多步骤处理** | 需要明确的数据传递 | 定义输出 schema |
| **类型安全** | 字符串 vs 数字混淆 | 强类型输出 |

### 1.2 主流结构化格式

| 格式 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **JSON** | 通用、易解析 | 不支持注释 | API 交互 |
| **XML** | 支持注释、Schema | 冗长 | 复杂文档 |
| **YAML** | 可读性好 | 缩进敏感 | 配置文件 |
| **Markdown Table** | 直观 | 类型不明确 | 简单表格 |

---

## 2. JSON 模式技术

### 2.1 基础 JSON 输出

**提示模板**：

```python
prompt = """
从以下文本中提取信息，输出 JSON 格式：

文本：
"""
{input_text}
"""

要求：
1. 输出必须是有效的 JSON
2. 不要添加任何解释文字
3. 使用以下字段：
   - name: 姓名（字符串）
   - age: 年龄（数字）
   - skills: 技能列表（字符串数组）
   - experience: 工作经验年数（数字）

输出：
"""
```

**示例输出**：

```json
{
  "name": "张三",
  "age": 28,
  "skills": ["Python", "机器学习", "数据分析"],
  "experience": 5
}
```

### 2.2 OpenAI JSON Mode

```python
import openai

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "你是一个信息提取助手。只输出 JSON，不要其他文字。"},
        {"role": "user", "content": "提取：张三，28 岁，会 Python 和机器学习，工作 5 年。"}
    ],
    response_format={"type": "json_object"}  # 强制 JSON 输出
)

# 解析结果
data = json.loads(response.choices[0].message.content)
```

### 2.3 使用 JSON Schema

```python
from pydantic import BaseModel, Field
from typing import List

class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄", ge=0, le=150)
    skills: List[str] = Field(description="技能列表")
    experience: int = Field(description="工作经验年数", ge=0)

# OpenAI 函数调用
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "提取：张三，28 岁..."}],
    functions=[{
        "name": "extract_person",
        "description": "提取人物信息",
        "parameters": Person.schema()
    }],
    function_call={"name": "extract_person"}
)

# 获取结构化结果
args = json.loads(response.choices[0].message.function_call.arguments)
person = Person(**args)
```

---

## 3. 高级技术

### 3.1 嵌套结构

```python
class Company(BaseModel):
    name: str
    founded_year: int
    employees: int
    
class Person(BaseModel):
    name: str
    age: int
    company: Company  # 嵌套对象
    projects: List[dict]  # 数组对象

# 示例输出
{
  "name": "张三",
  "age": 28,
  "company": {
    "name": "ABC 科技",
    "founded_year": 2015,
    "employees": 500
  },
  "projects": [
    {"name": "智能客服", "role": "负责人"},
    {"name": "推荐系统", "role": "核心开发"}
  ]
}
```

### 3.2 条件字段

```python
from typing import Optional, Union

class Result(BaseModel):
    success: bool
    data: Optional[dict] = None  # 成功时有值
    error: Optional[str] = None   # 失败时有值
    error_code: Optional[int] = None

# 成功时
{"success": true, "data": {"id": 123}}

# 失败时
{"success": false, "error": "未找到", "error_code": 404}
```

### 3.3 枚举类型

```python
from enum import Enum

class Status(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class Priority(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class Task(BaseModel):
    title: str
    status: Status
    priority: Priority
```

---

## 4. 实践模式

### 4.1 信息抽取

```python
# 定义 Schema
class NewsEvent(BaseModel):
    event_type: str = Field(description="事件类型：产品发布、收购、合作等")
    companies: List[str] = Field(description="涉及的公司")
    date: str = Field(description="发生日期，格式 YYYY-MM-DD")
    summary: str = Field(description="事件摘要，50字以内")
    impact: str = Field(description="影响评估：正面/负面/中性")

# 提示模板
prompt = """
分析以下新闻，提取关键信息：

新闻内容：
{news_text}

要求：
1. 严格按照 schema 输出 JSON
2. 如果信息不存在，使用 null
3. 日期必须标准化为 YYYY-MM-DD 格式
4. summary 不超过 50 字

Schema：
{schema}

只输出 JSON，不要其他文字：
"""
```

### 4.2 代码生成

```python
class FunctionSpec(BaseModel):
    name: str
    description: str
    parameters: List[dict]
    return_type: str
    examples: List[str]

prompt = """
根据需求生成 Python 函数，输出 JSON 格式：

需求：{requirement}

输出格式：
{
  "name": "函数名",
  "description": "功能描述",
  "parameters": [
    {"name": "参数名", "type": "类型", "description": "描述"}
  ],
  "return_type": "返回类型",
  "code": "完整代码",
  "examples": ["使用示例1", "使用示例2"]
}

只输出 JSON：
"""
```

### 4.3 多轮对话状态

```python
class DialogState(BaseModel):
    current_intent: str
    entities: dict
    missing_slots: List[str]
    context: dict
    next_action: str

# 对话管理
prompt = """
根据用户输入和对话历史，更新对话状态。

当前状态：
{current_state}

用户输入：{user_input}

输出新的对话状态（JSON）：
"""
```

---

## 5. 错误处理

### 5.1 常见错误

| 错误类型 | 示例 | 解决方案 |
|----------|------|----------|
| **格式错误** | `{name: "张三"}` | 要求严格 JSON 格式 |
| **类型错误** | `"age": "28"` | 明确类型要求 |
| **字段缺失** | 缺少必填字段 | 设置默认值或标记为可选 |
| **多余字段** | 输出未定义的字段 | 使用 Pydantic 过滤 |

### 5.2 验证与重试

```python
from pydantic import ValidationError

def safe_parse_json(text: str, schema: type) -> dict:
    """
    安全解析 JSON，失败时重试
    """
    # 尝试提取 JSON
    json_text = extract_json(text)
    
    try:
        # 第一次尝试
        data = json.loads(json_text)
        return schema(**data).dict()
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"解析失败：{e}")
        
        # 重试：让模型修正
        fix_prompt = f"""
        以下 JSON 有错误，请修正后重新输出：
        
        错误：{e}
        原始文本：{json_text}
        
        要求：
        1. 修复所有格式错误
        2. 确保字段类型正确
        3. 只输出修正后的 JSON
        """
        
        response = model.generate(fix_prompt)
        fixed_json = extract_json(response)
        
        try:
            data = json.loads(fixed_json)
            return schema(**data).dict()
        except Exception as e2:
            raise ValueError(f"无法修复：{e2}")
```

### 5.3 降级策略

```python
def robust_extraction(text: str, schema: type) -> dict:
    """
    鲁棒的信息提取，多种策略降级
    """
    strategies = [
        # 策略 1：标准 JSON 模式
        lambda: extract_with_json_mode(text, schema),
        # 策略 2：函数调用
        lambda: extract_with_function_call(text, schema),
        # 策略 3：正则提取
        lambda: extract_with_regex(text, schema),
        # 策略 4：手动解析
        lambda: extract_manually(text, schema),
    ]
    
    for strategy in strategies:
        try:
            result = strategy()
            if result:
                return result
        except Exception:
            continue
    
    # 所有策略失败，返回默认值
    return schema().dict()
```

---

## 6. 最佳实践

### 6.1 提示设计原则

```markdown
1. **明确格式**：明确要求 JSON 输出
2. **提供 Schema**：定义字段类型和描述
3. **示例驱动**：提供输入输出示例
4. **错误处理**：说明如何处理缺失数据
5. **验证规则**：添加约束条件（长度、范围等）
```

### 6.2 性能优化

```python
# 1. 使用缓存
@lru_cache(maxsize=1000)
def cached_extraction(text: str) -> dict:
    return extract_with_model(text)

# 2. 批量处理
def batch_extract(texts: List[str]) -> List[dict]:
    # 合并多个请求
    batch_prompt = build_batch_prompt(texts)
    response = model.generate(batch_prompt)
    return parse_batch_response(response)

# 3. 异步处理
async def async_extract(text: str) -> dict:
    return await model.agenerate(extraction_prompt(text))
```

### 6.3 安全检查

```python
def sanitize_output(data: dict) -> dict:
    """
    清理输出，防止注入攻击
    """
    # 移除危险字段
    dangerous_keys = ['__proto__', 'constructor', 'prototype']
    for key in dangerous_keys:
        if key in data:
            del data[key]
    
    # 转义 HTML
    for key, value in data.items():
        if isinstance(value, str):
            data[key] = html.escape(value)
    
    return data
```

---

## 参考资源

- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [JSON Schema](https://json-schema.org/)
- [LangChain Structured Output](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [Instructor - Structured LLM Outputs](https://python.useinstructor.com/)
