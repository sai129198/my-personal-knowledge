# 🤖 AI Engineering

> **一句话定位**：AI 工程实践知识库——覆盖模型选型、Prompt 工程、RAG 架构、Agent 设计等核心领域。

---

## 📂 板块导航

| 目录 | 内容 | 说明 |
|------|------|------|
| [`00-meta/`](./00-meta/) | 术语表、模板、推荐阅读、升级路径 | 子库的基础设施 |
| [`01-foundations/`](./01-foundations/) | AI 领域基础概念与核心技术 | 知识全景图、模型原理、评估指标 |
| [`02-model-selection/`](./02-model-selection/) | 模型选型与对比 | 各厂商模型能力矩阵、适用场景分析 |
| [`03-prompt-engineering/`](./03-prompt-engineering/) | Prompt 工程实践 | 模式、技巧、框架、bad case 复盘 |
| [`04-rag-architecture/`](./04-rag-architecture/) | RAG 架构设计 | 检索策略、向量数据库、评估方法 |
| [`05-agent-design/`](./05-agent-design/) | Agent 设计与实现 | 架构模式、工具调用、状态管理 |
| [`06-evaluation/`](./06-evaluation/) | 模型评估与测试 | 基准测试、红队测试、A/B Test、监控 |
| [`07-deployment/`](./07-deployment/) | AI 应用部署 | 推理优化、服务化、API 设计、成本控制 |
| [`08-data-engineering/`](./08-data-engineering/) | 数据工程 | 预处理、嵌入模型、向量数据库、合成数据 |
| [`09-security/`](./09-security/) | AI 安全与合规 | Prompt 注入防御、隐私保护、伦理治理 |
| [`99-personal/`](./99-personal/) | 个人思考、灵感、项目笔记 | 未经验证的想法、实验记录 |

> 💡 **目录编号规则**：`00` 为元信息，`01-89` 为主题内容，`90-99` 为个人空间。

---

## 🏷️ 本库专用标签

在全局标签体系基础上，本库常用标签：

| 标签 | 含义 |
|------|------|
| `#topic/llm` | 大语言模型 |
| `#topic/prompt-engineering` | Prompt 工程 |
| `#topic/rag` | RAG 架构 |
| `#topic/agent` | Agent 设计 |
| `#topic/evaluation` | 模型评估 |
| `#topic/ai-overview` | AI 领域概览 |
| `#vendor/openai` | OpenAI |
| `#vendor/anthropic` | Anthropic |
| `#vendor/google` | Google (Gemini) |
| `#product/langchain` | LangChain |
| `#product/llamaindex` | LlamaIndex |

---

## 📝 使用方式

### 新建笔记

1. 确定内容所属板块
2. 从 [`00-meta/templates/`](./00-meta/templates/) 选择对应模板
3. 按 `kebab-case` 命名文件，如 `chain-of-thought-prompting-guide.md`
4. 顶部填写标签，末尾写 `## 💡 我的思考`
5. 提交到对应板块目录

### 状态维护

```bash
# 查看所有 draft
grep -r "#status/draft" --include="*.md" .

# 查看某厂商相关内容
grep -r "#vendor/openai" --include="*.md" .
```

---

## 🔧 维护原则

- **少即是多**：每个条目都要能回答"我未来为什么要回来看它"
- **可追溯**：引用外部观点必须标注来源链接 + 访问日期
- **个人视角**：不只是信息搬运，每篇笔记必须有"我的思考"
- **定期归档**：每月检查 `#status/draft`，及时提升或归档

---

## 🗺️ 升级路径

| 阶段 | 目标 | 触发条件 |
|------|------|----------|
| L1 笔记 | 当前状态，Markdown 文件管理 | 现在 |
| L2 站点 | VitePress / Docusaurus 静态站点 | 笔记数量 > 50 |
| L3 搜索 | 接入全文检索（如 Algolia） | 笔记数量 > 100 |
| L4 问答 | 接入 RAG，支持自然语言问答 | 笔记质量稳定，canonical > 30 |
