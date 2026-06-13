# 🧠 My Personal Knowledge Base

> **一句话定位**：一个面向长期积累、结构化沉淀的个人知识管理系统。
> 
> 少即是多——每个条目都要能回答"我未来为什么要回来看它"。

---

## 📚 子库导航

| 子库 | 定位 | 状态 |
|------|------|------|
| [`ai-engineering/`](./ai-engineering/) | AI 工程实践：模型选型、Prompt 工程、RAG 架构、Agent 设计、AI 安全、推理部署 | 🚧 持续完善中 |
| [`product-thinking/`](./product-thinking/) | 产品思维与方法论：需求洞察、产品设计、增长策略、数据驱动决策 | 📝 框架已建立 |
| [`career-growth/`](./career-growth/) | 职业发展与个人成长：技术领导力、软技能、职业规划、学习效率 | 📝 框架已建立 |

> 💡 **未来扩展方向**：`system-design/`（系统设计）等，按需新建。

---

## 🏗️ 组织原则

### 1. 仓库级入口
- 根目录只放总导航、组织原则、维护节奏
- 具体知识内容全部下沉到子库中

### 2. 子库自治
- 每个子库独立演进，拥有自己的 README、板块目录、模板体系
- 子库之间互不耦合，可单独迁移或归档

### 3. 命名规范
- 所有文件名统一 **kebab-case**，如 `openai-gpt-overview.md`
- 目录名使用两位数字前缀 + kebab-case，如 `01-foundations/`

### 4. 版本控制
- 整库用 Git 管理，历史可追溯
- 重大结构调整记录于 [`CHANGELOG.md`](./CHANGELOG.md)

---

## 🏷️ 标签体系

所有笔记顶部必须包含行内标签，格式：`#category/value`

| 层级 | 格式 | 用途 | 示例 |
|------|------|------|------|
| 领域 | `#topic/{子主题}` | 标记技术/业务领域 | `#topic/rag` `#topic/prompt-engineering` |
| 来源 | `#vendor/{名称}` / `#product/{名称}` | 标记信息来源 | `#vendor/openai` `#product/langchain` |
| 状态 | `#status/{状态}` | 标记内容生命周期 | `#status/draft` `#status/canonical` |
| 时间 | `#year/{年份}` | 标记信息时效性 | `#year/2026` |

### 状态流转
```
draft → reviewed → canonical
  ↓
archived
```
- `draft`：初稿，观点未经验证
- `reviewed`：已审阅，基本可靠
- `canonical`：经反复验证，可作为权威参考
- `archived`：已过时或不再适用

---

## 🔧 维护节奏

| 周期 | 动作 | 耗时 |
|------|------|------|
| **日常** | 在对应子库下用对应模板新增条目 | 按需 |
| **每周** | 整理 `xx-personal/` 下的灵感与 bad case | ~30 min |
| **每月** | 翻一遍 `#status/draft`，决定提升为 canonical 或归档；检查子库结构是否需要重组 | ~1 h |
| **每年** | 复盘整体架构；考虑升级方案（如 VitePress 发布、接入 RAG 问答） | ~2 h |

---

## 🚀 快速开始

```bash
# 1. 确定内容所属子库
# 2. 选择对应模板（在子库的 00-meta/templates/ 中）
# 3. 按命名规范创建文件
# 4. 填写标签 + 内容 + 💡 我的思考
# 5. git add / commit / push
```

---

## 📜 License

本仓库内容采用 [MIT License](./LICENSE)。
