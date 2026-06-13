# 🚀 Career Growth

> **一句话定位**：职业发展与个人成长知识库——覆盖技术领导力、软技能、职业规划、学习效率等核心领域。

---

## 📂 板块导航

| 目录 | 内容 | 说明 |
|------|------|------|
| [`00-meta/`](./00-meta/) | 术语表、模板、推荐阅读、升级路径 | 子库的基础设施 |
| [`01-foundations/`](./01-foundations/) | 职业发展基础 | 职业阶段、能力模型、成长思维 |
| [`02-technical-leadership/`](./02-technical-leadership/) | 技术领导力 | 架构决策、团队管理、技术布道 |
| [`03-soft-skills/`](./03-soft-skills/) | 软技能 | 沟通、协作、谈判、时间管理 |
| [`04-learning-efficiency/`](./04-learning-efficiency/) | 学习效率 | 知识管理、笔记方法、刻意练习 |
| [`05-career-planning/`](./05-career-planning/) | 职业规划 | 路径选择、跳槽策略、薪资谈判 |
| [`06-engineering-culture/`](./06-engineering-culture/) | 工程文化 | 代码审查、技术债、DevOps 文化 |
| [`99-personal/`](./99-personal/) | 个人思考、灵感、项目笔记 | 未经验证的想法、实验记录 |

> 💡 **目录编号规则**：`00` 为元信息，`01-89` 为主题内容，`90-99` 为个人空间。

---

## 🏷️ 本库专用标签

在全局标签体系基础上，本库常用标签：

| 标签 | 含义 |
|------|------|
| `#topic/leadership` | 领导力 |
| `#topic/communication` | 沟通 |
| `#topic/learning` | 学习效率 |
| `#topic/career` | 职业规划 |
| `#topic/engineering-culture` | 工程文化 |
| `#topic/time-management` | 时间管理 |

---

## 📝 使用方式

### 新建笔记

1. 确定内容所属板块
2. 从 [`00-meta/templates/`](./00-meta/templates/) 选择对应模板
3. 按 `kebab-case` 命名文件，如 `effective-1on1-guide.md`
4. 顶部填写标签，末尾写 `## 💡 我的思考`
5. 提交到对应板块目录

### 状态维护

```bash
# 查看所有 draft
grep -r "#status/draft" --include="*.md" .

# 查看某主题相关内容
grep -r "#topic/leadership" --include="*.md" .
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
| L2 站点 | VitePress / Docusaurus 静态站点 | 笔记数量 > 30 |
| L3 搜索 | 接入全文检索（如 Algolia） | 笔记数量 > 60 |
| L4 问答 | 接入 RAG，支持自然语言问答 | 笔记质量稳定，canonical > 20 |
