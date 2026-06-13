# 🎯 Product Thinking

> **一句话定位**：产品思维与方法论知识库——覆盖需求洞察、产品设计、增长策略、数据驱动决策等核心领域。

---

## 📂 板块导航

| 目录 | 内容 | 说明 |
|------|------|------|
| [`00-meta/`](./00-meta/) | 术语表、模板、推荐阅读、升级路径 | 子库的基础设施 |
| [`01-foundations/`](./01-foundations/) | 产品思维基础 | 产品哲学、用户心理、决策框架 |
| [`02-user-research/`](./02-user-research/) | 用户研究 | 调研方法、用户画像、需求挖掘 |
| [`03-product-design/`](./03-product-design/) | 产品设计 | 交互设计、信息架构、原型方法 |
| [`04-product-strategy/`](./04-product-strategy/) | 产品战略 | 定位、路线图、竞争分析、商业模式 |
| [`05-growth/`](./05-growth/) | 增长与运营 | AARRR、增长实验、用户激活、留存 |
| [`06-data-driven/`](./06-data-driven/) | 数据驱动 | 指标体系、A/B测试、数据分析 |
| [`07-ai-product/`](./07-ai-product/) | AI 产品专题 | AI 原生产品、AI 功能集成、产品化策略 |
| [`99-personal/`](./99-personal/) | 个人思考、灵感、项目笔记 | 未经验证的想法、实验记录 |

> 💡 **目录编号规则**：`00` 为元信息，`01-89` 为主题内容，`90-99` 为个人空间。

---

## 🏷️ 本库专用标签

在全局标签体系基础上，本库常用标签：

| 标签 | 含义 |
|------|------|
| `#topic/ux` | 用户体验 |
| `#topic/growth` | 增长策略 |
| `#topic/product-strategy` | 产品战略 |
| `#topic/data-driven` | 数据驱动 |
| `#topic/user-research` | 用户研究 |
| `#topic/ai-product` | AI 产品 |
| `#vendor/figma` | Figma |
| `#vendor/mixpanel` | Mixpanel |
| `#vendor/amplitude` | Amplitude |

---

## 📝 使用方式

### 新建笔记

1. 确定内容所属板块
2. 从 [`00-meta/templates/`](./00-meta/templates/) 选择对应模板
3. 按 `kebab-case` 命名文件，如 `user-journey-mapping-guide.md`
4. 顶部填写标签，末尾写 `## 💡 我的思考`
5. 提交到对应板块目录

### 状态维护

```bash
# 查看所有 draft
grep -r "#status/draft" --include="*.md" .

# 查看某主题相关内容
grep -r "#topic/growth" --include="*.md" .
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
