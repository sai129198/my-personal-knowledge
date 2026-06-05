# 升级路径文档

> 记录 `ai-engineering` 子库从当前状态到未来形态的演进规划。

---

## 当前状态：L1 笔记

- Markdown 文件管理
- Git 版本控制
- 本地搜索（grep / VS Code）

---

## 阶段规划

### L2 静态站点（目标：笔记 > 50 篇）

**方案**：VitePress / Docusaurus

**收益**：
- 美观的在线阅读体验
- 自动生成侧边栏导航
- 支持全文搜索（本地或 Algolia）

**待决策**：
- [ ] 选择 VitePress 还是 Docusaurus？
- [ ] 是否启用评论功能？
- [ ] 部署到 Vercel / GitHub Pages？

---

### L3 增强搜索（目标：笔记 > 100 篇）

**方案**：接入 Algolia DocSearch 或自建 Elasticsearch

**收益**：
- 秒级全文检索
- 搜索建议与拼写纠错
- 搜索分析（了解高频查询）

---

### L4 智能问答（目标：canonical 笔记 > 30 篇）

**方案**：自建 RAG 系统，将笔记库作为知识源

**收益**：
- 自然语言问答
- 自动关联相关笔记
- 发现知识盲区

**技术栈候选**：
- 向量数据库：Chroma / Pinecone / Qdrant
- Embedding：OpenAI text-embedding-3 / 本地模型
- 框架：LlamaIndex / LangChain

---

## 里程碑检查点

| 检查日期 | 检查项 | 通过标准 |
|----------|--------|----------|
| 2026-12-31 | L2 评估 | 笔记数 ≥ 50，质量合格 |
| 2027-06-30 | L3 评估 | 笔记数 ≥ 100，搜索需求强烈 |
| 2027-12-31 | L4 评估 | canonical ≥ 30，有问答需求 |
