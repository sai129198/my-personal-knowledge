# Changelog

> 记录本知识库的重大结构调整与里程碑变更。
> 
> 格式参考：[Keep a Changelog](https://keepachangelog.com/)

---

## [Unreleased]

### Added
- **13-ai-infrastructure/huawei-ascend-llm.md** — 华为昇腾在 LLM 领域的角色与现状（推理/微调/预训练成熟度分析 + DeepSeek V4/GLM-5 芯片来源判断）
- **13-ai-infrastructure/cuda-pytorch-explained.md** — CUDA 与 PyTorch 关系通俗解读（分层理解 + 建筑类比 + 国产芯片生态对照）

### Changed
- **12-llm-training/training-infrastructure.md** — 训练硬件章节新增华为昇腾 NPU 及国产芯片生态分析

### Added
- **12-llm-training/post-training.md** — 后训练 (Post-Training) 全流程深度解析
  - 从 InstructGPT / RLHF / DPO / o1 等多源资料精读后沉淀
  - 覆盖 SFT → Reward Model → PPO → DPO → 推理增强 完整技术栈
  - 费曼化类比 + Llama 2 实践对比 + 常见陷阱与应对

## [Unreleased]

### Added
- **05-agent-design/cognee-deep-dive.md** — Cognee 开源 AI Memory 平台深度解析（架构/API/竞品对比/实战用例）
  - 从 GitHub、官方文档、学术论文、技术博客等多源资料精读后沉淀
  - 关联至现有 `agent-memory.md`
- **05-agent-design/cognee-api-reference.md** — Cognee 接口 API 参考（Python SDK 四核心动词 + 16 种 SearchType + MCP 14 工具 + Agent 集成模式）
- **05-agent-design/cognee-graph-visualization.md** — Cognee 图谱结构详解与前端渲染方案（vis-network / Cytoscape.js / D3-force + React 组件 + 后端 API 设计）
- `agent-memory.md` 更新交叉引用，形成 cognee 三件套网状关联

### Added
- `ai-engineering/02-model-selection/glm-5.2.md` — GLM-5.2 开源旗舰模型深度解析（2026-06-22 初版，2026-07-06 大幅增补）
- `docs/ai-industry-insights/2026-07/glm-5.2-community-deployment-update.md` — GLM-5.2 社区部署全景报告：数据中心/工作站/消费级三条路径（2026-07-06 留学调研成果）

### Changed
- `ai-engineering/02-model-selection/glm-5.2.md` 增补（2026-07-06）：
  - 新增 MTP 5-token 推测解码细节（纠正初版的 3-token 描述）
  - 新增 IndexCache/IndexShare 学术支撑（arXiv:2603.12201 论文解读）
  - 新增 NVFP4 + REAP 剪枝的 469B 消费级变体
  - 新增 AMD MI300X/MI355X ROCm 支持方案
  - 新增 Z.ai DevPack 完整 Coding Agent 生态
  - 新增精确 API 定价（$1.4/$4.4 per 1M tokens，纠正初版 $0.01-0.02/1K 的偏差估算）
  - 新增社区部署最新进展（renning22 从 24 卡升级到 32 卡，速度从 10→24 tok/s）
  - 新增 vLLM/SGLang 部署生态对比
  - 新增参考资源链接（7个）
- `docs/ai-industry-insights/README.md` — 更新目录和更新日志，添加 2026-07 条目
- `ai-engineering/06-evaluation/deepswe-benchmark.md` — DeepSWE 基准测试深度解析（2026-06-22 留学调研成果）
- 初始化知识库架构（根目录 README、.gitignore、CHANGELOG）
- 创建首个子库 `ai-engineering/`
- 建立标签体系与文档规范

## [2026-06-13] — career-growth 子库核心内容填充

### Added
- **01-foundations/**:
  - `career-stages.md` — 职业发展阶段模型：从初级到专家/管理者的成长路径
- **02-technical-leadership/**:
  - `tech-lead-role.md` — Tech Lead 角色指南：职责、能力模型、常见误区
- **03-soft-skills/**:
  - `effective-communication.md` — 高效沟通方法论：NVC、PREP、场景化策略
- **04-learning-efficiency/**:
  - `knowledge-management-system.md` — 知识管理系统：PARA、Zettelkasten、第二大脑
- **05-career-planning/**:
  - `job-switching-strategy.md` — 跳槽策略：从准备到谈判的全流程
- **06-engineering-culture/**:
  - `code-review-culture.md` — 代码审查文化：流程、规范、反模式

### Changed
- 更新 `CHANGELOG.md`

## [2026-06-13] — 新增 product-thinking 和 career-growth 子库

### Added
- **product-thinking/**:
  - 初始化子库架构（README、元信息层）
  - 建立 7 个主题板块：基础概念、用户研究、产品设计、产品战略、增长、数据驱动、AI 产品
  - 创建核心模板：产品拆解模板、用户研究模板
  - 建立术语表和推荐阅读清单
- **career-growth/**:
  - 初始化子库架构（README、元信息层）
  - 建立 6 个主题板块：基础概念、技术领导力、软技能、学习效率、职业规划、工程文化
  - 创建核心模板：技术方案模板、职业反思模板
  - 建立术语表和推荐阅读清单

### Changed
- 更新根目录 `README.md`，添加新子库导航
- 更新 `CHANGELOG.md`，记录本次变更

## [2026-06-12] — 留学模式：数据工程与安全合规

### Added
- **08-data-engineering/**:
  - `data-preprocessing.md` — 数据预处理：清洗、去重、质量评估、敏感信息处理
  - `embedding-models.md` — 嵌入模型：选型、评估、微调、部署优化
  - `data-labeling.md` — 数据标注：流程设计、质量控制、主动学习、LLM 辅助
  - `synthetic-data.md` — 合成数据：生成方法、验证、混合训练
  - `data-pipeline.md` — 数据流水线：架构设计、编排工具、质量监控
- **09-security/**:
  - `prompt-injection-defense.md` — Prompt 注入防御：检测、隔离、多层防御
  - `data-privacy-protection.md` — 数据隐私保护：差分隐私、联邦学习、脱敏
  - `model-safety-alignment.md` — 模型安全对齐：RLHF、DPO、红队测试
  - `content-moderation.md` — 内容审核：多层审核、图像审核、策略管理
  - `ai-ethics-governance.md` — AI 伦理治理：公平性、可解释性、合规
  - `red-teaming-guide.md` — 红队测试：攻击面分析、自动化测试、防御

### Changed
- 更新 `08-data-engineering/README.md`，完善目录索引
- 更新 `09-security/README.md`，完善目录索引

## [2026-06-10] — 留学模式：AI 安全与推理部署

### Added
- **06-ai-safety/**:
  - `ai-safety-fundamentals.md` — AI 安全基础：对齐问题、风险分类、安全开发生命周期
  - `guardrails-and-llm-shield.md` — 护栏与防护：输入过滤、输出审核、NeMo Guardrails
  - `red-teaming-adversarial.md` — 红队测试与对抗攻击：越狱、提示注入、防御策略
- **07-inference-deployment/**:
  - `model-quantization.md` — 模型量化：INT8、INT4、GPTQ、AWQ、GGUF
  - `inference-optimization.md` — 推理优化：KV Cache、FlashAttention、投机解码、连续批处理
  - `model-serving.md` — 模型服务化：vLLM、TensorRT-LLM、TGI、SGLang 对比

### Changed
- 更新根目录 `README.md`，扩展 ai-engineering 描述

## [2026-06-09] — 留学模式：AI 工程体系全面填充

### Added
- **01-foundations/**:
  - `model-evaluation-metrics.md` — 模型评估指标：Perplexity、BLEU、HumanEval、MMLU、LLM-as-a-Judge
- **03-prompt-engineering/**:
  - `prompt-design-patterns.md` — Prompt 设计模式大全：Zero-shot、Few-shot、CoT、ToT、ReAct 等
  - `prompt-optimization-techniques.md` — Prompt 优化技巧：APE、DSPy、多 Prompt 集成
  - `domain-specific-prompts.md` — 领域专用 Prompt 模板：代码、写作、数据分析、创意
- **04-rag-architecture/**:
  - `rag-core-concepts.md` — RAG 核心概念：Embedding、向量数据库、检索策略、生成增强
  - `rag-advanced-techniques.md` — RAG 高级技术：Hybrid Search、Reranking、Query 重写、多跳检索
  - `rag-production-system.md` — RAG 生产系统：架构设计、性能优化、评估体系、监控告警
- **05-agent-design/**:
  - `agent-fundamentals.md` — Agent 基础：定义、架构、工具调用、记忆系统
  - `agent-patterns.md` — Agent 设计模式：ReAct、Plan-and-Execute、Multi-Agent、Reflection
  - `agent-frameworks.md` — Agent 框架对比：LangChain、AutoGen、CrewAI、Dify、DSPy

### Changed
- 更新 `01-foundations/README.md`，添加 model-evaluation-metrics 索引
- 更新根目录 `README.md`，更新 ai-engineering 状态

## [2026-06-06] — 留学模式：AI Infra 与应用层知识梳理

### Added
- **01-foundations/**:
  - `ai-infra-landscape.md` — AI Infra 知识全景图：计算、数据、模型、平台、安全治理
  - `ai-application-layer.md` — AI 应用层知识全景图：内容生成、知识处理、智能代理、垂直行业
- **99-personal/**:
  - `study-notes-2026-06-06.md` — AI Infra 与应用层留学过程记录

### Changed
- 更新 `01-foundations/README.md`，添加新文件索引
- 更新 `99-personal/README.md`，添加新笔记索引

## [2026-06-05] — 留学模式：AI 领域知识体系梳理

### Added
- **01-foundations/**: `ai-knowledge-landscape.md` — AI 领域七层知识全景图
- **02-model-selection/**: 
  - `openai-gpt-overview.md` — OpenAI GPT 系列模型概览
  - `anthropic-claude-overview.md` — Anthropic Claude 系列模型概览
- **03-prompt-engineering/**: `prompt-patterns-cheat-sheet.md` — 常用 Prompt 模式速查表
- **04-rag-architecture/**: `rag-implementation-guide.md` — RAG 系统实现指南
- **05-agent-design/**: `agent-architecture-patterns.md` — Agent 架构模式详解
- **99-personal/**: `study-notes-2026-06-05.md` — 留学过程记录与思考

### Changed
- 更新各板块 README.md，添加文件索引和规划内容

---

## 模板

```markdown
## [版本号] - YYYY-MM-DD

### Added
- 新增内容

### Changed
- 变更内容

### Deprecated
- 即将移除的内容

### Removed
- 移除的内容

### Fixed
- 修复的问题

### Security
- 安全相关变更
```
