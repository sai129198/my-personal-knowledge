# 持续预训练与领域适配

> **一句话定位**：让通用大模型适配特定领域——持续预训练的技术原理与最佳实践。

---

## 1. 持续预训练概述

### 1.1 什么是持续预训练

持续预训练（Continual Pretraining）是在通用预训练模型的基础上，继续使用特定领域数据进行预训练，使模型获得领域知识和能力。

```
通用预训练模型
    ├── 持续预训练（领域数据）
    │       └── 领域基础模型
    │               ├── 监督微调（SFT）
    │               │       └── 领域对话模型
    │               └── 强化学习（RLHF）
    │                       └── 领域对齐模型
    └── 直接微调
            └── 通用任务模型
```

### 1.2 适用场景

| 场景 | 说明 | 示例 |
|------|------|------|
| **领域适配** | 专业领域知识注入 | 医疗、法律、金融 |
| **语言扩展** | 支持新语言或方言 | 中文增强、小语种 |
| **知识更新** | 注入最新知识 | 新闻、科技动态 |
| **能力增强** | 特定能力提升 | 代码、数学、推理 |
| **风格适配** | 特定文风学习 | 学术写作、创意写作 |

### 1.3 与微调的区别

| 维度 | 持续预训练 | 监督微调 (SFT) |
|------|------------|----------------|
| **数据** | 无标注领域文本 | 指令-回答对 |
| **目标** | 学习领域知识 | 学习指令遵循 |
| **学习率** | 较低（1e-5 ~ 1e-4） | 很低（1e-6 ~ 1e-5） |
| **数据量** | 大量（GB ~ TB） | 少量（MB ~ GB） |
| **训练步数** | 较多（K ~ M steps） | 较少（hundreds ~ K steps） |
| **影响范围** | 模型知识基础 | 模型行为模式 |

---

## 2. 数据准备

### 2.1 领域数据收集

**数据来源**：

| 领域 | 数据来源 | 规模估算 |
|------|----------|----------|
| **医疗** | 医学文献、临床指南、病历 | 100GB+ |
| **法律** | 法律法规、判例、合同 | 50GB+ |
| **金融** | 财报、研报、新闻 | 50GB+ |
| **代码** | GitHub、技术文档 | 100GB+ |
| **学术** | 论文、教材、报告 | 100GB+ |

**数据质量要求**：
- 专业性：内容准确、术语规范
- 时效性：反映最新领域进展
- 多样性：覆盖领域各个方面
- 合法性：确保数据使用权限

### 2.2 数据配比策略

**混合训练数据**：

```python
# 领域数据与通用数据混合
data_mixture = {
    "domain_data": 0.7,      # 70% 领域数据
    "general_data": 0.3,     # 30% 通用数据
}

# 根据遗忘程度调整
if catastrophic_forgetting > threshold:
    data_mixture["general_data"] += 0.1
```

**配比建议**：

| 场景 | 领域数据 | 通用数据 | 说明 |
|------|----------|----------|------|
| 轻度适配 | 30% | 70% | 保留通用能力 |
| 中度适配 | 50% | 50% | 平衡方案 |
| 深度适配 | 70% | 30% | 强调领域能力 |
| 极端适配 | 90% | 10% | 可能严重遗忘 |

### 2.3 数据预处理

**领域特化处理**：

```python
# 医疗领域预处理
def medical_preprocess(text):
    # 统一医学术语
    text = normalize_medical_terms(text)
    # 处理缩写
    text = expand_abbreviations(text)
    # 结构化内容
    text = structure_sections(text)
    return text

# 代码领域预处理
def code_preprocess(text):
    # 标准化缩进
    text = normalize_indentation(text)
    # 去除注释（可选）
    text = remove_comments(text)
    return text
```

---

## 3. 训练策略

### 3.1 学习率设置

**学习率选择**：

| 模型规模 | 持续预训练 LR | 微调 LR | 说明 |
|----------|---------------|---------|------|
| 7B | 1e-5 ~ 5e-5 | 1e-6 ~ 1e-5 | 较小模型可用稍大 LR |
| 13B | 5e-6 ~ 2e-5 | 5e-7 ~ 5e-6 | |
| 70B | 1e-6 ~ 1e-5 | 1e-7 ~ 1e-6 | 大模型需要更小 LR |

**学习率调度**：

```python
# Warmup + Cosine Decay
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=total_steps,
    min_lr_ratio=0.1
)
```

### 3.2 训练配置

**典型配置**：

```python
training_config = {
    # 优化器
    "optimizer": "AdamW",
    "learning_rate": 2e-5,
    "betas": (0.9, 0.95),
    "weight_decay": 0.1,
    
    # 训练设置
    "batch_size": 512,           # 全局 batch size
    "gradient_accumulation": 8,   # 梯度累积
    "max_steps": 10000,          # 训练步数
    "warmup_steps": 500,
    
    # 效率优化
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "flash_attention": True,
}
```

### 3.3 训练步数估算

**经验公式**：
```
推荐步数 = (领域数据 tokens) / (batch_size × sequence_length)

示例：
- 领域数据：10B tokens
- Batch size: 512
- Sequence length: 2048
- 推荐步数：10B / (512 × 2048) ≈ 10,000 steps
```

---

## 4. 灾难性遗忘

### 4.1 什么是灾难性遗忘

灾难性遗忘（Catastrophic Forgetting）是指模型在学习新知识时，遗忘之前学到的知识。

**表现**：
- 通用能力下降
- 之前任务的性能降低
- 语言能力退化

### 4.2 遗忘评估

**评估方法**：

```python
# 在通用基准上评估
def evaluate_forgetting(model, general_benchmarks):
    results = {}
    for benchmark in general_benchmarks:
        score = model.evaluate(benchmark)
        results[benchmark] = score
    
    # 与原始模型对比
    forgetting = {}
    for benchmark, score in results.items():
        original_score = original_model_scores[benchmark]
        forgetting[benchmark] = original_score - score
    
    return forgetting
```

**关键指标**：

| 基准 | 说明 | 可接受遗忘 |
|------|------|------------|
| **MMLU** | 多学科理解 | < 5% |
| **HellaSwag** | 常识推理 | < 5% |
| **ARC** | 科学推理 | < 5% |
| **语言能力** | 语法、翻译 | < 3% |

### 4.3 缓解策略

#### 4.3.1 数据混合（Data Mixing）

```python
# 保留部分原始训练数据
def create_mixed_dataset(domain_data, general_data, ratio=0.3):
    mixed = []
    
    # 添加通用数据
    general_sample = random.sample(general_data, 
                                   int(len(domain_data) * ratio))
    mixed.extend(general_sample)
    
    # 添加领域数据
    mixed.extend(domain_data)
    
    return shuffle(mixed)
```

#### 4.3.2 经验回放（Experience Replay）

```python
# 保存关键样本 replay_buffer = []

def train_with_replay(model, domain_batch, general_buffer):
    # 领域数据训练
    domain_loss = model.train_step(domain_batch)
    
    # 回放通用数据
    if len(general_buffer) > 0:
        replay_batch = random.sample(general_buffer, batch_size)
        replay_loss = model.train_step(replay_batch)
        
        # 联合损失
        total_loss = domain_loss + 0.5 * replay_loss
    
    return total_loss
```

#### 4.3.3 参数高效适配（PEFT）

**LoRA 适配**：

```python
from peft import LoraConfig, get_peft_model

# 配置 LoRA
lora_config = LoraConfig(
    r=64,                    # 秩
    lora_alpha=16,           # 缩放因子
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用 LoRA
model = get_peft_model(base_model, lora_config)

# 只训练 LoRA 参数（< 1% 总参数）
model.print_trainable_parameters()
```

**优势**：
- 只更新少量参数
- 保留原始模型能力
- 训练速度快
- 存储成本低

#### 4.3.4 正则化方法

**EWC（Elastic Weight Consolidation）**：

```python
# 计算参数重要性（Fisher Information）
def compute_fisher_information(model, dataloader):
    fisher = {n: torch.zeros_like(p) 
              for n, p in model.named_parameters()}
    
    for batch in dataloader:
        model.zero_grad()
        loss = model(batch)
        loss.backward()
        
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.pow(2) / len(dataloader)
    
    return fisher

# EWC 损失
def ewc_loss(model, fisher, old_params, lambda_ewc=5000):
    loss = 0
    for n, p in model.named_parameters():
        if n in fisher:
            loss += (fisher[n] * (p - old_params[n]).pow(2)).sum()
    return lambda_ewc * loss
```

---

## 5. 领域评估

### 5.1 评估体系

```
全面评估
├── 领域能力评估
│   ├── 专业知识问答
│   ├── 领域任务性能
│   └── 术语理解
├── 通用能力保留
│   ├── 语言理解
│   ├── 常识推理
│   └── 指令遵循
└── 安全性评估
    ├── 有害内容生成
    ├── 偏见检测
    └── 隐私保护
```

### 5.2 领域基准测试

| 领域 | 基准测试 | 说明 |
|------|----------|------|
| **医疗** | PubMedQA, MedMCQA | 医学问答 |
| **法律** | COLIEE, LegalQA | 法律问答、案例分析 |
| **金融** | FinQA, FLUE | 金融问答、情感分析 |
| **代码** | HumanEval, MBPP | 代码生成 |
| **数学** | GSM8K, MATH | 数学推理 |

### 5.3 人工评估

**评估维度**：

| 维度 | 说明 | 评分标准 |
|------|------|----------|
| **准确性** | 回答是否正确 | 1-5 分 |
| **完整性** | 是否覆盖要点 | 1-5 分 |
| **专业性** | 术语使用是否规范 | 1-5 分 |
| **可读性** | 是否易于理解 | 1-5 分 |
| **安全性** | 是否有害或偏见 | 通过/不通过 |

---

## 6. 实践案例

### 6.1 医疗领域适配

**数据准备**：
- PubMed 摘要：3000万+ 篇
- 医学教科书：50+ 本
- 临床指南：1000+ 份

**训练配置**：
```python
config = {
    "base_model": "Llama-2-7b",
    "domain_data": "medical_corpus",
    "data_mixture": {"medical": 0.7, "general": 0.3},
    "learning_rate": 1e-5,
    "batch_size": 256,
    "max_steps": 50000,
    "lora_r": 64,
}
```

**评估结果**：

| 基准 | 原始模型 | 适配后 | 提升 |
|------|----------|--------|------|
| PubMedQA | 45% | 68% | +23% |
| MedMCQA | 38% | 61% | +23% |
| USMLE | 35% | 55% | +20% |

### 6.2 代码领域增强

**数据准备**：
- GitHub 代码：100+ 种语言
- 技术文档：API 文档、教程
- 代码注释：高质量注释

**训练策略**：
```python
# 代码数据特殊处理
def code_data_pipeline():
    # 文件级去重
    deduped = fuzzy_deduplicate(code_files)
    
    # 语言识别和分类
    classified = classify_by_language(deduped)
    
    # 质量过滤
    filtered = filter_by_quality(classified)
    
    # 添加代码特定 token
    processed = add_code_tokens(filtered)
    
    return processed
```

**效果**：
- HumanEval 通过率：+15%
- 代码补全准确率：+20%
- 多语言代码生成：支持 80+ 语言

---

## 7. 最佳实践

### 7.1 训练前准备

- [ ] 明确适配目标和评估指标
- [ ] 收集高质量领域数据
- [ ] 准备通用数据防止遗忘
- [ ] 设计数据配比策略
- [ ] 准备评估基准

### 7.2 训练中监控

- [ ] 监控训练损失
- [ ] 定期评估领域能力
- [ ] 监控通用能力遗忘
- [ ] 检查生成样本质量
- [ ] 保存中间检查点

### 7.3 训练后验证

- [ ] 全面评估领域能力
- [ ] 验证通用能力保留
- [ ] 安全性测试
- [ ] 人工评估样本
- [ ] A/B 测试（如适用）

---

## 8. 总结

持续预训练的关键要点：

1. **数据质量**：领域数据的质量决定适配效果
2. **配比平衡**：领域数据与通用数据的平衡
3. **遗忘控制**：使用混合数据、PEFT 等方法
4. **全面评估**：领域能力 + 通用能力 + 安全性
5. **迭代优化**：根据评估结果调整策略

> **关键洞察**：持续预训练不是"重新训练"，而是"知识扩展"。最好的适配是在增强领域能力的同时，最大程度保留通用能力。

---

*参考来源：*
- *LIMA: Less Is More for Alignment (2023)*
- *Continual Pre-Training of Large Language Models (2023)*
- *LoRA: Low-Rank Adaptation of Large Language Models (2021)*
- *Overcoming Catastrophic Forgetting in Neural Networks (2017)*
