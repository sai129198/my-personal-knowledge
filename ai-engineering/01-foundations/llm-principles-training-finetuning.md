# 大模型原理、训练与微调详解

#topic/llm #topic/training #topic/finetuning #topic/transformer #topic/inference #year/2026 #status/reviewed

## 目录

- [1. 大模型核心原理](#1-大模型核心原理)
- [2. 大模型训练流程](#2-大模型训练流程)
- [3. 大模型微调技术](#3-大模型微调技术)
- [4. 对话模板与数据格式](#4-对话模板与数据格式)
- [5. 推理优化技术](#5-推理优化技术)
- [6. 核心概念速查表](#6-核心概念速查表)
- [7. 实际应用建议](#7-实际应用建议)
- [参考资料](#参考资料)

---

## 1. 大模型核心原理

### 1.1 什么是大语言模型（LLM）

大语言模型（Large Language Model, LLM）是基于Transformer架构的深度学习模型，通过海量文本数据训练，学习语言的统计规律和语义表示。

**核心特征：**
- **参数量巨大**：从数十亿到数千亿参数（如 GPT-3 1750亿，GPT-4 估计超过1万亿）
- **上下文学习（In-Context Learning）**：无需参数更新，仅通过提示词就能学习新任务
- **涌现能力（Emergent Abilities）**：当模型规模达到某个阈值时，会突然出现新的能力

### 1.2 Transformer 架构

Transformer 是 LLM 的基础架构，由 Google 在 2017 年提出。

#### 核心组件

```
输入 → [Embedding] → [Encoder/Decoder] → [Output]
              ↓
        [Self-Attention] → [Feed-Forward]
              ↓
        [Multi-Head Attention]
```

**1. Self-Attention（自注意力机制）**

自注意力机制让模型能够关注输入序列中不同位置的关系：

```python
# 简化的注意力计算
Q = X @ W_q  # Query
K = X @ W_k  # Key
V = X @ W_v  # Value

Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

**关键概念：**
- **Query (Q)**：当前位置想要查询的信息
- **Key (K)**：每个位置提供的标识信息
- **Value (V)**：每个位置的实际内容
- **Attention Score**：Q 和 K 的相似度，决定关注程度

**2. Multi-Head Attention（多头注意力）**

使用多组 Q、K、V 矩阵，让模型同时关注不同方面的信息：

```
Head 1: 关注语法关系
Head 2: 关注语义关系
Head 3: 关注指代关系
...
```

**3. Feed-Forward Network（前馈网络）**

对每个位置独立应用的全连接层：

```python
FFN(x) = max(0, x @ W1 + b1) @ W2 + b2
```

**4. Layer Normalization & Residual Connection**

- **残差连接**：解决梯度消失问题，`output = x + sublayer(x)`
- **层归一化**：稳定训练过程

### 1.3 大模型的核心能力来源

**1. 预训练目标：Next Token Prediction**

```
输入: "今天天气很"
目标: "好"

模型学习: P(好 | 今天天气很)
```

通过海量文本预测下一个词，模型被迫学习：
- 语法结构
- 语义关系
- 世界知识
- 推理模式

**2. 缩放定律（Scaling Laws）**

OpenAI 发现模型性能与以下因素呈幂律关系：

```
Loss ∝ (N^(-α)) + (D^(-β)) + (C^(-γ))

N: 参数量
D: 训练数据量
C: 计算量
```

**关键发现：**
- 模型越大，性能越好（但边际收益递减）
- 数据质量和多样性比单纯的数据量更重要
- 计算效率存在最优配置

**Chinchilla 最优配置（Hoffmann et al., 2022）**：

DeepMind 提出的计算最优训练策略：

| 发现 | 说明 |
|------|------|
| 等比例增长 | 模型参数和训练数据量应该等比例增长 |
| 最优比例 | 参数量 ≈ 训练 token 数 / 20 |
| 常见误区 | 很多模型 under-trained（数据量不足） |

| 模型 | 参数量 | 训练 token | 比例 | 评价 |
|------|--------|-----------|------|------|
| GPT-3 | 175B | 300B | 1.7:1 | under-trained |
| Chinchilla | 70B | 1.4T | 20:1 | 计算最优 |
| Llama 3 | 70B | 15T | 214:1 | 极度 over-trained |

> Llama 3 策略：固定模型大小，用海量数据训练，获得更好性能。

**3. 涌现能力（Emergent Abilities）**

当模型规模达到某个阈值时（通常 6B-10B），会突然出现小模型不具备的能力：

| 能力 | 说明 | 示例 |
|------|------|------|
| **上下文学习** | 给几个示例就能学会新任务 | Few-shot prompt |
| **思维链推理** | 逐步推理解决复杂问题 | Chain-of-Thought |
| **指令遵循** | 理解并执行自然语言指令 | Instruction tuning |

> 这些能力并非显式训练得到，而是大规模预训练的**涌现现象**。

---

## 2. 大模型训练流程

### 2.1 三阶段训练流程

```
预训练（Pre-training）
    ↓
监督微调（SFT - Supervised Fine-Tuning）
    ↓
对齐训练（Alignment：RLHF / DPO）
```

### 2.2 阶段一：预训练（Pre-training）

**目标**：让模型学习通用的语言表示和世界知识

**数据**：
- 互联网文本（Common Crawl、WebText）
- 书籍（BooksCorpus、Gutenberg）
- 代码（GitHub、Stack Overflow）
- 学术论文（arXiv、PubMed）

**数据规模**：
- GPT-3：约 5000 亿 tokens
- LLaMA 2：约 2 万亿 tokens
- 现代模型：通常 5-15 万亿 tokens

**训练细节**：

```python
# 简化的预训练流程
for batch in dataloader:
    # 1. 前向传播
    logits = model(batch.input_ids)
    
    # 2. 计算损失（Next Token Prediction）
    loss = cross_entropy(logits, batch.labels)
    
    # 3. 反向传播
    loss.backward()
    
    # 4. 更新参数
    optimizer.step()
```

**关键技术**：

| 技术 | 作用 |
|------|------|
| **混合精度训练** | FP16/BF16 减少显存占用，加速训练 |
| **梯度累积** | 模拟大批量训练 |
| **梯度裁剪** | 防止梯度爆炸 |
| **学习率预热** | 稳定早期训练 |
| **余弦退火** | 平滑降低学习率 |
| **数据并行** | 多 GPU 同时训练 |
| **模型并行** | 大模型分片到多 GPU |
| **流水线并行** | 层间并行，提高利用率 |

**计算资源**：
- GPT-3（175B）：约 3640 PetaFLOP/s-days
- LLaMA 2（70B）：约 1720 PetaFLOP/s-days
- 需要数千张 GPU，训练数周至数月

**Tokenization（分词）**：

文本需先转换为数字（token ID）才能被模型处理：

| 策略 | 方法 | 优点 | 缺点 |
|------|------|------|------|
| **Word-based** | 按空格分词 | 语义完整 | 词表大、OOV 问题 |
| **Character-based** | 按字符分词 | 词表小 | 序列长、语义弱 |
| **Subword** | BPE/WordPiece | 平衡 | 需要训练分词器 |

现代分词器词表大小：
- GPT-2/3/4: 50,257 (BPE)
- BERT: 30,522 (WordPiece)
- Llama 3: 128,000 (BPE)
- Qwen: 151,646 (BPE，中文优化)

### 2.3 阶段二：监督微调（SFT）

**目标**：让模型学会遵循指令、进行对话

**数据格式**：

```json
{
  "instruction": "请解释什么是机器学习",
  "input": "",
  "output": "机器学习是人工智能的一个分支..."
}
```

**数据特点**：
- 高质量人工标注（通常 10K-100K 条）
- 多样化任务类型
- 明确的输入输出格式

**训练方式**：

```python
# SFT 训练
for batch in sft_dataloader:
    logits = model(batch.input_ids)
    
    # 只计算 assistant 回复部分的损失
    loss = cross_entropy(logits[mask], batch.labels[mask])
    loss.backward()
    optimizer.step()
```

**关键要点**：
- 学习率通常比预训练小 10-100 倍
- 训练轮数少（通常 1-3 个 epoch）
- 容易过拟合，需要 early stopping

### 2.4 阶段三：对齐训练（Alignment）

**目标**：让模型行为符合人类价值观，减少有害输出

#### 方法 1：RLHF（Reinforcement Learning from Human Feedback）

```
步骤 1: 训练奖励模型（Reward Model）
   - 收集人类偏好数据（A vs B，哪个回答更好）
   - 训练模型预测人类偏好

步骤 2: 使用 PPO 强化学习优化策略
   - 策略模型生成回答
   - 奖励模型打分
   - PPO 算法更新策略
```

**PPO 训练流程**：

```python
# 简化的 PPO 流程
for episode in range(num_episodes):
    # 1. 生成回答
    response = policy_model.generate(prompt)
    
    # 2. 奖励模型打分
    reward = reward_model(prompt, response)
    
    # 3. 计算优势函数
    advantage = reward - baseline
    
    # 4. PPO 更新
    ratio = new_policy / old_policy
    clipped_ratio = clip(ratio, 1-ε, 1+ε)
    loss = -min(ratio * advantage, clipped_ratio * advantage)
    
    # 5. KL 散度约束（防止策略偏离太远）
    kl_penalty = KL(new_policy || ref_policy)
    total_loss = loss + β * kl_penalty
```

#### 方法 2：DPO（Direct Preference Optimization）

DPO 是 RLHF 的简化替代方案，无需显式训练奖励模型：

```python
# DPO 损失函数
loss = -log(σ(β * (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x)))))

# 其中：
# y_w: 人类偏好的回答
# y_l: 人类不喜欢的回答
# π: 当前策略
# π_ref: 参考策略（SFT 模型）
# β: 温度参数
```

**RLHF vs DPO 对比**：

| 特性 | RLHF (PPO) | DPO | GRPO |
|------|-----------|-----|------|
| 复杂度 | 高（奖励模型+PPO） | 低（直接优化） | 中（无需Critic） |
| 稳定性 | 较低 | 较高 | 较高 |
| 效果 | 好 | 接近 RLHF | 很好（可验证任务） |
| 计算成本 | 高 | 低 | 低 |
| Critic模型 | 需要 | 不需要 | 不需要 |
| 适用场景 | 通用 | 通用 | 数学/代码等可验证任务 |

---

## 3. 大模型微调技术

### 3.1 微调 vs 提示工程 vs RAG

```
┌─────────────────────────────────────────────────────────┐
│                    能力增强方式选择                        │
├─────────────────────────────────────────────────────────┤
│  简单任务 → 提示工程（Prompt Engineering）                │
│  需要知识 → RAG（检索增强生成）                           │
│  需要风格/格式 → 微调（Fine-tuning）                     │
│  复杂专业任务 → 微调 + RAG                              │
└─────────────────────────────────────────────────────────┘
```

### 3.2 全参数微调（Full Fine-tuning）

**方式**：更新模型所有参数

**适用场景**：
- 有大量标注数据（>100K）
- 计算资源充足
- 需要深度定制模型行为

**缺点**：
- 计算成本高（需要同等预训练资源）
- 容易灾难性遗忘（Catastrophic Forgetting）
- 存储成本高（每个任务一个完整模型）

### 3.3 参数高效微调（PEFT）

#### 1. LoRA（Low-Rank Adaptation）

**核心思想**：冻结原模型参数，只训练低秩分解矩阵

```python
# LoRA 原理
# 原始权重: W (d × k)
# 低秩分解: W' = W + ΔW = W + BA
# B: d × r, A: r × k, r << min(d, k)

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank
    
    def forward(self, x):
        # 原始输出 + LoRA 分支
        return x @ W + x @ self.lora_A @ self.lora_B * self.scaling
```

**超参数**：
- `rank`（r）：通常 4-64，越大表达能力越强
- `alpha`：缩放因子，通常等于 rank 或 2×rank
- `target_modules`：应用 LoRA 的层（通常是 q_proj, v_proj, k_proj, o_proj）

**优势**：
- 训练参数减少 99%+
- 不增加推理延迟（可合并权重）
- 多任务可切换（不同 LoRA 权重）

#### 2. QLoRA（Quantized LoRA）

**核心思想**：4-bit 量化 + LoRA

```python
# QLoRA 配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # 嵌套量化
    bnb_4bit_quant_type="nf4",       # 4-bit Normal Float
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# 然后应用 LoRA
model = get_peft_model(model, lora_config)
```

**优势**：
- 可在单张消费级 GPU（24GB）上微调 70B 模型
- 性能损失极小（<1%）

#### 3. Prefix Tuning / Prompt Tuning

**核心思想**：只训练输入前缀的嵌入向量

```python
# Prompt Tuning
# 冻结整个模型，只训练 soft prompt
soft_prompt = nn.Parameter(torch.randn(num_tokens, hidden_dim))

# 输入拼接
input_embeds = torch.cat([soft_prompt, text_embeds], dim=1)
```

**适用场景**：
- 参数极少（<1%）
- 适合简单任务适配
- 多任务切换方便

#### 4. Adapter

**核心思想**：在 Transformer 层中插入小型适配器模块

```python
# Adapter 结构
class Adapter(nn.Module):
    def __init__(self, hidden_dim, adapter_dim=64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, hidden_dim)
        self.activation = nn.GELU()
    
    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return x + residual  # 残差连接
```

### 3.4 微调数据准备

**数据格式示例**：

```json
[
  {
    "messages": [
      {"role": "system", "content": "你是一个专业的客服助手"},
      {"role": "user", "content": "我的订单什么时候到？"},
      {"role": "assistant", "content": "请提供您的订单号，我帮您查询..."}
    ]
  }
]
```

**数据质量要点**：

| 方面 | 建议 |
|------|------|
| **多样性** | 覆盖各种场景和边缘情况 |
| **质量** | 人工审核，确保正确性 |
| **数量** | 通常 1K-100K 即可，质量 > 数量 |
| **格式** | 与推理时格式完全一致 |
| **长度** | 控制序列长度，避免填充过多 |

### 3.5 微调最佳实践

**1. 学习率设置**

```python
# LoRA 学习率通常设置
lr = 1e-4  # 比全参数微调大 10 倍

# 学习率调度
scheduler = CosineAnnealingLR(optimizer, T_max=num_steps)
```

**2. 训练参数**

| 参数 | 建议值 |
|------|--------|
| batch size | 64-256（使用梯度累积） |
| epochs | 1-3（避免过拟合） |
| warmup steps | 100-500 |
| max sequence length | 根据任务调整 |
| LoRA rank | 8-64 |

**3. 防止过拟合**

```python
# 早停
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.001
)

# Dropout
lora_config = LoraConfig(
    r=16,
    lora_dropout=0.05,  # 添加 dropout
    target_modules=["q_proj", "v_proj"]
)
```

**4. 评估策略**

```python
# 定期评估
from transformers import TrainerCallback

class EvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # 保存最佳模型
        if metrics["eval_loss"] < best_loss:
            save_best_model()
        
        # 检查过拟合
        if metrics["eval_loss"] > metrics["train_loss"] * 1.5:
            print("警告：可能过拟合！")
```

---

---

## 4. 对话模板与数据格式

### 4.1 为什么需要对话模板？

不同模型使用不同的对话格式，因为预训练时使用的格式不同。使用 `tokenizer.apply_chat_template()` 自动处理，不要手写拼接。

### 4.2 主流模型模板对比

| 模型 | 模板风格 | 特点 |
|------|---------|------|
| **Llama-3** | `<|start_header_id|>user<|end_header_id|>...<|eot_id|>` | 支持 system prompt |
| **Qwen** | `<|im_start|>user\n...<|im_end|>` | 中文优化 |
| **Mistral** | `[INST] ... [/INST]` | 简洁 |
| **DeepSeek** | `<｜User｜>...<｜Assistant｜>` | 支持推理标签 |

### 4.3 标准数据格式

```json
{
  "messages": [
    {"role": "system", "content": "设定模型行为"},
    {"role": "user", "content": "用户问题"},
    {"role": "assistant", "content": "期望回答"}
  ]
}
```

**训练时只计算 assistant 部分的 loss**。

---

## 5. 推理优化技术

### 5.1 KV Cache

缓存已计算的 Key/Value，避免自回归生成时重复计算。显存占用约 2 × layers × heads × head_dim × seq_len × batch × 2 bytes。

### 5.2 量化推理

| 精度 | 显存节省 | 适用场景 |
|------|---------|---------|
| **FP16/BF16** | 1× | 标准推理 |
| **INT8** | 0.5× | 精度敏感 |
| **GPTQ/AWQ 4-bit** | 0.25× | 推荐方案 |

### 5.3 推理框架

| 框架 | 特点 | 适用场景 |
|------|------|---------|
| **vLLM** | PagedAttention，高吞吐 | 生产部署 |
| **TensorRT-LLM** | NVIDIA 优化 | NVIDIA GPU |
| **llama.cpp** | CPU 推理 | 边缘设备 |
| **Ollama** | 本地运行 | 本地开发 |

### 5.4 其他优化

- **投机解码**：小模型生成草稿，大模型验证，加速 2-3 倍
- **连续批处理**：动态插入请求，吞吐量提升 10-20 倍
- **Flash Attention**：内存高效注意力计算

---

## 6. 核心概念速查表

| 术语 | 解释 |
|------|------|
| **Token** | 模型处理的最小文本单位 |
| **KV Cache** | 缓存已计算的 Key/Value 加速生成 |
| **Temperature** | 控制采样随机性（越低越确定） |
| **Top-p/Top-k** | 限制候选词范围的采样策略 |
| **RoPE** | 旋转位置编码，现代主流 |
| **Decoder-only** | 只保留 Decoder 的架构（GPT/Llama） |
| **PEFT** | 参数高效微调（LoRA/QLoRA 等） |
| **BF16** | Brain Float 16，训练常用精度 |
| **Gradient Checkpointing** | 用计算换显存 |
| **Perplexity** | 困惑度，衡量模型预测能力 |

---

## 7. 实际应用建议

### 4.1 选择微调还是 RAG

```
决策树：
                    ┌─────────────────┐
                    │  需要外部知识？   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │ 是           │              │ 否
              ▼              │              ▼
        ┌─────────┐          │        ┌─────────┐
        │ 使用 RAG │          │        │ 需要特定风格？│
        └─────────┘          │        └────┬────┘
                             │             │
                    ┌────────┴────────┐    │
                    │ 需要改变模型行为？│    │
                    └────────┬────────┘    │
                             │             │
              ┌──────────────┼─────────────┘
              │ 是           │ 否
              ▼              ▼
        ┌─────────┐    ┌─────────┐
        │ 微调    │    │ 提示工程 │
        └─────────┘    └─────────┘
```

### 4.2 微调流程 checklist

```
□ 1. 明确目标：确定微调要解决的具体问题
□ 2. 准备数据：收集、清洗、格式化数据
□ 3. 选择基座模型：根据任务和预算选择
□ 4. 选择微调方法：Full / LoRA / QLoRA / Prompt Tuning
□ 5. 设置超参数：学习率、batch size、epochs 等
□ 6. 训练监控：观察 loss 曲线，防止过拟合
□ 7. 评估测试：在测试集上评估效果
□ 8. 部署上线：合并权重，优化推理
```

### 4.3 常见陷阱

| 陷阱 | 解决方案 |
|------|----------|
| 数据泄露 | 严格划分训练/验证/测试集 |
| 过拟合 | 早停、dropout、数据增强 |
| 灾难性遗忘 | 使用 LoRA、保留原始能力测试 |
| 评估偏差 | 使用多样化评估指标 |
| 数据偏见 | 审查训练数据，确保公平性 |

---

## 参考资料

1. **Transformer 论文**: "Attention Is All You Need" (Vaswani et al., 2017) — [arxiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. **GPT-3 论文**: "Language Models are Few-Shot Learners" (Brown et al., 2020)
3. **Chinchilla 论文**: "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022) — [arxiv:2203.15556](https://arxiv.org/abs/2203.15556)
4. **LoRA 论文**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021) — [arxiv:2106.09685](https://arxiv.org/abs/2106.09685)
5. **QLoRA 论文**: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023) — [arxiv:2305.14314](https://arxiv.org/abs/2305.14314)
6. **DPO 论文**: "Direct Preference Optimization" (Rafailov et al., 2023) — [arxiv:2305.18290](https://arxiv.org/abs/2305.18290)
7. **DeepSeek-R1 论文**: "Incentivizing Reasoning Capability in LLMs via RL" (DeepSeek-AI, 2025) — [arxiv:2501.12948](https://arxiv.org/abs/2501.12948)
8. **RLHF 论文**: "Training language models to follow instructions with human feedback" (Ouyang et al., 2022)
9. **Scaling Laws**: "Scaling Laws for Neural Language Models" (Kaplan et al., 2020)
10. **Hugging Face LLM Course**: [huggingface.co/learn/llm-course](https://huggingface.co/learn/llm-course)
11. **Hugging Face PEFT**: [huggingface.co/docs/peft](https://huggingface.co/docs/peft)
12. **vLLM 文档**: [docs.vllm.ai](https://docs.vllm.ai/)

---

*访问日期: 2026-06-06*

## 💡 我的思考

1. **预训练是根基**：大模型的能力主要来自预训练阶段，后续微调只是调整行为模式。理解这一点有助于合理设定微调预期。

2. **数据质量 > 数据数量**：在微调阶段，1000 条高质量数据往往比 10000 条低质量数据效果更好。人工审核和清洗数据的时间是值得的。

3. **LoRA 是性价比之王**：对于大多数应用场景，LoRA/QLoRA 是最佳选择。它既保留了预训练模型的通用能力，又能以极低成本适配特定任务。

4. **评估是关键**：没有好的评估，就无法判断微调是否成功。建议建立多维度的评估体系（自动指标 + 人工评估）。

5. **推理优化不可忽视**：模型训练完成后，推理阶段的优化（量化、KV Cache、批处理）直接影响用户体验和成本。

6. **Chat Template 是隐形陷阱**：不同模型的对话格式差异很大，使用 `apply_chat_template()` 是避免错误的最佳实践。

7. **持续学习是挑战**：当前微调方法容易导致灾难性遗忘。未来需要更好的持续学习方案，让模型能够不断吸收新知识而不遗忘旧知识。

8. **GRPO 值得关注**：DeepSeek-R1 的 GRPO 算法在可验证任务上表现出色，且无需 Critic 模型，大幅降低训练成本。
