#topic/data-engineering #topic/embedding #topic/vector #year/2026 #status/draft

# 嵌入模型选型与微调

> **一句话定位**：Embedding 是 RAG 和语义搜索的核心——选择、评估和微调适合业务场景的嵌入模型。

---

## 1. 嵌入模型基础

### 1.1 什么是 Embedding

Embedding 是将离散数据（文本、图像等）映射到连续向量空间的技术：

```
文本: "机器学习是人工智能的一个分支"
  ↓ Embedding Model
向量: [0.023, -0.156, 0.089, ..., 0.034]  (768 维)
```

**关键特性**：
- 语义相似的文本 → 向量距离近
- 不同模型生成的向量**不兼容**
- 维度固定（由模型决定）

### 1.2 嵌入模型的演进

| 时间 | 模型 | 创新 | 维度 |
|------|------|------|------|
| 2013 | Word2Vec | 词级别静态嵌入 | 300 |
| 2018 | BERT | 上下文相关嵌入 | 768 |
| 2019 | Sentence-BERT | 句子级别嵌入 | 768 |
| 2022 | OpenAI text-embedding-ada-002 | 通用文本嵌入 | 1536 |
| 2023 | E5 / BGE | 开源 SOTA | 768/1024 |
| 2024 | OpenAI text-embedding-3 | 多维度、可降维 | 3072/1536/512 |
| 2025 | Jina Embeddings / Nomic | 长上下文、多语言 | 768/1024 |

---

## 2. 主流嵌入模型对比

### 2.1 综合对比表

| 模型 | 维度 | 上下文长度 | 多语言 | 开源 | 适用场景 |
|------|------|-----------|--------|------|----------|
| **text-embedding-3-large** | 3072 | 8K | ✅ | ❌ | 通用、高精度 |
| **text-embedding-3-small** | 1536 | 8K | ✅ | ❌ | 通用、低成本 |
| **BGE-M3** | 1024 | 8K | ✅ | ✅ | 多语言、RAG |
| **E5-Mistral-7B** | 4096 | 32K | ✅ | ✅ | 长文档 |
| **Jina-Embeddings-v3** | 768 | 8K | ✅ | ✅ | 通用、轻量 |
| **Nomic-Embed-v1.5** | 768 | 8K | ✅ | ✅ | 开源、高质量 |
| **GTE-Qwen2-7B** | 3584 | 32K | ✅ | ✅ | 长上下文 |
| **bge-large-zh-v1.5** | 1024 | 512 | ❌ | ✅ | 中文专用 |

### 2.2 详细分析

#### OpenAI text-embedding-3 系列

**特点**：
- 支持维度裁剪（3072 → 1536 → 512）
- 性能随维度降低 gracefully 下降
- API 调用方便，但需联网

**选型建议**：
```python
# 高精度场景
text-embedding-3-large  # 3072 维

# 平衡场景
text-embedding-3-small  # 1536 维

# 低成本场景
text-embedding-3-large + dimensions=512  # 降维
```

#### BGE (BAAI General Embedding)

**特点**：
- 北京智源研究院开源
- BGE-M3 支持多语言、多粒度（句子/段落/文档）
- 中文效果优秀

**使用**：
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3')

# 编码时添加指令前缀（重要！）
instruction = "为这个句子生成表示："
texts = [instruction + t for t in texts]
embeddings = model.encode(texts, normalize_embeddings=True)
```

#### E5 系列

**特点**：
- Microsoft 开源
- 弱监督训练，效果优秀
- E5-Mistral-7B 支持 32K 上下文

**使用**：
```python
model = SentenceTransformer('intfloat/e5-mistral-7b-instruct')

# E5 需要特定格式
def format_e5(text: str, is_query: bool = False) -> str:
    if is_query:
        return f"query: {text}"
    return f"passage: {text}"
```

---

## 3. 嵌入模型评估

### 3.1 评估基准

| 基准 | 测试内容 | 代表模型分数 |
|------|----------|-------------|
| **MTEB** | 多任务嵌入基准（58 个数据集） | BGE-M3: 66.76 |
| **C-MTEB** | 中文嵌入基准 | bge-large-zh: 64.53 |
| **BEIR** | 零样本信息检索 | E5-large-v2: 50.0 |
| **LongEmbed** | 长上下文嵌入 | GTE-Qwen2-7B: 58.2 |

### 3.2 业务场景评估

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def evaluate_embedding_model(
    model_name: str,
    test_queries: list[str],
    test_documents: list[str],
    relevance_judgments: dict  # query_idx -> [relevant_doc_indices]
) -> dict:
    """评估嵌入模型在业务数据上的表现"""
    
    model = SentenceTransformer(model_name)
    
    # 编码
    query_embeddings = model.encode(test_queries)
    doc_embeddings = model.encode(test_documents)
    
    # 计算相似度
    similarity_matrix = cosine_similarity(query_embeddings, doc_embeddings)
    
    # 计算 Recall@K
    metrics = {}
    for k in [1, 3, 5, 10]:
        recalls = []
        for q_idx, relevant_docs in relevance_judgments.items():
            # 获取 Top-K 结果
            top_k_indices = np.argsort(similarity_matrix[q_idx])[-k:][::-1]
            
            # 计算召回率
            retrieved_relevant = len(set(top_k_indices) & set(relevant_docs))
            recall = retrieved_relevant / len(relevant_docs)
            recalls.append(recall)
        
        metrics[f'recall@{k}'] = np.mean(recalls)
    
    return metrics
```

---

## 4. 嵌入模型微调

### 4.1 何时需要微调

| 场景 | 是否需要微调 | 说明 |
|------|-------------|------|
| 通用语义搜索 | ❌ 不需要 | 预训练模型已足够 |
| 垂直领域（法律/医疗） | ✅ 建议微调 | 领域术语理解 |
| 特定格式数据 | ✅ 需要微调 | 如代码、表格 |
| 多语言混合 | ✅ 可能需要 | 特定语言组合 |

### 4.2 微调方法

#### 方法 1：Sentence Transformers 微调

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 准备训练数据
train_examples = [
    InputExample(texts=["查询1", "相关文档1"], label=1.0),
    InputExample(texts=["查询1", "不相关文档"], label=0.0),
    # ...
]

# 加载基础模型
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 定义损失函数
train_loss = losses.CosineSimilarityLoss(model)

# 训练
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path='./fine-tuned-embedding'
)
```

#### 方法 2：对比学习微调

```python
from sentence_transformers import SentenceTransformer, losses

# 使用 MultipleNegativesRankingLoss（无需显式负样本）
model = SentenceTransformer('BAAI/bge-m3')

# 准备 (query, positive) 对
train_examples = [
    InputExample(texts=["查询", "正例文档"]),
    # ...
]

train_loss = losses.MultipleNegativesRankingLoss(model)
```

#### 方法 3：领域自适应

```python
def domain_adaptation_finetune(
    base_model: str,
    domain_texts: list[str],
    output_path: str
):
    """
    使用领域文本进行无监督微调（MLM）
    """
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForMaskedLM.from_pretrained(base_model)
    
    # 编码文本
    encodings = tokenizer(
        domain_texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # MLM 训练
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=encodings
    )
    
    trainer.train()
```

---

## 5. 嵌入模型部署优化

### 5.1 ONNX 导出与加速

```python
from sentence_transformers import SentenceTransformer
import onnx
from onnxruntime import InferenceSession

# 导出 ONNX
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
model.save('./models/embedding', 'onnx')

# ONNX Runtime 推理
session = InferenceSession('./models/embedding/model.onnx')

# 比 PyTorch 快 2-3 倍
```

### 5.2 量化压缩

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction

# 动态量化（INT8）
model = ORTModelForFeatureExtraction.from_pretrained(
    './models/embedding',
    export=True,
    quantization_config={"is_static": False}
)

# 模型大小减少 75%，速度提升 2-3 倍
```

### 5.3 批处理优化

```python
def batch_encode(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """批量编码优化"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)
```

---

## 6. 多模态嵌入

### 6.1 CLIP 风格多模态嵌入

```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 文本嵌入
texts = ["一只猫", "一只狗"]
text_inputs = processor(text=texts, return_tensors="pt", padding=True)
text_embeddings = model.get_text_features(**text_inputs)

# 图像嵌入
from PIL import Image
image = Image.open("cat.jpg")
image_inputs = processor(images=image, return_tensors="pt")
image_embeddings = model.get_image_features(**image_inputs)

# 跨模态相似度
similarity = (text_embeddings @ image_embeddings.T).softmax(dim=-1)
```

### 6.2 统一多模态嵌入（Jina-CLIP）

```python
from transformers import AutoModel

model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)

# 统一接口处理文本和图像
text_embeddings = model.encode_text(["一只猫坐在沙发上"])
image_embeddings = model.encode_image(["cat.jpg"])
```

---

## 💡 我的思考

1. **Embedding 模型选择是 RAG 系统的关键决策**：向量质量直接决定检索效果，不能只看 benchmark 分数，要在自己的数据上测试。

2. **开源模型已经足够好**：BGE、E5、Jina 等开源模型在大多数场景下与 OpenAI 嵌入效果接近，且可本地部署。

3. **微调要谨慎**：微调可能提升领域效果，但也可能破坏通用能力。建议先尝试 Prompt 工程（如添加领域前缀），再考虑微调。

4. **维度不是越大越好**：3072 维不一定比 768 维好，要考虑存储成本和检索速度。OpenAI 的 text-embedding-3 支持动态降维是很好的设计。

5. **多模态嵌入是趋势**：CLIP 证明了图文可以共享嵌入空间，未来文本、图像、音频、视频的统一嵌入将成为标准。

---

## 参考来源

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — 访问日期：2026-06-10
- [BGE Paper: C-Pack](https://arxiv.org/abs/2309.07597) — 访问日期：2026-06-10
- [E5 Paper: Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533) — 访问日期：2026-06-10
- [OpenAI Embeddings Documentation](https://platform.openai.com/docs/guides/embeddings) — 访问日期：2026-06-10
- [Sentence Transformers Documentation](https://www.sbert.net/) — 访问日期：2026-06-10

---

*最后更新：2026-06-12*
