# 多模态 RAG 架构设计

> **一句话定位**：从文本 RAG 到多模态知识库，系统梳理图文音视频统一检索的架构设计与工程实践。
>
> #status/draft #topic/multimodal #topic/rag #year/2026

---

## 一、为什么需要多模态 RAG

### 1.1 传统文本 RAG 的局限

```
传统 RAG 只能处理：
├── 纯文本文档（txt, md, docx）
├── 结构化数据（csv, json）
└── 网页内容（HTML 转文本）

无法处理：
├── PDF 中的图表、图片
├── 产品图片、设计稿
├── 视频教程、会议录像
├── 语音留言、播客
└── 扫描件、手写笔记
```

### 1.2 多模态 RAG 的价值

| 场景 | 传统 RAG | 多模态 RAG |
|------|----------|------------|
| **产品手册** | 只提取文字 | 理解产品图片 + 规格表 |
| **医疗报告** | 丢失影像信息 | 结合 CT 图 + 诊断文字 |
| **电商客服** | 文字描述不清 | 用户上传图片直接识别 |
| **法律合同** | 忽略签名/印章 | 验证签名真伪 |
| **教育内容** | 纯文字枯燥 | 视频 + 图文综合回答 |

---

## 二、核心架构设计

### 2.1 统一嵌入空间

多模态 RAG 的核心挑战：**将不同模态映射到同一向量空间**

```
┌─────────────────────────────────────────────────────────┐
│                    统一嵌入空间                           │
│                                                         │
│   文本 ──→ [文本编码器] ──┐                             │
│                           │                             │
│   图像 ──→ [图像编码器] ──┼──→ 统一向量空间 ──→ 检索     │
│                           │                             │
│   音频 ──→ [音频编码器] ──┘                             │
│                                                         │
│   查询（任意模态）──→ 对应编码器 ──→ 同一空间检索          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 三种架构模式

#### 模式一：独立编码 + 融合检索

```python
# 各模态独立编码，检索时融合结果
from sentence_transformers import SentenceTransformer
import clip

# 文本编码器
text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

# 图像编码器
image_encoder, preprocess = clip.load("ViT-B/32")

# 分别建立索引
text_index = build_faiss_index(text_embeddings)
image_index = build_faiss_index(image_embeddings)

# 查询时融合
query = "红色连衣裙"
text_results = text_index.search(text_encoder.encode(query))
image_results = image_index.search(clip.encode_text(query))

# 重排序融合
final_results = reciprocal_rank_fusion(text_results, image_results)
```

**优点**：
- 各模态可用最佳编码器
- 灵活组合

**缺点**：
- 需要融合策略
- 不是真正统一空间

#### 模式二：统一多模态编码器

```python
# 使用原生多模态模型
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 文本和图像进入同一模型
text_features = model.get_text_features(**processor(text=["猫"], return_tensors="pt"))
image_features = model.get_image_features(**processor(images=[cat_image], return_tensors="pt"))

# 直接比较相似度
similarity = (text_features @ image_features.T).softmax(dim=-1)
```

**代表模型**：
| 模型 | 模态 | 特点 |
|------|------|------|
| **CLIP** | 图文 | 最经典，4亿图文对训练 |
| **OpenCLIP** | 图文 | 开源复刻，更多变体 |
| **SigLIP** | 图文 | Google，更优的对比学习 |
| **ImageBind** | 图文音深 | Meta，六种模态统一 |
| **Jina-CLIP** | 图文 | 针对 RAG 优化 |

**优点**：
- 真正统一空间
- 跨模态检索自然

**缺点**：
- 各模态可能不是最优
- 训练数据要求高

#### 模式三：多模态 LLM 理解

```python
# 用 VLM 直接理解多模态内容
from transformers import Qwen2VLForConditionalGeneration

model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B")

# 输入图文，输出结构化理解
response = model.chat([
    {"image": "chart.png"},
    {"text": "解释这张图表的关键趋势"}
])
# → "图表显示 2024 年 Q1 销售额同比增长 35%，主要驱动力是..."
```

**流程**：
```
文档 → [VLM 理解] → 生成文本描述 → [文本编码] → 向量库
查询 → [VLM 理解] → 生成查询意图 → [文本编码] → 检索
```

**优点**：
- 理解深度最高
- 可处理复杂推理

**缺点**：
- 成本高（每次需调用 VLM）
- 延迟大
- 索引阶段就需要 VLM

### 2.3 架构选型决策树

```
数据以什么模态为主？
├── 90%+ 文本，少量图片
│   └── 模式一：文本为主，图片单独索引
│
├── 图文各半（电商、媒体）
│   └── 模式二：CLIP 类统一编码
│
├── 复杂文档（PDF 含图表）
│   └── 模式三：VLM 生成描述 + 文本 RAG
│
└── 视频/音频为主
    └── 模式三：先转录/抽帧，再文本 RAG
```

---

## 三、文档处理流程

### 3.1 多模态文档解析

```python
# 使用 Unstructured 解析复杂文档
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    "report.pdf",
    extract_images_in_pdf=True,      # 提取图片
    infer_table_structure=True,       # 解析表格
    chunking_strategy="by_title",     # 按标题分块
    max_characters=4000,
    new_after_n_chars=3800,
)

for element in elements:
    if element.category == "Image":
        # 图片送入 VLM 生成描述
        description = vlm.describe(element.image)
        store(description, type="image", source=element.metadata)
    elif element.category == "Table":
        # 表格转为 Markdown
        markdown = table_to_markdown(element)
        store(markdown, type="table")
    else:
        # 文本直接存储
        store(element.text, type="text")
```

### 3.2 视频处理 Pipeline

```
视频文件
  ↓
┌─────────────────┐
│ 1. 音频提取     │ → Whisper ASR → 文本转录
└─────────────────┘
  ↓
┌─────────────────┐
│ 2. 关键帧提取   │ → 每秒/场景切换提取帧
└─────────────────┘
  ↓
┌─────────────────┐
│ 3. VLM 理解     │ → 生成帧描述
└─────────────────┘
  ↓
┌─────────────────┐
│ 4. 时间戳对齐   │ → "00:01:23 画面显示..."
└─────────────────┘
  ↓
统一存入向量库
```

### 3.3 图像索引策略

| 策略 | 方法 | 适用场景 |
|------|------|----------|
| **全局特征** | CLIP 整图嵌入 | 相似图片搜索 |
| **区域特征** | 目标检测 + 局部嵌入 | 细粒度检索 |
| **OCR + 理解** | 提取图中文字 + VLM 描述 | 图表、截图 |
| **多尺度** | 原图 + 局部 + OCR 分别索引 | 综合检索 |

---

## 四、检索与融合

### 4.1 跨模态检索

```python
# 文本查询检索图片
def retrieve_images(query: str, top_k: int = 5):
    # 文本编码
    query_embedding = clip.encode_text(query)
    
    # 在图像索引中检索
    results = image_index.search(query_embedding, k=top_k)
    
    return results

# 图片查询检索文本
def retrieve_texts(image, top_k: int = 5):
    # 图像编码
    query_embedding = clip.encode_image(image)
    
    # 在文本索引中检索
    results = text_index.search(query_embedding, k=top_k)
    
    return results
```

### 4.2 结果融合策略

#### 早期融合（推荐）

```python
# 查询时同时编码所有模态
query = {
    "text": "红色连衣裙",
    "image": reference_image  # 可选参考图
}

# 统一编码（如 ImageBind）
query_embedding = imagebind.encode(query)

# 在统一索引中检索
results = unified_index.search(query_embedding)
```

#### 晚期融合

```python
# 分别检索，再融合
results = []

if query.text:
    text_results = text_index.search(query.text_embedding)
    results.append(text_results)

if query.image:
    image_results = image_index.search(query.image_embedding)
    results.append(image_results)

# 融合策略
final = reciprocal_rank_fusion(results)  # RRF
# 或
final = weighted_merge(results, weights=[0.7, 0.3])
```

### 4.3 重排序优化

```python
# 使用多模态交叉编码器重排序
from transformers import AutoModelForVision2Seq

reranker = AutoModelForVision2Seq.from_pretrained("cross-encoder/vlm-reranker")

# 对初筛结果精排
candidates = initial_results[:50]
scores = []

for doc in candidates:
    score = reranker(
        query_image=query.image,
        query_text=query.text,
        doc_image=doc.image,
        doc_text=doc.text
    )
    scores.append(score)

# 按重排分数排序
final_results = sorted(candidates, key=lambda x: scores[x], reverse=True)[:10]
```

---

## 五、工程实现

### 5.1 完整 Pipeline 示例

```python
class MultimodalRAG:
    def __init__(self):
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.image_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B")
        self.vector_store = QdrantClient(path="/tmp/qdrant")
    
    def index_document(self, file_path: str):
        """索引多模态文档"""
        # 1. 解析文档
        elements = parse_document(file_path)
        
        for element in elements:
            if element.type == "image":
                # 生成多模态描述
                description = self.vlm.describe(element.content)
                embedding = self.image_encoder.encode(
                    image=element.content,
                    text=description
                )
            else:
                embedding = self.text_encoder.encode(element.content)
            
            # 存入向量库
            self.vector_store.upsert(
                collection="multimodal",
                points=[{
                    "id": element.id,
                    "vector": embedding,
                    "payload": {
                        "content": element.content,
                        "type": element.type,
                        "source": file_path,
                        "page": element.page
                    }
                }]
            )
    
    def query(self, query_text: str = None, query_image = None, top_k: int = 5):
        """多模态查询"""
        # 构建查询嵌入
        if query_image:
            query_embedding = self.image_encoder.encode(
                image=query_image,
                text=query_text or ""
            )
        else:
            query_embedding = self.text_encoder.encode(query_text)
        
        # 检索
        results = self.vector_store.search(
            collection="multimodal",
            query_vector=query_embedding,
            limit=top_k
        )
        
        return results
```

### 5.2 向量库选择

| 特性 | Qdrant | Milvus | Weaviate | PGVector |
|------|--------|--------|----------|----------|
| **多向量集合** | ✅ | ✅ | ✅ | ❌ |
| **混合搜索** | ✅ | ✅ | ✅ | ⚠️ |
| **多模态原生** | ⚠️ | ⚠️ | ✅ | ❌ |
| **开源** | ✅ | ✅ | ✅ | ✅ |
| **云托管** | ✅ | ✅ | ✅ | ⚠️ |

**Weaviate 多模态优势**：
- 原生支持多模态模块
- 内置 CLIP、ImageBind 集成
- 跨模态查询语法简洁

### 5.3 性能优化

```python
# 1. 分层索引
# 第一层：粗筛（快速）
coarse_results = hnsw_index.search(query, k=100)

# 第二层：精排（慢但准）
final_results = cross_encoder.rerank(query, coarse_results, k=10)

# 2. 缓存热门查询
cache = LRUCache(maxsize=1000)

# 3. 异步处理
async def index_batch(documents):
    tasks = [process_doc(doc) for doc in documents]
    await asyncio.gather(*tasks)

# 4. 量化降维
# 768维 → 128维 (PCA)
# float32 → int8 (量化)
```

---

## 六、评估方法

### 6.1 多模态 RAG 评估指标

| 指标 | 说明 | 测量方法 |
|------|------|----------|
| **Recall@K** | 前 K 结果中相关文档比例 | 人工标注相关性 |
| **MRR** | 首个相关结果排名倒数 | 平均排名位置 |
| **NDCG** | 考虑排序质量的相关性 | 加权折扣累积增益 |
| **跨模态准确率** | 图文互搜准确率 | 标注图文对测试 |
| **延迟** | 端到端响应时间 | 压测 |

### 6.2 测试集构建

```python
# 构建多模态测试集
test_cases = [
    {
        "query": {
            "text": "2024年Q1销售额",
            "image": None
        },
        "expected": ["report_q1.pdf#page=5", "chart_sales.png"],
        "modality": "text_to_multimodal"
    },
    {
        "query": {
            "text": "找类似这个的产品",
            "image": reference_product_image
        },
        "expected": ["product_123.jpg", "product_456.jpg"],
        "modality": "image_to_image"
    }
]
```

---

## 💡 我的思考

### 关键洞察

1. **统一嵌入空间是核心挑战** — CLIP 解决了图文，但音视频仍需努力
2. **VLM 是桥梁** — 用 VLM 将非文本模态转为文本描述，是最实用的过渡方案
3. **文档解析决定上限** — 再强的检索也救不了错误的文档切分
4. **延迟与质量的权衡** — 跨模态重排序提升质量但增加延迟

### 选型建议

| 场景 | 推荐架构 |
|------|----------|
| 快速启动 | VLM 生成描述 + 传统文本 RAG |
| 电商/媒体 | CLIP + 双塔架构 |
| 企业知识库 | Unstructured 解析 + 多模态索引 |
| 视频平台 | 抽帧 + ASR + 统一索引 |
| 高精度要求 | 三阶段：召回 → VLM 重排 → 生成 |

### 常见陷阱

- ❌ 忽视文档解析质量（PDF 解析错误导致检索失败）
- ❌ 所有模态用同一编码器（应该用最优的各自编码器）
- ❌ 忽略跨模态歧义（"苹果"是水果还是公司？）
- ❌ 不处理时间戳（视频检索需精确到秒）

### 下一步实践

- [ ] 用 CLIP 实现图文互搜 Demo
- [ ] 测试 Unstructured 解析复杂 PDF 的效果
- [ ] 对比早期融合 vs 晚期融合的检索质量
- [ ] 实现视频内容的秒级定位检索

---

## 参考来源

1. OpenAI - CLIP: Connecting Text and Images (2021)
2. Meta - ImageBind: One Embedding Space for Six Modalities (2023)
3. LlamaIndex - Multi-Modal Retrieval with GPT4V
4. Pinecone - Multi-modal ML with CLIP
5. Unstructured - Document Processing for LLMs
6. Qdrant - Vector Search Documentation
7. Weaviate - Multi-Modal AI Documentation

---

*最后更新：2026-06-08*
