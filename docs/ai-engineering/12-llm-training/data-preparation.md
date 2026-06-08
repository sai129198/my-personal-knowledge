# 数据准备与清洗

> **一句话定位**：数据是预训练的核心——高质量的数据准备决定模型上限。

---

## 1. 数据收集

### 1.1 公开数据集

| 数据集 | 规模 | 类型 | 说明 |
|--------|------|------|------|
| **Common Crawl** | 数百 TB | 网页 | 原始网页数据，需大量清洗 |
| **C4** | 800GB | 网页 | 清洗后的 Common Crawl |
| **The Pile** | 825GB | 混合 | 22 个高质量子集 |
| **BooksCorpus** | 11GB | 书籍 | 未出版书籍 |
| **Gutenberg** | 20GB | 书籍 | 公共领域书籍 |
| **Wikipedia** | 20GB | 百科 | 多语言百科 |
| **arXiv** | 100GB | 论文 | 学术论文 |
| **GitHub** | 100GB+ | 代码 | 开源代码 |
| **Reddit** | 50GB+ | 对话 | 社交媒体对话 |

### 1.2 数据收集策略

**质量分层**：
```
Tier 1（最高质量）：
- 专业书籍、学术论文
- 编辑过的百科内容
- 高质量新闻

Tier 2（高质量）：
- 技术博客、教程
- 论坛高质量回答
- 代码文档

Tier 3（一般质量）：
- 普通网页内容
- 社交媒体
- 用户生成内容
```

---

## 2. 数据清洗

### 2.1 清洗流程

```
原始数据
  ├── 文本提取 → 从 HTML/PDF 提取纯文本
  ├── 语言识别 → 保留目标语言
  ├── 质量过滤 → 去除低质量内容
  ├── 去重处理 → 去除重复内容
  ├── 敏感过滤 → 去除 PII 和有害内容
  ├── 格式标准化 → 统一编码和格式
  └── 最终数据集
```

### 2.2 文本提取

**HTML 处理**：
```python
from bs4 import BeautifulSoup

def extract_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    # 去除脚本和样式
    for script in soup(['script', 'style']):
        script.decompose()
    # 提取文本
    text = soup.get_text()
    # 清理空白
    lines = (line.strip() for line in text.splitlines())
    text = ' '.join(chunk for line in lines for chunk in line.split())
    return text
```

**PDF 处理**：
- 使用 pdfplumber 或 PyPDF2
- 处理表格和图表
- 保留段落结构

### 2.3 语言识别

**工具选择**：
- **fastText**：快速、准确
- **langdetect**：支持多种语言
- **cld3**：Google 的语言检测

```python
import fasttext

# 加载预训练模型
model = fasttext.load_model('lid.176.bin')

def detect_language(text):
    predictions = model.predict(text, k=1)
    lang = predictions[0][0].replace('__label__', '')
    score = predictions[1][0]
    return lang, score

# 过滤非目标语言
def filter_language(texts, target_lang='zh', threshold=0.9):
    filtered = []
    for text in texts:
        lang, score = detect_language(text)
        if lang == target_lang and score > threshold:
            filtered.append(text)
    return filtered
```

### 2.4 质量过滤

**基于规则**：
- 长度过滤：去除过短（< 100 字符）或过长的文档
- 符号比例：去除符号过多的文本
- 词汇多样性：去除重复词汇过多的文本

**基于模型**：
```python
# 使用语言模型评估质量
def quality_score(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
    # 困惑度越低，质量越高
    perplexity = torch.exp(loss).item()
    return perplexity

# 过滤低质量文本
def filter_quality(texts, threshold=100):
    filtered = []
    for text in texts:
        score = quality_score(text)
        if score < threshold:  # 困惑度低于阈值
            filtered.append(text)
    return filtered
```

### 2.5 去重处理

**精确去重**：
```python
# 基于哈希的去重
def exact_deduplicate(texts):
    seen = set()
    unique = []
    for text in texts:
        hash_val = hashlib.md5(text.encode()).hexdigest()
        if hash_val not in seen:
            seen.add(hash_val)
            unique.append(text)
    return unique
```

**模糊去重（MinHash + LSH）**：
```python
from datasketch import MinHash, MinHashLSH

def create_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    # 使用 5-gram
    for i in range(len(text) - 4):
        shingle = text[i:i+5]
        m.update(shingle.encode('utf8'))
    return m

def fuzzy_deduplicate(texts, threshold=0.9):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = {}
    
    for idx, text in enumerate(texts):
        m = create_minhash(text)
        # 查询相似文档
        result = lsh.query(m)
        if not result:  # 没有相似文档
            lsh.insert(idx, m)
            minhashes[idx] = text
    
    return list(minhashes.values())
```

### 2.6 敏感内容过滤

**PII 检测**：
```python
import re

# 常见 PII 模式
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
}

def remove_pii(text):
    for name, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f'[{name}_REDACTED]', text)
    return text
```

**毒性内容过滤**：
- 使用毒性分类器（如 Perspective API）
- 关键词过滤
- 人工审核抽样

---

## 3. 数据配比

### 3.1 配比策略

**LLaMA 配比**：

| 数据源 | 采样比例 | 说明 |
|--------|----------|------|
| Common Crawl | 67.0% | 经过清洗的网页 |
| C4 | 15.0% | 清洗后的网页 |
| GitHub | 4.5% | 代码数据 |
| Wikipedia | 4.5% | 百科数据 |
| Books | 4.5% | 书籍数据 |
| arXiv | 2.5% | 学术论文 |
| StackExchange | 2.0% | 问答数据 |

**The Pile 配比**：

| 数据源 | 占比 | 说明 |
|--------|------|------|
| Common Crawl | 18.1% | 网页数据 |
| PubMed Central | 14.4% | 医学论文 |
| Books3 | 12.1% | 书籍 |
| OpenWebText2 | 10.8% | Reddit 链接 |
| ArXiv | 8.6% | 学术论文 |
| GitHub | 7.6% | 代码 |
| FreeLaw | 6.1% | 法律文本 |
| Stack Exchange | 5.2% | 问答 |
| USPTO | 4.0% | 专利 |
| Others | 13.1% | 其他 |

### 3.2 配比原则

1. **质量优先**：高质量数据源可增加采样权重
2. **多样性**：覆盖不同领域和风格
3. **任务相关**：根据下游任务调整配比
4. **实验验证**：通过实验确定最优配比

---

## 4. 数据预处理

### 4.1 分词

**Tokenizer 选择**：

| Tokenizer | 特点 | 适用 |
|-----------|------|------|
| **BPE** | 字节对编码 | GPT 系列 |
| **WordPiece** | 子词单元 | BERT 系列 |
| **SentencePiece** | 语言无关 | 多语言模型 |
| **Unigram** | 概率模型 | T5、LLaMA |

**训练 Tokenizer**：
```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 初始化 BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 训练
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["<pad>", "<unk>", "<s>", "</s>"]
)

files = ["train.txt"]
tokenizer.train(files, trainer)

# 保存
tokenizer.save("tokenizer.json")
```

### 4.2 数据格式化

**JSONL 格式**：
```json
{"text": "文档内容...", "meta": {"source": "wikipedia", "language": "zh"}}
{"text": "文档内容...", "meta": {"source": "github", "language": "python"}}
```

**TFRecord / Arrow**：
- 高效存储和读取
- 支持流式加载
- 适合大规模训练

### 4.3 数据加载优化

**流式加载**：
```python
from datasets import load_dataset

# 流式加载大数据集
dataset = load_dataset(
    "json",
    data_files="path/to/data/*.jsonl",
    streaming=True
)

# 实时处理
def process_batch(batch):
    # 分词
    tokens = tokenizer(batch['text'], truncation=True, max_length=2048)
    return tokens

dataset = dataset.map(process_batch, batched=True)
```

---

## 5. 数据质量评估

### 5.1 自动评估

**统计指标**：

| 指标 | 说明 | 目标 |
|------|------|------|
| **文档数量** | 总文档数 | 根据模型规模确定 |
| **Token 数量** | 总 token 数 | 通常 1-2T for 7B |
| **平均长度** | 文档平均长度 | 1000-5000 tokens |
| **词汇多样性** | 唯一 token / 总 token | > 0.5 |
| **语言分布** | 各语言占比 | 符合目标 |

**质量分布**：
```python
def analyze_dataset(texts):
    lengths = [len(t) for t in texts]
    
    print(f"文档数量: {len(texts)}")
    print(f"平均长度: {np.mean(lengths):.0f}")
    print(f"中位数长度: {np.median(lengths):.0f}")
    print(f"长度标准差: {np.std(lengths):.0f}")
    print(f"最短文档: {min(lengths)}")
    print(f"最长文档: {max(lengths)}")
    
    # 长度分布
    plt.hist(lengths, bins=50)
    plt.xlabel('Document Length')
    plt.ylabel('Count')
    plt.title('Document Length Distribution')
    plt.show()
```

### 5.2 人工评估

**抽样检查**：
- 随机抽取 100-1000 条数据
- 评估内容质量、格式正确性
- 检查敏感内容残留

**评估维度**：
- 可读性：是否人类可读
- 连贯性：内容是否连贯
- 准确性：事实是否正确
- 安全性：是否包含有害内容

---

## 6. 数据版本管理

### 6.1 版本控制

**数据版本**：
```
dataset-v1.0/
  ├── raw/           # 原始数据
  ├── cleaned/       # 清洗后数据
  ├── deduped/       # 去重后数据
  ├── final/         # 最终数据集
  └── metadata.json  # 元数据
```

**元数据记录**：
```json
{
  "version": "1.0",
  "created": "2024-01-01",
  "sources": [
    {"name": "common_crawl", "size": "500GB", "filtering": "quality_score > 0.7"}
  ],
  "statistics": {
    "total_documents": 100000000,
    "total_tokens": 1000000000000,
    "average_length": 10000
  },
  "processing": {
    "language_filter": "zh",
    "quality_threshold": 0.7,
    "deduplication": "minhash_lsh"
  }
}
```

### 6.2 数据血缘

**追踪数据来源**：
- 原始数据来源
- 处理步骤记录
- 转换逻辑记录
- 质量评估结果

---

## 7. 最佳实践

### 7.1 数据准备清单

- [ ] 确定数据来源和收集策略
- [ ] 实施数据清洗流程
- [ ] 进行去重处理
- [ ] 过滤敏感内容
- [ ] 确定数据配比
- [ ] 训练 tokenizer
- [ ] 格式化数据
- [ ] 质量评估
- [ ] 版本管理

### 7.2 常见陷阱

| 陷阱 | 说明 | 避免方法 |
|------|------|----------|
| **数据污染** | 测试集混入训练集 | 严格分离数据 |
| **重复数据** | 训练/测试集重复 | 彻底去重 |
| **低质量数据** | 噪声数据影响训练 | 质量过滤 |
| **数据偏见** | 特定群体代表性不足 | 多样性检查 |
| **版权风险** | 使用受版权保护内容 | 合规审查 |

---

## 8. 总结

数据准备是预训练的基础工程：

1. **质量第一**：高质量数据比大量低质量数据更重要
2. **多样性**：覆盖不同领域、风格、语言
3. **彻底清洗**：去重、过滤、标准化
4. **持续监控**：建立数据质量监控体系

> **关键洞察**：在预训练中，数据质量往往比模型架构更重要。Garbage in, garbage out——高质量的数据是训练出优秀模型的前提。

---

*参考来源：*
- *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*
- *LLaMA: Open and Efficient Foundation Language Models*
- *Common Crawl: Web Data Commons*
- *Deduplicating Training Data Makes Language Models Better*
