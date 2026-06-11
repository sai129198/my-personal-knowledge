#topic/data-engineering #topic/preprocessing #year/2026 #status/draft

# 数据预处理与清洗

> **一句话定位**：AI 系统的数据质量决定模型上限——从原始数据到高质量训练/检索数据的全流程清洗策略。

---

## 1. 为什么数据预处理至关重要

### 1.1 数据质量对模型的影响

| 数据问题 | 对模型的影响 | 典型案例 |
|----------|-------------|----------|
| **噪声数据** | 模型学习错误模式 | 网页中的广告、导航栏文本 |
| **重复数据** | 训练效率降低、过拟合 | 同一新闻被多个网站转载 |
| **低质量内容** | 模型输出质量下降 | 机器翻译文本、SEO 垃圾内容 |
| **格式不一致** | 解析困难、信息丢失 | PDF 中的表格、多列布局 |
| **敏感信息** | 隐私泄露风险 | 身份证号、手机号、地址 |
| **语言混杂** | 模型语言能力下降 | 中英混杂、方言干扰 |

### 1.2 数据清洗的投入产出比

```
数据清洗投入 ↑  →  模型性能 ↑  →  后期调试成本 ↓
     ↑                    ↑
  10% 时间              50% 提升
```

**关键洞察**：在数据上投入 10% 的额外精力，往往能带来 50% 以上的模型性能提升。

---

## 2. 文本数据清洗流程

### 2.1 标准清洗 Pipeline

```
原始数据
  ├── 格式解析 → 从 HTML/PDF/Word 提取纯文本
  ├── 编码规范化 → UTF-8、去除 BOM
  ├── 语言识别 → 保留目标语言
  ├── 质量评分 → 过滤低质量内容
  ├── 去重处理 → 精确/模糊去重
  ├── 敏感过滤 → PII 检测与脱敏
  ├── 文本规范化 → 全半角、大小写、标点
  └── 最终数据集
```

### 2.2 HTML 解析与清洗

```python
from bs4 import BeautifulSoup
import re

def clean_html(html: str) -> str:
    """从 HTML 中提取干净文本"""
    soup = BeautifulSoup(html, 'html.parser')
    
    # 移除脚本、样式、导航等无关元素
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        tag.decompose()
    
    # 提取文本
    text = soup.get_text(separator='\n')
    
    # 清理空白
    lines = (line.strip() for line in text.splitlines())
    text = '\n'.join(line for line in lines if line)
    
    return text

# 处理常见噪声
NOISE_PATTERNS = [
    r'\[\d+\]'  # 引用标记 [1], [2]
    r'©\s*\d{4}.*'  # 版权声明
    r'All rights reserved'  # 版权文字
]

def remove_noise(text: str) -> str:
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text
```

### 2.3 PDF 文档处理

```python
import pdfplumber

def extract_pdf_text(pdf_path: str) -> list[dict]:
    """提取 PDF 文本，保留页面结构和表格信息"""
    documents = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            tables = page.extract_tables()
            
            documents.append({
                'page': i + 1,
                'text': text,
                'tables': tables,
                'metadata': {
                    'width': page.width,
                    'height': page.height
                }
            })
    
    return documents
```

### 2.4 编码与格式规范化

```python
import unicodedata

def normalize_text(text: str) -> str:
    """文本规范化"""
    # 1. Unicode 规范化（NFC 或 NFKC）
    text = unicodedata.normalize('NFKC', text)
    
    # 2. 全角转半角（特定场景）
    text = text.translate(str.maketrans(
        '０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ',
        '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ))
    
    # 3. 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 4. 合并连续空白
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
```

---

## 3. 语言识别与过滤

### 3.1 语言检测

```python
from langdetect import detect, detect_langs

def filter_by_language(texts: list[str], target_lang: str = 'zh') -> list[str]:
    """过滤非目标语言的文本"""
    filtered = []
    
    for text in texts:
        try:
            detected = detect(text)
            if detected == target_lang:
                filtered.append(text)
        except:
            # 检测失败时保留（可能是代码、公式等）
            filtered.append(text)
    
    return filtered

# 多语言置信度检测
def detect_with_confidence(text: str) -> dict:
    """返回语言检测结果及置信度"""
    try:
        langs = detect_langs(text)
        return {
            'primary': str(langs[0]).split(':')[0],
            'confidence': float(str(langs[0]).split(':')[1]),
            'all': [(str(l).split(':')[0], float(str(l).split(':')[1])) for l in langs]
        }
    except:
        return {'primary': 'unknown', 'confidence': 0.0}
```

### 3.2 语言质量评估

| 指标 | 方法 | 阈值 |
|------|------|------|
| **字符合法性** | 检查非法 Unicode 字符 | < 1% 非法字符 |
| **可读性** | 句子平均长度、词汇多样性 | 适中 |
| **语言纯度** | 目标语言字符占比 | > 90% |

---

## 4. 去重策略

### 4.1 精确去重

```python
def exact_deduplicate(texts: list[str]) -> list[str]:
    """基于哈希的精确去重"""
    seen = set()
    unique = []
    
    for text in texts:
        # 规范化后计算哈希
        normalized = text.lower().strip()
        text_hash = hash(normalized)
        
        if text_hash not in seen:
            seen.add(text_hash)
            unique.append(text)
    
    return unique
```

### 4.2 模糊去重（MinHash + LSH）

```python
from datasketch import MinHash, MinHashLSH

def fuzzy_deduplicate(texts: list[str], threshold: float = 0.8) -> list[str]:
    """基于 MinHash 的模糊去重"""
    
    def get_shingles(text: str, k: int = 5):
        """生成 k-gram"""
        return set(text[i:i+k] for i in range(len(text) - k + 1))
    
    # 创建 LSH 索引
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = {}
    
    for i, text in enumerate(texts):
        m = MinHash(num_perm=128)
        for s in get_shingles(text):
            m.update(s.encode('utf-8'))
        
        # 检查是否已有相似文档
        if not lsh.query(m):
            lsh.insert(f"doc_{i}", m)
            minhashes[i] = text
    
    return list(minhashes.values())
```

### 4.3 语义去重

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_deduplicate(texts: list[str], threshold: float = 0.95) -> list[str]:
    """基于语义相似度的去重"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 编码所有文本
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # 计算相似度矩阵
    similarity_matrix = np.inner(embeddings, embeddings)
    
    # 保留不相似的文档
    keep = [True] * len(texts)
    for i in range(len(texts)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(texts)):
            if similarity_matrix[i][j] > threshold:
                keep[j] = False
    
    return [text for text, k in zip(texts, keep) if k]
```

---

## 5. 质量评估与过滤

### 5.1 质量评分维度

| 维度 | 评估方法 | 过滤阈值 |
|------|----------|----------|
| **长度** | 字符/词数 | 剔除过短（< 50 字符） |
| **信息密度** | 去重后的熵值 | 剔除过低 |
| **语法正确性** | 语言模型困惑度 | 剔除过高 |
| **可读性** | Flesch Reading Ease | 适中范围 |
| **毒性** | 毒性分类器 | 剔除高毒性 |

### 5.2 质量评分实现

```python
import math
from collections import Counter

def calculate_entropy(text: str) -> float:
    """计算文本信息熵"""
    counter = Counter(text)
    length = len(text)
    
    entropy = 0
    for count in counter.values():
        p = count / length
        entropy -= p * math.log2(p)
    
    return entropy

def quality_score(text: str) -> dict:
    """综合质量评分"""
    scores = {}
    
    # 长度检查
    scores['length'] = len(text)
    scores['length_ok'] = 50 <= len(text) <= 100000
    
    # 信息熵
    scores['entropy'] = calculate_entropy(text)
    scores['entropy_ok'] = scores['entropy'] > 2.0  # 过滤低熵（重复字符）
    
    # 词汇多样性
    words = text.split()
    unique_words = set(words)
    scores['diversity'] = len(unique_words) / len(words) if words else 0
    scores['diversity_ok'] = scores['diversity'] > 0.3
    
    # 综合判断
    scores['is_quality'] = all([
        scores['length_ok'],
        scores['entropy_ok'],
        scores['diversity_ok']
    ])
    
    return scores
```

---

## 6. 敏感信息处理

### 6.1 PII（个人身份信息）检测

```python
import re

PII_PATTERNS = {
    'phone': r'1[3-9]\d{9}',
    'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    'id_card': r'\d{17}[\dXx]',
    'bank_card': r'\d{16,19}',
    'address': r'(?:省|市|区|县|街道|路|号)',
}

def detect_pii(text: str) -> list[dict]:
    """检测文本中的 PII"""
    findings = []
    
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            findings.append({
                'type': pii_type,
                'value': match.group(),
                'position': (match.start(), match.end())
            })
    
    return findings

def mask_pii(text: str) -> str:
    """脱敏处理"""
    # 手机号
    text = re.sub(r'(1[3-9]\d)\d{4}(\d{4})', r'\1****\2', text)
    # 身份证号
    text = re.sub(r'(\d{6})\d{8}(\d{4})', r'\1********\2', text)
    # 邮箱
    text = re.sub(r'(\w{2})\w+(@\w+)', r'\1***\2', text)
    
    return text
```

### 6.2 内容安全过滤

```python
from transformers import pipeline

# 使用预训练的内容审核模型
content_classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    device=-1  # CPU
)

def filter_toxic_content(texts: list[str], threshold: float = 0.5) -> list[str]:
    """过滤有毒内容"""
    safe_texts = []
    
    for text in texts:
        results = content_classifier(text[:512])  # 截断处理
        toxicity_score = results[0]['score'] if results[0]['label'] == 'toxic' else 0
        
        if toxicity_score < threshold:
            safe_texts.append(text)
    
    return safe_texts
```

---

## 7. 领域特定清洗

### 7.1 代码数据清洗

```python
def clean_code_data(code: str) -> str:
    """清洗代码数据"""
    # 移除 shebang
    code = re.sub(r'^#!.*\n', '', code)
    
    # 统一缩进（空格）
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        # 将 tab 转为 4 空格
        line = line.replace('\t', '    ')
        # 移除行尾空格
        line = line.rstrip()
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)
```

### 7.2 对话数据清洗

```python
def clean_conversation_data(dialogues: list[dict]) -> list[dict]:
    """清洗对话数据"""
    cleaned = []
    
    for dialogue in dialogues:
        # 过滤过短对话
        if len(dialogue.get('turns', [])) < 2:
            continue
        
        # 过滤角色混乱的对话
        roles = [turn.get('role') for turn in dialogue['turns']]
        if not all(role in ['user', 'assistant', 'system'] for role in roles):
            continue
        
        # 过滤包含敏感信息的对话
        full_text = ' '.join(turn.get('content', '') for turn in dialogue['turns'])
        if detect_pii(full_text):
            continue
        
        cleaned.append(dialogue)
    
    return cleaned
```

---

## 8. 数据版本管理

### 8.1 使用 DVC 管理数据

```bash
# 初始化 DVC
dvc init

# 跟踪数据集
dvc add data/raw_dataset.jsonl

# 推送到远程存储
dvc remote add -d myremote s3://mybucket/dvc
dvc push

# 版本标签
git tag -a v1.0 -m "初始数据集"
dvc tag -a v1.0 data/raw_dataset.jsonl
```

### 8.2 数据血缘追踪

```python
from datetime import datetime
import json

def create_data_card(
    source: str,
    processing_steps: list[str],
    statistics: dict
) -> dict:
    """创建数据卡片"""
    return {
        'version': datetime.now().isoformat(),
        'source': source,
        'processing_steps': processing_steps,
        'statistics': statistics,
        'quality_metrics': {
            'total_records': statistics.get('total', 0),
            'avg_length': statistics.get('avg_length', 0),
            'language_distribution': statistics.get('languages', {})
        }
    }

# 保存数据卡片
with open('data/data_card.json', 'w') as f:
    json.dump(create_data_card(...), f, indent=2, ensure_ascii=False)
```

---

## 💡 我的思考

1. **数据清洗是"隐性工作"，但决定上限**：好的数据清洗不会出现在论文标题里，但它是所有成功 AI 项目的基础。

2. **清洗策略要因领域而异**：通用文本、代码、对话、医疗数据的清洗重点完全不同，没有万能方案。

3. **去重比想象中更重要**：Common Crawl 中超过 50% 的内容是重复的，不彻底去重会严重浪费训练资源。

4. **质量评估需要多维度**：单一指标（如长度）不够，需要结合信息熵、多样性、语法正确性等。

5. **敏感信息处理是红线**：PII 泄露不仅是技术问题，更是法律和伦理问题，必须在清洗阶段严格处理。

6. **数据版本管理不可忽视**：数据变更对模型的影响往往比代码变更更大，需要同样严格的版本控制。

---

## 参考来源

- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) — 访问日期：2026-06-10
- [Deduplicating Training Data Mitigates Privacy Risks in Language Models](https://arxiv.org/abs/2202.06539) — 访问日期：2026-06-10
- [Data Filtering Networks](https://arxiv.org/abs/2309.08600) — 访问日期：2026-06-10
- [Hugging Face: Datasets Documentation](https://huggingface.co/docs/datasets/) — 访问日期：2026-06-10

---

*最后更新：2026-06-12*
