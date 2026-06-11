#topic/security #topic/moderation #topic/content-safety #year/2026 #status/draft

# 内容审核系统

> **一句话定位**：自动化检测和过滤有害内容——从规则引擎到 AI 模型的多层审核架构。

---

## 1. 内容审核概述

### 1.1 审核内容类型

| 类别 | 说明 | 示例 |
|------|------|------|
| **仇恨言论** | 针对群体的攻击性内容 | 种族歧视、性别歧视 |
| **暴力** | 暴力描述或威胁 | 打斗场景、威胁伤害 |
| **色情** | 性相关内容 | 裸露、性暗示 |
| **骚扰** | 针对个人的恶意行为 | 网络霸凌、人肉搜索 |
| **虚假信息** | 故意传播的错误信息 | 假新闻、阴谋论 |
| **垃圾信息** | 无意义的重复内容 | 广告轰炸、机器人发帖 |
| **非法内容** | 违反法律的内容 | 毒品交易、诈骗 |

### 1.2 审核挑战

```
语境依赖："杀"在"杀病毒"vs"杀人"中含义不同
    ↓
文化差异：不同地区对"敏感"的定义不同
    ↓
尺度把握：过度审核 vs 审核不足
    ↓
实时性要求：海量内容需要快速处理
```

---

## 2. 审核技术架构

### 2.1 多层审核系统

```
用户内容
    ↓
┌─────────────────────────────────────┐
│  第一层：规则引擎（毫秒级）           │
│  - 关键词匹配                        │
│  - 正则表达式                        │
│  - 黑名单/白名单                     │
└─────────────────────────────────────┘
    ↓（可疑内容）
┌─────────────────────────────────────┐
│  第二层：机器学习模型（百毫秒级）      │
│  - 毒性分类器                        │
│  - 图像识别模型                      │
│  - 垃圾信息检测                      │
└─────────────────────────────────────┘
    ↓（高风险内容）
┌─────────────────────────────────────┐
│  第三层：人工审核（分钟级）            │
│  - 专业审核团队                      │
│  - 申诉处理                          │
└─────────────────────────────────────┘
```

### 2.2 规则引擎实现

```python
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Rule:
    """审核规则"""
    id: str
    name: str
    pattern: str
    action: str  # 'block', 'flag', 'warn'
    category: str
    severity: int  # 1-5
    enabled: bool = True

class RuleEngine:
    """规则引擎"""
    
    def __init__(self):
        self.rules: List[Rule] = []
        self.compiled_patterns: Dict[str, re.Pattern] = {}
    
    def add_rule(self, rule: Rule):
        """添加规则"""
        self.rules.append(rule)
        self.compiled_patterns[rule.id] = re.compile(
            rule.pattern, 
            re.IGNORECASE | re.UNICODE
        )
    
    def check(self, text: str) -> List[Dict]:
        """检查文本是否违反规则"""
        violations = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            pattern = self.compiled_patterns[rule.id]
            matches = pattern.finditer(text)
            
            for match in matches:
                violations.append({
                    'rule_id': rule.id,
                    'rule_name': rule.name,
                    'category': rule.category,
                    'severity': rule.severity,
                    'action': rule.action,
                    'matched_text': match.group(),
                    'position': (match.start(), match.end())
                })
        
        return violations
    
    def get_action(self, violations: List[Dict]) -> str:
        """根据违规决定操作"""
        if not violations:
            return 'allow'
        
        # 取最严重的操作
        actions = [v['action'] for v in violations]
        if 'block' in actions:
            return 'block'
        elif 'flag' in actions:
            return 'flag'
        else:
            return 'warn'

# 预定义规则集
DEFAULT_RULES = [
    Rule('hate_1', '种族歧视词汇', r'\b(种族歧视词列表)\b', 'block', 'hate', 5),
    Rule('violence_1', '暴力威胁', r'(杀|打|伤害).*?(你|他|她)', 'flag', 'violence', 4),
    Rule('spam_1', '垃圾信息', r'(加微信|扫码|点击链接).*\1{2,}', 'block', 'spam', 3),
    Rule('pii_1', '手机号泄露', r'1[3-9]\d{9}', 'flag', 'privacy', 3),
]
```

### 2.3 ML 分类器

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class ContentClassifier:
    """内容分类器"""
    
    def __init__(self, model_name: str = "unitary/toxic-bert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
        # 标签映射
        self.id2label = {
            0: 'toxic',
            1: 'severe_toxic',
            2: 'obscene',
            3: 'threat',
            4: 'insult',
            5: 'identity_hate'
        }
    
    def predict(self, text: str) -> Dict:
        """预测内容类别"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = F.sigmoid(outputs.logits)
        
        # 解析结果
        results = {}
        for i, label in self.id2label.items():
            prob = probabilities[0][i].item()
            results[label] = {
                'probability': prob,
                'flagged': prob > 0.5
            }
        
        # 综合评分
        max_prob = max(r['probability'] for r in results.values())
        results['overall'] = {
            'is_toxic': max_prob > 0.5,
            'max_probability': max_prob
        }
        
        return results
    
    def batch_predict(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """批量预测"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.sigmoid(outputs.logits)
            
            for j, text in enumerate(batch):
                text_results = {}
                for k, label in self.id2label.items():
                    prob = probabilities[j][k].item()
                    text_results[label] = {
                        'probability': prob,
                        'flagged': prob > 0.5
                    }
                results.append(text_results)
        
        return results
```

---

## 3. 图像内容审核

### 3.1 图像分类审核

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

class ImageModerator:
    """图像内容审核器"""
    
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(
            "Falconsai/nsfw_image_detection"
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            "Falconsai/nsfw_image_detection"
        )
        self.model.eval()
    
    def moderate(self, image_path: str) -> Dict:
        """审核图像"""
        image = Image.open(image_path)
        
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        
        # 解析结果
        labels = ['safe', 'unsafe']
        results = {
            label: prob.item()
            for label, prob in zip(labels, probabilities[0])
        }
        
        results['is_unsafe'] = results['unsafe'] > 0.5
        results['confidence'] = max(results.values())
        
        return results
    
    def moderate_batch(self, image_paths: List[str]) -> List[Dict]:
        """批量审核图像"""
        images = [Image.open(path) for path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        
        results = []
        for i, path in enumerate(image_paths):
            result = {
                'path': path,
                'safe': probabilities[i][0].item(),
                'unsafe': probabilities[i][1].item(),
                'is_unsafe': probabilities[i][1].item() > 0.5
            }
            results.append(result)
        
        return results
```

### 3.2 OCR + 文本审核

```python
import pytesseract
from PIL import Image

class OCRModerator:
    """OCR 文本审核"""
    
    def __init__(self):
        self.text_classifier = ContentClassifier()
    
    def moderate(self, image_path: str) -> Dict:
        """审核图像中的文字"""
        # 1. OCR 提取文字
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang='chi_sim+eng')
        
        # 2. 文本审核
        text_result = self.text_classifier.predict(text)
        
        return {
            'extracted_text': text,
            'text_analysis': text_result,
            'has_violation': text_result['overall']['is_toxic']
        }
```

---

## 4. 审核策略管理

### 4.1 动态策略配置

```python
from enum import Enum
from typing import Optional

class Action(Enum):
    ALLOW = "allow"
    WARN = "warn"
    FLAG = "flag"
    BLOCK = "block"
    HUMAN_REVIEW = "human_review"

class ModerationPolicy:
    """审核策略"""
    
    def __init__(self):
        self.thresholds = {
            'toxicity': 0.7,
            'severe_toxicity': 0.5,
            'obscene': 0.6,
            'threat': 0.4,
            'insult': 0.8,
            'identity_hate': 0.3
        }
        self.default_action = Action.FLAG
        self.severity_actions = {
            1: Action.WARN,
            2: Action.WARN,
            3: Action.FLAG,
            4: Action.BLOCK,
            5: Action.BLOCK
        }
    
    def evaluate(self, classification_result: Dict) -> Action:
        """根据分类结果决定操作"""
        max_severity = 0
        
        for category, result in classification_result.items():
            if category == 'overall':
                continue
            
            if result.get('flagged', False):
                probability = result.get('probability', 0)
                
                # 根据概率确定严重程度
                if probability > 0.9:
                    severity = 5
                elif probability > 0.7:
                    severity = 4
                elif probability > 0.5:
                    severity = 3
                else:
                    severity = 2
                
                max_severity = max(max_severity, severity)
        
        return self.severity_actions.get(max_severity, self.default_action)
    
    def update_threshold(self, category: str, threshold: float):
        """更新阈值"""
        self.thresholds[category] = threshold
```

### 4.2 用户分级审核

```python
class UserTrustLevel(Enum):
    NEW = "new"
    NORMAL = "normal"
    TRUSTED = "trusted"
    VIP = "vip"

class TieredModeration:
    """分级审核系统"""
    
    def __init__(self):
        self.user_levels: Dict[str, UserTrustLevel] = {}
        self.level_policies = {
            UserTrustLevel.NEW: {
                'strictness': 1.5,  # 更严格
                'require_review': True
            },
            UserTrustLevel.NORMAL: {
                'strictness': 1.0,
                'require_review': False
            },
            UserTrustLevel.TRUSTED: {
                'strictness': 0.8,  # 较宽松
                'require_review': False
            },
            UserTrustLevel.VIP: {
                'strictness': 0.7,
                'require_review': False
            }
        }
    
    def get_user_level(self, user_id: str) -> UserTrustLevel:
        """获取用户信任等级"""
        return self.user_levels.get(user_id, UserTrustLevel.NEW)
    
    def moderate_with_user_level(self, 
                                 content: str, 
                                 user_id: str,
                                 base_result: Dict) -> Dict:
        """根据用户等级调整审核结果"""
        level = self.get_user_level(user_id)
        policy = self.level_policies[level]
        
        # 调整阈值
        adjusted_result = base_result.copy()
        for category, result in adjusted_result.items():
            if isinstance(result, dict) and 'probability' in result:
                # 根据严格度调整有效概率
                effective_prob = result['probability'] * policy['strictness']
                result['effective_probability'] = min(effective_prob, 1.0)
                result['flagged'] = effective_prob > 0.5
        
        # 决定是否需要人工审核
        adjusted_result['require_human_review'] = (
            policy['require_review'] or 
            base_result.get('overall', {}).get('is_toxic', False)
        )
        
        return adjusted_result
```

---

## 5. 审核系统部署

### 5.1 实时审核 API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

app = FastAPI(title="Content Moderation API")

# 初始化审核组件
rule_engine = RuleEngine()
classifier = ContentClassifier()
image_moderator = ImageModerator()

class TextModerationRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    context: Optional[str] = None

class ModerationResponse(BaseModel):
    allowed: bool
    action: str
    violations: List[Dict]
    confidence: float
    processing_time_ms: float

@app.post("/moderate/text", response_model=ModerationResponse)
async def moderate_text(request: TextModerationRequest):
    """文本内容审核"""
    start_time = asyncio.get_event_loop().time()
    
    # 1. 规则引擎检查
    rule_violations = rule_engine.check(request.text)
    
    # 2. ML 分类器检查
    ml_result = classifier.predict(request.text)
    
    # 3. 综合判断
    all_violations = rule_violations.copy()
    if ml_result['overall']['is_toxic']:
        all_violations.append({
            'type': 'ml_classification',
            'category': 'toxicity',
            'confidence': ml_result['overall']['max_probability']
        })
    
    # 4. 决定操作
    action = rule_engine.get_action(rule_violations)
    if ml_result['overall']['is_toxic'] and action == 'allow':
        action = 'flag'
    
    processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
    
    return ModerationResponse(
        allowed=action not in ['block'],
        action=action,
        violations=all_violations,
        confidence=ml_result['overall']['max_probability'],
        processing_time_ms=processing_time
    )

@app.post("/moderate/image")
async def moderate_image(image_url: str):
    """图像内容审核"""
    # 下载图像
    import requests
    response = requests.get(image_url)
    
    # 保存临时文件
    temp_path = f"/tmp/{hash(image_url)}.jpg"
    with open(temp_path, 'wb') as f:
        f.write(response.content)
    
    # 审核
    result = image_moderator.moderate(temp_path)
    
    # 清理
    import os
    os.remove(temp_path)
    
    return result
```

### 5.2 异步审核队列

```python
from celery import Celery
import redis

celery_app = Celery('moderation', broker='redis://localhost:6379')

@celery_app.task
def async_moderate(content_id: str, content_type: str, content: str):
    """异步审核任务"""
    
    # 执行审核
    if content_type == 'text':
        result = classifier.predict(content)
    elif content_type == 'image':
        result = image_moderator.moderate(content)
    else:
        result = {'error': 'Unsupported content type'}
    
    # 存储结果
    store_moderation_result(content_id, result)
    
    # 如果违规，触发后续处理
    if result.get('overall', {}).get('is_toxic', False):
        handle_violation(content_id, result)
    
    return result

def submit_for_moderation(content_id: str, content_type: str, content: str):
    """提交审核"""
    task = async_moderate.delay(content_id, content_type, content)
    return task.id
```

---

## 💡 我的思考

1. **审核是平衡艺术**：过度审核会伤害用户体验和言论自由，审核不足会带来风险。需要根据平台定位找到平衡点。

2. **规则引擎 + ML 是最佳组合**：规则引擎处理已知模式（快），ML 处理未知模式（准），两者互补。

3. **上下文是关键**：同样的词在不同语境下含义完全不同，审核系统需要理解上下文。

4. **透明性建立信任**：用户应该知道为什么内容被审核，有申诉渠道。

5. **审核标准需要本地化**：不同国家/地区的法律和文化差异很大，不能一套标准全球通用。

---

## 参考来源

- [Toxic-BERT: A Universal Strategy to Combat Toxic Comments](https://arxiv.org/abs/2010.05387) — 访问日期：2026-06-10
- [Perspective API Documentation](https://developers.perspectiveapi.com/s/docs) — 访问日期：2026-06-10
- [Content Moderation at Scale](https://cyber.harvard.edu/story/2019-06/content-moderation-scale) — 访问日期：2026-06-10
- [AWS Comprehend Moderation](https://docs.aws.amazon.com/comprehend/) — 访问日期：2026-06-10

---

*最后更新：2026-06-12*
