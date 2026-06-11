#topic/security #topic/privacy #topic/data-protection #year/2026 #status/draft

# 数据隐私保护

> **一句话定位**：在 AI 系统全生命周期中保护敏感数据——从采集、训练到推理的隐私保护技术与合规实践。

---

## 1. AI 中的隐私风险

### 1.1 隐私泄露途径

```
训练数据 → 模型记忆 → 恶意提取
    ↓
用户输入 → 日志记录 → 未授权访问
    ↓
模型输出 → 敏感信息泄露
```

### 1.2 隐私风险类型

| 风险类型 | 说明 | 示例 |
|----------|------|------|
| **成员推断攻击** | 判断某条记录是否在训练集中 | 推断某患者数据用于医疗模型训练 |
| **模型反演** | 从模型输出反推训练数据 | 从人脸模型反推训练人脸图像 |
| **属性推断** | 推断数据中的敏感属性 | 从文本推断作者性别、年龄 |
| **提示泄露** | 模型输出训练数据中的敏感信息 | 输出包含真实姓名、地址 |
| **数据残留** | 模型权重中残留训练数据信息 | 微调后仍保留预训练数据记忆 |

---

## 2. 隐私保护技术

### 2.1 差分隐私 (Differential Privacy)

**核心思想**：在数据或模型中添加可控噪声，使得无法确定某条特定记录是否被使用。

```python
import numpy as np

def add_laplace_noise(value: float, sensitivity: float, epsilon: float) -> float:
    """添加拉普拉斯噪声实现差分隐私"""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise

# 示例：统计查询的差分隐私保护
def private_count(records: list, condition: callable, epsilon: float = 1.0) -> int:
    """差分隐私计数"""
    true_count = sum(1 for r in records if condition(r))
    # 敏感度为 1（添加/删除一条记录最多改变计数 1）
    return int(add_laplace_noise(true_count, sensitivity=1, epsilon=epsilon))

# 使用
users = [{'age': 25}, {'age': 30}, {'age': 35}]
count = private_count(users, lambda x: x['age'] > 25, epsilon=1.0)
print(f"估计数量: {count}")  # 结果有噪声，但保护了单条记录隐私
```

**隐私预算 (Epsilon)**：
- ε < 1：强隐私保护
- ε = 1-10：中等隐私保护
- ε > 10：弱隐私保护

### 2.2 联邦学习 (Federated Learning)

**核心思想**：数据不出本地，只共享模型更新。

```
┌─────────┐    ┌─────────┐    ┌─────────┐
│ 设备 A  │    │ 设备 B  │    │ 设备 C  │
│ 数据    │    │ 数据    │    │ 数据    │
│ ↓       │    │ ↓       │    │ ↓       │
│ 本地训练 │    │ 本地训练 │    │ 本地训练 │
│ ↓       │    │ ↓       │    │ ↓       │
│ 梯度更新 │───→│  聚合   │←───│ 梯度更新 │
└─────────┘    └─────────┘    └─────────┘
```

```python
# 简化的联邦学习示例
class FederatedLearning:
    def __init__(self, global_model, num_clients):
        self.global_model = global_model
        self.num_clients = num_clients
    
    def train_round(self, client_data):
        """一轮联邦学习"""
        client_updates = []
        
        # 每个客户端本地训练
        for client_id, data in enumerate(client_data):
            local_model = self._local_train(self.global_model, data)
            
            # 计算模型更新（梯度）
            update = self._compute_update(self.global_model, local_model)
            client_updates.append(update)
        
        # 聚合更新（FedAvg）
        self.global_model = self._aggregate_updates(client_updates)
        
        return self.global_model
    
    def _aggregate_updates(self, updates):
        """FedAvg 聚合"""
        # 简单平均（实际应按数据量加权）
        aggregated = {}
        for key in updates[0].keys():
            aggregated[key] = sum(u[key] for u in updates) / len(updates)
        return aggregated
```

### 2.3 同态加密 (Homomorphic Encryption)

**核心思想**：在加密数据上直接进行计算，无需解密。

```python
# 使用 TenSEAL 进行同态加密计算
import tenseal as ts

# 创建加密上下文
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()

# 加密向量
vector = [1.5, 2.3, 3.7]
encrypted_vector = ts.ckks_vector(context, vector)

# 在加密状态下计算
encrypted_result = encrypted_vector * 2 + 1

# 解密结果
decrypted_result = encrypted_result.decrypt()
print(decrypted_result)  # [4.0, 5.6, 8.4]
```

### 2.4 数据脱敏与匿名化

```python
import hashlib
import re

class DataAnonymizer:
    """数据匿名化工具"""
    
    def __init__(self, salt: str = "random_salt"):
        self.salt = salt
    
    def pseudonymize(self, identifier: str) -> str:
        """假名化：将标识符转为不可逆哈希"""
        return hashlib.sha256(
            f"{identifier}{self.salt}".encode()
        ).hexdigest()[:16]
    
    def mask_email(self, email: str) -> str:
        """邮箱脱敏"""
        if '@' not in email:
            return email
        local, domain = email.split('@')
        masked_local = local[:2] + '*' * (len(local) - 2)
        return f"{masked_local}@{domain}"
    
    def mask_phone(self, phone: str) -> str:
        """手机号脱敏"""
        return re.sub(r'(\d{3})\d{4}(\d{4})', r'\1****\2', phone)
    
    def generalize_age(self, age: int) -> str:
        """年龄泛化"""
        if age < 18:
            return "<18"
        elif age < 30:
            return "18-29"
        elif age < 50:
            return "30-49"
        else:
            return "50+"
    
    def k_anonymize(self, data: list[dict], 
                    quasi_identifiers: list[str], 
                    k: int = 5) -> list[dict]:
        """K-匿名化"""
        from collections import defaultdict
        
        # 按准标识符分组
        groups = defaultdict(list)
        for record in data:
            key = tuple(record[q] for q in quasi_identifiers)
            groups[key].append(record)
        
        # 对不满足 K-匿名的组进行泛化
        anonymized = []
        for key, group in groups.items():
            if len(group) >= k:
                anonymized.extend(group)
            else:
                # 泛化处理（简化示例）
                for record in group:
                    for qi in quasi_identifiers:
                        if isinstance(record[qi], int):
                            record[qi] = self.generalize_age(record[qi])
                    anonymized.append(record)
        
        return anonymized
```

---

## 3. 训练数据隐私保护

### 3.1 PII 检测与移除

```python
from transformers import pipeline
import re

class PIIDetector:
    """PII 检测器"""
    
    def __init__(self):
        # 使用预训练的 NER 模型
        self.ner = pipeline("ner", model="dslim/bert-base-NER")
        
        # 正则表达式模式
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b1[3-9]\d{9}\b',
            'id_card': r'\b\d{17}[\dXx]\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        }
    
    def detect(self, text: str) -> list[dict]:
        """检测文本中的 PII"""
        findings = []
        
        # 正则匹配
        for pii_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                findings.append({
                    'type': pii_type,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 1.0
                })
        
        # NER 检测（人名、组织、地点）
        ner_results = self.ner(text)
        for entity in ner_results:
            if entity['entity'] in ['B-PER', 'I-PER']:
                findings.append({
                    'type': 'person_name',
                    'value': entity['word'],
                    'start': entity['start'],
                    'end': entity['end'],
                    'confidence': entity['score']
                })
        
        return findings
    
    def remove_pii(self, text: str, replacement: str = '[PII]') -> str:
        """移除 PII"""
        findings = self.detect(text)
        
        # 从后向前替换（避免位置偏移）
        for finding in sorted(findings, key=lambda x: x['start'], reverse=True):
            text = text[:finding['start']] + replacement + text[finding['end']:]
        
        return text
```

### 3.2 数据去标识化 Pipeline

```python
class DataDeidentificationPipeline:
    """数据去标识化流水线"""
    
    def __init__(self):
        self.pii_detector = PIIDetector()
        self.anonymizer = DataAnonymizer()
    
    def process(self, dataset: list[dict], text_fields: list[str]) -> list[dict]:
        """处理数据集"""
        processed = []
        
        for record in dataset:
            clean_record = record.copy()
            
            for field in text_fields:
                if field in clean_record:
                    text = clean_record[field]
                    
                    # 1. 检测 PII
                    pii_findings = self.pii_detector.detect(text)
                    
                    # 2. 移除 PII
                    clean_text = self.pii_detector.remove_pii(text)
                    
                    # 3. 记录处理日志
                    clean_record[f'{field}_pii_count'] = len(pii_findings)
                    clean_record[field] = clean_text
            
            processed.append(clean_record)
        
        return processed
```

---

## 4. 模型层面的隐私保护

### 4.1 梯度裁剪与噪声（DP-SGD）

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class DPSGD:
    """差分隐私随机梯度下降"""
    
    def __init__(self, model, noise_multiplier, max_grad_norm):
        self.model = model
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
    
    def compute_per_sample_gradients(self, loss):
        """计算每个样本的梯度"""
        return torch.autograd.grad(
            loss, self.model.parameters(),
            create_graph=True, retain_graph=True
        )
    
    def clip_and_accumulate(self, per_sample_gradients):
        """裁剪梯度并累积"""
        # 计算每个样本的梯度范数
        per_sample_norms = [
            g.norm(2, dim=list(range(1, g.dim())))
            for g in per_sample_gradients
        ]
        
        # 裁剪因子
        clip_factors = [
            torch.clamp(max_grad_norm / norm, max=1.0)
            for norm, max_grad_norm in zip(
                per_sample_norms, 
                [self.max_grad_norm] * len(per_sample_norms)
            )
        ]
        
        # 裁剪并求和
        clipped_gradients = []
        for grad in per_sample_gradients:
            clipped = grad * clip_factors[0].view(-1, *([1] * (grad.dim() - 1)))
            clipped_gradients.append(clipped.sum(dim=0))
        
        return clipped_gradients
    
    def add_noise(self, gradients, batch_size):
        """添加高斯噪声"""
        noisy_gradients = []
        for grad in gradients:
            noise = torch.normal(
                0, 
                self.noise_multiplier * self.max_grad_norm / batch_size,
                grad.shape
            )
            noisy_gradients.append(grad + noise)
        
        return noisy_gradients
```

### 4.2 模型水印

```python
class ModelWatermark:
    """模型水印技术"""
    
    def __init__(self, watermark_key: str):
        self.watermark_key = watermark_key
        self.watermark_pattern = self._generate_pattern()
    
    def _generate_pattern(self):
        """生成水印模式"""
        import hashlib
        hash_val = hashlib.md5(self.watermark_key.encode()).hexdigest()
        # 转换为二进制模式
        pattern = [int(b) for b in bin(int(hash_val, 16))[2:].zfill(128)]
        return pattern
    
    def embed(self, model: nn.Module):
        """嵌入水印到模型权重"""
        params = list(model.parameters())
        
        for i, bit in enumerate(self.watermark_pattern):
            if i < len(params):
                param = params[i]
                # 在权重最低有效位嵌入水印
                if bit == 1:
                    param.data += 1e-5
                else:
                    param.data -= 1e-5
    
    def verify(self, model: nn.Module) -> float:
        """验证水印"""
        params = list(model.parameters())
        matches = 0
        
        for i, bit in enumerate(self.watermark_pattern):
            if i < len(params):
                param = params[i]
                # 检查最低有效位
                detected_bit = 1 if param.data.mean() > 0 else 0
                if detected_bit == bit:
                    matches += 1
        
        return matches / len(self.watermark_pattern)
```

---

## 5. 合规要求

### 5.1 GDPR（欧盟）

| 要求 | 说明 | AI 系统影响 |
|------|------|-------------|
| **数据最小化** | 只收集必要数据 | 限制训练数据范围 |
| **目的限制** | 按声明目的使用 | 不能随意复用数据 |
| **存储限制** | 不永久保存 | 定期清理训练数据 |
| **可删除权** | 用户可要求删除 | 需支持"被遗忘权" |
| **可携带权** | 用户可导出数据 | 提供数据导出功能 |

### 5.2 中国个人信息保护法

```python
class PrivacyComplianceChecker:
    """隐私合规检查器"""
    
    def __init__(self):
        self.required_consents = [
            'data_collection',
            'data_processing',
            'model_training',
            'third_party_sharing'
        ]
    
    def check_consent(self, user_consents: dict) -> dict:
        """检查用户授权"""
        missing = []
        for consent in self.required_consents:
            if not user_consents.get(consent):
                missing.append(consent)
        
        return {
            'compliant': len(missing) == 0,
            'missing_consents': missing
        }
    
    def check_data_retention(self, 
                            data_age_days: int,
                            max_retention_days: int = 365) -> bool:
        """检查数据保留期限"""
        return data_age_days <= max_retention_days
    
    def generate_privacy_report(self, 
                               data_usage_stats: dict) -> dict:
        """生成隐私报告"""
        return {
            'total_users': data_usage_stats['total_users'],
            'data_categories': data_usage_stats['categories'],
            'processing_purposes': data_usage_stats['purposes'],
            'third_parties': data_usage_stats['third_parties'],
            'retention_periods': data_usage_stats['retention_periods'],
            'user_rights_requests': data_usage_stats['rights_requests']
        }
```

---

## 6. 隐私保护最佳实践

### 6.1 数据生命周期隐私保护

```
数据采集 → 明确告知 + 获取同意
    ↓
数据存储 → 加密 + 访问控制
    ↓
数据使用 → 最小化 + 审计日志
    ↓
数据共享 → 脱敏 + 合同约束
    ↓
数据销毁 → 安全删除 + 证书
```

### 6.2 隐私设计原则

| 原则 | 实践 |
|------|------|
| **默认隐私** | 系统默认最高隐私级别 |
| **数据最小化** | 只收集必要数据 |
| **目的限制** | 数据使用不超过声明范围 |
| **透明性** | 清晰告知用户数据使用方式 |
| **用户控制** | 提供查看、修改、删除功能 |

---

## 💡 我的思考

1. **隐私保护是设计约束，不是事后补丁**：必须在系统设计之初就考虑隐私，事后补救成本高且效果差。

2. **差分隐私的实用性在提升**：以前认为差分隐私会严重降低模型性能，但现在技术成熟，可以在保护隐私的同时保持可用性。

3. **联邦学习不是银弹**：它解决了数据不出域的问题，但梯度本身也可能泄露信息，需要结合差分隐私。

4. **合规是底线，信任是目标**：满足 GDPR 等法规只是底线，真正赢得用户信任需要超越合规的隐私保护。

5. **"被遗忘权"是巨大挑战**：要求从训练好的模型中删除特定用户数据，目前技术上非常困难，是活跃研究领域。

---

## 参考来源

- [Differential Privacy: A Primer for a Non-Technical Audience](https://privacytools.seas.harvard.edu/publications/differential-privacy-primer-non-technical-audience) — 访问日期：2026-06-10
- [Federated Learning: Collaborative Machine Learning without Centralized Training Data](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html) — 访问日期：2026-06-10
- [Membership Inference Attacks Against Machine Learning Models](https://arxiv.org/abs/1610.05820) — 访问日期：2026-06-10
- [GDPR and AI: A Guide for Compliance](https://ico.org.uk/for-organisations/guide-to-data-protection/guide-to-the-uk-gdpr/artificial-intelligence/) — 访问日期：2026-06-10

---

*最后更新：2026-06-12*
