#topic/data-engineering #topic/synthetic-data #topic/data-augmentation #year/2026 #status/draft

# 合成数据生成技术

> **一句话定位**：当真实数据不足或敏感时，合成数据是扩展训练集、提升模型泛化能力的有效手段。

---

## 1. 为什么需要合成数据

### 1.1 真实数据的局限

| 问题 | 说明 | 影响 |
|------|------|------|
| **数据稀缺** | 特定领域标注数据少 | 模型性能不足 |
| **隐私敏感** | 医疗、金融数据涉及隐私 | 无法直接使用 |
| **成本高昂** | 人工标注费用高 | 项目预算受限 |
| **分布不均** | 长尾类别样本少 | 模型偏见 |
| **边界情况** | 危险/罕见场景难采集 | 模型鲁棒性差 |

### 1.2 合成数据的价值

```
真实数据 + 合成数据 → 更丰富的训练集 → 更好的泛化能力
```

**关键洞察**：合成数据不是替代真实数据，而是补充和增强。

---

## 2. 合成数据生成方法

### 2.1 基于规则生成

```python
import random
from faker import Faker

fake = Faker('zh_CN')

def generate_synthetic_customer_data(n: int = 1000) -> list[dict]:
    """生成合成客户数据（用于测试和演示）"""
    data = []
    
    for _ in range(n):
        customer = {
            'id': fake.uuid4(),
            'name': fake.name(),
            'age': random.randint(18, 80),
            'city': fake.city(),
            'phone': fake.phone_number(),
            'email': fake.email(),
            'registration_date': fake.date_between(start_date='-5y', end_date='today'),
            'purchase_history': [
                {
                    'product': fake.word(),
                    'amount': round(random.uniform(10, 1000), 2),
                    'date': fake.date_between(start_date='-1y', end_date='today')
                }
                for _ in range(random.randint(0, 10))
            ]
        }
        data.append(customer)
    
    return data
```

### 2.2 基于模板生成

```python
import random

def generate_qa_from_template(
    templates: list[dict],
    n_samples: int = 100
) -> list[dict]:
    """基于模板生成问答数据"""
    
    data = []
    
    for _ in range(n_samples):
        template = random.choice(templates)
        
        # 填充模板变量
        variables = {
            'product': random.choice(['手机', '电脑', '耳机', '手表']),
            'brand': random.choice(['苹果', '华为', '小米', '三星']),
            'price': random.randint(1000, 10000),
            'feature': random.choice(['续航', '拍照', '性能', '屏幕'])
        }
        
        question = template['question'].format(**variables)
        answer = template['answer'].format(**variables)
        
        data.append({
            'instruction': question,
            'input': '',
            'output': answer
        })
    
    return data

# 模板示例
QA_TEMPLATES = [
    {
        'question': '{brand}的{product}怎么样？',
        'answer': '{brand}的{product}在{feature}方面表现不错，价格约{price}元。'
    },
    {
        'question': '推荐一款{feature}好的{product}',
        'answer': '如果您注重{feature}，可以考虑{brand}的{product}，售价约{price}元。'
    }
]
```

### 2.3 基于 LLM 生成

```python
import openai
import json

def generate_synthetic_dialogue(
    topic: str,
    n_turns: int = 5,
    style: str = "casual"
) -> dict:
    """使用 LLM 生成合成对话数据"""
    
    prompt = f"""生成一段关于"{topic}"的合成对话数据。

要求：
- 对话风格：{style}
- 轮数：{n_turns} 轮
- 包含用户和助手角色
- 对话应该自然、有信息量
- 助手回答应该准确、有帮助

输出格式（JSON）：
{{
    "topic": "主题",
    "style": "风格",
    "turns": [
        {{"role": "user", "content": "用户问题"}},
        {{"role": "assistant", "content": "助手回答"}}
    ]
}}
"""
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.8
    )
    
    return json.loads(response.choices[0].message.content)

# 批量生成
def batch_generate_synthetic_data(
    topics: list[str],
    n_per_topic: int = 10
) -> list[dict]:
    """批量生成合成数据"""
    
    all_data = []
    
    for topic in topics:
        for _ in range(n_per_topic):
            try:
                dialogue = generate_synthetic_dialogue(topic)
                all_data.append(dialogue)
            except Exception as e:
                print(f"生成失败: {topic}, 错误: {e}")
    
    return all_data
```

### 2.4 数据增强技术

```python
import nlpaug.augmenter.word as naw

def augment_text_data(
    texts: list[str],
    augmentation_type: str = 'synonym',
    n_augmentations: int = 2
) -> list[str]:
    """文本数据增强"""
    
    if augmentation_type == 'synonym':
        # 同义词替换
        aug = naw.SynonymAug(aug_src='wordnet')
    elif augmentation_type == 'back_translation':
        # 回译（中文 -> 英文 -> 中文）
        aug = naw.BackTranslationAug(
            from_model_name='facebook/wmt19-en-de',
            to_model_name='facebook/wmt19-de-en'
        )
    elif augmentation_type == 'contextual':
        # 基于上下文的词替换（BERT）
        aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-chinese',
            action="substitute"
        )
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
    
    augmented = []
    for text in texts:
        for _ in range(n_augmentations):
            augmented_text = aug.augment(text)
            augmented.append(augmented_text)
    
    return augmented
```

---

## 3. 领域特定合成数据

### 3.1 代码数据合成

```python
def generate_synthetic_code_data(
    programming_language: str = 'python',
    n_samples: int = 100
) -> list[dict]:
    """生成合成代码数据"""
    
    templates = {
        'python': [
            {
                'instruction': '写一个函数，计算列表的平均值',
                'template': '''def average(numbers):
    """计算列表平均值"""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)'''
            },
            {
                'instruction': '写一个函数，检查字符串是否为回文',
                'template': '''def is_palindrome(s):
    """检查字符串是否为回文"""
    s = s.lower().replace(' ', '')
    return s == s[::-1]'''
            }
        ]
    }
    
    data = []
    for _ in range(n_samples):
        template = random.choice(templates.get(programming_language, []))
        
        # 可以添加变量替换使代码更多样
        data.append({
            'instruction': template['instruction'],
            'output': template['template'],
            'language': programming_language
        })
    
    return data
```

### 3.2 对话数据合成

```python
def generate_multi_turn_dialogue(
    scenario: str,
    n_turns: int = 5
) -> list[dict]:
    """生成多轮对话数据"""
    
    scenarios = {
        'customer_service': {
            'system': '你是一位专业的客服代表',
            'topics': ['退货', '换货', '产品咨询', '投诉', '账户问题']
        },
        'medical_consultation': {
            'system': '你是一位专业的医生',
            'topics': ['感冒', '头痛', '胃痛', '过敏', '体检报告']
        }
    }
    
    scenario_config = scenarios.get(scenario, scenarios['customer_service'])
    topic = random.choice(scenario_config['topics'])
    
    # 使用 LLM 生成对话
    prompt = f"""生成一段{scenario}场景的多轮对话。

主题：{topic}
系统设定：{scenario_config['system']}
轮数：{n_turns}

要求：
- 对话自然、有逻辑
- 用户问题逐步深入
- 助手回答专业、有帮助
- 最后应该解决问题或给出明确建议

输出格式：
[
    {{"role": "system", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
]
"""
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)['dialogue']
```

### 3.3 结构化数据合成

```python
def generate_synthetic_table_data(
    schema: dict,
    n_rows: int = 1000
) -> list[dict]:
    """生成合成表格数据"""
    
    import random
    from datetime import datetime, timedelta
    
    data = []
    
    for _ in range(n_rows):
        row = {}
        
        for column, config in schema.items():
            col_type = config['type']
            
            if col_type == 'int':
                row[column] = random.randint(
                    config.get('min', 0),
                    config.get('max', 100)
                )
            elif col_type == 'float':
                row[column] = round(random.uniform(
                    config.get('min', 0.0),
                    config.get('max', 1.0)
                ), config.get('precision', 2))
            elif col_type == 'choice':
                row[column] = random.choice(config['options'])
            elif col_type == 'date':
                start = config.get('start', datetime.now() - timedelta(days=365))
                end = config.get('end', datetime.now())
                delta = end - start
                random_days = random.randint(0, delta.days)
                row[column] = (start + timedelta(days=random_days)).isoformat()
            elif col_type == 'text':
                length = random.randint(
                    config.get('min_length', 10),
                    config.get('max_length', 100)
                )
                row[column] = fake.text(max_nb_chars=length)
        
        data.append(row)
    
    return data

# 示例 schema
SALES_SCHEMA = {
    'order_id': {'type': 'int', 'min': 10000, 'max': 99999},
    'product': {'type': 'choice', 'options': ['手机', '电脑', '耳机', '手表', '平板']},
    'quantity': {'type': 'int', 'min': 1, 'max': 10},
    'price': {'type': 'float', 'min': 100, 'max': 10000, 'precision': 2},
    'region': {'type': 'choice', 'options': ['华北', '华东', '华南', '西南', '西北']},
    'order_date': {'type': 'date'}
}
```

---

## 4. 合成数据质量控制

### 4.1 真实性验证

```python
def validate_synthetic_data(
    synthetic_data: list[dict],
    real_data_sample: list[dict],
    validation_metrics: list[str] = ['distribution', 'correlation', 'uniqueness']
) -> dict:
    """验证合成数据的真实性"""
    
    results = {}
    
    if 'distribution' in validation_metrics:
        # 检查分布相似性
        results['distribution'] = compare_distributions(
            synthetic_data, real_data_sample
        )
    
    if 'correlation' in validation_metrics:
        # 检查特征相关性
        results['correlation'] = compare_correlations(
            synthetic_data, real_data_sample
        )
    
    if 'uniqueness' in validation_metrics:
        # 检查重复率
        results['uniqueness'] = check_uniqueness(synthetic_data)
    
    return results

def compare_distributions(synthetic: list[dict], real: list[dict]) -> dict:
    """比较合成数据和真实数据的分布"""
    
    from scipy import stats
    
    results = {}
    
    # 假设我们比较数值型字段
    numeric_fields = ['age', 'price', 'quantity']
    
    for field in numeric_fields:
        if field in synthetic[0] and field in real[0]:
            synthetic_values = [d[field] for d in synthetic]
            real_values = [d[field] for d in real]
            
            # KS 检验
            ks_stat, p_value = stats.ks_2samp(synthetic_values, real_values)
            
            results[field] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'similar': p_value > 0.05  # p > 0.05 认为分布相似
            }
    
    return results
```

### 4.2 隐私保护检查

```python
def check_privacy_leakage(
    synthetic_data: list[dict],
    real_data: list[dict],
    sensitive_fields: list[str]
) -> dict:
    """检查合成数据是否泄露真实数据隐私"""
    
    results = {
        'exact_matches': 0,
        'near_matches': 0,
        'risk_level': 'low'
    }
    
    # 检查精确匹配
    real_set = set(
        tuple(d[field] for field in sensitive_fields)
        for d in real_data
    )
    
    for synthetic_record in synthetic_data:
        synthetic_tuple = tuple(
            synthetic_record[field] for field in sensitive_fields
        )
        
        if synthetic_tuple in real_set:
            results['exact_matches'] += 1
    
    # 评估风险
    match_rate = results['exact_matches'] / len(synthetic_data)
    
    if match_rate > 0.01:
        results['risk_level'] = 'high'
    elif match_rate > 0.001:
        results['risk_level'] = 'medium'
    
    return results
```

---

## 5. 合成数据最佳实践

### 5.1 使用原则

| 原则 | 说明 |
|------|------|
| **补充而非替代** | 合成数据应补充真实数据，而非完全替代 |
| **质量验证** | 必须验证合成数据的质量和真实性 |
| **隐私保护** | 确保合成数据不会泄露真实数据隐私 |
| **多样性** | 合成数据应覆盖真实数据的分布 |
| **标注一致性** | 合成数据的标注标准应与真实数据一致 |

### 5.2 混合训练策略

```python
def create_mixed_dataset(
    real_data: list[dict],
    synthetic_data: list[dict],
    synthetic_ratio: float = 0.3
) -> list[dict]:
    """创建真实数据和合成数据的混合数据集"""
    
    n_real = len(real_data)
    n_synthetic = int(n_real * synthetic_ratio / (1 - synthetic_ratio))
    
    # 如果合成数据不足，可以重复采样或生成更多
    if len(synthetic_data) < n_synthetic:
        synthetic_sample = random.choices(
            synthetic_data,
            k=n_synthetic
        )
    else:
        synthetic_sample = random.sample(
            synthetic_data,
            k=n_synthetic
        )
    
    # 混合并打乱
    mixed = real_data + synthetic_sample
    random.shuffle(mixed)
    
    return mixed
```

---

## 💡 我的思考

1. **合成数据是"放大器"不是"替代品"**：它能放大真实数据的模式，但如果真实数据有偏见，合成数据会放大这种偏见。

2. **LLM 生成数据的质量取决于 Prompt 设计**：好的 Prompt 能生成高质量数据，差的 Prompt 会生成"看起来像那么回事但毫无价值"的数据。

3. **验证比生成更重要**：生成合成数据容易，验证其真实性和有用性难。必须建立严格的验证流程。

4. **隐私保护是红线**：即使合成数据也不应包含可追溯到真实个体的信息，特别是医疗、金融等敏感领域。

5. **领域知识是关键**：通用合成数据价值有限，结合领域知识的定向合成才能产生真正有用的训练数据。

---

## 参考来源

- [The Pile v1](https://arxiv.org/abs/2101.00027) — 访问日期：2026-06-10
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) — 访问日期：2026-06-10
- [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html) — 访问日期：2026-06-10
- [Synthetic Data Generation: A Survey](https://arxiv.org/abs/2304.03276) — 访问日期：2026-06-10

---

*最后更新：2026-06-12*
