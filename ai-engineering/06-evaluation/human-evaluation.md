# 人工评估方法与最佳实践

> **一句话定位**：设计可靠、高效的人工评估流程，获取高质量的模型能力判断。
>
> #status/canonical #topic/evaluation #topic/human-evaluation #year/2026

---

## 1. 人工评估基础

### 1.1 为什么需要人工评估

```
自动评估的局限：
- 无法捕捉语义细微差别
- 对创意性内容评估困难
- 不能评估安全性和价值观
- 与真实用户体验有差距

人工评估的优势：
- 理解上下文和意图
- 评估创造性和风格
- 判断安全性和适当性
- 反映真实用户满意度
```

### 1.2 评估类型

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| **绝对评分** | 按标准打分 | 质量评估 |
| **相对评分** | 比较两个输出 | A/B 测试 |
| **成对比较** | 二选一 | 模型对比 |
| **排序** | 多个输出排序 | 多模型比较 |

---

## 2. 评估维度设计

### 2.1 通用维度

```python
class EvaluationDimensions:
    """
    通用评估维度
    """
    
    DIMENSIONS = {
        "accuracy": {
            "name": "准确性",
            "description": "信息是否正确、事实是否准确",
            "scale": [1, 5],
            "criteria": {
                1: "大量错误，完全不可信",
                3: "部分正确，有一些错误",
                5: "完全正确，所有信息准确"
            }
        },
        "relevance": {
            "name": "相关性",
            "description": "回答是否与问题相关",
            "scale": [1, 5],
            "criteria": {
                1: "完全无关",
                3: "部分相关",
                5: "完全相关，直接回答"
            }
        },
        "completeness": {
            "name": "完整性",
            "description": "回答是否全面",
            "scale": [1, 5],
            "criteria": {
                1: "严重遗漏",
                3: "基本完整，有小遗漏",
                5: "非常全面"
            }
        },
        "clarity": {
            "name": "清晰度",
            "description": "表达是否清晰易懂",
            "scale": [1, 5],
            "criteria": {
                1: "难以理解",
                3: "基本清晰",
                5: "非常清晰"
            }
        },
        "helpfulness": {
            "name": "有用性",
            "description": "对用户是否有帮助",
            "scale": [1, 5],
            "criteria": {
                1: "完全没用",
                3: "有一定帮助",
                5: "非常有帮助"
            }
        },
        "safety": {
            "name": "安全性",
            "description": "是否包含有害内容",
            "scale": [1, 5],
            "criteria": {
                1: "包含有害内容",
                3: "基本安全",
                5: "完全安全"
            }
        }
    }
```

### 2.2 任务特定维度

| 任务类型 | 特定维度 |
|----------|----------|
| **代码生成** | 可运行性、效率、可读性、正确性 |
| **文本摘要** | 信息保留、简洁性、流畅性 |
| **对话系统** | 连贯性、上下文理解、个性一致性 |
| **创意写作** | 创造性、风格一致性、吸引力 |

---

## 3. 评估流程设计

### 3.1 标准评估流程

```python
class HumanEvaluationPipeline:
    """
    人工评估流程
    """
    
    def __init__(self, annotators: List[str], dimensions: List[str]):
        self.annotators = annotators
        self.dimensions = dimensions
        self.results = []
    
    def create_evaluation_task(self, samples: List[Dict]) -> List[Dict]:
        """
        创建评估任务
        """
        tasks = []
        for sample in samples:
            task = {
                "id": sample["id"],
                "input": sample["input"],
                "output": sample["output"],
                "reference": sample.get("reference"),
                "dimensions": self.dimensions,
                "instructions": self._generate_instructions()
            }
            tasks.append(task)
        return tasks
    
    def _generate_instructions(self) -> str:
        """生成评估说明"""
        instructions = """
        # 评估说明
        
        请根据以下维度评估模型输出：
        
        """
        for dim in self.dimensions:
            dim_info = EvaluationDimensions.DIMENSIONS[dim]
            instructions += f"""
            ## {dim_info['name']}
            {dim_info['description']}
            
            评分标准：
            """
            for score, desc in dim_info['criteria'].items():
                instructions += f"- {score}分: {desc}\n"
        
        return instructions
    
    def collect_annotations(self, tasks: List[Dict]) -> Dict:
        """
        收集标注结果
        """
        annotations = {}
        
        for annotator in self.annotators:
            annotator_results = []
            for task in tasks:
                # 分配任务给标注员
                result = self._annotate_task(annotator, task)
                annotator_results.append(result)
            
            annotations[annotator] = annotator_results
        
        return annotations
    
    def _annotate_task(self, annotator: str, task: Dict) -> Dict:
        """
        单个任务标注（实际实现中这是人工操作）
        """
        # 这里模拟标注过程
        # 实际应用中，这会是一个 Web 界面或 API
        return {
            "task_id": task["id"],
            "annotator": annotator,
            "scores": {dim: random.randint(1, 5) for dim in task["dimensions"]},
            "comments": "",
            "timestamp": datetime.now()
        }
```

### 3.2 质量控制系统

```python
class QualityControl:
    """
    标注质量控制
    """
    
    def __init__(self):
        self.gold_standard = []  # 标准答案
        self.min_agreement = 0.7  # 最低一致性
    
    def add_gold_standard(self, task_id: str, expected_scores: Dict):
        """添加标准答案"""
        self.gold_standard.append({
            "task_id": task_id,
            "scores": expected_scores
        })
    
    def calculate_agreement(self, annotations: Dict) -> float:
        """
        计算标注员间一致性
        """
        from sklearn.metrics import cohen_kappa_score
        
        annotator_ids = list(annotations.keys())
        if len(annotator_ids) < 2:
            return 1.0
        
        # 计算两两一致性
        agreements = []
        for i in range(len(annotator_ids)):
            for j in range(i + 1, len(annotator_ids)):
                ann1 = annotations[annotator_ids[i]]
                ann2 = annotations[annotator_ids[j]]
                
                # 提取分数
                scores1 = [a["scores"]["overall"] for a in ann1]
                scores2 = [a["scores"]["overall"] for a in ann2]
                
                # 计算 Cohen's Kappa
                kappa = cohen_kappa_score(scores1, scores2)
                agreements.append(kappa)
        
        return sum(agreements) / len(agreements)
    
    def filter_low_quality(self, annotations: Dict) -> Dict:
        """
        过滤低质量标注
        """
        quality_scores = {}
        
        for annotator, results in annotations.items():
            # 检查与标准答案的一致性
            gold_scores = []
            for result in results:
                gold = self._find_gold_standard(result["task_id"])
                if gold:
                    agreement = self._calculate_score_agreement(
                        result["scores"], gold["scores"]
                    )
                    gold_scores.append(agreement)
            
            if gold_scores:
                quality_scores[annotator] = sum(gold_scores) / len(gold_scores)
            else:
                quality_scores[annotator] = 1.0
        
        # 过滤低质量标注员
        filtered = {}
        for annotator, score in quality_scores.items():
            if score >= self.min_agreement:
                filtered[annotator] = annotations[annotator]
            else:
                print(f"Warning: Annotator {annotator} has low quality score: {score:.2f}")
        
        return filtered
    
    def _calculate_score_agreement(self, scores1: Dict, scores2: Dict) -> float:
        """计算分数一致性"""
        common_dims = set(scores1.keys()) & set(scores2.keys())
        if not common_dims:
            return 0.0
        
        agreements = []
        for dim in common_dims:
            agreements.append(1 - abs(scores1[dim] - scores2[dim]) / 4)
        
        return sum(agreements) / len(agreements)
```

---

## 4. 评估工具

### 4.1 标注界面

```python
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/annotate/<task_id>')
def annotate_task(task_id):
    """标注界面"""
    task = load_task(task_id)
    return render_template('annotate.html', task=task)

@app.route('/api/submit', methods=['POST'])
def submit_annotation():
    """提交标注"""
    data = request.json
    
    annotation = {
        "task_id": data["task_id"],
        "annotator": data["annotator"],
        "scores": data["scores"],
        "comments": data.get("comments", ""),
        "timestamp": datetime.now()
    }
    
    save_annotation(annotation)
    return jsonify({"status": "success"})

# HTML 模板示例
'''
<!DOCTYPE html>
<html>
<head>
    <title>模型输出评估</title>
</head>
<body>
    <h1>任务 {{ task.id }}</h1>
    
    <div class="input">
        <h2>输入</h2>
        <p>{{ task.input }}</p>
    </div>
    
    <div class="output">
        <h2>模型输出</h2>
        <p>{{ task.output }}</p>
    </div>
    
    <form id="evaluation-form">
        {% for dim in task.dimensions %}
        <div class="dimension">
            <h3>{{ dim.name }}</h3>
            <p>{{ dim.description }}</p>
            
            <div class="rating">
                {% for score in range(1, 6) %}
                <label>
                    <input type="radio" name="{{ dim.id }}" value="{{ score }}">
                    {{ score }} - {{ dim.criteria[score] }}
                </label>
                {% endfor %}
            </div>
        </div>
        {% endfor %}
        
        <div class="comments">
            <h3>其他意见</h3>
            <textarea name="comments" rows="4"></textarea>
        </div>
        
        <button type="submit">提交</button>
    </form>
</body>
</html>
'''
```

### 4.2 批量评估

```python
class BatchEvaluator:
    """
    批量评估工具
    """
    
    def __init__(self, evaluators: List[str]):
        self.evaluators = evaluators
        self.assignments = {}
    
    def assign_tasks(self, tasks: List[Dict], overlap: int = 2):
        """
        分配任务给评估员
        
        Args:
            tasks: 任务列表
            overlap: 每个任务的评估员数量
        """
        import random
        
        for task in tasks:
            # 随机选择评估员
            selected = random.sample(self.evaluators, min(overlap, len(self.evaluators)))
            self.assignments[task["id"]] = selected
    
    def calculate_final_scores(self, annotations: Dict) -> Dict:
        """
        计算最终分数
        """
        final_scores = {}
        
        for task_id, annotators in self.assignments.items():
            scores = []
            for annotator in annotators:
                if annotator in annotations:
                    annotation = self._find_annotation(annotations[annotator], task_id)
                    if annotation:
                        scores.append(annotation["scores"])
            
            if scores:
                # 取平均
                final_scores[task_id] = {
                    dim: sum(s[dim] for s in scores) / len(scores)
                    for dim in scores[0].keys()
                }
        
        return final_scores
```

---

## 5. 成本优化

### 5.1 采样策略

```python
class SamplingStrategy:
    """
    评估采样策略
    """
    
    @staticmethod
    def random_sample(data: List, n: int) -> List:
        """随机采样"""
        import random
        return random.sample(data, min(n, len(data)))
    
    @staticmethod
    def stratified_sample(data: List, strata_key: str, n_per_stratum: int) -> List:
        """分层采样"""
        from collections import defaultdict
        
        strata = defaultdict(list)
        for item in data:
            strata[item[strata_key]].append(item)
        
        sampled = []
        for stratum, items in strata.items():
            sampled.extend(random.sample(items, min(n_per_stratum, len(items))))
        
        return sampled
    
    @staticmethod
    def uncertainty_sampling(model, data: List, n: int) -> List:
        """
        不确定性采样
        选择模型最不确定的样本
        """
        uncertainties = []
        for item in data:
            # 计算预测不确定性
            predictions = model.predict_multiple(item, n=5)
            uncertainty = len(set(predictions)) / len(predictions)
            uncertainties.append((item, uncertainty))
        
        # 选择不确定性最高的
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in uncertainties[:n]]
```

### 5.2 主动学习

```python
class ActiveLearningEvaluator:
    """
    主动学习评估
    """
    
    def __init__(self, model, initial_labeled: List):
        self.model = model
        self.labeled_data = initial_labeled
        self.unlabeled_pool = []
    
    def select_samples(self, batch_size: int = 10) -> List:
        """
        选择最有价值的样本进行标注
        """
        # 计算每个未标注样本的信息量
        informativeness = []
        for sample in self.unlabeled_pool:
            # 模型预测的不确定性
            uncertainty = self._calculate_uncertainty(sample)
            
            # 样本代表性
            representativeness = self._calculate_representativeness(sample)
            
            # 综合评分
            score = uncertainty * 0.7 + representativeness * 0.3
            informativeness.append((sample, score))
        
        # 选择得分最高的
        informativeness.sort(key=lambda x: x[1], reverse=True)
        return [sample for sample, _ in informativeness[:batch_size]]
    
    def _calculate_uncertainty(self, sample: Dict) -> float:
        """计算预测不确定性"""
        predictions = self.model.predict_proba(sample)
        # 使用熵作为不确定性度量
        entropy = -sum(p * math.log(p) for p in predictions if p > 0)
        return entropy
    
    def _calculate_representativeness(self, sample: Dict) -> float:
        """计算样本代表性"""
        # 计算与已标注样本的平均距离
        distances = []
        for labeled in self.labeled_data:
            dist = self._distance(sample, labeled)
            distances.append(dist)
        
        # 距离越大，代表性越高
        return sum(distances) / len(distances)
```

---

## 参考资源

- [Human Evaluation of Natural Language Generation](https://aclanthology.org/2020.inlg-1.25/) - van der Lee et al., 2020
- [Best Practices for Crowdsourcing Evaluation](https://aclanthology.org/2021.emnlp-main.670/) - Graham et al., 2021
- [The Trouble with BLEU](https://aclanthology.org/W19-5358/) - Reiter, 2019
- [Scaling Evaluation](https://arxiv.org/abs/2211.09110) - Liang et al., 2022
- [Label Studio](https://labelstud.io/) - 开源标注工具
- [Prodigy](https://prodi.gy/) - 商业标注工具
