# 主流基准测试套件

> **一句话定位**：了解主流 LLM 基准测试套件，选择合适的测试评估模型能力。
>
> #status/canonical #topic/evaluation #topic/benchmark #year/2026

---

## 1. 综合能力评估

### 1.1 MMLU（Massive Multitask Language Understanding）

**简介**：
- 涵盖 57 个学科的多项选择题测试
- 包括数学、历史、计算机科学、法律等
- 测试模型的知识和推理能力

**特点**：
- 零样本（Zero-shot）和少样本（Few-shot）测试
- 标准化多项选择格式
- 广泛被用于模型比较

```python
# MMLU 评估示例
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("cais/mmlu", "all")

# 评估函数
def evaluate_mmlu(model, tokenizer):
    correct = 0
    total = 0
    
    for example in dataset["test"]:
        question = example["question"]
        choices = example["choices"]
        answer = example["answer"]
        
        # 构建提示
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"
        
        # 生成答案
        prediction = generate_answer(model, tokenizer, prompt)
        
        # 检查正确性
        if prediction.strip().upper() == chr(65 + answer):
            correct += 1
        total += 1
    
    return correct / total
```

### 1.2 HELM（Holistic Evaluation of Language Models）

**简介**：
- 斯坦福大学提出的全面评估框架
- 覆盖多种场景和指标
- 强调透明性和可复现性

**评估维度**：

| 维度 | 说明 |
|------|------|
| **准确性** | 预测正确性 |
| **校准性** | 置信度与准确率匹配 |
| **鲁棒性** | 对输入扰动的稳定性 |
| **公平性** | 不同群体的表现差异 |
| **效率** | 计算资源使用 |
| **偏见** | 社会偏见检测 |

```python
# HELM 评估框架使用
from helm.benchmark.run import run_benchmark

# 配置评估
config = {
    "model": "openai/gpt-4",
    "scenarios": [
        "mmlu",
        "truthful_qa",
        "bbq"
    ],
    "metrics": [
        "accuracy",
        "calibration",
        "robustness"
    ]
}

# 运行评估
results = run_benchmark(config)
```

### 1.3 Big-Bench（Beyond the Imitation Game）

**简介**：
- Google 提出的综合性基准测试
- 包含 200+ 个任务
- 测试模型的多样化能力

**任务类型**：
- 逻辑推理
- 数学计算
- 语言理解
- 代码生成
- 创意写作
- 社会推理

```python
# Big-Bench 任务示例
import bigbench.api.task as task
import bigbench.api.model as model

class MyTask(task.Task):
    """自定义 Big-Bench 任务"""
    
    def get_task_details(self):
        return task.TaskMetadata(
            name="my_task",
            description="A custom evaluation task",
            keywords=["logical reasoning"],
            max_input_length_per_query=1000,
            max_queries=100
        )
    
    def evaluate_model(self, model, max_examples=None, random_seed=0):
        """评估模型"""
        score_data = []
        
        # 测试问题
        questions = self._load_questions()
        
        for question in questions[:max_examples]:
            # 获取模型回答
            response = model.generate_text(question)
            
            # 评分
            score = self._score_response(response, question)
            score_data.append(score)
        
        return task.ScoreData(
            score_dict={"accuracy": sum(score_data) / len(score_data)},
            preferred_score="accuracy",
            subtask_description="My task evaluation",
            low_score=0.0,
            high_score=1.0
        )
```

---

## 2. 推理能力评估

### 2.1 GSM8K（Grade School Math）

**简介**：
- 小学数学应用题
- 需要多步推理
- 测试数学推理能力

```python
# GSM8K 评估
def evaluate_gsm8k(model, tokenizer):
    dataset = load_dataset("gsm8k", "main")
    
    correct = 0
    for example in dataset["test"]:
        question = example["question"]
        answer = example["answer"].split("####")[-1].strip()
        
        # 生成答案
        prompt = f"Question: {question}\nLet's think step by step.\n"
        response = generate(model, tokenizer, prompt)
        
        # 提取数字答案
        predicted = extract_number(response)
        
        if predicted == float(answer):
            correct += 1
    
    return correct / len(dataset["test"])
```

### 2.2 MATH Dataset

**简介**：
- 高中和大学级别的数学问题
- 涵盖代数、几何、微积分等
- 需要复杂推理

**难度分级**：
- Level 1-5，从简单到困难

### 2.3 HumanEval（代码生成）

**简介**：
- OpenAI 提出的代码生成基准
- 164 个编程问题
- 测试函数级代码生成能力

```python
# HumanEval 评估
from human_eval.evaluation import evaluate_functional_correctness

# 生成代码样本
samples = []
for problem in problems:
    prompt = problem["prompt"]
    
    # 使用模型生成代码
    completion = model.generate(prompt)
    
    samples.append({
        "task_id": problem["task_id"],
        "completion": completion
    })

# 评估正确性
results = evaluate_functional_correctness(samples)
print(f"Pass@1: {results['pass@1']:.2f}")
print(f"Pass@10: {results['pass@10']:.2f}")
```

**指标**：
- Pass@1：一次生成的通过率
- Pass@10：10 次中至少一次通过
- Pass@100：100 次中至少一次通过

---

## 3. 知识评估

### 3.1 TriviaQA

**简介**：
- 阅读理解式问答
- 650K 问题-答案对
- 测试事实知识

### 3.2 Natural Questions

**简介**：
- Google 搜索的真实问题
- 需要长文档理解
- 测试信息检索能力

### 3.3 TruthfulQA

**简介**：
- 测试模型是否说真话
- 包含常见误解和谣言
- 评估事实准确性

```python
# TruthfulQA 评估
def evaluate_truthfulqa(model, tokenizer):
    dataset = load_dataset("truthful_qa", "generation")
    
    scores = []
    for example in dataset["validation"]:
        question = example["question"]
        best_answer = example["best_answer"]
        
        # 生成答案
        response = model.generate(question)
        
        # 评估真实性（使用 judge 模型）
        truth_score = judge_truthfulness(question, response)
        scores.append(truth_score)
    
    return sum(scores) / len(scores)
```

---

## 4. 安全评估

### 4.1 BBQ（Bias Benchmark for QA）

**简介**：
- 社会偏见检测
- 测试模型在不同群体上的表现
- 评估刻板印象

### 4.2 RealToxicityPrompts

**简介**：
- 毒性内容生成测试
- 25K 提示词
- 评估生成内容的安全性

```python
# 毒性评估
from perspective import PerspectiveAPI

def evaluate_toxicity(text: str) -> Dict:
    """
    使用 Perspective API 评估毒性
    """
    client = PerspectiveAPI(api_key="YOUR_API_KEY")
    
    result = client.analyze(
        text,
        attributes=[
            "TOXICITY",
            "SEVERE_TOXICITY",
            "IDENTITY_ATTACK",
            "INSULT",
            "THREAT"
        ]
    )
    
    return {
        attr: result[attr]["summaryScore"]["value"]
        for attr in result
    }
```

### 4.3 Red Teaming 数据集

**简介**：
- 对抗性测试
- 越狱尝试
- 有害请求

---

## 5. 中文评估基准

### 5.1 C-Eval

**简介**：
- 中文综合评估基准
- 涵盖 52 个学科
- 高中、大学、专业考试级别

```python
# C-Eval 评估
def evaluate_ceval(model, tokenizer):
    dataset = load_dataset("ceval/ceval-exam", "all")
    
    results = {}
    for subject in dataset.keys():
        correct = 0
        total = 0
        
        for example in dataset[subject]:
            question = example["question"]
            choices = example["choices"]
            answer = example["answer"]
            
            # 构建提示
            prompt = f"问题：{question}\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "答案："
            
            # 生成答案
            prediction = model.generate(prompt)
            
            if prediction.strip().upper() == answer:
                correct += 1
            total += 1
        
        results[subject] = correct / total
    
    return results
```

### 5.2 CMMLU

**简介**：
- 中文多任务语言理解
- 67 个主题
- 测试中文知识和推理

### 5.3 Gaokao

**简介**：
- 中国高考题目
- 真实考试难度
- 综合知识测试

---

## 6. 自定义基准测试

### 6.1 构建自定义评估

```python
class CustomBenchmark:
    """
    自定义基准测试
    """
    
    def __init__(self, name: str, tasks: List[Dict]):
        self.name = name
        self.tasks = tasks
        self.results = {}
    
    def add_task(self, task: Dict):
        """添加任务"""
        self.tasks.append(task)
    
    def evaluate(self, model) -> Dict:
        """
        评估模型
        """
        for task in self.tasks:
            task_name = task["name"]
            task_type = task["type"]
            
            if task_type == "multiple_choice":
                score = self._evaluate_multiple_choice(model, task)
            elif task_type == "generation":
                score = self._evaluate_generation(model, task)
            elif task_type == "classification":
                score = self._evaluate_classification(model, task)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            self.results[task_name] = score
        
        return self.results
    
    def _evaluate_multiple_choice(self, model, task: Dict) -> float:
        """评估选择题任务"""
        correct = 0
        for example in task["examples"]:
            prediction = model.predict(example["question"])
            if prediction == example["answer"]:
                correct += 1
        return correct / len(task["examples"])
    
    def _evaluate_generation(self, model, task: Dict) -> float:
        """评估生成任务"""
        scores = []
        for example in task["examples"]:
            prediction = model.generate(example["prompt"])
            score = self._score_generation(prediction, example["reference"])
            scores.append(score)
        return sum(scores) / len(scores)
    
    def generate_report(self) -> str:
        """生成评估报告"""
        report = f"# {self.name} Evaluation Report\n\n"
        
        for task_name, score in self.results.items():
            report += f"## {task_name}\n"
            report += f"Score: {score:.4f}\n\n"
        
        overall = sum(self.results.values()) / len(self.results)
        report += f"## Overall\n"
        report += f"Average Score: {overall:.4f}\n"
        
        return report
```

### 6.2 评估框架选择

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| **EleutherAI LM Evaluation Harness** | 支持多种模型和任务 | 学术研究 |
| **OpenCompass** | 中文支持好 | 中文模型评估 |
| **Big-Bench** | 任务丰富 | 综合能力测试 |
| **HELM** | 全面透明 | 系统评估 |

---

## 参考资源

- [MMLU Paper](https://arxiv.org/abs/2009.03300) - Hendrycks et al., 2020
- [HELM Paper](https://arxiv.org/abs/2211.09110) - Liang et al., 2022
- [Big-Bench Paper](https://arxiv.org/abs/2206.04615) - Srivastava et al., 2022
- [GSM8K Dataset](https://arxiv.org/abs/2110.14168) - Cobbe et al., 2021
- [HumanEval Paper](https://arxiv.org/abs/2107.03374) - Chen et al., 2021
- [C-Eval](https://arxiv.org/abs/2305.08322) - Huang et al., 2023
- [OpenCompass](https://github.com/open-compass/opencompass)
