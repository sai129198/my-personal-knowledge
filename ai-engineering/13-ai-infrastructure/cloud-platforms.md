# 云平台 AI 服务

> **一句话定位**：主流云平台的 AI 基础设施对比与选型指南。
>
> #status/canonical #topic/cloud #topic/platform #year/2026

---

## 1. 云平台概览

### 1.1 主流平台对比

| 平台 | 厂商 | 核心优势 | 适用场景 |
|------|------|----------|----------|
| **AWS SageMaker** | Amazon | 生态完整、功能丰富 | 企业级 ML 全流程 |
| **Azure ML** | Microsoft | 与 Azure 集成、企业特性 | Microsoft 生态 |
| **Google Vertex AI** | Google | AutoML、TPU、开源友好 | 研究、创新项目 |
| **阿里云 PAI** | 阿里 | 国内领先、中文优化 | 国内业务 |
| **腾讯云 TI** | 腾讯 | 游戏/社交场景优化 | 泛娱乐行业 |

### 1.2 服务层次

```
┌─────────────────────────────────────────────┐
│  Layer 3: AI 应用层                           │
│  (预训练模型、AI 应用、智能体)                  │
├─────────────────────────────────────────────┤
│  Layer 2: ML 平台层                           │
│  (训练、推理、标注、实验管理)                   │
├─────────────────────────────────────────────┤
│  Layer 1: 基础设施层                          │
│  (计算、存储、网络、容器编排)                   │
└─────────────────────────────────────────────┘
```

---

## 2. AWS AI/ML 服务

### 2.1 核心服务

| 服务 | 功能 | 定价模式 |
|------|------|----------|
| **SageMaker** | 全托管 ML 平台 | 实例按时计费 |
| **Bedrock** | 基础模型 API | 按 token 计费 |
| **Trainium/Inferentia** | 自研 AI 芯片 | 实例按时计费 |
| **EC2 P5** | H100 GPU 实例 | 实例按时计费 |
| **EKS** | K8s 托管服务 | 控制面 + 节点 |

### 2.2 SageMaker 核心功能

```python
# SageMaker 训练作业
import sagemaker
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    role=sagemaker.get_execution_role(),
    instance_type='ml.p4d.24xlarge',  # 8x A100
    instance_count=2,
    distribution={'pytorchddp': {'enabled': True}},
    hyperparameters={
        'epochs': 3,
        'batch_size': 64,
    }
)

estimator.fit('s3://my-bucket/training-data/')
```

### 2.3 SageMaker 推理部署

```python
# 部署模型端点
from sagemaker.huggingface import HuggingFaceModel

huggingface_model = HuggingFaceModel(
    model_data='s3://my-bucket/model.tar.gz',
    transformers_version='4.28',
    pytorch_version='2.0',
    py_version='py310',
    role=role,
)

# 部署到 GPU 实例
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g5.2xlarge',
    endpoint_name='llm-endpoint',
)

# 异步推理（大模型）
async_predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g5.48xlarge',
    async_inference_config={
        'OutputConfig': {
            'S3OutputPath': 's3://my-bucket/async-output/'
        }
    }
)
```

### 2.4 AWS 成本优化

```markdown
1. **SageMaker Savings Plans**
   - 1-3 年承诺，最高节省 64%

2. **Spot 实例**
   - 训练：使用 Managed Spot Training
   - 可节省高达 90%

3. **多模型端点 (MME)**
   - 多个模型共享 GPU
   - 提高利用率

4. **Serverless 推理**
   - 按调用付费
   - 适合低频请求
```

---

## 3. Azure AI/ML 服务

### 3.1 核心服务

| 服务 | 功能 | 特点 |
|------|------|------|
| **Azure ML** | ML 平台 | 与 Azure 深度集成 |
| **Azure OpenAI** | OpenAI 模型 API | GPT-4、DALL-E |
| **Azure AI Studio** | 模型开发平台 | 低代码 |
| **ND H100 v5** | GPU 虚拟机 | 8x H100 |
| **AKS** | K8s 服务 | GPU 节点支持 |

### 3.2 Azure ML 工作流

```python
# Azure ML 训练
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="...",
    resource_group="...",
    workspace_name="..."
)

# 创建训练作业
job = command(
    name="llm-finetune",
    display_name="LLM Fine-tuning",
    command="python train.py --data ${{inputs.data}}",
    environment="azureml:pytorch-gpu:1",
    compute="gpu-cluster",
    inputs={
        "data": "azureml:training-data:1"
    },
    resources={
        "instance_count": 2,
        "shm_size": "256g"
    }
)

ml_client.jobs.create_or_update(job)
```

### 3.3 Azure OpenAI 服务

```python
# Azure OpenAI API
import openai

openai.api_type = "azure"
openai.api_base = "https://my-resource.openai.azure.com/"
openai.api_version = "2024-02-01"
openai.api_key = "..."

response = openai.ChatCompletion.create(
    engine="gpt-4",  # 部署名称
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=100
)
```

---

## 4. Google Cloud AI/ML 服务

### 4.1 核心服务

| 服务 | 功能 | 特点 |
|------|------|------|
| **Vertex AI** | 统一 AI 平台 | AutoML、MLOps |
| **TPU** | 自研 AI 加速器 | 训练/推理优化 |
| **Gemini API** | 多模态模型 | 原生多模态 |
| **GKE** | K8s 服务 | Autopilot 模式 |
| **Cloud Storage** | 对象存储 | 与 AI 服务集成 |

### 4.2 Vertex AI 工作流

```python
# Vertex AI 训练
from google.cloud import aiplatform

aiplatform.init(
    project='my-project',
    location='us-central1',
    staging_bucket='gs://my-bucket'
)

# 自定义训练作业
job = aiplatform.CustomTrainingJob(
    display_name='llm-training',
    script_path='train.py',
    container_uri='us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest',
    model_serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest',
)

model = job.run(
    args=['--epochs=3', '--batch-size=32'],
    replica_count=2,
    machine_type='a2-highgpu-1g',  # A100
    accelerator_type='NVIDIA_TESLA_A100',
    accelerator_count=1,
)
```

### 4.3 TPU 使用

```python
# TPU 训练 (JAX/Flax)
import jax
from jax import random
import jax.numpy as jnp

# 检测 TPU
device_count = jax.device_count()
print(f"TPU devices: {device_count}")

# 数据并行
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

sharding = PositionalSharding(mesh_utils.create_device_mesh((device_count,)))

# 分片数据
data = jnp.ones((device_count, 1024))
data = jax.device_put(data, sharding)
```

---

## 5. 国内云平台

### 5.1 阿里云 PAI

| 服务 | 功能 |
|------|------|
| **PAI-DSW** | 交互式开发环境 |
| **PAI-DLC** | 深度学习训练 |
| **PAI-EAS** | 模型在线服务 |
| **PAI-Blade** | 推理优化 |
| **灵骏智算** | 高性能计算集群 |

```python
# PAI-EAS 部署
from pai.eas import PredictClient

client = PredictClient(
    endpoint='http://my-service.cn-beijing.pai-eas.aliyuncs.com',
    token='...'
)

response = client.predict({
    'prompt': '你好',
    'max_length': 100
})
```

### 5.2 腾讯云 TI

| 服务 | 功能 |
|------|------|
| **TI-ONE** | ML 平台 |
| **TI-EMS** | 弹性模型服务 |
| **TI-ACC** | 训练加速 |
| **大模型知识引擎** | 知识库 RAG |

---

## 6. GPU 实例选型

### 6.1 云厂商 GPU 实例对比

| 实例类型 | 平台 | GPU | 显存 | 适用场景 |
|----------|------|-----|------|----------|
| p5.48xlarge | AWS | 8x H100 | 80GBx8 | 大模型训练 |
| p4d.24xlarge | AWS | 8x A100 | 40GBx8 | 训练/推理 |
| g5.48xlarge | AWS | 8x A10G | 24GBx8 | 推理 |
| ND H100 v5 | Azure | 8x H100 | 80GBx8 | 训练 |
| NC A100 v4 | Azure | 1-8x A100 | 80GB | 训练/推理 |
| a2-highgpu-8g | GCP | 8x A100 | 40GBx8 | 训练 |
| a3-highgpu-8g | GCP | 8x H100 | 80GBx8 | 大模型训练 |
| gn7 | 阿里云 | V100/A100 | 多种 | 通用 |
| GN10Xp | 腾讯云 | A100/H800 | 多种 | 训练/推理 |

### 6.2 选型决策树

```
预算?
├── 高 → 选择最新代 GPU (H100/H800)
│         └── 需要大显存? → 80GB 版本
└── 有限 → 选择性价比方案
          ├── 推理为主 → A10G/T4
          ├── 训练中小模型 → A100 40GB
          └── 边缘部署 → T4/L4
```

---

## 7. 混合云与多云策略

### 7.1 混合云架构

```
┌─────────────────────────────────────────────┐
│                 公有云                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ 训练集群  │  │ 推理服务  │  │ 数据存储  │  │
│  │ (GPU)    │  │ (AutoScale)│  │ (S3/GCS) │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└──────────┬──────────────────────────────────┘
           │ VPN/专线
┌──────────┴──────────────────────────────────┐
│                 私有云/本地                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ 数据预处理│  │ 模型开发  │  │ 敏感数据  │  │
│  │ (CPU)    │  │ (GPU)    │  │ (本地存储)│  │
│  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────┘
```

### 7.2 多云管理

| 工具 | 功能 |
|------|------|
| **Terraform** | 基础设施即代码 |
| **Crossplane** | K8s 多云管理 |
| **Karmada** | 多云 K8s 编排 |
| **Rancher** | 多集群管理 |

```hcl
# Terraform 多云部署
# AWS
module "aws_gpu" {
  source = "./modules/aws-gpu"
  instance_type = "p4d.24xlarge"
  count = 2
}

# Azure
module "azure_gpu" {
  source = "./modules/azure-gpu"
  vm_size = "Standard_ND96asr_v4"
  count = 2
}

# GCP
module "gcp_gpu" {
  source = "./modules/gcp-gpu"
  machine_type = "a2-highgpu-1g"
  count = 2
}
```

---

## 8. 成本对比分析

### 8.1 按需价格对比 (参考)

| 实例 | AWS | Azure | GCP | 阿里云 |
|------|-----|-------|-----|--------|
| 1x A100 40GB | $3.06/h | $3.60/h | $3.67/h | ¥15/h |
| 1x H100 80GB | $8.80/h | $9.20/h | $8.50/h | ¥40/h |
| 1x A10G | $1.20/h | $1.35/h | - | ¥6/h |
| 1x T4 | $0.50/h | $0.55/h | $0.35/h | ¥2/h |

### 8.2 成本优化策略

```markdown
1. **预留实例 (RI) / 承诺使用折扣 (CUD)**
   - 1 年承诺：节省 30-40%
   - 3 年承诺：节省 50-60%

2. **Spot/Preemptible 实例**
   - 训练任务：节省 60-90%
   - 需要容错机制

3. **自动启停**
   - 非工作时间关闭开发环境
   - 使用 Serverless 推理

4. **跨区部署**
   - 选择低价区域
   - 注意数据合规
```

---

## 9. 平台选型指南

### 9.1 选型矩阵

| 需求 | 推荐平台 | 理由 |
|------|----------|------|
| 企业级 ML 全流程 | AWS SageMaker | 功能最完整 |
| Microsoft 生态 | Azure ML | 深度集成 |
| 研究/创新 | GCP Vertex AI | TPU、开源友好 |
| 国内业务 | 阿里云 PAI | 合规、中文支持 |
| 成本敏感 | GCP / 阿里云 | 价格优势 |
| 大模型训练 | AWS p5 / GCP a3 | 最新 H100 |

### 9.2 迁移考虑

```markdown
1. **数据迁移**
   - 评估数据量和带宽
   - 使用专线或数据迁移服务

2. **模型兼容性**
   - 框架版本一致性
   - 自定义算子支持

3. **API 兼容性**
   - 推理接口标准化
   - 使用 OpenAI API 格式

4. **成本评估**
   - 总拥有成本 (TCO)
   - 隐藏成本（网络、存储）
```

---

## 参考资源

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Azure Machine Learning Documentation](https://docs.microsoft.com/azure/machine-learning/)
- [Google Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [阿里云 PAI 文档](https://help.aliyun.com/pai/)
- [腾讯云 TI 文档](https://cloud.tencent.com/document/product/851)
- [Cloud GPU Pricing Comparison](https://cloud-gpus.com/)
