# 训练基础设施与分布式训练

> **一句话定位**：大规模 LLM 训练需要强大的基础设施——理解分布式训练策略和工程优化。

---

## 1. 训练硬件

### 1.1 GPU / NPU 选择

**NVIDIA GPU（主流选择）：**

| GPU | 显存 | 算力 (FP16) | 适用规模 |
|-----|------|-------------|----------|
| **A100 40GB** | 40GB | 312 TFLOPS | 7B-13B |
| **A100 80GB** | 80GB | 312 TFLOPS | 13B-30B |
| **H100 80GB** | 80GB | 989 TFLOPS | 30B-70B |
| **H100 NVL** | 188GB | 989 TFLOPS | 70B+ |
| **MI250X** | 128GB | 383 TFLOPS | 30B-70B |

**国产芯片（华为昇腾 NPU）：**

| NPU | 显存 | 生态系统 | 成熟度 |
|-----|------|----------|--------|
| **昇腾 910B** | 64GB HBM2e | CANN + torch_npu + MindSpore | ✅ 推理成熟，微调可用 |
| **昇腾 910C** | 128GB HBM2e | CANN（持续完善中） | 🟡 推理可用，预训练在探索 |

> 📖 详见 [华为昇腾在 LLM 领域的角色与现状](../13-ai-infrastructure/huawei-ascend-llm.md)

**昇腾在 LLM 各阶段的成熟度：**

```
🏆 推理服务        ← 最成熟，已经大规模商用
🏆 信创市场        ← 政策护城河，别无选择
🥈 后训练/微调     ← 能做，成本可接受
🥉 小规模预训练    ← 能做但费劲
🤔 千亿级预训练    ← 理论上能，实际上极少团队敢赌
❌ 前沿探索训练    ← 目前不现实
```

**关键瓶颈不在单卡算力，在软件生态**：NVIDIA 的 CUDA 生态经过 20 年打磨，算子库（cuBLAS/cuDNN）覆盖极全，PyTorch 开箱即用；昇腾的 CANN 仍在追赶，算子覆盖度、调试工具链、社区生态都有明显差距。DeepSeek 等极致效率团队深度依赖 CUDA 优化，迁移到昇腾成本不可接受。

### 1.2 集群架构

```
┌─────────────────────────────────────────┐
│              计算节点 (Node)              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ GPU 0   │ │ GPU 1   │ │ GPU 2   │   │
│  │ GPU 3   │ │ GPU 4   │ │ GPU 5   │   │
│  │ GPU 6   │ │ GPU 7   │ │ ...     │   │
│  └─────────┘ └─────────┘ └─────────┘   │
│         NVLink / NVSwitch               │
└─────────────────────────────────────────┘
              │
         InfiniBand / RoCE
              │
┌─────────────────────────────────────────┐
│              存储系统                     │
│    并行文件系统 (Lustre / GPFS)          │
└─────────────────────────────────────────┘
```

### 1.3 网络拓扑

| 网络类型 | 带宽 | 延迟 | 用途 |
|----------|------|------|------|
| **NVLink** | 900GB/s | < 1μs | GPU 间通信 |
| **InfiniBand** | 400Gbps | 1-2μs | 节点间通信 |
| **RoCE** | 100Gbps | 2-5μs | 节点间通信 |
| **以太网** | 10-100Gbps | 10μs+ | 管理网络 |

---

## 2. 分布式训练基础

### 2.1 并行策略

```
┌─────────────────────────────────────────┐
│           数据并行 (Data Parallelism)     │
│                                         │
│  Batch ──► [Split] ──► GPU 0           │
│              │         GPU 1           │
│              │         GPU 2           │
│              ▼         ...             │
│           [Gradient AllReduce]          │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│           模型并行 (Model Parallelism)    │
│                                         │
│  Layer 1-4 ──► GPU 0                   │
│  Layer 5-8 ──► GPU 1                   │
│  Layer 9-12 ──► GPU 2                  │
│      ...                               │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│           流水线并行 (Pipeline Parallel)  │
│                                         │
│  Micro-batch 1 ──► Stage 1 ──► Stage 2 │
│  Micro-batch 2 ──► Stage 1 ──► Stage 2 │
│  Micro-batch 3 ──► Stage 1 ──► Stage 2 │
│      ...         (流水线填充)            │
└─────────────────────────────────────────┘
```

### 2.2 数据并行 (DP)

**原理**：
- 每个 GPU 持有完整模型副本
- 数据分片到不同 GPU
- 反向传播后 AllReduce 梯度

**实现**：
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group(backend='nccl')

# 包装模型
model = DDP(model, device_ids=[local_rank])

# 训练循环
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()  # DDP 自动 AllReduce 梯度
```

**适用场景**：
- 模型可以放入单 GPU
- 需要加速训练
- 实现简单

### 2.3 模型并行 (MP / Tensor Parallelism)

**原理**：
- 模型层切分到不同 GPU
- 每 GPU 只存储部分参数
- 前向/反向传播需要跨 GPU 通信

**实现**：
```python
# Megatron-LM 张量并行
from megatron import get_args
from megatron.model import GPTModel

# 初始化张量并行
initialize_megatron()

# 模型自动切分
model = GPTModel(
    num_tokentypes=0,
    parallel_output=True
)
```

**切分方式**：
- **列并行**：线性层输出切分
- **行并行**：线性层输入切分
- **注意力并行**：多头注意力切分到不同 GPU

**适用场景**：
- 模型太大无法放入单 GPU
- 层间计算量均衡

### 2.4 流水线并行 (PP)

**原理**：
- 模型按层分组（Stage）
- 每个 Stage 分配到不同 GPU
- 微批次（Micro-batch）流水线执行

**调度策略**：

| 策略 | 说明 | 气泡率 |
|------|------|--------|
| **F-then-B** | 全部前向再反向 | 高 |
| **1F1B** | 一个前向一个反向 | 中 |
| **Interleaved 1F1B** | 交错调度 | 低 |

**实现**：
```python
from deepspeed.pipe import PipelineModule

# 定义层
layers = [
    lambda: EmbeddingLayer(),
    lambda: TransformerBlock(),
    lambda: TransformerBlock(),
    lambda: OutputLayer()
]

# 流水线模型
model = PipelineModule(
    layers=layers,
    num_stages=4,
    loss_fn=loss_function
)
```

**适用场景**：
- 模型层数多
- 节点间通信带宽高

### 2.5 3D 并行

**组合策略**：
```
数据并行 (DP) × 张量并行 (TP) × 流水线并行 (PP)

总 GPU 数 = DP × TP × PP

示例：
- 8 节点 × 8 GPU = 64 GPU
- DP=4, TP=4, PP=4
- 每个流水线 Stage：4 GPU (TP)
- 4 个流水线副本 (DP)
```

**配置示例**：
```python
# Megatron-LM 3D 并行配置
--tensor-model-parallel-size 4      # TP=4
--pipeline-model-parallel-size 4    # PP=4
--num-layers 32
--hidden-size 4096
--num-attention-heads 32
```

---

## 3. DeepSpeed ZeRO

### 3.1 ZeRO 概述

**ZeRO（Zero Redundancy Optimizer）**：
- 消除数据并行中的冗余内存
- 将优化器状态、梯度、参数分片到不同 GPU

**内存分解**：
```
模型状态内存：
├── 参数 (Parameters): 4 bytes/param (FP32)
├── 梯度 (Gradients): 4 bytes/param (FP32)
└── 优化器状态 (Optimizer States): 8 bytes/param (Adam: momentum + variance)

总内存 = 16 × 参数量 bytes (FP32)
```

### 3.2 ZeRO Stage

| Stage | 分片内容 | 内存节省 | 通信量 |
|-------|----------|----------|--------|
| **ZeRO-1** | 优化器状态 | 4× | 与 DP 相同 |
| **ZeRO-2** | 优化器状态 + 梯度 | 8× | 1.5× DP |
| **ZeRO-3** | 优化器状态 + 梯度 + 参数 | 与 DP 度数线性相关 | 3× DP |

**ZeRO-3 配置**：
```python
from deepspeed import DeepSpeedEngine

deepspeed_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e5
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto"
}

model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=deepspeed_config
)
```

### 3.3 ZeRO-Infinity

**特性**：
- 支持 NVMe 存储卸载
- 可训练万亿参数模型
- 自动划分和调度

```python
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": True
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": True
        }
    }
}
```

---

## 4. 训练优化技术

### 4.1 混合精度训练

**FP16 / BF16**：
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast(dtype=torch.bfloat16):
        loss = model(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**BF16 vs FP16**：

| 特性 | BF16 | FP16 |
|------|------|------|
| 指数位 | 8 bit | 5 bit |
| 尾数位 | 7 bit | 10 bit |
| 动态范围 | 与 FP32 相同 | 较小 |
| 精度 | 较低 | 较高 |
| 需要 Loss Scaling | 否 | 是 |

### 4.2 梯度检查点 (Gradient Checkpointing)

**原理**：时间换空间，不保存中间激活值

```python
from torch.utils.checkpoint import checkpoint

class TransformerLayer(nn.Module):
    def forward(self, x):
        # 使用 checkpoint
        return checkpoint(self._forward, x)
    
    def _forward(self, x):
        # 正常的层前向传播
        return x
```

**内存节省**：
- 标准：O(L) 层数
- Checkpointing：O(1)
- 计算增加：~30%

### 4.3 Flash Attention

**原理**：IO-Aware 的注意力计算，减少 HBM 访问

```python
from flash_attn import flash_attn_func

# 使用 Flash Attention
out = flash_attn_func(q, k, v, causal=True)
```

**优势**：
- 内存：O(N) 而非 O(N²)
- 速度：2-4× 加速
- 支持更长序列

### 4.4 序列并行

**原理**：将序列维度切分到不同 GPU

```python
# Megatron-LM 序列并行
--sequence-parallel

# 适用于长序列训练
# 将 LayerNorm 和 Dropout 的激活分片
```

---

## 5. 训练监控与调试

### 5.1 监控指标

| 指标 | 说明 | 告警阈值 |
|------|------|----------|
| **Loss** | 训练损失 | 突然上升/NaN |
| **Perplexity** | 困惑度 | 持续上升 |
| **Learning Rate** | 学习率 | 异常值 |
| **Gradient Norm** | 梯度范数 | > 1.0 或 NaN |
| **Throughput** | 每秒处理的 tokens | 突然下降 |
| **GPU Memory** | 显存使用 | > 95% |
| **GPU Utilization** | GPU 利用率 | < 50% |

### 5.2 日志与可视化

**Weights & Biases**：
```python
import wandb

wandb.init(project="llm-training")
wandb.config.update(config)

for step in range(total_steps):
    loss = train_step()
    wandb.log({
        "loss": loss,
        "learning_rate": scheduler.get_last_lr()[0],
        "throughput": tokens_per_sec
    })
```

**TensorBoard**：
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
writer.add_scalar("Loss/train", loss, step)
writer.add_scalar("Perplexity", ppl, step)
```

### 5.3 常见问题排查

| 问题 | 诊断 | 解决 |
|------|------|------|
| **Loss NaN** | 梯度爆炸、数值溢出 | 梯度裁剪、降低学习率、检查数据 |
| **Loss 不下降** | 学习率过低、数据问题 | 调整学习率、检查数据质量 |
| **OOM** | 显存不足 | 减小 batch size、启用 ZeRO、梯度检查点 |
| **训练慢** | GPU 利用率低 | 检查数据加载、减少 CPU-GPU 传输 |
| **收敛不稳定** | 学习率过高 | 增加 warmup、降低学习率 |

---

## 6. 训练成本估算

### 6.1 计算量估算

**公式**：
```
总 FLOPs ≈ 6 × 参数量 × Token 数

示例：
- 7B 模型，1T tokens
- FLOPs = 6 × 7B × 1T = 4.2e19

A100 理论峰值：312 TFLOPS (FP16)
实际效率：~30-50%
有效算力：~100-150 TFLOPS

训练时间 = 4.2e19 / (100e12 × GPU数)
         = 4.2e5 / GPU数 秒
         = 117 / GPU数 小时

1024 张 A100：~0.1 小时 = 约 7 分钟/步
```

### 6.2 成本估算

| 模型规模 | GPU 数量 | 训练时间 | 云成本 (A100) |
|----------|----------|----------|---------------|
| 7B | 64 | ~1 周 | ~$50K |
| 13B | 128 | ~1-2 周 | ~$150K |
| 30B | 256 | ~2-3 周 | ~$500K |
| 65B | 512 | ~3-4 周 | ~$1.5M |
| 175B | 1024 | ~4-8 周 | ~$5-10M |

---

## 7. 最佳实践

### 7.1 训练 checklist

- [ ] 硬件环境检查（GPU、网络、存储）
- [ ] 分布式配置验证
- [ ] 小规模测试（1-8 GPU）
- [ ] 损失收敛验证
- [ ] 梯度检查（无 NaN/Inf）
- [ ] 检查点保存测试
- [ ] 监控告警配置
- [ ] 容错恢复测试

### 7.2 性能优化 checklist

- [ ] 启用混合精度（BF16/FP16）
- [ ] 使用 Flash Attention
- [ ] 优化数据加载（num_workers、pin_memory）
- [ ] 启用梯度检查点（如果需要）
- [ ] 调整 ZeRO 配置
- [ ] 优化通信（NCCL、InfiniBand）
- [ ] 使用编译优化（torch.compile）

---

## 8. 总结

大规模 LLM 训练是系统工程：

1. **硬件基础**：选择合适的 GPU 和网络架构
2. **并行策略**：DP + TP + PP 组合使用
3. **内存优化**：ZeRO、梯度检查点、Flash Attention
4. **监控调试**：实时跟踪训练状态
5. **成本控制**：估算和优化训练成本

> **关键洞察**：分布式训练的核心是**在通信和计算之间找到平衡**。过多的并行会增加通信开销，过少的并行则无法充分利用硬件。

---

*参考来源：*
- *NVIDIA: Megatron-LM (2021)*
- *Microsoft: DeepSpeed ZeRO (2019)*
- *FlashAttention: Fast and Memory-Efficient Exact Attention (2022)*
- *Efficient Large-Scale Language Model Training on GPU Clusters (2021)*
