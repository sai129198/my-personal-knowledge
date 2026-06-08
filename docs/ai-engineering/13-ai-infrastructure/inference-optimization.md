# AI 推理优化技术

> **一句话定位**：从模型量化到服务编排，全方位提升 AI 推理性能与效率。
>
> #status/canonical #topic/inference #topic/optimization #year/2026

---

## 1. 推理性能基础

### 1.1 关键性能指标

| 指标 | 定义 | 优化目标 |
|------|------|----------|
| **吞吐量 (Throughput)** | 单位时间处理的请求数 (req/s) | 最大化 |
| **延迟 (Latency)** | 单个请求的处理时间 (ms) | 最小化 |
| **首 Token 延迟 (TTFT)** | 第一个 token 的生成时间 | < 100ms |
| **Token 间延迟 (ITL)** | 连续 token 的生成间隔 | < 50ms |
| **GPU 利用率** | GPU 计算时间占比 | > 80% |
| **显存利用率** | 显存使用比例 | 平衡 |

### 1.2 延迟构成分析

```
总延迟 = 网络传输 + 队列等待 + 预处理 + 模型推理 + 后处理

模型推理 = Attention 计算 + FFN 计算 + KV Cache 读取
```

**瓶颈识别**：
- **计算瓶颈**：GPU 利用率 > 90%，batch size 不足
- **内存瓶颈**：显存带宽饱和，KV Cache 过大
- **通信瓶颈**：多卡间通信耗时，TP/PP 开销

---

## 2. 模型优化技术

### 2.1 量化 (Quantization)

**精度等级对比**：

| 精度 | 位宽 | 显存节省 | 精度损失 | 适用场景 |
|------|------|----------|----------|----------|
| FP32 | 32-bit | 基准 | 0% | 训练 |
| FP16/BF16 | 16-bit | 50% | < 1% | 通用推理 |
| INT8 | 8-bit | 75% | 1-3% | 高吞吐 |
| INT4/FP4 | 4-bit | 87.5% | 3-5% | 边缘部署 |
| GPTQ/AWQ | 4-bit | 87.5% | 2-4% | 大模型 |

**主流量化方法**：

```python
# GPTQ 量化示例 (4-bit)
from transformers import AutoModelForCausalLM, GPTQConfig

quantization_config = GPTQConfig(
    bits=4,
    group_size=128,
    dataset="c4",
    desc_act=False,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=quantization_config,
    device_map="auto"
)
```

```python
# AWQ 量化 (更优的 4-bit)
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-AWQ",
    fuse_layers=True,
    use_cache=True
)
```

**量化策略选择**：
- **训练后量化 (PTQ)**：快速部署，适合静态场景
- **量化感知训练 (QAT)**：精度最高，需要重新训练
- **动态量化**：运行时量化，灵活性高

### 2.2 剪枝 (Pruning)

**剪枝类型**：

| 类型 | 描述 | 压缩率 |
|------|------|--------|
| **非结构化剪枝** | 移除单个权重 | 2-10x |
| **结构化剪枝** | 移除整个通道/头 | 2-5x |
| **半结构化 (2:4)** | NVIDIA 稀疏格式 | 2x |

**SparseGPT 剪枝流程**：

```
1. 计算每层权重的重要性 (Hessian 矩阵)
2. 按重要性排序，移除最不重要的权重
3. 使用剩余权重重建输出
4. 迭代直至达到目标稀疏度
```

### 2.3 知识蒸馏 (Knowledge Distillation)

**蒸馏策略**：

```
教师模型 (大) ──→ 软标签 (logits) ──┐
                                      ├──→ 学生模型 (小)
真实标签 ─────────────────────────────┘

损失函数 = α × CE(学生输出, 软标签) + (1-α) × CE(学生输出, 真实标签)
```

**适用场景**：
- 大模型 → 小模型（如 GPT-4 → Llama-7B）
- 多任务蒸馏
- 自蒸馏（模型压缩自身）

---

## 3. 推理引擎优化

### 3.1 vLLM：PagedAttention

**核心创新**：PagedAttention

```
传统 KV Cache：连续内存分配 → 内存碎片 + 浪费
PagedAttention：分页内存管理 → 灵活分配 + 共享

类比：操作系统虚拟内存 vs 物理内存
```

**vLLM 部署**：

```bash
# 安装
pip install vllm

# 启动服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256
```

**关键参数**：

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--tensor-parallel-size` | TP 并行度 | GPU 数 |
| `--gpu-memory-utilization` | GPU 显存使用上限 | 0.85-0.95 |
| `--max-num-seqs` | 最大并发序列数 | 256-512 |
| `--max-model-len` | 最大序列长度 | 模型支持 |
| `--quantization` | 量化类型 | awq/gptq/squeezellm |

### 3.2 TensorRT-LLM

**NVIDIA 优化特性**：

- **Kernel 融合**：合并小算子减少启动开销
- **FP8 支持**：Hopper 架构原生 FP8
- **In-flight Batching**：动态批处理
- **Paged KV Cache**：类似 vLLM
- **Multi-GPU**：TP + PP 支持

**构建流程**：

```bash
# 1. 安装 TensorRT-LLM
pip install tensorrt_llm

# 2. 构建引擎
python build.py --model_dir ./Llama-2-7b \
    --dtype float16 \
    --remove_input_padding \
    --use_gpt_attention_plugin float16 \
    --enable_context_fmha \
    --output_dir ./engine

# 3. 运行推理
python run.py --engine_dir ./engine \
    --max_output_len 512 \
    --tokenizer_dir ./Llama-2-7b \
    --input_text "What is AI?"
```

### 3.3 其他推理引擎对比

| 引擎 | 厂商 | 特点 | 适用场景 |
|------|------|------|----------|
| **vLLM** | 开源 | PagedAttention、高吞吐 | 通用 LLM 服务 |
| **TensorRT-LLM** | NVIDIA | 极致性能、FP8 | NVIDIA GPU 生产环境 |
| **TGI** | HuggingFace | 易用、集成好 | 快速原型、HF 生态 |
| **DeepSpeed-Inference** | Microsoft | ZeRO、MIIG | 超大模型 |
| **SGLang** | 开源 | 结构化生成、RadixAttention | 复杂推理任务 |
| **llama.cpp** | 开源 | CPU 优化、量化 | 边缘/本地部署 |
| **ONNX Runtime** | Microsoft | 跨平台、通用 | 非 LLM 模型 |

---

## 4. 服务层优化

### 4.1 动态批处理 (Dynamic Batching)

**连续批处理 (Continuous Batching)**：

```
传统批处理：等待 batch 满 → 一起处理 → 一起返回
连续批处理：新请求随时加入 → 完成后立即返回 → 新请求补位

优势：
- 减少等待时间
- 提高 GPU 利用率
- 降低尾部延迟
```

**实现机制**：

```python
# vLLM 中的连续批处理
class ContinuousBatchingScheduler:
    def step(self):
        # 1. 运行当前 batch 的 forward
        outputs = self.model_runner.execute_model(batch)
        
        # 2. 检查完成的序列
        finished = [seq for seq in batch if seq.is_finished()]
        
        # 3. 从 waiting queue 补充新请求
        new_requests = self.waiting_queue.get_up_to(batch.remaining_slots())
        batch.add(new_requests)
        
        return outputs
```

### 4.2 请求调度策略

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| **FCFS** | 先到先服务 | 公平性优先 |
| **最短作业优先 (SJF)** | 优先处理短请求 | 降低平均延迟 |
| **优先级调度** | VIP 用户优先 | 多租户 |
| **抢占式调度** | 长请求可被中断 | 实时性要求 |

### 4.3 投机解码 (Speculative Decoding)

**原理**：

```
小模型 (草稿) 快速生成候选 token
大模型 (目标) 并行验证候选 token

速度提升：2-3x (取决于接受率)
```

```python
# 投机解码流程
def speculative_decode(draft_model, target_model, prompt, K=5):
    tokens = tokenize(prompt)
    
    while not done:
        # 1. 草稿模型快速生成 K 个 token
        draft_tokens = draft_model.generate(tokens, max_new_tokens=K)
        
        # 2. 目标模型并行验证
        logits = target_model.forward(tokens + draft_tokens)
        
        # 3. 接受或拒绝
        accepted = verify_tokens(logits, draft_tokens)
        tokens.extend(accepted)
        
        if len(accepted) < K:
            # 重新采样
            tokens.append(resample(logits[len(accepted)]))
    
    return tokens
```

---

## 5. 内存优化

### 5.1 KV Cache 管理

**优化策略**：

| 技术 | 描述 | 效果 |
|------|------|------|
| **PagedAttention** | 分页管理 KV Cache | 减少碎片 |
| **KV Cache 量化** | INT8/FP8 KV Cache | 50% 显存节省 |
| **滑动窗口注意力** | 只缓存最近 N 个 token | 固定显存 |
| **MQA/GQA** | 多查询/分组查询注意力 | 减少 KV 头数 |

**KV Cache 显存计算**：

```
KV Cache 显存 = 2 × num_layers × num_heads × head_dim × seq_len × batch_size × precision

示例：Llama-2-70B, batch=32, seq_len=4096, FP16
= 2 × 80 × 8 × 128 × 4096 × 32 × 2 bytes
= 343 GB (需要 4-5 张 A100 80GB)
```

### 5.2 显存优化技术

```python
# 1. 梯度检查点 (训练时)
from torch.utils.checkpoint import checkpoint

# 2. 混合精度
from torch.cuda.amp import autocast

with autocast(dtype=torch.float16):
    output = model(input)

# 3. 显存碎片整理
torch.cuda.empty_cache()

# 4. 使用更高效的注意力实现
from xformers.ops import memory_efficient_attention
output = memory_efficient_attention(q, k, v)
```

---

## 6. 网络与通信优化

### 6.1 多 GPU 通信

**并行策略**：

| 策略 | 描述 | 通信量 | 适用 |
|------|------|--------|------|
| **Tensor Parallel (TP)** | 层内切分 | 大 (每 layer) | 单节点多卡 |
| **Pipeline Parallel (PP)** | 层间切分 | 小 (每 stage) | 多节点 |
| **Data Parallel (DP)** | 数据切分 | 梯度同步 | 大规模训练 |
| **Sequence Parallel (SP)** | 序列切分 | 中等 | 长序列 |
| **Expert Parallel (EP)** | MoE 专家切分 | 中等 | MoE 模型 |

**NCCL 优化**：

```bash
# 环境变量优化
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0  # 启用 InfiniBand
export NCCL_SOCKET_IFNAME=eth0
export NCCL_TREE_THRESHOLD=0
```

### 6.2 网络拓扑优化

```
理想拓扑：全连接 (all-to-all)
实际拓扑：树形、环形、2D/3D Torus

优化方向：
1. 减少跨节点通信
2. 利用节点内高带宽 (NVLink)
3. 通信与计算重叠
```

---

## 7. 性能 profiling 与调优

### 7.1 性能分析工具

| 工具 | 用途 | 关键指标 |
|------|------|----------|
| **PyTorch Profiler** | 算子级分析 | kernel 时间、内存 |
| **Nsight Systems** | 系统级分析 | GPU/CPU 时间线 |
| **Nsight Compute** | Kernel 级分析 | 占用率、带宽 |
| **TensorBoard** | 可视化 | 训练/推理曲线 |
| **vLLM Profiler** | 服务级分析 | 吞吐量、延迟分布 |

### 7.2 调优检查清单

```markdown
□ 模型优化
  □ 是否使用合适的量化精度？
  □ 是否启用 FlashAttention？
  □ 是否使用高效的 KV Cache 管理？

□ 服务配置
  □ batch size 是否最优？
  □ 并发数是否合理？
  □ 是否启用连续批处理？

□ 硬件利用
  □ GPU 利用率是否 > 80%？
  □ 显存是否充分利用？
  □ 是否存在内存瓶颈？

□ 网络通信
  □ 多卡通信是否优化？
  □ 是否使用 RDMA/InfiniBand？
  □ 通信与计算是否重叠？
```

---

## 8. 边缘与移动端部署

### 8.1 边缘部署方案

| 方案 | 工具 | 特点 |
|------|------|------|
| **ONNX Runtime** | onnxruntime | 跨平台、轻量 |
| **TensorFlow Lite** | tflite | 移动端优化 |
| **Core ML** | coremltools | Apple 生态 |
| **llama.cpp** | llama.cpp | CPU 推理、量化 |
| **MLC LLM** | mlc-llm | 多平台、编译优化 |
| **ExecuTorch** | executorch | PyTorch 移动端 |

### 8.2 移动端优化

```python
# TensorFlow Lite 量化
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
tflite_model = converter.convert()

# Core ML 转换
import coremltools as ct

mlmodel = ct.convert(
    pytorch_model,
    inputs=[ct.TensorType(shape=(1, 3, 224, 224))],
    compute_units=ct.ComputeUnit.ALL
)
```

---

## 9. 性能基准测试

### 9.1 标准化测试方法

```bash
# vLLM 基准测试
python benchmarks/benchmark_throughput.py \
    --model meta-llama/Llama-2-7b \
    --dataset ShareGPT \
    --num-prompts 1000 \
    --max-model-len 4096

# 关键输出：
# - Total throughput: X tokens/s
# - Average latency: X ms
# - P99 latency: X ms
```

### 9.2 不同模型性能参考

| 模型 | 精度 | GPU | 吞吐量 (tokens/s) | 延迟 (ms) |
|------|------|-----|-------------------|-----------|
| Llama-2-7B | FP16 | A100 | 1500 | 50 |
| Llama-2-7B | AWQ-4bit | A100 | 2000 | 40 |
| Llama-2-70B | TP8 FP16 | 8xA100 | 800 | 120 |
| GPT-4 (API) | - | - | - | 300-500 |

---

## 10. 最佳实践总结

### 10.1 优化决策树

```
开始
├── 延迟敏感 (< 100ms)?
│   ├── 是 → 小模型 + 量化 + 缓存
│   └── 否 → 继续
├── 吞吐量优先?
│   ├── 是 → 大 batch + 连续批处理 + TP
│   └── 否 → 继续
├── 显存受限?
│   ├── 是 → 量化 + 滑动窗口 + GQA
│   └── 否 → 继续
└── 多租户?
    ├── 是 → 动态调度 + 资源隔离
    └── 否 → 标准部署
```

### 10.2 生产环境配置模板

```yaml
# vLLM 生产配置
model: meta-llama/Llama-2-70b
tensor_parallel_size: 8
gpu_memory_utilization: 0.90
max_num_seqs: 512
max_model_len: 8192
quantization: awq

# 调度
scheduling_policy: fcfs
enable_chunked_prefill: true

# 性能
num_scheduler_steps: 1
enable_prefix_caching: true
```

---

## 参考资源

- [vLLM Documentation](https://docs.vllm.ai/)
- [TensorRT-LLM User Guide](https://nvidia.github.io/TensorRT-LLM/)
- [Efficiently Scaling Transformer Inference - Kwon et al.](https://arxiv.org/abs/2211.05102)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering)
- [Quantization Methods for LLMs: A Survey](https://arxiv.org/abs/2308.14103)
