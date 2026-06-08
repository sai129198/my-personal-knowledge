# 推理优化技术详解

> **一句话定位**：从模型量化到服务框架，系统梳理 LLM 推理加速的核心技术与选型策略。
>
> #status/draft #topic/deployment #topic/inference #topic/optimization #year/2026

---

## 一、推理瓶颈分析

### 1.1 内存瓶颈（最主要）

LLM 推理的痛点不在计算，而在**内存**：

| 组件 | 内存占用 | 说明 |
|------|----------|------|
| 模型权重 | 70B 模型 ≈ 140GB (FP16) | 最大头 |
| KV Cache | 单序列 ≈ 数 GB | 随序列长度线性增长 |
| 激活值 | 相对较小 | 批处理时显著 |

**关键洞察**：模型权重是静态的，KV Cache 是动态的且**不可预测**。

### 1.2 计算瓶颈

- **预填充阶段 (Prefill)**：计算密集型，处理输入 prompt
- **解码阶段 (Decode)**：内存带宽密集型，逐个生成 token

**Decode 阶段是瓶颈**：每次只生成一个 token，但需加载全部模型权重。

---

## 二、模型量化 (Quantization)

### 2.1 量化原理

将高精度权重（FP32/FP16）映射到低精度（INT8/INT4）：

```
FP16: [-0.0234, 0.1567, -0.0089, ...]  16-bit
  ↓ 线性映射
INT4: [3, 12, 5, ...]                    4-bit
```

**收益**：
- 内存占用 ↓ 50-75%
- 推理速度 ↑（低精度运算更快）

**代价**：
- 精度损失（需评估可接受范围）

### 2.2 主流量化方法对比

| 方法 | 精度 | 特点 | 适用场景 |
|------|------|------|----------|
| **RTN (Round-to-Nearest)** | INT8/INT4 | 最简单，直接四舍五入 | 快速实验 |
| **GPTQ** | INT4/INT3/INT2 | 逐层量化，考虑权重重要性 | 生产部署 |
| **AWQ** | INT4 | 保护"重要"权重通道 | 精度敏感场景 |
| **GGUF/GGML** | Q4_0, Q5_K_M 等 | 多种预设方案，CPU 友好 | 本地/边缘部署 |
| **BitsAndBytes** | NF4/FP4 | HuggingFace 集成最好 | 快速微调+推理 |
| **AQLM** | 1-2 bit | 极端压缩，需校准 | 资源极度受限 |
| **FP8 (NVIDIA)** | FP8 | 硬件原生支持，无损精度 | H100/H200 |

### 2.3 量化选型决策树

```
你有 NVIDIA H100/H200 吗？
├── 是 → 使用 FP8（硬件原生，几乎无损）
└── 否 → 需要 CPU 运行吗？
    ├── 是 → GGUF/GGML (llama.cpp)
    └── 否 → 精度要求极高？
        ├── 是 → AWQ 或 GPTQ INT4
        └── 否 → BitsAndBytes NF4（最简单）
```

### 2.4 实践代码示例

**BitsAndBytes 4-bit 量化**：

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",  # Normal Float 4
    bnb_4bit_use_double_quant=True,  # 嵌套量化
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=quantization_config,
    device_map="auto",
)
```

**GPTQ 量化**：

```python
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    device="cuda:0",
)
```

---

## 三、KV Cache 优化

### 3.1 问题：KV Cache 内存浪费

传统系统为每个请求**预先分配**最大长度的 KV Cache：

- 浪费 60-80% 内存（碎片 + 过度预留）
- 限制 batch size，降低吞吐量

### 3.2 PagedAttention (vLLM)

**核心思想**：借鉴操作系统虚拟内存的**分页机制**。

```
传统方式：
[请求 A: 预分配 2048 tokens] [空闲] [请求 B: 预分配 2048 tokens] [空闲]
→ 大量空闲碎片

PagedAttention：
[Block 0: A的前16个token] [Block 1: B的前16个token] [Block 2: A的16-32token]
→ 非连续存储，按需分配
```

**关键机制**：
- **Block Table**：逻辑块到物理块的映射表
- **按需分配**：新 token 生成时才分配新块
- **Copy-on-Write**：共享块直到需要修改

**效果**：
- 内存浪费 < 4%（接近最优）
- 吞吐量提升 **2-4x**

### 3.3 其他 KV Cache 优化

| 技术 | 原理 | 效果 |
|------|------|------|
| **Multi-Query Attention (MQA)** | 所有头共享 K/V | 内存 ↓，速度 ↑，质量微降 |
| **Grouped-Query Attention (GQA)** | 头分组共享 K/V | 平衡内存和质量 |
| **KV Cache 压缩** | 量化/剪枝 KV Cache | 进一步减少内存 |
| **Prefix Caching** | 共享公共 prompt 前缀 | 多轮对话场景显著 |

---

## 四、服务框架对比

### 4.1 主流框架

| 框架 | 核心特点 | 适用场景 | 维护状态 |
|------|----------|----------|----------|
| **vLLM** | PagedAttention、高吞吐 | 生产部署、高并发 | ✅ 活跃 |
| **TGI** | HuggingFace 生态、功能全 | 快速部署、实验 | ⚠️ 维护模式 |
| **TensorRT-LLM** | NVIDIA 优化、极致性能 | H100/A100 生产 | ✅ 活跃 |
| **llama.cpp** | C++、CPU 友好、GGUF | 本地/边缘部署 | ✅ 活跃 |
| **SGLang** | 结构化生成、RadixAttention | 复杂工作流 | ✅ 活跃 |
| **MLC LLM** | 跨平台（手机/浏览器） | 移动端部署 | ✅ 活跃 |

### 4.2 性能对比（吞吐量）

```
场景：LLaMA-7B，ShareGPT 数据集

vLLM:        ████████████████████  24x (vs HF Transformers)
TGI:         ████████              3.5x
HF Transformers: █                 1x (baseline)
```

### 4.3 vLLM 快速开始

```bash
# 安装
pip install vllm

# 启动服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 1 \
    --quantization awq

# 调用（OpenAI 兼容 API）
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-hf",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

---

## 五、其他优化技术

### 5.1 连续批处理 (Continuous Batching)

**传统**：静态 batch，等最慢的请求完成才能返回

**连续批处理**：动态插入新请求，无需等待

```
时间线：
传统: [AAAA] [BBBB] [CCCC]  → 等 A 完成才能开始 B
连续: [AAAB] [BBBC] [BCCC]  → B 在 A 未完成时插入
```

### 5.2 投机解码 (Speculative Decoding)

**核心思想**：用小模型"草稿"生成多个 token，大模型一次性验证。

```
小模型（快）：生成 draft tokens [t1, t2, t3, t4]
大模型（慢）：并行验证 [t1, t2, t3, t4] 是否都正确
              ↓
如果前3个正确，接受；第4个错误，修正后继续
```

**效果**：速度提升 **2-3x**，质量无损。

### 5.3 张量并行 vs 流水线并行

| 并行方式 | 切分维度 | 适用场景 | 通信开销 |
|----------|----------|----------|----------|
| **张量并行** | 层内切分 | 单节点多 GPU | 高（频繁同步） |
| **流水线并行** | 层间切分 | 跨节点 | 中（气泡问题） |
| **数据并行** | 批次切分 | 独立请求 | 低 |

**实践建议**：
- 单节点：张量并行（TP）
- 跨节点：流水线并行（PP）+ 张量并行
- 大 batch：数据并行（DP）

---

## 六、优化策略总结

### 6.1 按场景选择

| 场景 | 推荐方案 |
|------|----------|
| **高吞吐在线服务** | vLLM + AWQ/GPTQ 量化 + 连续批处理 |
| **低延迟交互** | TensorRT-LLM + FP8 + 投机解码 |
| **本地开发** | llama.cpp + GGUF Q4_K_M |
| **边缘/手机** | MLC LLM / llama.cpp |
| **极致压缩** | AQLM 2-bit（需评估质量） |

### 6.2 优化检查清单

- [ ] 是否使用了量化？（INT4/FP8）
- [ ] KV Cache 是否高效管理？（PagedAttention）
- [ ] 是否启用连续批处理？
- [ ] 是否使用了 Flash Attention？
- [ ] 并行策略是否合理？（TP/PP/DP）
- [ ] 是否考虑投机解码？（低延迟场景）

---

## 💡 我的思考

### 关键洞察

1. **内存是主要瓶颈**：优化 KV Cache 比优化计算更有收益
2. **量化是性价比最高的优化**：几乎无损的 INT4/FP8 量化应作为默认选项
3. **框架选型比模型选型更重要**：同样的模型，vLLM 比 naive 实现快 10x+
4. **没有银弹**：需要根据场景（吞吐 vs 延迟 vs 成本）做权衡

### 常见误区

- ❌ 盲目追求最低精度（INT2 可能质量不可接受）
- ❌ 忽视 batch size 的影响（大 batch 才能发挥量化优势）
- ❌ 混淆吞吐和延迟（高吞吐 ≠ 低延迟）

### 下一步实践

- [ ] 对比同一模型在 vLLM / TGI / llama.cpp 上的性能
- [ ] 测试 AWQ vs GPTQ 在业务数据上的质量差异
- [ ] 测量投机解码在实际工作负载中的加速比

---

## 参考来源

1. vLLM Paper - Efficient Memory Management for LLM Serving with PagedAttention (SOSP 2023)
2. vLLM Blog - Easy, Fast, and Cheap LLM Serving (2023)
3. Hugging Face - Quantization Overview (2024)
4. Hugging Face - Text Generation Inference Docs (2024)
5. llama.cpp GitHub - LLM inference in C/C++
6. NVIDIA - TensorRT-LLM Documentation

---

*最后更新：2026-06-08*
