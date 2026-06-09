#topic/quantization #topic/model-compression #topic/inference #year/2026 #status/draft

# 模型量化

> 用更少的比特表示模型权重：从 INT8 到 INT4，从 GPTQ 到 AWQ，量化技术的原理与实践。

---

## 1. 为什么需要量化？

### 1.1 模型大小与显存

| 模型 | FP16 大小 | INT8 大小 | INT4 大小 |
|------|----------|----------|----------|
| Llama-2-7B | 14 GB | 7 GB | 3.5 GB |
| Llama-2-70B | 140 GB | 70 GB | 35 GB |
| Mixtral-8x7B | 96 GB | 48 GB | 24 GB |

**收益**：
- ✅ 降低显存占用
- ✅ 提高推理速度
- ✅ 降低部署成本
- ✅ 支持更大模型

### 1.2 量化损失

```
FP16 权重: [0.0012, -0.0234, 0.1567, ...]
    │
    ▼ 量化
INT8 权重: [1, -24, 160, ...]  (缩放因子: 0.0012)
    │
    ▼ 反量化
恢复: [0.0012, -0.0288, 0.192, ...]  ← 有误差
```

---

## 2. 量化基础

### 2.1 线性量化

**公式**：

```
Q = round((R - Z) / S)
R = S * (Q - Z)

其中：
Q: 量化后的整数值
R: 原始浮点值
S: 缩放因子 (Scale)
Z: 零点 (Zero Point)
```

**对称量化 vs 非对称量化**：

| 类型 | 公式 | 适用 |
|------|------|------|
| **对称** | Z = 0，S = max(\|R\|) / 127 | 权重（分布对称） |
| **非对称** | Z = min(R)，S = (max(R) - min(R)) / 255 | 激活值（分布偏移） |

### 2.2 量化粒度

| 粒度 | 说明 | 精度 | 开销 |
|------|------|------|------|
| **Per-tensor** | 整个张量一个缩放因子 | 低 | 最小 |
| **Per-channel** | 每个输出通道一个因子 | 中 | 小 |
| **Per-group** | 每 K 个元素一个因子 | 高 | 中 |
| **Per-token** | 每个 token 动态计算 | 最高 | 大 |

---

## 3. INT8 量化

### 3.1 平滑量化（SmoothQuant）

**核心思想**：将激活值的量化难度转移到权重上。

```
问题：激活值有异常值（outliers），难以量化
解决：Y = (X * diag(s)^(-1)) * (diag(s) * W)
      激活值除以 s，权重乘以 s
```

**效果**：
- 保持 INT8 精度接近 FP16
- 无需反向传播
- 易于实现

### 3.2 LLM.int8()

**核心思想**：对异常值保持 FP16，其他用 INT8。

```python
# 混合精度分解
# 找出异常值（> 阈值）
outliers = abs(X) > threshold

# 异常值用 FP16
X_fp16 = X[outliers]

# 正常值用 INT8
X_int8 = quantize(X[~outliers])

# 分别计算后合并
Y = X_fp16 @ W_fp16 + X_int8 @ W_int8
```

**效果**：
- 几乎无损精度
- 但速度提升有限（异常值计算仍是 FP16）

---

## 4. INT4 量化

### 4.1 GPTQ

**核心思想**：逐层量化，用 Hessian 矩阵信息补偿误差。

```
步骤：
1. 选择一层权重 W
2. 计算该层的 Hessian 矩阵 H
3. 逐行/逐列量化权重
4. 用 H 计算最优补偿
5. 更新剩余未量化权重
```

**特点**：
- 一次性量化，无需微调
- 支持 group-size（如 128）
- 精度损失较小

**使用**：

```bash
# AutoGPTQ
pip install auto-gptq

# 量化
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_pretrained(
    model_id,
    quantize_config=BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
    )
)
model.quantize(examples)
model.save_quantized(quantized_model_dir)
```

### 4.2 AWQ（Activation-aware Weight Quantization）

**核心思想**：保护对激活值影响大的权重（salient weights）。

```
观察：只有 ~1% 的权重对输出影响特别大
策略：
1. 识别 salient weights（通过激活值幅度）
2. 对这些权重进行缩放保护
3. 其他权重正常量化
```

**优势**：
- 比 GPTQ 更好的精度
- 更快的推理速度
- 支持 fused kernel

**使用**：

```bash
# AutoAWQ
pip install autoawq

from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_quantized(quant_path)
```

### 4.3 GGUF（GGML Universal Format）

**llama.cpp 的量化格式**。

**量化类型**：

| 类型 | 说明 | 适用 |
|------|------|------|
| Q4_0 | 4-bit，块大小 32 | 速度优先 |
| Q4_K_M | 4-bit，混合精度 | 平衡 |
| Q5_K_M | 5-bit，混合精度 | 精度优先 |
| Q6_K | 6-bit | 高精度 |
| Q8_0 | 8-bit | 几乎无损 |

**使用**：

```bash
# 转换模型
python convert.py models/llama-7b/

# 量化
./quantize models/llama-7b/ggml-model-f16.gguf models/llama-7b/ggml-model-q4_0.gguf q4_0

# 推理
./main -m models/llama-7b/ggml-model-q4_0.gguf -p "Hello"
```

---

## 5. 量化对比

### 5.1 精度 vs 速度 vs 大小

| 方法 | 精度 | 速度 | 大小 | 适用场景 |
|------|------|------|------|----------|
| FP16 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 1× | 训练、高精度推理 |
| INT8 (SmoothQuant) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 0.5× | 通用推理 |
| INT8 (LLM.int8) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 0.5× | 精度敏感 |
| INT4 (GPTQ) | ⭐⭐⭐ | ⭐⭐⭐⭐ | 0.25× | 显存受限 |
| INT4 (AWQ) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 0.25× | 推荐方案 |
| GGUF Q4_K_M | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 0.25× | CPU 推理 |

### 5.2 选型指南

```
硬件条件
    │
    ├─ GPU 显存充足？
    │   ├─ 是 → FP16 / INT8
    │   └─ 否 → 继续
    │
    ├─ GPU 显存紧张？
    │   ├─ 是 → AWQ / GPTQ (INT4)
    │   └─ 否 → 继续
    │
    ├─ CPU 推理？
    │   ├─ 是 → GGUF (Q4_K_M / Q5_K_M)
    │   └─ 否 → 继续
    │
    └─ 边缘设备？
        ├─ 是 → GGUF Q4_0 / 专用芯片
        └─ 否 → AWQ
```

---

## 6. 量化最佳实践

### 6.1 评估量化效果

```python
def evaluate_quantization(model_fp16, model_quantized, test_dataset):
    results = {
        "perplexity": {},
        "accuracy": {},
        "speed": {},
    }
    
    # 1. 困惑度对比
    results["perplexity"]["fp16"] = compute_ppl(model_fp16, test_dataset)
    results["perplexity"]["quantized"] = compute_ppl(model_quantized, test_dataset)
    
    # 2. 下游任务准确率
    results["accuracy"]["fp16"] = evaluate_tasks(model_fp16)
    results["accuracy"]["quantized"] = evaluate_tasks(model_quantized)
    
    # 3. 推理速度
    results["speed"]["fp16"] = benchmark_speed(model_fp16)
    results["speed"]["quantized"] = benchmark_speed(model_quantized)
    
    return results
```

### 6.2 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 量化后精度下降明显 | 异常值处理不当 | 使用 SmoothQuant / AWQ |
| 推理速度未提升 | 未使用优化 kernel | 使用 fused kernel |
| 显存占用未减少 | 未正确加载量化模型 | 检查加载方式 |
| 特定任务效果差 | 任务对精度敏感 | 提高量化位数 |

---

## 💡 我的思考

1. **量化是性价比最高的优化**：相比买更好的 GPU，量化可以以几乎零成本获得 2-4× 的显存节省。

2. **不是越小越好**：4-bit 不是总是比 8-bit 好，需要在精度和效率之间权衡。

3. **AWQ 是当前最佳实践**：在速度和精度之间取得了很好的平衡。

4. **量化后仍需评估**：量化效果因模型和任务而异，必须在自己的数据上验证。

5. **未来方向**：
   - 更低比特（2-bit, 1-bit）
   - 动态量化（根据输入调整）
   - 硬件感知量化（针对特定芯片优化）

---

## 参考来源

- **GPTQ**: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al., 2022) — [arxiv:2210.17323](https://arxiv.org/abs/2210.17323)
- **AWQ**: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (Lin et al., 2023) — [arxiv:2306.00978](https://arxiv.org/abs/2306.00978)
- **SmoothQuant**: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (Xiao et al., 2022) — [arxiv:2211.10438](https://arxiv.org/abs/2211.10438)
- **GGUF**: [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) — 访问日期：2026-06-09
- **TheBloke**: [huggingface.co/TheBloke](https://huggingface.co/TheBloke) — 访问日期：2026-06-09

---

*访问日期: 2026-06-09*
