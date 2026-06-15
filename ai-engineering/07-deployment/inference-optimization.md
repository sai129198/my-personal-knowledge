#topic/inference #topic/optimization #topic/kv-cache #topic/flash-attention #year/2026 #status/draft

# 推理优化技术

> 让大模型跑得更快：KV Cache、FlashAttention、投机解码、连续批处理等核心优化技术。

---

## 1. KV Cache

### 1.1 为什么需要 KV Cache？

**问题**：自回归生成时，每次都要重新计算所有 token 的 Key 和 Value。

```
生成第 1 个 token: 计算 K1, V1
生成第 2 个 token: 计算 K1, V1, K2, V2  ← K1, V1 重复计算！
生成第 3 个 token: 计算 K1, V1, K2, V2, K3, V3  ← 更多重复！
```

**解决**：缓存已计算的 K 和 V。

```
生成第 1 个 token: 计算 K1, V1 → 缓存 [K1, V1]
生成第 2 个 token: 取缓存 [K1, V1] + 计算 K2, V2 → 缓存 [K1, V1, K2, V2]
生成第 3 个 token: 取缓存 [K1..V2] + 计算 K3, V3 → 缓存 [K1..V3]
```

### 1.2 KV Cache 显存计算

```python
def kv_cache_memory(
    batch_size,
    seq_length,
    num_layers,
    num_heads,
    head_dim,
    dtype_bytes=2  # FP16
):
    """
    KV Cache 显存 = 2 * batch * seq * layers * heads * head_dim * dtype_bytes
    """
    return (2 * batch_size * seq_length * num_layers * 
            num_heads * head_dim * dtype_bytes)

# 示例：Llama-2-7B, batch=1, seq=4096
# layers=32, heads=32, head_dim=128
memory = kv_cache_memory(1, 4096, 32, 32, 128)
# = 2 * 1 * 4096 * 32 * 32 * 128 * 2
# = 2,147,483,648 bytes ≈ 2 GB
```

### 1.3 KV Cache 优化

| 优化 | 方法 | 效果 |
|------|------|------|
| **Multi-Query Attention** | 所有头共享 K, V | 显存减少 8× |
| **Group-Query Attention** | 头分组共享 K, V | 显存减少 4× |
| **PagedAttention** | 非连续存储 | 减少碎片 |
| **KV Cache 压缩** | 量化、剪枝 | 显存减少 2-4× |

---

## 2. FlashAttention

### 2.1 问题：标准 Attention 的内存瓶颈

```
标准 Attention:
Q, K, V (N × d) → 计算 S = QK^T (N × N) → 计算 P = softmax(S) (N × N) → 计算 O = PV (N × d)

内存访问: O(N^2) 的中间结果必须写入 HBM
```

**HBM 带宽瓶颈**：
- 计算速度 >> 内存带宽
- 大量时间花在读写内存上

### 2.2 FlashAttention 核心思想

**Tiling + 重计算**：

```
1. 将 Q, K, V 分块（tile）放入 SRAM
2. 在 SRAM 中完成分块计算
3. 只输出最终结果 O，不保存中间 S, P
4. 反向传播时重计算中间结果
```

**效果**：
- 减少 HBM 访问次数
- 接近计算峰值性能
- 内存复杂度从 O(N^2) 降到 O(N)

### 2.3 FlashAttention-2 改进

| 改进 | 效果 |
|------|------|
| 减少非矩阵乘法操作 | 1.5-2× 加速 |
| 更好的并行性 | 更长序列支持 |
| 支持 head 维度到 256 | 更大模型支持 |

### 2.4 使用

```python
# PyTorch 2.0+ 内置
import torch.nn.functional as F

# 自动使用 FlashAttention（如果可用）
output = F.scaled_dot_product_attention(Q, K, V)

# 或显式启用
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    output = F.scaled_dot_product_attention(Q, K, V)
```

---

## 3. 投机解码（Speculative Decoding）

### 3.1 核心思想

**用小模型生成草稿，大模型验证。**

```
小模型（Draft Model）：
- 快速生成 K 个候选 token
- 例如：Llama-68M

大模型（Target Model）：
- 并行验证 K 个 token
- 接受匹配的，拒绝不匹配的
- 例如：Llama-70B
```

### 3.2 算法流程

```python
def speculative_decoding(draft_model, target_model, prompt, K=5):
    tokens = tokenize(prompt)
    
    while not_done(tokens):
        # 1. 小模型快速生成 K 个草稿 token
        draft_tokens = draft_model.generate(tokens, max_new_tokens=K)
        
        # 2. 大模型并行验证
        # 计算所有 K 个位置的概率
        target_probs = target_model.get_probs(tokens + draft_tokens)
        draft_probs = draft_model.get_probs(tokens + draft_tokens)
        
        # 3. 逐个接受或拒绝
        accepted = []
        for i in range(K):
            if accept(target_probs[i], draft_probs[i]):
                accepted.append(draft_tokens[i])
            else:
                # 从修正分布采样
                new_token = sample(target_probs[i], draft_probs[i])
                accepted.append(new_token)
                break
        
        tokens.extend(accepted)
    
    return tokens
```

### 3.3 效果

| 场景 | 加速比 | 条件 |
|------|--------|------|
| 理想情况 | 2-3× | 草稿模型准确率高 |
| 一般情况 | 1.5-2× | 草稿质量中等 |
| 差情况 | ~1× | 草稿质量低 |

### 3.4 变体

| 方法 | 草稿来源 | 特点 |
|------|----------|------|
| **Speculative Decoding** | 小模型 | 通用，需要额外模型 |
| **Medusa** | 额外头 | 单模型，训练开销 |
| **Lookahead Decoding** | 自身生成 | 无需额外模型 |
| **EAGLE** | 特征层 | 更高接受率 |

---

## 4. 连续批处理（Continuous Batching）

### 4.1 问题：静态批处理

```
静态批处理：
Batch 1: [请求A(10 tokens), 请求B(100 tokens)]
→ 必须等 B 完成才能处理新请求
→ A 完成后 GPU 空闲

时间线：
A: ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
B: ██████████████████████████████████████████████████
    ↑ A 完成后 GPU 空闲
```

### 4.2 连续批处理

```
新请求随时加入批处理：

时间线：
A: ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
B: ██████████████████████████████████████████████████
C:        ███████████████████████████████████████████  ← A 完成后加入
D:               █████████████████████████████████████  ← 随时加入

GPU 利用率：███████████████████████████████████████████
```

### 4.3 In-flight Batching

**vLLM 实现**：

```python
# vLLM 自动处理连续批处理
from vllm import LLM, SamplingParams

llm = LLM(model="llama-2-7b")

# 发送请求（非阻塞）
request1 = llm.generate_async("Hello", sampling_params)
request2 = llm.generate_async("World", sampling_params)

# 结果自动返回
for output in llm.get_outputs():
    print(output.text)
```

**效果**：
- 吞吐量提升 10-20×
- 延迟更稳定
- GPU 利用率接近 100%

---

## 5. 其他优化技术

### 5.1 算子融合（Operator Fusion）

**将多个小算子合并为一个大算子**：

```
融合前：
LayerNorm → QKV Projection → Attention → Projection → Residual → LayerNorm
   ↓           ↓              ↓            ↓           ↓          ↓
  6 次内核启动

融合后：
[Fused Attention Block]
   ↓
  1 次内核启动
```

**效果**：减少内核启动开销，提高内存局部性。

### 5.2 动态批处理（Dynamic Batching）

| 策略 | 说明 | 适用 |
|------|------|------|
| **按长度分组** | 相似长度的请求一起处理 | 减少填充 |
| **按模型分组** | 同一模型的请求一起处理 | 多模型服务 |
| **优先级调度** | 高优先级请求优先 | 实时应用 |

### 5.3 模型并行

| 类型 | 分割维度 | 适用 |
|------|----------|------|
| **张量并行** | 层内分割 | 单节点多 GPU |
| **流水线并行** | 层间分割 | 多节点 |
| **序列并行** | 序列维度 | 超长序列 |

---

## 6. 优化效果汇总

| 技术 | 延迟优化 | 吞吐优化 | 显存优化 | 实现难度 |
|------|----------|----------|----------|----------|
| **KV Cache** | ⭐⭐⭐ | - | - | 低 |
| **FlashAttention** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 低 |
| **投机解码** | ⭐⭐⭐⭐ | - | - | 中 |
| **连续批处理** | - | ⭐⭐⭐⭐⭐ | - | 中 |
| **量化** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | 低 |
| **算子融合** | ⭐⭐ | ⭐⭐⭐ | - | 高 |

---

## 💡 我的思考

1. **优化是系统工程**：单一技术的效果有限，组合使用才能达到最佳效果。

2. **FlashAttention 是必选项**：几乎无损的优化，没有理由不用。

3. **连续批处理改变游戏规则**：从静态到动态，吞吐量提升一个数量级。

4. **投机解码的权衡**：需要额外模型或训练，适合高并发场景。

5. **硬件协同设计**：未来的优化需要与硬件架构更紧密结合（如专用芯片）。

---

## 参考来源

- **FlashAttention**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022) — [arxiv:2205.14135](https://arxiv.org/abs/2205.14135)
- **FlashAttention-2**: [arxiv:2307.08691](https://arxiv.org/abs/2307.08691) — 访问日期：2026-06-09
- **Speculative Decoding**: "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2022) — [arxiv:2211.17192](https://arxiv.org/abs/2211.17192)
- **Continuous Batching**: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) — 访问日期：2026-06-09
- **PagedAttention**: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al., 2023) — [arxiv:2309.06180](https://arxiv.org/abs/2309.06180)

---

*访问日期: 2026-06-09*
