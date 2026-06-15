#topic/model-serving #topic/inference-engine #topic/vllm #topic/tensorrt #year/2026 #status/draft

# 模型服务化框架

> 生产级 LLM 推理服务：vLLM、TensorRT-LLM、TGI、SGLang 等框架的对比与选型。

---

## 1. 框架概览

| 框架 | 开发方 | 定位 | 成熟度 | 适用场景 |
|------|--------|------|--------|----------|
| **vLLM** | UC Berkeley | 高吞吐推理服务 | ⭐⭐⭐⭐⭐ | 生产部署首选 |
| **TensorRT-LLM** | NVIDIA | NVIDIA GPU 优化 | ⭐⭐⭐⭐⭐ | NVIDIA 环境 |
| **TGI (Text Generation Inference)** | Hugging Face | HuggingFace 模型服务 | ⭐⭐⭐⭐ | HF 生态 |
| **SGLang** | UC Berkeley | 结构化生成 | ⭐⭐⭐⭐ | 复杂生成任务 |
| **llama.cpp** | ggerganov | CPU/边缘推理 | ⭐⭐⭐⭐⭐ | 本地/边缘 |
| **Ollama** | Ollama | 本地模型运行 | ⭐⭐⭐⭐ | 开发测试 |
| **DeepSpeed-MII** | Microsoft | 微软生态推理 | ⭐⭐⭐⭐ | Azure 环境 |
| **OpenLLM** | BentoML | 模型部署平台 | ⭐⭐⭐ | 快速部署 |

---

## 2. vLLM

### 2.1 核心特性

- **PagedAttention**：解决 KV Cache 内存碎片
- **Continuous Batching**：动态批处理
- **张量并行**：多 GPU 支持
- **量化支持**：AWQ、GPTQ、SqueezeLLM
- **OpenAI 兼容 API**

### 2.2 PagedAttention 原理

```
传统 KV Cache：
┌─────────────────────────────────────┐
│ 请求A: [███████░░░░░░░░░░░░░░░░░░░░] │
│ 请求B: [████████████░░░░░░░░░░░░░░░░] │
│ 请求C: [████░░░░░░░░░░░░░░░░░░░░░░░░] │
└─────────────────────────────────────┘
        ↑ 大量内存碎片

PagedAttention（类似 OS 虚拟内存）：
┌─────────────────────────────────────┐
│ 块 1: [███][██][█][████]            │
│ 块 2: [██][████][███][█]            │
│ 块 3: [█][██][████][██]            │
└─────────────────────────────────────┘
        ↑ 非连续存储，无碎片
```

### 2.3 部署示例

```bash
# 安装
pip install vllm

# 启动服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 1 \
    --quantization awq

# 调用（OpenAI 兼容）
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-hf",
        "prompt": "Hello",
        "max_tokens": 100
    }'
```

### 2.4 性能特点

| 指标 | 表现 |
|------|------|
| 吞吐量 | 比 HF Transformers 高 10-20× |
| 延迟 | P50 ~50ms, P99 ~200ms（7B 模型） |
| 显存效率 | 接近 100%（PagedAttention） |
| 并发 | 支持 1000+ 并发请求 |

---

## 3. TensorRT-LLM

### 3.1 核心特性

- **NVIDIA 官方优化**：针对 NVIDIA GPU 深度优化
- **Plugin 生态**：丰富的自定义算子
- **多精度支持**：FP8、INT8、INT4
- **多 GPU 支持**：张量并行、流水线并行
- **与 Triton 集成**：生产级服务框架

### 3.2 优化技术

| 技术 | 说明 |
|------|------|
| **Layer Fusion** | 层融合减少内核启动 |
| **Kernel Auto-Tuning** | 自动选择最优 kernel |
| **Multi-Block Mode** | 大 batch 多 block 并行 |
| **Paged KV Cache** | 类似 vLLM 的内存管理 |

### 3.3 部署示例

```python
# 1. 转换模型
python convert_checkpoint.py --model_dir ./llama-7b \
    --output_dir ./tllm_checkpoint \
    --dtype float16

# 2. 构建引擎
trllm-build --checkpoint_dir ./tllm_checkpoint \
    --output_dir ./llama_7b_trt_engine \
    --gemm_plugin float16

# 3. 运行
python run.py --engine_dir ./llama_7b_trt_engine \
    --max_output_len 100
```

### 3.4 性能特点

| 指标 | 表现 |
|------|------|
| 吞吐量 | 比 vLLM 高 10-30%（NVIDIA GPU） |
| 延迟 | 最低延迟方案 |
| 精度 | 支持 FP8，几乎无损 |
| 限制 | 仅 NVIDIA GPU |

---

## 4. TGI (Text Generation Inference)

### 4.1 核心特性

- **HuggingFace 原生**：无缝集成 HF 生态
- **Safetensors 支持**：快速模型加载
- **FlashAttention**：内置优化
- **Sharded 加载**：大模型分片加载
- **Prometheus 监控**：内置指标暴露

### 4.2 部署示例

```bash
# Docker 部署
docker run --gpus all \
    -p 8080:80 \
    -v $(pwd)/data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-hf \
    --quantize awq

# 调用
curl http://localhost:8080/generate \
    -X POST \
    -d '{
        "inputs": "Hello",
        "parameters": {"max_new_tokens": 100}
    }' \
    -H 'Content-Type: application/json'
```

### 4.3 与 vLLM 对比

| 特性 | TGI | vLLM |
|------|-----|------|
| HF 集成 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 吞吐量 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 监控 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 量化 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 5. SGLang

### 5.1 核心特性

- **结构化生成**：强制 JSON、正则约束
- **RadixAttention**：复用 KV Cache
- **前端语言**：Pythonic 的编程接口
- **多模型支持**：同时服务多个模型

### 5.2 结构化生成

```python
import sglang as sgl

@sgl.function
def text_qa(s, question):
    s += "Question: " + question + "\n"
    s += "Answer: " + sgl.gen("answer", max_tokens=256)

# 强制 JSON 输出
@sgl.function
def extract_info(s, text):
    s += text + "\n"
    s += sgl.gen(
        "json_output",
        max_tokens=256,
        regex=r'\{[^{}]*\}'  # 正则约束
    )

# 运行
state = text_qa.run(
    question="What is AI?",
    backend=sgl.RuntimeEndpoint("http://localhost:30000")
)
print(state["answer"])
```

### 5.3 RadixAttention

**复用前缀 KV Cache**：

```
请求 1: "请翻译以下英文：Hello" → 缓存 [请翻译以下英文：]
请求 2: "请翻译以下英文：World" → 复用缓存！
请求 3: "请总结以下文章：[长文章]" → 缓存 [长文章]
请求 4: "请提取关键词：[长文章]" → 复用 [长文章] 缓存！
```

---

## 6. 本地/边缘推理

### 6.1 llama.cpp

**特性**：
- CPU 推理（也支持 GPU）
- GGUF 格式
- 多平台（Windows/Mac/Linux）
- 绑定多语言（Python、Go、Rust）

**使用**：

```bash
# 下载量化模型
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

# 运行
./main -m llama-2-7b.Q4_K_M.gguf -p "Hello" -n 100

# Python 绑定
from llama_cpp import Llama

llm = Llama(model_path="./llama-2-7b.Q4_K_M.gguf")
output = llm("Hello", max_tokens=100)
```

### 6.2 Ollama

**特性**：
- 一键运行模型
- 模型管理（pull、run、list）
- REST API
- Modelfile 自定义

**使用**：

```bash
# 安装
curl -fsSL https://ollama.com/install.sh | sh

# 运行模型
ollama run llama2

# REST API
curl http://localhost:11434/api/generate -d '{
    "model": "llama2",
    "prompt": "Hello"
}'
```

---

## 7. 选型决策树

```
部署环境
    │
    ├─ NVIDIA GPU 服务器？
    │   ├─ 追求极致性能？
    │   │   ├─ 是 → TensorRT-LLM
    │   │   └─ 否 → vLLM
    │   └─ HF 生态深度集成？
    │       ├─ 是 → TGI
    │       └─ 否 → vLLM
    │
    ├─ 需要结构化生成？
    │   ├─ 是 → SGLang
    │   └─ 否 → 继续
    │
    ├─ CPU / 边缘设备？
    │   ├─ 是 → llama.cpp
    │   └─ 否 → 继续
    │
    ├─ 本地开发测试？
    │   ├─ 是 → Ollama
    │   └─ 否 → 继续
    │
    ├─ 多云/混合云？
    │   ├─ 是 → vLLM（通用性最好）
    │   └─ 否 → 继续
    │
    └─ 默认推荐 → vLLM
```

---

## 8. 生产部署架构

### 8.1 典型架构

```
                    ┌─────────────┐
                    │   Load      │
                    │  Balancer   │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  vLLM       │ │  vLLM       │ │  vLLM       │
    │  Instance 1 │ │  Instance 2 │ │  Instance 3 │
    │  (GPU 1-2)  │ │  (GPU 3-4)  │ │  (GPU 5-6)  │
    └─────────────┘ └─────────────┘ └─────────────┘
           │               │               │
           └───────────────┼───────────────┘
                           │
                    ┌──────┴──────┐
                    │   Redis     │
                    │   Cache     │
                    └─────────────┘
```

### 8.2 监控指标

| 指标 | 说明 | 告警阈值 |
|------|------|----------|
| **Throughput** | tokens/秒 | 低于基线 20% |
| **Latency** | P50/P95/P99 | P99 > 2s |
| **Queue Size** | 等待队列长度 | > 100 |
| **GPU Util** | GPU 利用率 | < 50% 或 > 95% |
| **GPU Memory** | 显存使用 | > 90% |
| **Error Rate** | 错误率 | > 1% |

---

## 💡 我的思考

1. **vLLM 是通用最佳选择**：除非有特殊需求（NVIDIA 极致优化、HF 深度集成、结构化生成），否则 vLLM 是最稳妥的选择。

2. **框架在快速收敛**：各框架都在互相学习（vLLM 借鉴 TensorRT 的 kernel，TGI 借鉴 vLLM 的批处理），差异在缩小。

3. **本地部署很重要**：不是所有场景都需要云端大模型，llama.cpp 和 Ollama 让本地部署变得简单。

4. **监控是生产关键**：推理服务的监控比功能更重要，需要建立完善的 observability。

5. **成本优化是持续过程**：模型量化、动态批处理、缓存策略，每层优化都能显著降低成本。

---

## 参考来源

- **vLLM**: [docs.vllm.ai](https://docs.vllm.ai/) — 访问日期：2026-06-09
- **TensorRT-LLM**: [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — 访问日期：2026-06-09
- **TGI**: [huggingface.co/docs/text-generation-inference](https://huggingface.co/docs/text-generation-inference) — 访问日期：2026-06-09
- **SGLang**: [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) — 访问日期：2026-06-09
- **llama.cpp**: [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) — 访问日期：2026-06-09
- **Ollama**: [ollama.com](https://ollama.com/) — 访问日期：2026-06-09

---

*访问日期: 2026-06-09*
