# 视觉语言模型（VLM）选型与应用

> **一句话定位**：让 AI"看懂"世界——VLM 架构原理、主流模型对比与工程实践。
>
> #status/draft #topic/multimodal #topic/vision #year/2026

---

## 一、什么是视觉语言模型？

### 1.1 核心能力

VLM（Vision-Language Model）能够同时理解**图像**和**文本**，实现跨模态推理：

```
输入：
  图像：[照片：一只猫坐在沙发上]
  文本："这只猫在做什么？"

输出：
  "这只猫正舒适地坐在沙发上，看起来很放松。"
```

### 1.2 应用场景

| 场景 | 示例 |
|------|------|
| **图像描述** | 自动生成图片 ALT 文本 |
| **视觉问答** | "图中有几个人？" |
| **文档理解** | 解析发票、表格、PDF |
| **UI 自动化** | 根据界面截图执行操作 |
| **医疗影像** | 辅助诊断 X 光、CT |
| **自动驾驶** | 理解道路场景 |

---

## 二、架构原理

### 2.1 典型架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   图像输入   │     │   文本输入   │     │   视觉编码器 │
│  (PNG/JPG)  │────→│  (Tokenizer)│     │  (ViT/ResNet)│
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                       ┌────────────────────────┘
                       ▼
              ┌─────────────────┐
              │  投影层/适配器   │  ← 对齐视觉和文本特征空间
              │  (Projection/   │
              │   Adapter)      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   大语言模型     │  ← 核心推理能力
              │   (LLM Backbone)│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    文本输出      │
              └─────────────────┘
```

### 2.2 关键组件

#### 视觉编码器 (Vision Encoder)

| 类型 | 代表模型 | 特点 |
|------|----------|------|
| **ViT** | CLIP-ViT, SigLIP | Transformer 架构，全局特征 |
| **CNN** | ResNet, ConvNeXt | 局部特征强，计算效率高 |
| **混合** | CoAtNet, MaxViT | 结合 CNN 和 Transformer |

#### 投影层 (Projection Layer)

将视觉特征映射到 LLM 的输入空间：

```python
# 简单线性投影
vision_tokens = Linear(vision_dim, llm_dim)(vision_features)

# 或使用 Q-Former (BLIP-2)
vision_tokens = QFormer(vision_features, query_tokens)
```

#### 对齐策略

| 策略 | 代表模型 | 原理 |
|------|----------|------|
| **预训练对齐** | CLIP | 对比学习对齐图文特征 |
| **指令微调** | LLaVA, MiniGPT-4 | 用指令数据微调对齐 |
| **冻结视觉** | Frozen, Flamingo | 冻结编码器，只训练投影层 |

---

## 三、主流模型对比

### 3.1 闭源模型

| 模型 | 厂商 | 视觉能力 | 文本能力 | 特点 |
|------|------|----------|----------|------|
| **GPT-4o** | OpenAI | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 原生多模态，实时交互 |
| **Claude 3.5 Sonnet** | Anthropic | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 视觉推理强，文档理解 |
| **Gemini 1.5 Pro** | Google | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 100万 token 上下文 |
| **Qwen-VL-Max** | 阿里 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中文场景优化 |

### 3.2 开源模型

| 模型 | 参数 | 视觉编码器 | LLM | 特点 |
|------|------|------------|-----|------|
| **LLaVA-1.6** | 7B/13B/34B | CLIP ViT | Vicuna/Llama 2 | 社区最活跃 |
| **Qwen2-VL** | 7B/72B | Qwen2-ViT | Qwen2 | 中文支持好 |
| **InternVL2** | 8B/26B/40B | InternViT | InternLM2 | 文档理解强 |
| **MiniCPM-V** | 2.8B | SigLIP | MiniCPM | 端侧部署 |
| **Moondream** | 1.6B | SigLIP | Phi-2 | 超轻量级 |

### 3.3 能力评估维度

```
评估 VLM 时关注：
├─ 基础识别
│  ├─ 物体检测与分类
│  ├─ OCR（文字识别）
│  └─ 人脸/场景识别
│
├─ 高级理解
│  ├─ 空间关系（上下左右）
│  ├─ 数量统计
│  ├─ 属性识别（颜色、材质）
│  └─ 情感/意图理解
│
├─ 推理能力
│  ├─ 因果推理
│  ├─ 逻辑推理
│  └─ 多步推理
│
└─ 特殊场景
   ├─ 低分辨率图像
   ├─ 文档/表格
   ├─ 多图对比
   └─ 视频理解
```

---

## 四、工程实践

### 4.1 模型选型决策树

```
预算充足？
├── 是 → 需要实时交互？
│   ├── 是 → GPT-4o / Gemini
│   └── 否 → Claude 3.5（文档理解）
└── 否 → 需要端侧部署？
    ├── 是 → MiniCPM-V / Moondream
    └── 否 → 需要中文优化？
        ├── 是 → Qwen2-VL
        └── 否 → LLaVA-1.6
```

### 4.2 部署优化

#### 量化策略

```python
# VLM 量化通常分两部分

# 1. 视觉编码器量化（影响较小）
vision_encoder = quantize(vision_encoder, bits=8)

# 2. LLM 部分量化（影响较大，用 GPTQ/AWQ）
llm = load_quantized_model("llava-llama-2-13b-4bit-gptq")
```

**注意**：VLM 量化需谨慎，视觉-文本对齐容易被破坏。

#### 推理加速

| 技术 | 效果 | 适用场景 |
|------|------|----------|
| **FlashAttention-2** | 2-4x 加速 | 长序列 |
| **vLLM** | 高吞吐服务 | 在线 API |
| **TensorRT-LLM** | 极致性能 | NVIDIA GPU |
| **Batch 推理** | 提高 GPU 利用率 | 离线处理 |

### 4.3 提示工程

#### 图像描述优化

```markdown
## 基础提示
"描述这张图片"

## 优化后
"请详细描述这张图片，包括：
1. 主要物体和人物
2. 场景和环境
3. 物体之间的空间关系
4. 颜色和光线
5. 任何文字内容（OCR）"
```

#### 视觉问答优化

```markdown
## 基础提示
"图中有几个人？"

## 优化后（Few-shot）
"我会问你关于图片的问题。请仔细数清楚。

Q: [示例图1] 图中有几只猫？
A: 3 只

Q: [示例图2] 图中有几辆车？
A: 2 辆

Q: [目标图] 图中有几个人？
A:"
```

### 4.4 处理高分辨率图像

```python
def process_high_res_image(image, max_size=1024):
    """处理高分辨率图像"""
    
    # 1. 调整尺寸（保持比例）
    image.thumbnail((max_size, max_size))
    
    # 2. 分块处理（用于细节识别）
    patches = split_into_patches(image, patch_size=448)
    
    # 3. 全局 + 局部特征融合
    global_features = encode_image(image)
    local_features = [encode_image(patch) for patch in patches]
    
    return fuse_features(global_features, local_features)
```

---

## 五、多模态 RAG

### 5.1 架构设计

```
┌─────────────────────────────────────────┐
│           多模态 RAG 架构                │
├─────────────────────────────────────────┤
│                                         │
│  文档 → [文本提取] ─→ 文本向量 ─┐       │
│       ↓                        │       │
│       [图像提取] ─→ 图像向量 ──┼──→ 向量数据库 │
│       ↓                        │       │
│       [表格提取] ─→ 表格向量 ──┘       │
│                                         │
│  查询 → [查询理解] ──→ 多模态检索 ──→ 重排序 │
│                              ↓         │
│                         生成回答        │
└─────────────────────────────────────────┘
```

### 5.2 关键技术

| 技术 | 说明 |
|------|------|
| **统一嵌入空间** | 用 CLIP 等模型将图文映射到同一空间 |
| **多向量表示** | 一个文档用多个向量表示（文本、图像、表格） |
| **跨模态检索** | 文本查询检索图像，图像查询检索文本 |
| **多模态重排序** | 用 VLM 对检索结果精排 |

---

## 六、评估与基准

### 6.1 评估基准

| 基准 | 测试内容 | 代表模型分数 |
|------|----------|-------------|
| **MMBench** | 多模态综合能力 | GPT-4V: 86.1 |
| **MMMU** | 大学级别多学科 | GPT-4V: 62.0 |
| **TextVQA** | 图中文字理解 | GPT-4V: 87.2 |
| **DocVQA** | 文档理解 | Claude 3.5: 95.2 |
| **MathVista** | 数学图表推理 | GPT-4o: 73.8 |

### 6.2 自建评估集

```python
def evaluate_vlm(model, test_cases):
    """评估 VLM 在业务场景的表现"""
    results = []
    
    for case in test_cases:
        # 1. 准备输入
        image = load_image(case.image_path)
        prompt = case.question
        
        # 2. 模型推理
        response = model.generate(image=image, prompt=prompt)
        
        # 3. 评估答案
        if case.evaluation_type == "exact":
            correct = normalize(response) == normalize(case.answer)
        elif case.evaluation_type == "contain":
            correct = case.answer in response
        elif case.evaluation_type == "llm_judge":
            correct = llm_judge(response, case.answer)
        
        results.append({
            "question": prompt,
            "predicted": response,
            "expected": case.answer,
            "correct": correct
        })
    
    return calculate_metrics(results)
```

---

## 💡 我的思考

### 关键洞察

1. **VLM 正在快速收敛**：开源模型（LLaVA、Qwen2-VL）与闭源差距在缩小
2. **文档理解是杀手级应用**：发票、合同、表格解析需求巨大
3. **端侧部署是趋势**：MiniCPM-V、Moondream 让 VLM 能在手机运行
4. **多模态 RAG 是下一个热点**：统一处理图文视频的知识库

### 选型建议

| 场景 | 推荐方案 |
|------|----------|
| 快速原型 | Claude 3.5 / GPT-4o API |
| 生产部署（云端） | Qwen2-VL-72B + vLLM |
| 端侧/私有化 | MiniCPM-V 2.8B |
| 文档解析专用 | InternVL2-40B |

### 常见陷阱

- ❌ 忽视图像分辨率（高分辨率需要特殊处理）
- ❌ 直接信任 VLM 的计数（容易出错，需验证）
- ❌ 混淆 OCR 和 VLM（纯 OCR 任务用 PaddleOCR 更高效）

### 下一步实践

- [ ] 用 LLaVA 和 Qwen2-VL 在业务数据上对比测试
- [ ] 实现多模态 RAG pipeline（图文混合检索）
- [ ] 评估 VLM 量化对业务场景的影响
- [ ] 研究视频理解模型（Video-LLaMA、LLaVA-NeXT-Video）

---

## 参考来源

1. LLaVA Paper - Visual Instruction Tuning (NeurIPS 2023)
2. Qwen2-VL Technical Report (2024)
3. InternVL2 Paper - Better than the Best (2024)
4. GPT-4V System Card (OpenAI, 2023)
5. Claude 3 Model Card (Anthropic, 2024)
6. MMBench / MMMU Benchmark Papers

---

*最后更新：2026-06-08*
