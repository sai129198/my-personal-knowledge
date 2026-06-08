# 视频理解技术

> **一句话定位**：从视频分类到视频生成，系统梳理视频 AI 的理解、分析、生成技术栈与工程实践。
>
> #status/draft #topic/multimodal #topic/video #year/2026

---

## 一、视频 AI 技术概览

### 1.1 视频理解 vs 视频生成

```
视频理解（分析）              视频生成（合成）
├── 视频分类                   ├── 文生视频 (T2V)
├── 动作识别                   ├── 图生视频 (I2V)
├── 时序定位                   ├── 视频编辑
├── 视频描述                   ├── 视频补全
├── 视频问答                   └── 风格迁移
└── 视频检索
```

### 1.2 核心挑战

| 挑战 | 说明 | 影响 |
|------|------|------|
| **数据量巨大** | 1分钟视频 ≈ 1800帧 (30fps) | 计算成本高 |
| **时序依赖** | 动作需要连续多帧理解 | 需建模时间关系 |
| **多模态融合** | 画面 + 音频 + 字幕 | 信息整合复杂 |
| **长视频处理** | 电影/直播可达数小时 | 内存和计算瓶颈 |
| **标注成本高** | 视频标注比图片贵 10-100 倍 | 训练数据稀缺 |

---

## 二、视频理解技术

### 2.1 技术演进

```
传统方法 (2014-2018)
  ├─ Two-Stream CNN (RGB + 光流)
  ├─ C3D (3D 卷积)
  └─ I3D (Inflated 3D ConvNet)

Transformer 时代 (2019-2022)
  ├─ TimeSformer (时空注意力)
  ├─ ViViT (Video Vision Transformer)
  └─ VideoMAE (自监督预训练)

多模态大模型 (2023-至今)
  ├─ Video-LLaVA (视频 + 语言)
  ├─ Qwen2-VL (统一视觉语言)
  └─ Gemini 1.5 Pro (百万 token 上下文)
```

### 2.2 核心架构

#### 3D CNN 架构

```
输入: T × H × W × C (时间 × 高 × 宽 × 通道)
  ↓
[3D Conv] → 提取时空特征
  ↓
[3D Pooling] → 降维
  ↓
[FC] → 分类/回归
```

**代表模型**：C3D、I3D、SlowFast

**特点**：
- 显式建模时空关系
- 计算量大（3D 卷积参数量高）
- 适合短视频分类

#### Video Transformer 架构

```
输入视频
  ↓
[Patch Embedding] → 将时空切分为 token
  ↓
[Transformer Encoder] × N
  ├─ 空间注意力 (同一帧内)
  └─ 时间注意力 (不同帧间)
  ↓
[CLS Token] → 分类/理解
```

**代表模型**：ViViT、TimeSformer、VideoMAE

**VideoMAE 创新点**：
- 90-95% 的高掩码率（视频冗余度高）
- Tube masking（时空管状掩码）
- 小数据即可微调（3K-4K 视频）

#### 多模态视频理解架构

```
视频输入                    文本查询
  ↓                          ↓
[视觉编码器]              [文本编码器]
  (ViT/VideoMAE)           (LLM)
  ↓                          ↓
[投影层] ←── 对齐 ──→ [投影层]
  ↓
[多模态 LLM]
  (LLaVA/Qwen2-VL)
  ↓
生成回答/描述/分析
```

### 2.3 主流模型对比

| 模型 | 类型 | 参数 | 特点 | 适用场景 |
|------|------|------|------|----------|
| **Video-LLaVA** | 开源 VLM | 7B | 视频+图片统一理解 | 视频问答 |
| **Qwen2-VL** | 开源 VLM | 7B/72B | 阿里，视频理解强 | 多模态对话 |
| **LLaVA-NeXT** | 开源 VLM | 7B/34B | 支持视频帧序列 | 视频分析 |
| **Gemini 1.5 Pro** | 闭源 | - | 1M 上下文，长视频 | 长视频理解 |
| **GPT-4o** | 闭源 | - | 多模态，视频理解 | 通用分析 |
| **VideoMAE** | 自监督 | 86M | 高效预训练 | 视频分类 |
| **ViViT** | Transformer | 86M | 纯 Transformer | 动作识别 |
| **InternVid** | 开源 | - | 视频-文本对训练 | 视频检索 |

### 2.4 视频理解任务

#### 视频分类

```python
from transformers import VideoMAEForVideoClassification
import torch

model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics"
)

# 输入: 采样 16 帧，每帧 224x224
# 输出: 400 类动作分类
outputs = model(pixel_values=video_frames)
predicted_class = outputs.logits.argmax(-1).item()
```

**常用数据集**：
- Kinetics-400/600/700（动作识别）
- UCF101（动作识别）
- Something-Something V2（时序推理）

#### 视频时序定位

```
任务：在视频中找到"某人做某事"的时间段

输入：
- 视频（10分钟）
- 查询："展示产品功能的片段"

输出：
- [02:15 - 03:30] 产品功能介绍
- [05:40 - 06:10] 功能演示
```

**方法**：
- 滑动窗口 + 相似度匹配
- 注意力机制定位
- 边界回归

#### 视频描述生成

```python
# 使用 Video-LLaVA 生成视频描述
from transformers import VideoLlavaForConditionalGeneration

model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B")

# 输入视频帧 + 提示
prompt = "USER: <video>\nDescribe this video.\nASSISTANT:"
output = model.generate(**inputs)
# → "A person is cooking pasta in the kitchen..."
```

---

## 三、视频生成技术

### 3.1 技术演进

```
早期方法 (2018-2022)
  ├─ GAN-based (TGAN, MoCoGAN)
  ├─ VAE-based
  └─ 质量低，时序不稳定

扩散模型 (2022-2024)
  ├─ Make-A-Video (Meta)
  ├─ Imagen Video (Google)
  ├─ AnimateDiff (开源)
  └─ 质量提升，但时长受限

DiT 时代 (2024-至今)
  ├─ Sora (OpenAI) - 里程碑
  ├─ Open-Sora (开源复刻)
  ├─ CogVideo (智谱)
  └─ 分钟级高质量视频
```

### 3.2 Sora 架构解析

```
文本提示 → [文本编码器] → 语义向量
                              ↓
噪声视频 → [DiT (Diffusion Transformer)] → 去噪迭代
                              ↑
                    时空 Patch 化
                    (Spacetime Patches)
                              ↓
                        生成视频
```

**关键技术**：

| 技术 | 说明 |
|------|------|
| **Spacetime Patches** | 将视频切分为时空块，类似 ViT 的 patch |
| **DiT 架构** | Diffusion + Transformer，可扩展性强 |
| **视频压缩网络** | 类似 VAE，将视频压缩到潜空间 |
| **长上下文** | 支持分钟级视频生成 |

### 3.3 开源视频生成模型

| 模型 | 参数 | 时长 | 分辨率 | 特点 |
|------|------|------|--------|------|
| **Open-Sora Plan v1.5** | 8B | 93帧 | 480p | 北大，昇腾训练 |
| **CogVideoX** | 5B | 6秒 | 720p | 智谱，开源 |
| **Mochi** | 10B | 5秒 | 480p | Genmo，开源 |
| **LTX-Video** | 3B | 5秒 | 768p | Lightricks，实时 |
| **HunyuanVideo** | - | 5秒 | 720p | 腾讯，开源 |
| **Wan 2.1** | 14B | 5秒 | 720p | 阿里，开源 |

### 3.4 视频生成 Pipeline

```python
# 使用 Open-Sora 生成视频
from opensora.models import OpenSora

model = OpenSora.from_pretrained("hpcai-tech/Open-Sora-v1.5")

# 文本生成视频
video = model.generate(
    prompt="A cat playing with a ball of yarn in a sunny garden",
    num_frames=93,
    height=480,
    width=640,
    fps=24
)

# 图生视频
video = model.generate(
    prompt="Make the image animate",
    image="input.png",
    num_frames=93
)
```

---

## 四、工程实践

### 4.1 视频预处理

```python
import cv2
import numpy as np
from torchvision import transforms

def preprocess_video(video_path, num_frames=16, target_size=224):
    """视频预处理标准流程"""
    
    # 1. 读取视频
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 2. 均匀采样
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 3. 调整大小
            frame = cv2.resize(frame, (target_size, target_size))
            # 4. 颜色空间转换
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    # 5. 归一化
    frames = np.stack(frames) / 255.0
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # T, C, H, W
    
    return frames
```

### 4.2 长视频处理策略

```
策略一：滑动窗口
├── 将长视频切分为重叠的短片段
├── 每个片段独立处理
└── 融合结果

策略二：关键帧提取
├── 场景切换检测
├── 只处理关键帧
└── 大幅降低计算量

策略三：层次化理解
├── 第一层：快速浏览（抽帧）
├── 第二层：重点区域（时序定位）
└── 第三层：精细分析（完整片段）

策略四：Gemini 1.5 式长上下文
├── 直接处理百万 token
├── 无需切分
└── 目前仅限闭源模型
```

### 4.3 视频 RAG 实现

```python
class VideoRAG:
    def __init__(self):
        self.frame_extractor = VideoFrameExtractor(fps=1)
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B")
        self.asr = WhisperModel("large-v3")
        self.vector_store = QdrantClient()
    
    def index_video(self, video_path: str):
        """索引视频内容"""
        # 1. 提取关键帧
        frames = self.frame_extractor.extract(video_path)
        
        # 2. 生成帧描述
        descriptions = []
        for frame in frames:
            desc = self.vlm.describe(frame)
            descriptions.append(desc)
        
        # 3. 语音转录
        transcript = self.asr.transcribe(video_path)
        
        # 4. 构建时间戳对齐的文档
        documents = []
        for i, (frame, desc) in enumerate(zip(frames, descriptions)):
            timestamp = frame.timestamp
            documents.append({
                "content": f"[{timestamp}] 画面: {desc}\n音频: {transcript.get(timestamp, '')}",
                "timestamp": timestamp,
                "frame_path": frame.path
            })
        
        # 5. 存入向量库
        self.vector_store.upsert(documents)
    
    def query(self, question: str):
        """查询视频内容"""
        # 检索相关片段
        results = self.vector_store.search(question, k=5)
        
        # 组装上下文
        context = "\n".join([r.content for r in results])
        
        # 生成回答
        answer = self.vlm.generate(
            prompt=f"基于以下视频内容回答问题：\n{context}\n\n问题：{question}"
        )
        
        return {
            "answer": answer,
            "sources": [r.timestamp for r in results]
        }
```

### 4.4 性能优化

| 优化点 | 方法 | 效果 |
|--------|------|------|
| **帧采样** | 从 30fps 降到 1fps | 30x 加速 |
| **分辨率** | 1080p → 224p | 25x 加速 |
| **模型量化** | FP16/INT8 | 2-4x 加速 |
| **批处理** | 多视频并行 | 线性提升 |
| **缓存** | 关键帧特征缓存 | 避免重复计算 |
| **硬件** | TensorRT / ONNX | 2-3x 加速 |

---

## 五、应用场景

### 5.1 智能监控

```
输入：摄像头实时视频流
处理：
  ├─ 人员检测/跟踪
  ├─ 异常行为识别
  ├─ 人群密度分析
  └─ 事件预警
输出：实时告警 + 录像片段
```

### 5.2 内容审核

```
输入：用户上传视频
处理：
  ├─ 画面审核（暴力/色情/政治）
  ├─ 音频审核（脏话/敏感词）
  ├─ 字幕审核
  └─ 综合风险评分
输出：通过/拒绝/人工复核
```

### 5.3 视频搜索

```
输入："找昨天会议中关于预算的讨论"
处理：
  ├─ 语音识别转录
  ├─ 画面内容理解
  ├─ 多模态检索
  └─ 时间戳定位
输出：[15:23-18:45] 预算讨论片段
```

### 5.4 教育/培训

```
输入：教学视频
处理：
  ├─ 自动生成章节
  ├─ 提取关键知识点
  ├─ 生成测验题
  └─ 生成文字摘要
输出：结构化学习材料
```

---

## 💡 我的思考

### 关键洞察

1. **视频理解正在从"分类"走向"理解"** — Video-LLaVA、Qwen2-VL 让视频可以对话
2. **视频生成是下一个爆发点** — Sora 证明了分钟级高质量视频生成的可行性
3. **长视频处理仍是挑战** — 大多数模型只能处理几秒到几十秒
4. **多模态融合是正确方向** — 画面 + 音频 + 字幕联合理解更准确

### 选型建议

| 场景 | 推荐方案 |
|------|----------|
| 视频分类/动作识别 | VideoMAE / ViViT |
| 视频问答/对话 | Qwen2-VL / Video-LLaVA |
| 长视频理解 | Gemini 1.5 Pro / GPT-4o |
| 视频生成 | Open-Sora / CogVideoX |
| 实时视频分析 | 轻量化模型 + 边缘部署 |
| 视频 RAG | 抽帧 + VLM + ASR + 向量库 |

### 常见陷阱

- ❌ 忽略帧采样策略（均匀采样可能错过关键动作）
- ❌ 高分辨率直接处理（应先降采样）
- ❌ 忽视音频信息（很多信息在音频中）
- ❌ 不考虑时序关系（单帧理解 vs 连续动作）

### 下一步实践

- [ ] 搭建 Video-LLaVA 本地 Demo，测试视频问答
- [ ] 实现视频 RAG：抽帧 + ASR + 向量检索
- [ ] 对比不同帧采样策略对理解准确率的影响
- [ ] 测试 Open-Sora 生成视频的质量和速度

---

## 参考来源

1. OpenAI - Sora: First Impressions (2024)
2. PKU - Open-Sora Plan v1.5 Technical Report (2025)
3. Alibaba - Qwen2-VL Paper (2024)
4. Hugging Face - VideoMAE Documentation
5. Hugging Face - ViViT Documentation
6. Video-LLaVA Paper (2023)
7. VideoMAE Paper (NeurIPS 2022)

---

*最后更新：2026-06-08*
