# 语音识别与合成技术

> **一句话定位**：从 Whisper 到 ElevenLabs，系统梳理语音 AI 的 ASR/TTS 技术栈与选型策略。
>
> #status/draft #topic/multimodal #topic/speech #year/2026

---

## 一、技术概览

### 1.1 语音 AI 两大核心任务

```
┌─────────────────────────────────────────────────┐
│                  语音 AI 技术栈                   │
├─────────────────────────────────────────────────┤
│                                                 │
│   语音输入 ──→ 语音识别 (ASR) ──→ 文本           │
│   (Audio)      (Whisper/Wav2Vec)   (Text)       │
│                                                 │
│   文本输入 ──→ 语音合成 (TTS) ──→ 语音           │
│   (Text)       (VITS/XTTS)         (Audio)      │
│                                                 │
│   语音输入 ──→ 语音转换 (VC) ──→ 语音            │
│   (Audio)      (音色/风格转换)      (Audio)      │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 1.2 应用场景

| 场景 | 技术组合 | 示例 |
|------|----------|------|
| **智能助手** | ASR + LLM + TTS | Siri, Alexa, 小爱同学 |
| **实时字幕** | ASR + 时间戳对齐 | 会议记录、直播字幕 |
| **有声书** | TTS + 情感控制 | 电子书朗读 |
| **配音** | TTS + 声音克隆 | 视频多语言配音 |
| **呼叫中心** | ASR + 意图识别 | 客服质检、语音机器人 |
| **无障碍** | ASR + TTS | 听障/视障辅助 |

---

## 二、语音识别 (ASR)

### 2.1 技术演进

```
传统方法 (2010-2018)
  ├─ HMM-GMM → HMM-DNN → CTC
  └─ 需要大量标注数据，准确率有限

深度学习 (2018-2022)
  ├─ Wav2Vec 2.0 (自监督学习)
  ├─ 仅需少量标注即可微调
  └─ 准确率大幅提升

端到端 (2022-至今)
  ├─ Whisper (Encoder-Decoder)
  ├─ 多语言、多任务统一模型
  └─ 接近人类水平
```

### 2.2 核心原理

#### CTC (Connectionist Temporal Classification)

解决输入输出长度不对齐的问题：

```
音频帧: [f1] [f2] [f3] [f4] [f5] [f6] ...
           ↓   ↓   ↓   ↓   ↓   ↓
预测:    [h] [h] [e] [e] [l] [l] [l] [o] [o]
           ↓ 合并重复 + 移除空白
输出:     "hello"
```

**特点**：
- 帧级独立预测，速度快
- 但缺乏语言模型，需外部语言模型辅助

#### Attention-based Seq2Seq

```
音频编码器          注意力机制           文本解码器
[音频特征] ──→ [上下文向量] ──→ ["你好"]
                ↑
         动态对齐音频和文本
```

**特点**：
- 自动学习对齐关系
- 生成质量更高，但速度较慢

#### Whisper 架构

```
音频 (30秒, 16kHz)
  ↓
Log-Mel 频谱图 (80维, 3000帧)
  ↓
Encoder (Transformer, 32层)
  ↓
Decoder (Transformer, 自回归生成文本)
  ↓
"<|startoftranscript|> <|zh|> <|transcribe|> 你好世界 <|endoftext|>"
```

**特殊 token 控制任务**：
- `<|zh|>` / `<|en|>`：语言标记
- `<|transcribe|>` / `<|translate|>`：转录或翻译
- `<|notimestamps|>`：不输出时间戳

### 2.3 主流模型对比

| 模型 | 参数 | 语言支持 | 特点 | 适用场景 |
|------|------|----------|------|----------|
| **Whisper v3** | 1.5B | 99 种 | 开源最强，多任务 | 通用转录 |
| **Whisper Large v3 Turbo** | 809M | 99 种 | 速度优化版 | 实时应用 |
| **Wav2Vec 2.0** | 1B | 多语言 | 自监督，易微调 | 领域适配 |
| **Wav2Vec-BERT 2.0** | 1B | 多语言 | 4.5M 小时预训练 | 低资源语言 |
| **Canary** | 1B | 多语言 | NVIDIA 出品 | 企业级部署 |
| **SenseVoice** | 300M | 中文优化 | 阿里开源 | 中文场景 |
| **Paraformer** | 220M | 中文 | 非自回归，速度快 | 实时识别 |

### 2.4 性能指标

| 指标 | 说明 | 优秀水平 |
|------|------|----------|
| **WER** (Word Error Rate) | 词错误率 | < 5% (干净音频) |
| **CER** (Character Error Rate) | 字错误率 | < 3% (中文) |
| **RTF** (Real-Time Factor) | 实时率 | < 0.1x (10倍实时) |
| **Latency** | 首字延迟 | < 300ms |

---

## 三、语音合成 (TTS)

### 3.1 技术演进

```
拼接合成 (2010前)
  └─ 录制音素片段拼接，机械感强

参数合成 (2010-2018)
  ├─ Tacotron / Tacotron 2
  ├─ 生成 Mel 频谱 + Griffin-Lim/ WaveNet 声码器
  └─ 自然度提升，但仍有 artifacts

端到端 (2018-2022)
  ├─ VITS / Glow-TTS
  ├─ 直接生成波形，一步合成
  └─ 自然度接近真人

大模型时代 (2022-至今)
  ├─ VALL-E / VoiceBox / XTTS
  ├─ 上下文学习，声音克隆
  └─ 情感/风格控制
```

### 3.2 核心原理

#### 两阶段 TTS

```
文本 → [文本编码器] → 声学特征 (Mel 频谱) → [声码器] → 波形

Tacotron2          Griffin-Lim / WaveNet / HiFi-GAN
(生成 Mel)          (Mel → 波形)
```

#### 端到端 TTS (VITS)

```
文本 → [VITS] → 波形

特点：
- 单阶段，直接生成波形
- 使用 Flow-based 模型 + GAN
- 训练稳定，音质好
```

#### 大模型 TTS (VALL-E / XTTS)

```
文本 + 参考音频 → [大模型] → 波形

核心能力：
- 3 秒参考音频即可克隆声音
- 保持说话人音色、语调、情感
- 支持跨语言语音克隆
```

### 3.3 主流模型/服务对比

| 服务/模型 | 类型 | 音质 | 延迟 | 定制能力 | 价格 |
|-----------|------|------|------|----------|------|
| **ElevenLabs** | 云端 API | ⭐⭐⭐⭐⭐ | 低 | 声音克隆、情感 | $5/月 |
| **Azure TTS** | 云端 API | ⭐⭐⭐⭐ | 低 | 多种音色、SSML | 按量 |
| **Amazon Polly** | 云端 API | ⭐⭐⭐ | 低 | 标准/神经音色 | 按量 |
| **Google Cloud TTS** | 云端 API | ⭐⭐⭐⭐ | 低 | WaveNet/Studio | 按量 |
| **XTTS v2** | 开源 | ⭐⭐⭐⭐ | 中 | 本地克隆 | 免费 |
| **VITS** | 开源 | ⭐⭐⭐ | 中 | 需训练 | 免费 |
| **PaddleSpeech** | 开源 | ⭐⭐⭐ | 中 | 中文优化 | 免费 |
| **ChatTTS** | 开源 | ⭐⭐⭐⭐ | 高 | 中文情感 | 免费 |

### 3.4 声音克隆技术

```python
# XTTS v2 声音克隆示例
from TTS.api import TTS

# 加载模型
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# 3 秒参考音频克隆声音
tts.tts_to_file(
    text="这是一段用克隆声音合成的测试文本。",
    speaker_wav="reference_3sec.wav",  # 参考音频
    language="zh",
    file_path="output.wav"
)
```

**关键要求**：
- 参考音频质量 > 音质
- 3-10 秒足够，需清晰无噪音
- 说话风格会被继承（语速、语调）

---

## 四、工程实践

### 4.1 ASR 部署优化

#### Whisper 优化

```python
from transformers import pipeline
import torch

# 1. 使用更快的模型
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",  # 809M，速度更快
    torch_dtype=torch.float16,
    device="cuda"
)

# 2. 批处理
results = pipe(audio_files, batch_size=8)

# 3. 使用 faster-whisper (CTranslate2 加速)
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.mp3", beam_size=5)
```

**性能对比**：

| 实现 | RTF | 加速比 |
|------|-----|--------|
| Whisper (PyTorch) | 0.15x | 1x |
| faster-whisper | 0.03x | 5x |
| Whisper.cpp | 0.02x | 7x |
| NVIDIA Riva | 0.01x | 15x |

### 4.2 实时语音识别

```python
import whisper
import numpy as np

class RealtimeTranscriber:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.buffer = np.array([], dtype=np.float32)
        self.chunk_duration = 3  # 秒
    
    def process_audio(self, audio_chunk):
        """处理实时音频流"""
        self.buffer = np.concatenate([self.buffer, audio_chunk])
        
        # 累积足够数据后识别
        if len(self.buffer) >= self.chunk_duration * 16000:
            result = self.model.transcribe(self.buffer)
            text = result["text"]
            
            # 滑动窗口保留上下文
            self.buffer = self.buffer[-16000:]  # 保留 1 秒
            
            return text
        return None
```

**优化策略**：
- VAD (Voice Activity Detection)：只处理有语音的片段
- 滑动窗口：保留上下文提高准确率
- 流式解码：减少首字延迟

### 4.3 TTS 部署优化

```python
from TTS.api import TTS

# 1. 预热模型
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(text="warmup", speaker_wav="ref.wav", language="zh")

# 2. 批量合成
texts = ["第一句", "第二句", "第三句"]
for text in texts:
    wav = tts.tts(text=text, speaker_wav="ref.wav", language="zh")
    # 流式输出或保存

# 3. 使用 ONNX / TensorRT 加速
# 导出 ONNX 后使用 ONNX Runtime
```

---

## 五、选型决策

### 5.1 ASR 选型

```
场景分析：
├── 多语言/通用场景
│   └── Whisper Large v3 (开源) 或 Azure Speech (商用)
├── 中文专用
│   └── SenseVoice / Paraformer (开源) 或 讯飞 (商用)
├── 实时性要求高
│   └── Whisper.cpp / faster-whisper / NVIDIA Riva
├── 领域适配（医疗/法律）
│   └── Wav2Vec 2.0 + 领域微调
└── 端侧部署
    └── Whisper Tiny / EdgeTPU 量化模型
```

### 5.2 TTS 选型

```
场景分析：
├── 最高音质 + 声音克隆
│   └── ElevenLabs (商用) / XTTS v2 (开源)
├── 企业级稳定
│   └── Azure TTS / Amazon Polly
├── 中文情感丰富
│   └── ChatTTS / 阿里云
├── 完全离线
│   └── Coqui TTS (XTTS) / PaddleSpeech
└── 超低成本
    └── EdgeTTS (基于 Azure 免费额度)
```

### 5.3 成本对比（每小时音频）

| 服务 | ASR 成本 | TTS 成本 |
|------|----------|----------|
| OpenAI Whisper API | $0.006/分钟 | - |
| Azure Speech | $1/小时 | $16/百万字符 |
| ElevenLabs | - | $0.18/千字符 |
| 开源自托管 | GPU 电费 | GPU 电费 |

---

## 六、高级话题

### 6.1 语音情感控制

```python
# 使用 SSML 控制情感（Azure TTS）
ssml = """
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="zh-CN">
    <voice name="zh-CN-XiaoxiaoNeural">
        <mstts:express-as style="cheerful" styledegree="2">
            今天天气真好！
        </mstts:express-as>
        <mstts:express-as style="sad" styledegree="1">
            但是明天要下雨了。
        </mstts:express-as>
    </voice>
</speak>
"""
```

### 6.2 语音翻译 Pipeline

```
语音输入 (中文)
  ↓ Whisper (transcribe)
中文文本
  ↓ LLM 翻译
英文文本
  ↓ TTS (英文声音)
语音输出 (英文)
```

**优化**：Whisper 可直接 `<|translate|>` 到英文，省去 LLM 步骤。

### 6.3 说话人分离 (Diarization)

```python
from pyannote.audio import Pipeline

# 识别"谁说了什么"
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
diarization = pipeline("meeting.wav")

# 输出：
# [00:00:00 - 00:00:05] Speaker A: "大家好..."
# [00:00:05 - 00:00:10] Speaker B: "我是..."
```

---

## 💡 我的思考

### 关键洞察

1. **ASR 已基本解决**：Whisper 让语音识别进入"够用"阶段，重点转向场景适配
2. **TTS 进入大模型时代**：声音克隆 3 秒即可，情感控制越来越精细
3. **实时性是工程重点**：模型再好，延迟高也无法用于对话场景
4. **多模态融合是趋势**：语音 + 视觉 + 文本的统一理解

### 选型建议

| 场景 | 推荐方案 |
|------|----------|
| 快速上线 | Azure Speech (ASR) + ElevenLabs (TTS) |
| 成本控制 | Whisper + XTTS v2 自托管 |
| 中文优化 | SenseVoice + ChatTTS |
| 实时对话 | faster-whisper + 流式 TTS |
| 声音克隆 | XTTS v2 (开源) / ElevenLabs (商用) |

### 常见陷阱

- ❌ 忽视音频预处理（降噪、增益、重采样）
- ❌ 长音频直接处理（需切分 + 后合并）
- ❌ 忽略 VAD（静音段浪费计算）
- ❌ 声音克隆用低质量参考音频

### 下一步实践

- [ ] 搭建 Whisper + XTTS 的本地语音助手
- [ ] 测试不同场景下的 WER（噪音、口音、专业术语）
- [ ] 实现流式 ASR + TTS 的实时对话系统
- [ ] 探索多模态语音理解（语音 + 唇读）

---

## 参考来源

1. OpenAI - Whisper Paper (2022)
2. Meta - wav2vec 2.0 Paper (NeurIPS 2020)
3. Microsoft - SpeechT5 Paper (2021)
4. Coqui - XTTS v2 Documentation
5. ElevenLabs - Text-to-Speech API Docs
6. Hugging Face - Transformers Speech Processing
7. NVIDIA - Riva Speech AI Documentation

---

*最后更新：2026-06-08*
