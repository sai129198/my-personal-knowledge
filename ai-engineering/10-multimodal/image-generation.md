# 图像生成技术详解

> **一句话定位**：从 Diffusion 到 GAN，系统梳理图像生成模型的原理、演进与工程实践。
>
> #status/draft #topic/multimodal #topic/image-generation #year/2026

---

## 一、技术演进时间线

```
2014  GAN 诞生 (Goodfellow)
  ↓
2020  DDPM (Denoising Diffusion Probabilistic Models)
  ↓
2021  Latent Diffusion (Stable Diffusion)
  ↓
2022  DALL·E 2, Midjourney, Stable Diffusion 开源
  ↓
2023  DALL·E 3, SDXL, ControlNet, LoRA
  ↓
2024  Stable Diffusion 3, DiT (Diffusion Transformer), Flux
  ↓
2025  视频生成爆发 (Sora, Runway, Pika)
```

---

## 二、核心原理

### 2.1 Diffusion 模型（当前主流）

#### 前向过程（加噪）

```
原始图像 x₀ → 加噪 → x₁ → 加噪 → x₂ → ... → x_T (纯噪声)

     x₀        x₁₀₀      x₂₀₀      x_T
     🐱   →    🙈   →    😵   →   📺
    清晰      模糊      很模糊     纯噪声
```

数学表达：

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

#### 反向过程（去噪）

训练神经网络预测噪声，逐步从纯噪声恢复图像：

```
纯噪声 x_T → 去噪 → x_{T-1} → ... → x₁ → x₀ (清晰图像)

   📺   →    😵   →    🙈   →    🐱
  纯噪声    很模糊     模糊      清晰
```

**关键洞察**：模型学习的是"噪声预测"，而不是直接生成图像。

### 2.2 Latent Diffusion（Stable Diffusion 核心）

**优化思路**：不在像素空间，而在**压缩的隐空间**（Latent Space）做扩散。

```
像素空间 (512×512×3 = 786K 维)
         ↓ VAE 编码器
隐空间 (64×64×4 = 16K 维，压缩 48x)
         ↓ Diffusion 过程
隐空间噪声 → 去噪 → 清晰隐表示
         ↓ VAE 解码器
像素空间图像
```

**优势**：
- 计算量减少 10-100x
- 内存占用大幅降低
- 生成速度显著提升

### 2.3 Diffusion Transformer (DiT)

**Stable Diffusion 3 和 Sora 的核心架构**。

**传统 U-Net vs DiT**：

| 特性 | U-Net (SD 1.x/2.x) | DiT (SD 3/Sora) |
|------|---------------------|-----------------|
| 架构 | CNN + Attention | 纯 Transformer |
| 可扩展性 | 有限 | 随模型大小线性提升 |
| 训练稳定性 | 一般 | 更好 |
| 文本理解 | 较弱 | 更强 |

**核心思想**：将图像 patch 化为 token，用 Transformer 处理。

```
图像 (256×256)
  ↓ 分成 16×16 patches
32×32 = 1024 个 patch tokens
  ↓ + 文本 tokens
Transformer 处理
  ↓
预测噪声
```

---

## 三、主流模型对比

### 3.1 闭源模型

| 模型 | 厂商 | 特点 | 访问方式 |
|------|------|------|----------|
| **DALL·E 3** | OpenAI | 提示词遵循度最高 | ChatGPT / API |
| **Midjourney v6** | Midjourney | 艺术质量顶尖 | Discord |
| **Imagen 3** | Google | 照片级真实感 | Vertex AI |
| **Firefly 3** | Adobe | 商业安全，集成 PS | Adobe 产品 |

### 3.2 开源模型

| 模型 | 参数 | 架构 | 特点 |
|------|------|------|------|
| **Stable Diffusion 3** | 800M-8B | DiT + Flow Matching | 开源最强 |
| **SDXL** | 3.5B | U-Net | 生态最丰富 |
| **Flux** | 12B | Transformer | 黑马，质量接近闭源 |
| **PixArt-Σ** | 600M | DiT | 高效，4K 生成 |
| **Kandinsky** | 3B | Diffusion | 多语言支持 |

### 3.3 能力对比

```
提示词遵循度：DALL·E 3 > Flux > SD3 > SDXL > Midjourney
艺术质量：    Midjourney > Flux > DALL·E 3 > SD3
真实感：      Imagen 3 > Flux > SD3 > DALL·E 3
开源生态：    SDXL > Flux > SD3 > Kandinsky
```

---

## 四、关键技术

### 4.1 条件控制机制

| 技术 | 原理 | 应用 |
|------|------|------|
| **Text Conditioning** | CLIP 文本编码器 | 文生图 |
| **ControlNet** | 边缘/深度/姿态等条件 | 精确控制构图 |
| **IP-Adapter** | 参考图像特征注入 | 风格迁移 |
| **T2I-Adapter** | 轻量级条件控制 | 快速控制 |

**ControlNet 示例**：

```python
from diffusers import StableDiffusionControlNetPipeline

# 用 Canny 边缘控制生成
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet
)

# 输入：边缘图 + 提示词
image = pipe(
    prompt="a beautiful landscape",
    image=edge_image,  # Canny 边缘图
).images[0]
```

### 4.2 微调技术

#### LoRA (Low-Rank Adaptation)

**核心思想**：只训练低秩矩阵，冻结原模型权重。

```python
# LoRA 微调只需训练 1-10% 的参数
from peft import LoraConfig

lora_config = LoraConfig(
    r=4,                    # 秩，通常 4-128
    lora_alpha=32,          # 缩放因子
    target_modules=["to_q", "to_k", "to_v"],  # 应用位置
)

# 训练后得到 .safetensors 文件（仅几十 MB）
```

**适用场景**：
- 特定风格训练（动漫、水墨画）
- 特定人物/物体训练
- 快速实验（训练时间 < 1 小时）

#### DreamBooth

**核心思想**：用 3-5 张图片让模型学习新概念。

```python
# 训练命令示例
accelerate launch train_dreambooth.py \
    --pretrained_model_name="runwayml/stable-diffusion-v1-5" \
    --instance_data_dir="./my-dog/" \
    --instance_prompt="a photo of sks dog" \
    --class_data_dir="./dog-class/" \
    --class_prompt="a photo of dog"
```

**关键技巧**：使用 rare token（如 "sks"）避免与已有概念冲突。

#### Textual Inversion

**核心思想**：学习新 token 的嵌入，不改变模型权重。

```python
# 只训练几个新 token
learned_embeds = train_textual_inversion(
    images=concept_images,
    initializer_token="toy"
)
# 生成时使用新 token: "a photo of <toy-concept>"
```

### 4.3 采样器与调度器

| 采样器 | 特点 | 适用场景 |
|--------|------|----------|
| **DDIM** | 确定性，可复现 | 需要相同结果 |
| **DPM++ 2M Karras** | 质量高，速度快 | 默认推荐 |
| **Euler a** | 有随机性，创意好 | 探索性生成 |
| **UniPC** | 步数少，质量高 | 快速生成 |

**步数选择**：
- 20-30 步：大多数场景足够
- 50 步：追求极致质量
- < 10 步：快速预览（配合特定采样器）

---

## 五、工程实践

### 5.1 提示词工程

#### 结构模板

```markdown
[主体], [细节描述], [环境/背景], [风格], [光照], [相机/镜头], [质量词]

示例：
"a majestic lion, detailed fur texture, golden mane flowing in wind, 
savanna at sunset, cinematic lighting, 85mm lens, 8k, highly detailed"
```

#### 负面提示词

```markdown
常见负面词：
blurry, low quality, distorted, deformed, ugly, duplicate, 
watermark, signature, text, cropped, out of frame
```

### 5.2 分辨率与比例

| 比例 | 适用场景 |
|------|----------|
| 1:1 (512×512) | 头像、图标 |
| 4:3 (768×576) | 照片、插画 |
| 16:9 (1024×576) | 风景、壁纸 |
| 9:16 (576×1024) | 手机壁纸、海报 |
| 21:9 | 电影感、超宽屏 |

**SDXL 推荐**：直接生成 1024×1024 或更高，再裁剪。

### 5.3 性能优化

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,  # 半精度
).to("cuda")

# 1. 使用 xFormers / FlashAttention
pipe.enable_xformers_memory_efficient_attention()

# 2. VAE 切片（大分辨率）
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# 3. CPU  offloading（显存不足）
pipe.enable_model_cpu_offload()

# 4. 批量生成
images = pipe([prompt1, prompt2, prompt3], num_inference_steps=25).images
```

### 5.4 部署方案

| 方案 | 延迟 | 成本 | 适用场景 |
|------|------|------|----------|
| **API 服务** (DALL·E/SD API) | 2-5s | 按次计费 | 快速上线 |
| **自托管** (A100) | 1-3s | 固定成本 | 高并发 |
| **Serverless** (Replicate) | 3-10s | 按秒计费 | 波动流量 |
| **端侧** (ONNX/CoreML) | 10-60s | 免费 | 隐私敏感 |

---

## 六、评估与质量

### 6.1 自动评估指标

| 指标 | 说明 | 局限 |
|------|------|------|
| **FID** (Fréchet Inception Distance) | 生成图像与真实分布的距离 | 不反映文本对齐 |
| **CLIP Score** | 图像-文本相似度 | 不反映图像质量 |
| **PickScore** | 人类偏好预测 | 需要训练数据 |

### 6.2 人工评估维度

```
图像质量：
  ├─ 清晰度（无模糊、伪影）
  ├─ 色彩自然度
  └─ 细节丰富度

文本对齐：
  ├─ 主体正确性
  ├─ 属性准确性（颜色、数量）
  └─ 关系正确性（空间、动作）

美学质量：
  ├─ 构图
  ├─ 风格一致性
  └─ 创意性
```

---

## 七、安全与伦理

### 7.1 内容安全

| 风险 | 防护措施 |
|------|----------|
| 深度伪造 | 水印、溯源、检测模型 |
| 有害内容 | 提示词过滤、输出审核 |
| 版权争议 | 训练数据授权、风格过滤 |
| 虚假信息 | 元数据标注、平台政策 |

### 7.2 合规建议

- [ ] 用户生成内容免责声明
- [ ] 禁止生成真实人物（未经同意）
- [ ] 水印或元数据标识 AI 生成
- [ ] 建立举报和申诉机制

---

## 💡 我的思考

### 关键洞察

1. **DiT 是下一代架构**：SD3 和 Sora 验证了 Transformer 在生成任务的可扩展性
2. **开源正在追赶闭源**：Flux 质量已接近 Midjourney，生态爆发式增长
3. **ControlNet 是生产力工具**：精确控制能力让 AI 生成进入工作流
4. **LoRA 降低创新门槛**：个人开发者也能训练专属风格模型

### 选型建议

| 场景 | 推荐方案 |
|------|----------|
| 快速原型 / API | DALL·E 3 API |
| 艺术创作 | Midjourney + Photoshop |
| 自托管 / 定制 | Flux + ComfyUI |
| 移动端 / 边缘 | SD 1.5 + ONNX |
| 视频生成 | Sora (等开放) / Runway |

### 常见陷阱

- ❌ 忽视负面提示词（导致低质量输出）
- ❌ 直接用 SD 1.5 生成高分辨率（需用 HiRes Fix）
- ❌ 训练 LoRA 数据太少（过拟合）
- ❌ 忽略版权风险（商用需谨慎）

### 下一步实践

- [ ] 用 ComfyUI 搭建工作流（节点式生成 pipeline）
- [ ] 训练一个 LoRA（特定风格或物体）
- [ ] 对比 Flux / SD3 / DALL·E 3 在业务场景的表现
- [ ] 研究视频生成（AnimateDiff / SVD）

---

## 参考来源

1. Stable Diffusion Paper - Latent Diffusion Models (CVPR 2022)
2. DiT Paper - Scalable Diffusion Models with Transformers (ICCV 2023)
3. SD3 Technical Report - Stability AI (2024)
4. LoRA Paper - Low-Rank Adaptation of Large Language Models (ICLR 2022)
5. ControlNet Paper - Adding Conditional Control to Diffusion Models (ICCV 2023)
6. Hugging Face Diffusers Documentation (2024)
7. DALL·E 3 System Card (OpenAI, 2023)

---

*最后更新：2026-06-08*
