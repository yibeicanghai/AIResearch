# 三类多模态生成模型技术分析洞察报告

> **分析对象**：MiniMax-H3、MAGI-2 Preview、LTX-2.5
> **报告日期**：2026-11-03
> **资料来源**：ModelScope 模型页面、官方 GitHub 仓库、官方技术博客、arXiv 论文、官方文档

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [MiniMax-H3 技术分析](#2-minimax-h3-技术分析)
3. [MAGI-2 Preview 技术分析](#3-magi-2-preview-技术分析)
4. [LTX-2.5 技术分析](#4-ltx-25-技术分析)
5. [对比分析](#5-对比分析)
6. [关键技术深度对比](#6-关键技术深度对比)
7. [洞察与结论](#7-洞察与结论)
8. [深度架构问题分析](#8-深度架构问题分析)
   - 8.1 单流与多流架构的概念解释
   - 8.2 MiniMax-H3 模态特定参数与 AdaLN 机制详解
   - 8.3 多流架构的注意力机制与推理流程详解
   - 8.4 单流架构的 token 融合与解码机制详解
   - 8.5 注意力结构与稀疏机制详解
   - 8.6 单流架构中文本 token 的注意力机制与模态区分
9. [PixelUMM：像素空间统一多模态模型分析](#9-pixelumm像素空间统一多模态模型分析)
10. [PixelUMM 与三类多模态生成模型的对比分析](#10-pixelumm-与三类多模态生成模型的对比分析)
11. [PixelUMM MoT 架构深度解析（源码级）](#11-pixelumm-mot-架构深度解析源码级)
12. [采用 MoT 架构的其他多模态模型](#12-采用-mot-架构的其他多模态模型)
13. [BAGEL 模型架构详解](#13-bagel-模型架构详解)
14. [像素空间生成模型全景与技术成熟度分析](#14-像素空间生成模型全景与技术成熟度分析)

---

## 1. 执行摘要

本报告对三类多模态音视频生成模型进行深度技术分析。这三个模型代表了当前多模态生成领域的三种不同技术路线：

| 维度 | MiniMax-H3 | MAGI-2 Preview | LTX-2.5 |
|------|-----------|----------------|---------|
| **模型类型** | 全模态生成系统 | MoE 视频生成模型 | 双流 Diffusion Transformer |
| **总参数量** | ~33B（H3-Base，dense） | ~114B（MoE，激活~6B/token） | ~22B（双流 DiT） |
| **核心架构** | 单流 Omni-Transformer | 单流 MagiMoE Transformer | 非对称双流 DiT |
| **输入模态** | 文本/图像/视频/音频/参考 | 文本/图像 | 文本/图像/视频 |
| **输出模态** | 视频 + 立体声音频 | 视频 + 音频 | 视频 + 立体声音频 |
| **最大分辨率** | 2K（通过 Regenerate） | 1080p（通过 Refiner） | 1080p（通过多尺度推理） |
| **最大时长** | 15 秒 | 10 秒 | 20 秒 |
| **开源状态** | H3-Base 开源（Context-IR/2K 未开源） | 推理代码 + 权重开源 | 全部权重 + 代码开源 |

**核心洞察**：
- **MiniMax-H3** 走的是"全模态理解+生成"的系统路线，通过 Context-IR 中间表示实现复杂多模态指令理解，架构上是最简洁的单流 dense Transformer，但系统设计最复杂。
- **MAGI-2 Preview** 走的是"极致 MoE 扩展"路线，用 114B 总参数换取 6B 激活参数的效率，核心创新在于 Ultra-Fine-Grained MoE 和配套的分布式训练系统。
- **LTX-2.5** 走的是"非对称双流+高效推理"路线，用分离的视频/音频流配合跨模态注意力，在保持质量的同时实现极高推理速度（比 Wan 2.2-14B 快约 18 倍）。

---

## 2. MiniMax-H3 技术分析

### 2.1 模型概述

MiniMax-H3 是 MiniMax 推出的通用全模态（omni-modal）生成系统。它支持对文本、图像、视频、音频组成的多模态上下文进行统一理解，并能生成分辨率高达 2K、时长达 15 秒的带原生立体声音频的视频。

> **出处**：[ModelScope MiniMax-H3 模型页面](https://www.modelscope.cn/models/MiniMax/MiniMax-H3)、[HuggingFace MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)、[GitHub MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)

### 2.2 输入输出能力支持

#### 输入规格

H3 系统支持两种主要输入模式，由两个独立 checkpoint 处理：

| 模型变体 | 输入模式 | 规格 |
|---------|---------|------|
| H3-Base-FL2VA | 首尾帧模式 | 支持零、一或两张输入图像：无图像=T2V；一张=首帧或尾帧 I2V；两张=首尾帧 I2V |
| H3-Base-Ref2VA | 全参考模式 | 图像≤9张；视频≤3段（每段2-15秒，总时长≤15秒）；音频≤3段（每段2-15秒，总时长≤15秒）；混合输入总文件数≤12 |

#### 输出规格

| 类别 | 规格 |
|------|------|
| 输出时长 | 4-15 秒 |
| 宽高比 | 21:9, 16:9, 4:3, 1:1, 3:4, 9:16 等 |
| 输出分辨率 | 默认短边 768px；2K 通过 H3-Regenerate-2K 实现 |
| 帧率 | 24 FPS |
| 音频 | 32 kHz 立体声 |
| 支持语言 | 11种：阿拉伯语、中文、英语、法语、德语、意大利语、日语、韩语、葡萄牙语、俄语、西班牙语 |

> **出处**：ModelScope MiniMax-H3 README — System Overview 章节

### 2.3 模型架构

H3 系统由三个模块组成：

```
用户输入 → H3-Context-IR → Context中间表示 → H3-Base (768p) → H3-Regenerate-2K (2K) → 最终输出
```

#### 2.3.1 H3-Context-IR（上下文中间表示模块）

- **功能**：解释文本、图像、音频和参考视频之间的关系，将自由形式的多模态输入转换为 H3-Base 可理解的结构化 Context 中间表示。
- **内部工作流**：指令解析 → 跨模态关联 → 时间理解 → 复杂逻辑推理
- **输出**：包含 `integrated_multimodal_description`、`overall_soundscape`、`non_diegetic_music` 等字段的结构化文本表示
- **开源状态**：**未开源**（依赖多阶段工作流和多个托管模型），提供 API 调用

> **出处**：ModelScope MiniMax-H3 README — H3-Context-IR 章节

#### 2.3.2 H3-Base（核心生成模块）

**整体架构**：
- H3-Base 使用不同模态的编码器或 VAE 编码各种模态，将编码后的表示组织成统一的 packed 多模态序列
- 使用 RoPE 捕获 token 间的空间和时间关系
- 整个序列传入 H3-Omni-Transformer 进行联合预测
- 视频和音频 latents 分别解码为视频和立体声音频

**H3-Omni-Transformer**：

| 属性 | 值 |
|------|-----|
| 参数量 | ~33B（dense 单流 Transformer） |
| AdaLN 参数 | ~13B（可预计算并缓存，推理时无需加载） |
| 架构类型 | Dense 单流 Transformer |
| 位置编码 | 三维 Multimodal RoPE (MM-RoPE)，表示 `(t, h, w)` |
| 注意力/FFN | 不含模态特定结构，模态特定参数仅存在于输入/输出层和 AdaLN 分支 |
| 稀疏注意力 | 训练阶段引入原生稀疏注意力（**尚未在开源版本中提供**） |
| 调制方式 | 模态特定 AdaLN（以较低训练/推理成本提升生成质量） |

**H3-Encoder**：
- 基于 **Qwen3-VL-32B** 的完整预训练权重
- 使用其第 50 层的隐藏状态提供给 H3-Omni-Transformer
- 添加了特殊 token（如 `<d>`）到 tokenizer

**H3-VisualVAE**（视频变分自编码器）：

| 属性 | 值 |
|------|-----|
| 类型 | 时间因果视频自编码器 |
| 空间压缩 | 16× |
| 时间压缩 | 4× |
| latent 通道数 | 24 |
| 标记法 | f16t4d24 |
| Patchify | 1×2×2（time×height×width） |
| 有效空间下采样 | 32×（16×2） |
| 有效时间下采样 | 4×（不变） |
| 解码器 | ViT-based 解码器（降低解码成本，提升重建质量） |

**H3-AudioVAE**（音频变分自编码器）：

| 属性 | 值 |
|------|-----|
| 通道处理 | 左右声道使用相同的编码器/解码器，独立处理后重组 |
| 输入采样率 | 32 kHz |
| 时间压缩率 | 40 Hz（每秒 40 个 latent token） |
| 设计灵感 | VA-VAE |

> **出处**：ModelScope MiniMax-H3 README — H3-Base 章节

#### 2.3.3 H3-Regenerate-2K（2K 重生成模块）

- **方法**：不使用传统超分辨率模块，而是将 H3 基础模型以 in-context 方式重新生成低分辨率结果
- **优势**：(1) 复用 H3 基础模型的生成能力；(2) in-context 格式可复用原始多模态上下文，恢复传统超分辨率需要"猜测"的信息（如小文字和精细细节）
- **开源状态**：**未开源**，提供 API 调用

> **出处**：ModelScope MiniMax-H3 README — H3-Regenerate-2K 章节

### 2.4 推理流程

```
1. 用户输入（文本/图像/视频/音频）
       ↓
2. H3-Context-IR（API 调用，未开源）
   - 指令解析 → 跨模态关联 → 时间理解 → 逻辑推理
   - 输出：结构化 Context 中间表示
       ↓
3. H3-Base 本地推理（开源，BF16）
   - H3-Encoder（Qwen3-VL-32B）编码文本和视觉输入
   - H3-VisualVAE 编码视觉输入
   - H3-AudioVAE 编码音频输入
   - 所有模态打包为统一多模态序列
   - H3-Omni-Transformer 联合预测视频和音频 latents
   - 分别解码为 768p 视频 + 立体声音频
       ↓
4. H3-Regenerate-2K（API 调用，未开源）
   - 将 768p 结果 + 原始上下文 → in-context 重生成 → 2K 输出
       ↓
5. 最终输出：2K 分辨率视频 + 32kHz 立体声音频
```

**部署方式**：
- 推荐框架：SGLang、vLLM、diffusers、ComfyUI
- GPU 需求：4 GPU（SGLang，Ulysses degree=4）
- 精度：BF16

> **出处**：ModelScope MiniMax-H3 README — Recommended Workflow 章节

### 2.5 关键技术

1. **全模态统一理解与生成**：在预训练阶段就具备广泛的多模态上下文理解和生成能力，实现任务泛化。

2. **Context 中间表示（Context-IR）**：将复杂多模态指令深度理解和精炼为结构化表示，包含 `integrated_multimodal_description`、`overall_soundscape`、`non_diegetic_music` 三部分，是输出质量的关键。

3. **单流 Dense Transformer + 模态特定 AdaLN**：注意力层和 FFN 层不含模态特定结构，模态特定参数仅限于输入/输出层和 AdaLN 分支，以最低成本实现模态区分。

4. **AdaLN 参数缓存优化**：~13B AdaLN 参数可预计算并缓存，推理时无需加载，有效推理参数约 20B。

5. **In-Context 2K 重生成**：不使用专用超分模块，而是让基础模型以 in-context 方式重新生成，复用生成能力和原始上下文信息。

6. **原生稀疏注意力**：训练阶段已引入，降低长序列计算成本（推理版本尚未发布）。

7. **ViT-based VAE 解码器**：替代传统卷积解码器，降低解码成本并提升重建质量。

---

## 3. MAGI-2 Preview 技术分析

### 3.1 模型概述

MAGI-2 Preview 是 Sand AI 发布的 114B 参数统一音视频生成模型，每个 token 仅激活约 6B 参数。基于 MagiMoE 架构，在架构、系统和数据三个维度协同设计，探索视频生成模型的高效扩展路径。

> **出处**：[ModelScope MAGI-2-preview 模型页面](https://www.modelscope.cn/models/sand-ai/MAGI-2-preview)、[Sand.ai 技术博客](https://sand.ai/blog/magi-2-preview)、[HuggingFace MAGI-2-preview](https://huggingface.co/sand-ai/MAGI-2-preview)

### 3.2 输入输出能力支持

#### 输入

| 模式 | 说明 |
|------|------|
| T2V（文本到视频） | 文本提示词 → 视频 + 音频 |
| I2V（图像到视频） | 文本提示词 + 静态图像 → 视频 + 音频 |

#### 输出

| 属性 | 值 |
|------|-----|
| 时长 | 10 秒（当前唯一支持时长） |
| 分辨率 | 1080p（通过 Refiner 阶段） |
| 音频 | 与视频同步生成的音频，mux 到输出文件 |
| 预览阶段分辨率 | 512×896 |
| 精炼阶段分辨率 | 1088×1920（输出可缩放到 1080×1920） |

**Prompt 增强**：训练用的 caption 长且结构化，手写 prompt 会欠用模型能力。提供了 T2V 和 I2V 两个 prompt 增强系统提示词，通过指令遵循 LLM 将原始 prompt 扩展为 JSON caption。

> **出处**：ModelScope MAGI-2-preview README、Sand.ai 技术博客

### 3.3 模型架构

#### 3.3.1 整体架构

MAGI-2 采用**单流（single-stream）架构**，源自 MagiHuman 的设计理念：文本、视频和音频作为统一的 token 序列，由同一个 Transformer backbone 处理，使用纯 self-attention，避免多模态塔和 cross-attention 接口的额外复杂性。

> **出处**：Sand.ai 技术博客 — Section 1: Starting from Single-Stream Architecture；参考论文 [2] *Speed by Simplicity: A Single-Stream Architecture for Fast Audio-Video Generative Foundation Model* (arXiv:2603.21986)

#### 3.3.2 MagiMoE：Ultra-Fine-Grained MoE

| 组件 | 配置 |
|------|------|
| Backbone | 40 层 Transformer |
| 稀疏核心 | 中间 36 层使用 Multi-Head MoE；4 个边界层保持 dense |
| 模型宽度 | 3,072 |
| 路由表示 | 12 heads × 256 dimensions |
| 专家池 | 每个 head 256 个专家 |
| 激活专家 | 每个 head Top-6 |
| Expert FFN | 256 → 1,280 → 256，fused SwiGLU |

**每个 sparse 层**：
- 总 expert 单元数：12 × 256 = 3,072
- 每 token 激活 expert 数：12 × 6 = 72
- 每个 expert 操作 256 维子空间（非全宽度 expert）

**路由机制**：
1. Split projection：将 3,072 维隐藏状态映射到路由表示
2. 分割为 12 个 256 维子空间
3. 每个 head 独立 router，从 256 个专家中选 Top-6
4. 12 个 head 输出拼接后通过 merge projection 映射回 Transformer 隐藏状态

**关键设计**：不是添加必须作为不可分割单元调用的宽 FFN 集合，而是添加可独立选择和重组的低维变换集合。同一 token 的不同表示子空间可以形成不同的专家组合。

> **出处**：Sand.ai 技术博客 — Section 2: MagiMoE

#### 3.3.3 配套系统设计

**Multi-Head LatentMoE + Head Parallel**：

核心思想来自 Multi-Head LatentMoE（arXiv:2602.04870）：每个 latent head 拥有独立的 router 和 expert 集合，使得 Head Parallel 成为可能——sub-token 先沿 head 维度分配，然后每个 head owner 本地执行路由和专家计算。

通信量对比：
- 标准 MoE：`C ∝ N·k·H`（随 Top-k 增长）
- Head Parallel：`C ∝ N·M·H_head = N·H`（不随 Top-k 增长，且可静态预分配缓冲区）

> **出处**：Sand.ai 技术博客 — Section 2.1；参考论文 [6] arXiv:2602.04870

**Hierarchical Head Parallel（分层 Head Parallel）**：
- **跨节点（InfiniBand）**：固定形状的 head slice 交换，非动态 token 复制
- **节点内（NVLink）**：ZeRO/FSDP 式全分片分发专家权重、梯度和优化器状态

**MagiMoE 执行优化**：
- 融合路由和 Top-K → 紧凑的 expert-wise token 布局 → 融合 grouped expert FFN → 路由加权输出合并
- 路由分数 FP32 计算（数值稳定），专家权重和主计算 BF16
- 融合 up/gate 投影与 SwiGLU
- 辅助损失无关的专家 bias 平衡（arXiv:2408.15664）

**MagiMuon 优化器**：
- 保持 head × expert 结构化布局
- Router gates 按 head 分组的矩阵批处理
- Expert up/gate/down 投影沿 head 和 expert 扁平化到 batch 维度
- Muon 正交化独立应用于每个矩阵
- 混合设计：主矩阵参数用 MagiMuon，adapters/mHC/attention sinks/gating 用 Adam

> **出处**：Sand.ai 技术博客 — Section 3

#### 3.3.4 模型组件与权重大小

| 目录 | 大小 | 内容 |
|------|------|------|
| preview | 228 GB | Preview 阶段 Transformer（56 shards） |
| text_encoder | 56 GB | 文本编码器（Qwen3.5-27B） |
| refiner | 14 GB | Refiner 阶段 Transformer（3 shards） |
| stable-audio-open-1.0 | 5 GB | Audio VAE（解码生成的音频 latents） |
| vae | 3 GB | Video VAE（来自 Wan-AI/Wan2.2-TI2V-5B） |
| turbo_vae | 2 GB | 蒸馏 VAE 解码器（默认解码使用） |
| **总计** | **~307 GB** | |

> **出处**：ModelScope MAGI-2-preview README — Checkpoints 章节

### 3.4 推理流程

```
1. 用户输入（文本 prompt / prompt + 静态图像）
       ↓
2. （可选）Prompt 增强：通过指令遵循 LLM 将原始 prompt 扩展为结构化 JSON caption
       ↓
3. Stage 1 — Preview 阶段（magi2_preview）
   - 文本编码器（Qwen3.5-27B）编码 prompt
   - Video VAE（Wan2.2 VAE）处理视觉条件
   - 114B MoE Transformer 在低分辨率（512×896）去噪
   - 生成低分辨率视频 latents + 音频 latents
       ↓
4. Stage 2 — Refiner 阶段（magi2_refiner）
   - 接收 Preview 阶段结果
   - 提升至 1080p（1088×1920）
   - 14B Refiner Transformer 精炼
       ↓
5. VAE 解码
   - 视频解码：turbo_vae（蒸馏 VAE 解码器，时间滑动窗口，单 rank 处理）
   - 音频解码：stable-audio-open-1.0
       ↓
6. 音视频 mux（需要 ffmpeg）→ 输出 MP4 文件（1080p + 音频）
```

**部署要求**：
- 硬件：8× NVIDIA Hopper GPU
- Python 3.12 + CUDA toolkit
- ffmpeg（用于音频 mux）
- Offload 模式：preview/refiner 默认 `roundtrip`（分阶段进出 GPU），因为 1080p 时两者无法同时驻留在 80GB 显卡上
- `MAGI2_DETERMINISTIC=1`：使 MoE scatter 和 attention kernels bit-exact（牺牲速度）

> **出处**：ModelScope MAGI-2-preview README — Running inference 章节

### 3.5 关键技术

1. **Ultra-Fine-Grained MoE**：114B 总参数，6B 激活参数/token。将容量组织为低维专家单元，可独立路由和重组。每个 sparse 层 3,072 个 expert 单元，每 token 激活 72 个。

2. **Multi-Head LatentMoE + Head Parallel**：通信量不随 Top-k 增长，跨设备通信形状可静态确定，实现可预分配的通信边界。

3. **Hierarchical Head Parallel**：InfiniBand 处理固定形状 head slice 交换，NVLink 处理专家状态的全分片分发，分层利用带宽。

4. **MagiMoE 执行系统**：端到端优化路由→token 重排→专家计算→输出合并，专为数千个细粒度专家设计。

5. **MagiMuon 优化器**：匹配 head × expert 结构的矩阵批处理 Muon 正交化，平衡优化器计算。

6. **从过滤到高吞吐生产的数据范式转变**：从"过滤为中心"转向"高吞吐数据生产 + 精确多模态标注"，保留复杂运动、多人交互、镜头切换、长尾主题、字幕、复杂音频关系。

7. **两阶段生成（Preview + Refiner）**：Preview 在低分辨率去噪，Refiner 提升至 1080p。

8. **蒸馏 VAE 解码器（Turbo VAE）**：用于快速解码，时间滑动窗口。

---

## 4. LTX-2.5 技术分析

### 4.1 模型概述

LTX-2.5 是 Lightricks 推出的开源世界模型（open world model），专为本地执行和微调设计。基于 DiT（Diffusion Transformer）架构，可从文本、图像和视频输入生成同步的高保真视频和音频。LTX-2.5 是 LTX-2 的升级版本，论文发表于 arXiv:2601.03233。

> **出处**：[ModelScope LTX-2.5 模型页面](https://www.modelscope.cn/models/Lightricks/LTX-2.5)、[arXiv:2601.03233](https://arxiv.org/abs/2601.03233)、[LTX-2 GitHub](https://github.com/Lightricks/LTX-2)、[LTX 文档](https://docs.ltx.io/open-source-model/getting-started/overview)

### 4.2 输入输出能力支持

#### 输入

| 模式 | 说明 |
|------|------|
| T2V（文本到视频） | 文本提示词 → 视频 + 音频 |
| I2V（图像到视频） | 文本 + 一张或多张图像（指定帧索引和强度） → 视频 + 音频 |
| V2V（视频到视频） | 文本 + 视频输入 |
| 多镜头生成 | 单次生成可产生多个连接镜头，保持角色/场景/光照/风格/声音一致 |

#### 输出

| 属性 | 值 |
|------|-----|
| 最大时长 | 20 秒（论文指出 LTX-2 可达 20 秒，超过 Veo 3 的 12 秒和 Sora 2 的 16 秒） |
| 分辨率 | 通过多尺度推理可达 1080p Full-HD |
| 音频 | 24 kHz 立体声（论文）；LTX-2.5 有独立 Audio VAE + vocoder |
| 帧率 | 24 FPS（推理代码示例） |
| 帧数约束 | `num_frames % 8 == 1`（1, 9, 17, …, 121, …） |
| 尺寸约束 | 宽高必须被 32 整除 |
| 多语言 | 支持 9 种语言：en, de, es, fr, ja, ko, zh, it, pt |

> **出处**：ModelScope LTX-2.5 README、arXiv:2601.03233

### 4.3 模型架构

#### 4.3.1 非对称双流 DiT

LTX-2/2.5 的核心是**非对称双流 Diffusion Transformer**：

| 属性 | LTX-2（论文） | LTX-2.5（模型卡片） |
|------|-------------|-------------------|
| 总参数量 | 19B（14B + 5B） | 22B |
| 视频流 | 14B 参数 | 扩展至约 17B（推测） |
| 音频流 | 5B 参数 | 扩展至约 5B（推测） |
| 架构 | 双流 DiT | 双流 DiT |

> **注**：LTX-2 论文中 Table 1 标注为 19B，结论中提到"extending a pretrained 13B video diffusion transformer with a lightweight 3B audio stream"，但架构章节和摘要明确为 14B video + 5B audio = 19B。LTX-2.5 模型卡片明确标注为 22B-dev-transformer，表明 LTX-2.5 在 LTX-2 基础上进行了扩参数。

**双流 block 操作顺序**（每个 dual-stream block 执行四步）：
1. **Self-Attention**（模态内自注意力）
2. **Text Cross-Attention**（文本条件交叉注意力）
3. **Audio-Visual Cross-Attention**（跨模态交叉注意力）
4. **FFN**（前馈网络）

> **出处**：arXiv:2601.03233 — Section 3.1

#### 4.3.2 位置编码

| 流 | 位置编码 | 说明 |
|----|---------|------|
| 视频流 | 3D RoPE `(x, y, t)` | 空间+时间轴 |
| 音频流 | 1D temporal RoPE | 仅时间轴 |
| 跨模态交互 | 仅 temporal RoPE | 强制跨模态注意力聚焦于时间同步而非空间对齐 |

> **出处**：arXiv:2601.03233 — Section 3.1.1

#### 4.3.3 音视频交叉注意力

- 双向信息流：video queries → audio keys/values，以及反方向
- AdaLN 调制：基于流的 diffusion timestep 独立缩放和偏移 Q 和 (K,V)
- 附加 AdaLN 门：参数依赖另一模态的 timestep，调节跨模态信息整合程度
- 交叉注意力可视化证明模型能空间跟踪移动物体、动态切换说话者注意力、聚焦唇部区域

> **出处**：arXiv:2601.03233 — Section 3.1.2, Figure 3

#### 4.3.4 文本条件化与 Thinking Tokens

**文本编码器**：
- LTX-2：Gemma3-12B（论文）
- LTX-2.5：Custom Gemma 4 12B（模型卡片）

**多层特征提取器（Multi-Layer Feature Extractor）**：
1. 从 LLM 所有 decoder 层提取中间特征（形状 `[B, T, D, L]`）
2. 对每层做 mean-centered scaling
3. 扁平化为 `[B, T, D×L]`
4. 通过可学习投影矩阵 W 投影到目标维度 D
5. W 与 LTX-2 联合优化初始阶段后冻结

**Thinking Tokens**：
- 可学习的 token 集合，追加到输入 token 序列（替换 padded 位置）
- 通过全双向注意力 transformer block（text connector）处理
- 作为全局信息载体，携带聚合的上下文信息或缺失细节
- 每个流（视频/音频）有独立的 text connector

> **出处**：arXiv:2601.03233 — Section 3.2

#### 4.3.5 音频 VAE 与 Latent 空间

| 属性 | 值 |
|------|-----|
| 输入 | 立体声 16 kHz 波形 |
| 表示 | 每声道 mel spectrogram，沿通道维度拼接 |
| Latent token | 每个对应约 1/25 秒音频，128 维特征向量 |
| VAE 类型 | Causal audio VAE |
| Vocoder | HiFi-GAN 架构修改版，双声道联合合成 |
| 输出采样率 | 24 kHz |
| 通道扩展 | 生成器通道数相比 HiFi-GAN V1 翻倍 |

> **出处**：arXiv:2601.03233 — Section 3.3

#### 4.3.6 视频 VAE

- LTX-2：基于 LTX-Video 的时空因果 VAE
- LTX-2.5：提供两种选择：
  - **DiffVAE**：更高质量，更重
  - **Conv VAE**：更快，更轻

#### 4.3.7 LTX-2.5 新增组件

| 组件 | 说明 |
|------|------|
| 新 Diffusion 视频解码器 | 时间步条件化解码，替代 VAE 重建阶段，改善面部/纹理/屏幕文字/运动 |
| Diffusion Fidelity Rendering | 按场景复杂度和预算动态分配压缩率 |
| Duration Predictor | 可选节点，从 prompt 预测片段时长并设置帧数 |
| Prompt Enhancer | 将短 prompt 扩展为丰富的电影指令 |
| Spatial Upscaler x2 | 潜在空间上采样器 |
| Temporal Upscaler x2 | 时间维度上采样器 |
| Distilled Model | 8 步固定调度，CFG=1，保留更多完整模型质量 |
| LoRA / IC-LoRA | 支持微调定制 |

> **出处**：ModelScope LTX-2.5 README、LTX 文档

### 4.4 推理流程

```
1. 用户输入（文本 prompt / prompt + 图像 / prompt + 视频）
       ↓
2. （可选）Prompt Enhancer：短 prompt → 丰富的电影指令
       ↓
3. （可选）Duration Predictor：prompt → 预测片段时长 → 设置帧数
       ↓
4. 文本编码
   - Gemma 4 12B 编码 prompt
   - Multi-Layer Feature Extractor 提取所有层特征
   - Text Connector（含 Thinking Tokens）精炼文本嵌入
       ↓
5. Stage 1 — 基础生成（~0.5 MP）
   - 视频/音频 VAE 编码输入
   - 双流 DiT 联合去噪（视频流 3D RoPE + 音频流 1D RoPE）
   - 跨模态交叉注意力实现音视频同步
   - Modality-CFG：s_t（文本引导）+ s_m（跨模态引导）
       ↓
6. Latent Upscaling
   - 专用 latent upscaler 提升空间分辨率
   - 保持时间一致性和听觉对齐
       ↓
7. Stage 2 — Tiled Refinement（1080p）
   - 上采样 latents 分为重叠的空间和时间 tiles
   - 同一基础模型参数精炼每个 tile
   - latent 空间混合 tiles
       ↓
8. VAE 解码
   - 视频 VAE（DiffVAE 或 Conv VAE）→ 视频帧
   - Audio VAE + Vocoder → 24 kHz 立体声
       ↓
9. 输出：1080p 视频 + 同步立体声音频
```

**蒸馏模型推理**：
- 固定 8 步调度，CFG=1
- 显著减少计算量，保留更多完整模型质量
- 支持 `--quantization fp8-cast` + `--offload cpu`（低 VRAM 模式）

> **出处**：arXiv:2601.03233 — Section 4；ModelScope LTX-2.5 README — Usage 章节

### 4.5 关键技术

1. **非对称双流架构**：视频流（高容量）+ 音频流（窄），按信息密度分配计算资源，不复制视觉 backbone。

2. **双向音视频交叉注意力 + 时间 RoPE**：sub-frame 精度的音视频同步，捕获唇同步和环境声学依赖。

3. **跨模态 AdaLN 门**：一个模态的 scaling/shift 参数由另一模态的隐藏状态条件化，实现去噪过程中的跨模态同步。

4. **Thinking Tokens**：可学习 token 追加到文本序列，通过双向注意力处理，作为全局信息载体，改善复杂 prompt 的语义稳定性和语音准确性。

5. **多层特征提取器**：从 LLM 所有层提取特征而非仅最终层，捕获从语音到语义的层次化语言信息。

6. **Modality-Aware CFG（modality-CFG）**：扩展标准 CFG，引入独立的文本引导项（s_t）和跨模态引导项（s_m），独立控制文本条件和模态间对齐。
   - 视频流：s_t=3, s_m=3
   - 音频流：s_t=7, s_m=3

7. **多尺度多 tile 推理**：基础生成（~0.5MP）→ latent 上采样 → tiled 精炼（1080p），无需单次处理高分辨率 latents。

8. **极高推理效率**：LTX-2（19B，Audio+Video）在 H100 上每步 1.22 秒，比 Wan 2.2-14B（Video-only，22.30 秒/步）快约 18 倍。

9. **原生多镜头生成（LTX-2.5 新增）**：单次生成多个连接镜头，保持角色/场景/光照/风格/声音跨切一致性。

10. **Diffusion Fidelity Rendering（LTX-2.5 新增）**：按场景复杂度动态分配计算，而非锁定单一压缩率。

> **出处**：arXiv:2601.03233 — Section 3-4, Table 1；LTX 文档

---

## 5. 对比分析

### 5.1 架构对比总览

| 维度 | MiniMax-H3 | MAGI-2 Preview | LTX-2.5 |
|------|-----------|----------------|---------|
| **核心范式** | Dense 单流 Transformer | MoE 单流 Transformer | 非对称双流 DiT |
| **总参数** | ~33B | ~114B | ~22B |
| **激活参数/token** | ~33B（dense，~20B 推理时扣除 AdaLN 缓存） | ~6B | ~22B |
| **Transformer 层数** | 未公开（33B dense） | 40 层（4 dense + 36 MoE） | 未公开（双流） |
| **模态融合方式** | 统一 packed 序列 + 单流 self-attention | 统一 token 序列 + 单流 self-attention | 分离双流 + 双向 cross-attention |
| **文本编码器** | Qwen3-VL-32B（第50层） | Qwen3.5-27B | Gemma 4 12B（所有层） |
| **视频 VAE** | H3-VisualVAE (f16t4d24, ViT decoder) | Wan2.2 VAE | DiffVAE / Conv VAE |
| **音频 VAE** | H3-AudioVAE (32kHz, 40Hz latent) | Stable Audio Open 1.0 | Causal Audio VAE (16kHz, 25Hz latent) |
| **位置编码** | MM-RoPE (3D: t,h,w) | 未明确（单流，推测 3D RoPE） | 视频 3D RoPE + 音频 1D RoPE |
| **MoE** | 无（dense） | Ultra-Fine-Grained MoE | 无（dense） |
| **稀疏注意力** | 有（训练阶段，未开源推理） | 未明确 | 无 |
| **精度** | BF16 | BF16 / F32 | BF16（支持 fp8/int8/nvfp4 量化） |

### 5.2 输入输出能力对比

| 维度 | MiniMax-H3 | MAGI-2 Preview | LTX-2.5 |
|------|-----------|----------------|---------|
| **文本输入** | ✅ | ✅ | ✅ |
| **图像输入** | ✅（≤9张） | ✅（1张静态图） | ✅（多张，指定帧和强度） |
| **视频输入** | ✅（≤3段） | ❌ | ✅ |
| **音频输入** | ✅（≤3段） | ❌ | ❌（推理代码中未见） |
| **参考模式** | ✅（Ref2VA） | ❌ | ❌ |
| **首尾帧模式** | ✅（FL2VA） | ❌（仅首帧 I2V） | ✅（多帧条件化） |
| **视频输出** | ✅ | ✅ | ✅ |
| **音频输出** | ✅（32kHz 立体声） | ✅ | ✅（24kHz 立体声） |
| **最大分辨率** | 2K（Regenerate） | 1080p（Refiner） | 1080p（多尺度推理） |
| **最大时长** | 15 秒 | 10 秒（固定） | 20 秒 |
| **多镜头** | ✅（通过 Context-IR） | 未明确 | ✅（原生） |
| **多语言** | 11 种 | 未明确 | 9 种 |
| **LoRA/微调** | 支持（diffusers） | 不支持 | ✅（LoRA/IC-LoRA） |

### 5.3 推理流程对比

| 维度 | MiniMax-H3 | MAGI-2 Preview | LTX-2.5 |
|------|-----------|----------------|---------|
| **预处理** | H3-Context-IR（API，结构化表示） | Prompt 增强 LLM（可选，JSON caption） | Prompt Enhancer（可选） |
| **生成阶段** | H3-Base → H3-Regenerate-2K | Preview → Refiner | Stage 1 → Latent Upscale → Stage 2 Tiled |
| **分辨率提升方式** | In-context 重生成 | Refiner Transformer 精炼 | Latent 上采样 + Tiled 精炼 |
| **GPU 需求** | 4 GPU（SGLang） | 8× Hopper GPU | 消费级 GPU（支持 CPU offload） |
| **CFG 机制** | CFG-distilled（权重已蒸馏） | 未明确 | Modality-CFG（s_t + s_m） |
| **蒸馏模型** | CFG-distilled 权重 | Turbo VAE 解码器 | 8 步蒸馏模型（CFG=1） |
| **推理框架** | SGLang/vLLM/diffusers/ComfyUI | 自研（torchrun） | ltx-pipelines/ComfyUI/Diffusers |
| **开源完整度** | H3-Base 开源，Context-IR/2K 仅 API | 推理代码 + 权重 | 全部开源 |

---

## 6. 关键技术深度对比

### 6.1 模态融合策略

这是三个模型最核心的架构差异：

| 策略 | 模型 | 优势 | 劣势 |
|------|------|------|------|
| **单流统一序列** | MiniMax-H3, MAGI-2 | 模态间信息交换贯穿整个 backbone；无预定义接口瓶颈；架构简单 | 计算量随总 token 数增长；难以按模态差异化分配容量 |
| **非对称双流 + 交叉注意力** | LTX-2.5 | 按信息密度差异化分配容量；支持独立 V2A/A2V 工作流；计算效率高 | 跨模态交互仅发生在特定层；需要额外的 cross-attention 参数 |

**洞察**：
- MiniMax-H3 和 MAGI-2 都选择了单流架构（MAGI-2 引用 MagiHuman 的设计理念），但出发点不同：H3 是为了全模态统一理解和任务泛化，MAGI-2 是为了在 MoE 扩展时保持架构简洁。
- LTX-2.5 选择双流架构是出于效率考虑——"视频和音频有根本不同的信息密度"，非对称设计确保计算资源花在需要的地方。
- LTX-2.5 论文明确指出解耦序列管线（V2A 或 A2V）是次优的，因为它们无法建模两个模态的完整联合分布。双流+交叉注意力是在"完全分离"和"完全融合"之间的中间路线。

### 6.2 扩展策略

| 维度 | MiniMax-H3 | MAGI-2 Preview | LTX-2.5 |
|------|-----------|----------------|---------|
| **参数扩展** | Dense 扩展至 33B | MoE 扩展至 114B（6B 激活） | 双流扩展至 22B |
| **效率优化** | AdaLN 缓存（~13B 可缓存） | MoE 稀疏激活（6B/token） | 非对称容量分配 + 高效 latent 空间 |
| **推理成本** | 中（33B dense，~20B 推理） | 低（6B 激活，但需 8 GPU） | 极低（18× 快于同等模型） |

**洞察**：
- MAGI-2 的核心论点是"视频生成处理大量时空 token 并反复执行去噪网络，天然比文本生成更昂贵"，因此在架构设计阶段就考虑推理效率，而非训练后再优化。114B 总参数提供了大容量，6B 激活参数控制了推理成本。
- MiniMax-H3 的 AdaLN 缓存是一种独特的效率优化——13B 参数在推理时不需要加载，但这仍然是一个 dense 模型，所有 token 都激活所有（非缓存）参数。
- LTX-2.5 的效率来自多个层面：高效 latent 空间（来自 LTX-Video 的设计）、非对称双流（音频流更轻）、以及优化的 latent 压缩机制。

### 6.3 文本条件化策略

| 维度 | MiniMax-H3 | MAGI-2 Preview | LTX-2.5 |
|------|-----------|----------------|---------|
| **文本编码器** | Qwen3-VL-32B | Qwen3.5-27B | Gemma 4 12B |
| **特征提取** | 第 50 层隐藏状态 | 未明确 | 所有层特征（Multi-Layer Feature Extractor） |
| **文本增强** | Context-IR（系统级） | Prompt 增强 LLM（外部） | Thinking Tokens + Text Connector（模型内） |
| **多语言** | 11 种 | 未明确 | 9 种 |

**洞察**：
- MiniMax-H3 使用最大的文本编码器（Qwen3-VL-32B，32B 参数），并通过 Context-IR 系统将复杂多模态指令转换为结构化中间表示。这是一种"系统级"的文本增强。
- LTX-2.5 虽然文本编码器较小（Gemma 4 12B），但通过多层特征提取器（从所有层提取特征）和 Thinking Tokens（模型内可学习 token）补偿。论文指出这对语音生成的音素精度至关重要。
- MAGI-2 的 prompt 增强是外部可选步骤，通过另一个指令遵循 LLM 实现，不是模型架构的一部分。

### 6.4 高分辨率策略

| 维度 | MiniMax-H3 | MAGI-2 Preview | LTX-2.5 |
|------|-----------|----------------|---------|
| **方法** | In-context 重生成 | Refiner Transformer | Latent 上采样 + Tiled 精炼 |
| **基础分辨率** | 768p | 512×896 | ~0.5 MP |
| **目标分辨率** | 2K | 1080p | 1080p |
| **是否复用基础模型** | ✅（同一 H3-Base 模型） | ❌（独立 Refiner 模型，14GB） | ✅（同一基础模型参数） |
| **是否利用原始上下文** | ✅（in-context 格式复用原始多模态上下文） | 未明确 | 未明确 |

**洞察**：
- MiniMax-H3 的 in-context 重生成是最独特的方法——不使用专用超分模块，而是让基础模型自己重新生成，并能复用原始上下文信息恢复传统超分需要"猜测"的细节（如小文字）。这是"任务泛化"的体现。
- LTX-2.5 的多尺度多 tile 策略也复用同一基础模型参数，但通过 latent 空间的分块处理实现高分辨率，更偏向工程优化。
- MAGI-2 使用独立的 Refiner 模型（14GB，远小于 Preview 的 228GB），是一种轻量级精炼策略。

### 6.5 音频生成策略

| 维度 | MiniMax-H3 | MAGI-2 Preview | LTX-2.5 |
|------|-----------|----------------|---------|
| **音频表示** | H3-AudioVAE (32kHz, 40Hz latent) | Stable Audio Open 1.0 | Causal Audio VAE (16kHz input, 25Hz latent, 24kHz output) |
| **声道** | 立体声（左右独立编码后重组） | 与视频同步 | 立体声（双声道 mel spectrogram） |
| **音频-视频同步** | 统一序列 self-attention | 统一序列 self-attention | 双向 cross-attention + temporal RoPE |
| **语音生成** | ✅（通过 `<d>` 对话 token） | 未明确 | ✅（phonetic 精度，唇同步） |
| **音频类型** | 对话 + 环境声 + 非叙事音乐 | 未明确 | 语音 + foley + 环境声 + 音乐 |

**洞察**：
- MiniMax-H3 的 Context-IR 输出明确包含 `overall_soundscape`（环境声）和 `non_diegetic_music`（非叙事音乐）字段，以及 `<d>[语言] 对话内容</d>` 格式的对话指令，表明音频生成是被系统级精确定向的。
- LTX-2.5 论文详细描述了音频-视频交叉注意力的机制和可视化（Figure 3），证明了 sub-frame 级别的同步能力，以及 Modality-CFG 对跨模态对齐的控制。
- MAGI-2 使用现成的 Stable Audio Open 1.0 作为音频 VAE，视频 VAE 来自 Wan2.2，体现了"复用已有组件"的策略，重心在 MoE 架构本身。

### 6.6 数据策略

| 维度 | MiniMax-H3 | MAGI-2 Preview | LTX-2.5 |
|------|-----------|----------------|---------|
| **核心理念** | 任务泛化导向的预训练 | 从过滤转向高吞吐生产+精确标注 | 视频字幕系统捕获完整声景 |
| **标注内容** | 结构化多模态描述（通过 Context-IR） | 主体/场景/动作/镜头/对话/环境声/字幕 | 动作/外观/声音/对话/说话者/语言/口音 |
| **数据过滤** | 未明确 | 明确提出"数据过滤陷阱"，减少过滤 | 子集聚焦于含丰富音频的视频 |
| **涌现能力** | 任务泛化（in-context 2K 重生成） | 多镜头身份一致、字幕效果联合出现 | 多镜头一致性、跨切声音一致 |

**洞察**：
- MAGI-2 提出了最深度的数据方法论论述——"数据过滤陷阱"：当训练数据持续简化以适应当前模型能力时，数据本身就成为了更强模型的能力天花板。这是对视频生成领域数据策略的重要反思。
- 三个模型都认识到"保留复杂关系"的重要性：MAGI-2 保留多镜头/字幕/复杂音频关系，LTX-2.5 开发了详尽的字幕系统捕获完整声景，MiniMax-H3 通过 Context-IR 精确描述声景和音乐。

---

## 7. 洞察与结论

### 7.1 三种技术路线的本质差异

这三个模型代表了多模态音视频生成的三种不同技术哲学：

**1. MiniMax-H3 — "系统主义"路线**
- 核心信念：多模态生成的关键不在模型架构本身，而在系统设计。通过 Context-IR 将复杂多模态指令精炼为结构化中间表示，通过 in-context 重生成实现高分辨率，通过任务泛化实现新能力。
- 架构选择：最简洁的 dense 单流 Transformer，但系统最复杂（三模块协作）。
- 独特价值：全模态理解+生成，支持最丰富的输入组合（图像+视频+音频+参考），2K 输出。

**2. MAGI-2 Preview — "扩展主义"路线**
- 核心信念：视频生成模型需要更大的容量，但 dense 扩展会遇到训练基础设施和推理成本的双重约束。MoE 是解耦总容量和单 token 计算的通用结构。
- 架构选择：114B MoE，6B 激活，Ultra-Fine-Grained 专家设计，配套分布式系统。
- 独特价值：验证了 100B 级视频 MoE 的可行性，提出了数据过滤陷阱的重要警示。

**3. LTX-2.5 — "效率主义"路线**
- 核心信念：视频和音频有根本不同的信息密度，应该差异化分配计算资源。通过非对称双流和跨模态注意力，在保持质量的同时实现极高推理效率。
- 架构选择：非对称双流 DiT（14B 视频 + 5B 音频 → 22B），多层文本条件化。
- 独特价值：18× 快于同等模型，最长时长（20 秒），完全开源，支持消费级 GPU。

### 7.2 共同趋势

1. **音视频联合生成成为标准**：三个模型都生成同步的音频和视频，而非先生成视频再"填充"音频。这反映了对联合分布建模重要性的共识。

2. **结构化 Prompt 是质量关键**：MiniMax-H3 的 Context-IR、MAGI-2 的 prompt 增强 LLM、LTX-2.5 的 prompt enhancer——三个模型都认识到手写短 prompt 会欠用模型能力，需要结构化扩展。

3. **多阶段生成策略**：三个模型都采用基础生成+分辨率提升的两阶段策略（H3-Base→Regenerate、Preview→Refiner、Stage 1→Stage 2 Tiled），而非单阶段直接生成高分辨率。

4. **预训练 LLM 作为文本编码器**：三个模型都使用大型预训练 LLM（Qwen3-VL-32B、Qwen3.5-27B、Gemma 4 12B）作为文本编码器，而非从头训练。

5. **复用已有 VAE 组件**：MAGI-2 直接使用 Wan2.2 VAE 和 Stable Audio Open 1.0；LTX-2.5 基于 LTX-Video 的 latent 空间；MiniMax-H3 的 AudioVAE 灵感来自 VA-VAE。

### 7.3 关键差异维度

| 差异维度 | MiniMax-H3 | MAGI-2 Preview | LTX-2.5 |
|---------|-----------|----------------|---------|
| **MoE vs Dense** | Dense | MoE（核心创新） | Dense |
| **单流 vs 双流** | 单流 | 单流 | 双流 |
| **系统完整度** | 最完整（三模块系统） | 中（两阶段+配套系统） | 最简洁（单模型+多尺度推理） |
| **开源程度** | 部分（Base 开源，IR/2K 仅 API） | 完整（代码+权重） | 完全（代码+权重+论文+文档） |
| **推理效率** | 中 | 中（6B 激活但需 8 GPU） | 最高（消费级 GPU 可用） |
| **输入丰富度** | 最高（图/视频/音频/参考） | 最低（文本/图像） | 中（文本/图像/视频） |
| **输出分辨率** | 最高（2K） | 中（1080p） | 中（1080p） |
| **输出时长** | 15 秒 | 10 秒（固定） | 20 秒 |

### 7.4 技术发展趋势判断

1. **MoE 在视频生成中的潜力**：MAGI-2 验证了 100B 级视频 MoE 的可行性。Ultra-Fine-Grained MoE 的"低维专家独立重组"理念，可能成为视频生成模型扩展的主流方向。但 MAGI-2 也坦诚这只是"中间研究发布"，更多 scaling 曲线和消融实验待后续。

2. **系统级多模态理解的价值**：MiniMax-H3 的 Context-IR 表明，对于复杂多模态指令（如"参考视频+音频音色+对话内容"），系统级的上下文理解比模型架构本身更重要。这可能推动更多"预处理系统+生成模型"的分层架构。

3. **效率优先的设计哲学**：LTX-2.5 的 18× 加速表明，通过架构设计（非对称双流、高效 latent 空间）而非仅靠模型压缩，可以实现显著的推理效率提升。在端侧部署和实时生成场景中，这种路线有独特优势。

4. **数据策略的范式转变**：MAGI-2 提出的"数据过滤陷阱"是对整个领域的警示——随着模型规模增长，数据策略必须从"适应当前模型"转向"保留真实世界的复杂关系并精确标注"，否则更大的模型仍只能看到简化后的世界。

5. **跨模态同步的技术分化**：单流架构（H3、MAGI-2）通过统一序列 self-attention 实现模态融合，双流架构（LTX-2.5）通过跨模态 cross-attention + temporal RoPE 实现同步。两种路线的 trade-off（信息交换深度 vs 计算效率）仍需更多消融实验验证。

---

## 附录：资料来源汇总

| 模型 | 资料类型 | 来源 |
|------|---------|------|
| MiniMax-H3 | ModelScope 模型页面 | https://www.modelscope.cn/models/MiniMax/MiniMax-H3 |
| MiniMax-H3 | ModelScope Markdown | https://www.modelscope.cn/models/MiniMax/MiniMax-H3.md |
| MiniMax-H3 | GitHub | https://github.com/MiniMax-AI/MiniMax-H3 |
| MiniMax-H3 | HuggingFace | https://huggingface.co/MiniMaxAI/MiniMax-H3 |
| MAGI-2 Preview | ModelScope 模型页面 | https://www.modelscope.cn/models/sand-ai/MAGI-2-preview |
| MAGI-2 Preview | HuggingFace | https://huggingface.co/sand-ai/MAGI-2-preview |
| MAGI-2 Preview | 技术博客 | https://sand.ai/blog/magi-2-preview |
| MAGI-2 Preview | 参考论文 [1] | arXiv:2505.13211 (Magi-1) |
| MAGI-2 Preview | 参考论文 [2] | arXiv:2603.21986 (MagiHuman) |
| MAGI-2 Preview | 参考论文 [6] | arXiv:2602.04870 (Multi-Head LatentMoE) |
| LTX-2.5 | ModelScope 模型页面 | https://www.modelscope.cn/models/Lightricks/LTX-2.5 |
| LTX-2.5 | HuggingFace | https://huggingface.co/Lightricks/LTX-2.5 |
| LTX-2.5 | arXiv 论文 | https://arxiv.org/abs/2601.03233 |
| LTX-2.5 | arXiv HTML 全文 | https://arxiv.org/html/2601.03233v1 |
| LTX-2.5 | GitHub | https://github.com/Lightricks/LTX-2 |
| LTX-2.5 | 官方文档 | https://docs.ltx.io/open-source-model/getting-started/overview |

---

*本报告基于截至 2026-11-03 可获取的公开资料编写。部分架构细节（如 MiniMax-H3 的 Transformer 层数、LTX-2.5 的精确双流参数分配）在公开资料中未完全披露，已在相应位置标注。*

---

## 8. 深度架构问题分析

> 本章节针对报告中的核心架构概念进行深入解释，涵盖单流/多流架构原理、模态特定参数机制、注意力结构等。

### 8.1 单流与多流架构的概念解释

#### 8.1.1 "流"（Stream）的定义

在多模态生成模型中，**"流"指的是模型骨干网络中处理特定模态 token 的独立处理路径**。一个"流"包含：

- **一组连续的 Transformer 层**，专门处理某种（或所有）模态的 token
- **独立的参数集**（权重矩阵），可以被该流独占或与其他流共享
- **特定的序列空间**，token 在该空间内做 self-attention
- **可选的独立宽度（维度）和深度（层数）**

#### 8.1.2 单流架构（Single-stream）

**定义**：所有模态（文本、视频、音频）的 token 被拼接（packed）成一个统一序列，通过**同一个 Transformer 骨干网络**处理。所有 token 在同一个序列空间中通过 self-attention 相互交互。

**结构示意**：
```
[文本token_1, 文本token_2, ..., 视频token_1, ..., 音频token_1, ...]
                              ↓
                    ┌─────────────────┐
                    │  统一 Transformer │  ← 一个骨干网络处理所有token
                    │  (Self-Attention) │  ← 所有token互相attend
                    │       FFN        │
                    └─────────────────┘
                              ↓
                    [统一hidden states]
                              ↓
                    分离解码视频和音频
```

**实例**：
- **MiniMax-H3**：H3-Omni-Transformer 是一个 33B dense 单流 Transformer。文本（经 Qwen3-VL-32B 编码）、视频（经 VisualVAE 编码）、音频（经 AudioVAE 编码）的 token 被 packed 成一个统一序列，全部通过同一个 Transformer 的 self-attention 处理。一个视频 token 可以直接 attend 到一个音频 token 或文本 token。
- **MAGI-2 Preview**：同样采用单流架构（源自 MagiHuman 设计），文本、视频、音频作为统一 token 序列，由同一个 114B MoE Transformer backbone 处理。

> **出处**：MiniMax-H3 README — "organizes the encoded representations into a unified packed multimodal sequence"；Sand.ai 技术博客 — "a single Transformer to process text, video, and audio as one unified token sequence using self-attention only"

#### 8.1.3 多流架构（Multi-stream / Dual-stream）

**定义**：不同模态有各自**独立的 Transformer 骨干网络（流）**，各模态 token 在各自的序列空间中独立处理，通过 **cross-attention 层**在特定位置交换信息。

**结构示意**：
```
文本嵌入                        文本嵌入
    ↓                               ↓
    ↓ (cross-attn)           (cross-attn) ↓
┌──────────┐    (AV cross-attn)    ┌──────────┐
│ 视频流    │ ←──────────────────→ │ 音频流    │
│ Self-Attn │    (双向cross-attn)   │ Self-Attn │
│   FFN     │                      │   FFN     │
│ 14B参数   │                      │ 5B参数    │
└──────────┘                       └──────────┘
    ↓                                   ↓
视频latents                         音频latents
    ↓                                   ↓
Video VAE解码                      Audio VAE解码
```

**实例**：
- **LTX-2.5**：非对称双流 DiT，14B 视频流 + 5B 音频流。视频 token 在视频流的序列空间内做 self-attention，音频 token 在音频流的序列空间内做 self-attention。两者通过双向 cross-attention 关联。

> **出处**：arXiv:2601.03233 — "an asymmetric dual-stream transformer with a 14B-parameter video stream and a 5B-parameter audio stream, coupled through bidirectional audio-video cross-attention layers"

#### 8.1.4 在模型结构和推理过程中的含义

| 维度 | 单流 | 多流 |
|------|------|------|
| **模型结构** | 一个共享 Transformer backbone | 多个独立 Transformer backbone |
| **序列空间** | 所有模态 token 在同一序列 | 各模态 token 在各自独立的序列 |
| **模态交互** | self-attention 中直接交互（全深度） | cross-attention 在特定层交互 |
| **参数分配** | 所有模态共享同一组 attention/FFN 参数 | 各流有独立参数，可差异化容量 |
| **推理过程** | 所有 token 流过同一条路径 | 各流可并行计算，通过 cross-attention 同步 |
| **计算特点** | 序列长度 = 所有模态 token 总和 | 各流序列长度独立，但需额外 cross-attention 开销 |

---

### 8.2 MiniMax-H3 模态特定参数与 AdaLN 机制详解

#### 8.2.1 AdaLN（Adaptive Layer Normalization）是什么

AdaLN 是 Diffusion Transformer（DiT）中的核心条件化组件，用于将条件信号（如扩散时间步 embedding）注入到 Transformer 层中，控制每一层的归一化参数。

**标准 DiT 中的 AdaLN-Zero 公式**：
```
输入：条件信号 c（通常是时间步 embedding）
     隐藏状态 x

过程：
  γ₁, β₁, α₁, γ₂, β₂, α₂ = Linear(SiLU(Linear(c)))   ← 6×d 个参数
  x = x + α₁ · (γ₁ · LayerNorm(Attention(x)) + β₁)     ← 调制 self-attention 输出
  x = x + α₂ · (γ₂ · LayerNorm(FFN(x)) + β₂)           ← 调制 FFN 输出

其中：
  γ = scale（缩放）
  β = shift（偏移）
  α = gate（门控，初始为0，实现zero-init）
```

AdaLN 的作用是：**根据当前扩散步骤的条件，动态调整每一层 attention 和 FFN 输出的缩放和偏移**。这使得模型在不同去噪阶段表现出不同的行为。

> **出处**：Peebles & Xie, "Scalable Diffusion Models with Transformers", arXiv:2212.09748

#### 8.2.2 模态特定 AdaLN 的含义

MiniMax-H3 的 AdaLN 是**模态特定的**，这意味着：

```
对于视频 token：
  γ₁ᵛ, β₁ᵛ, α₁ᵛ, γ₂ᵛ, β₂ᵛ, α₂ᵛ = MLP_video(c)    ← 视频专用投影层

对于音频 token：
  γ₁ᵃ, β₁ᵃ, α₁ᵃ, γ₂ᵃ, β₂ᵃ, α₂ᵃ = MLP_audio(c)    ← 音频专用投影层

对于文本 token：
  γ₁ᵗ, β₁ᵗ, α₁ᵗ, γ₂ᵗ, β₂ᵗ, α₂ᵗ = MLP_text(c)     ← 文本专用投影层
```

**关键点**：
- 不同模态的 token，即使在同一扩散时间步，也获得**不同的 scale/shift/gate 参数**
- 每层每模态都有独立的 MLP 投影层，这就是 ~13B AdaLN 参数的来源
- 模型通过知道每个 token 属于哪种模态来选择对应的 AdaLN 参数集

**~13B 参数的构成估算**：
- 假设 d_model ≈ 6144（33B 模型的典型宽度），N_layers 层，3 种模态
- 每层每模态的 AdaLN 参数 ≈ 2 × d_model × d_model × 6（两层 Linear，输出 6×d）
- 总量 ≈ N_layers × 3 × 2 × d² × 6 ≈ 数十亿级参数

> **出处**：MiniMax-H3 README — "approximately 13B parameters residing in AdaLN-related branches. Because the AdaLN modulation outputs can be precomputed and cached, these parameters do not need to be loaded for inference-only deployment."

#### 8.2.3 输入/输出层的模态特定处理

**输入层**（模态进入 Transformer 的接口）：

不同模态有不同的线性投影矩阵，将各自编码器的输出映射到 Transformer 的统一隐藏维度 d_model：

| 模态 | 编码器输出 | 投影路径 |
|------|-----------|---------|
| 文本 | Qwen3-VL-32B 第 50 层隐藏状态（维度 d_qwen） | W_text: d_qwen → d_model |
| 视频 | VisualVAE 24 通道 latent → patchify(1×2×2) → flatten（维度 24×4=96） | W_video: 96 → d_model |
| 音频 | AudioVAE latent（维度 d_audio） | W_audio: d_audio → d_model |

**输出层**（模态离开 Transformer 的接口）：

不同模态有不同的输出投影矩阵，从 d_model 映射回各自的 latent 空间：

| 模态 | 投影路径 | 后续解码 |
|------|---------|---------|
| 视频 | W_video_out: d_model → 24 通道 | 反 patchify → VisualVAE ViT 解码器 → 视频帧 |
| 音频 | W_audio_out: d_model → audio latent 维度 | AudioVAE 解码器 → 立体声音频 |
| 文本 | 无输出投影（文本 token 仅作条件输入，不需解码） | — |

#### 8.2.4 模型如何标识不同模态的 token

这是单流架构的核心问题：在一个 packed 序列中，模型如何知道哪些 token 是视频、哪些是音频、哪些是文本？

MiniMax-H3 通过以下**多重机制**实现模态标识：

**1. MM-RoPE（多模态旋转位置编码）—— 主要标识机制**

三维位置 `(t, h, w)` 自然区分不同模态：
- **视频 token**：`(t_frame, h_spatial, w_spatial)` — 有真实的空间坐标（像素位置映射）和时间坐标（帧索引），h 和 w 维度有有意义的变异
- **音频 token**：`(t_audio, h_placeholder, w_placeholder)` — 只有时间坐标有意义（对应 40Hz 的音频时间戳），空间维度使用占位值或固定值
- **文本 token**：映射到 3D 空间的特定方案 — 可能使用 1D 序列位置扩展到 3D，或特定的时间-空间映射

这种位置编码的差异使得 self-attention 中的 Q·K 点积天然反映模态差异——视频 token 之间的空间位置关系会产生特定的 attention 模式，而音频 token 不会。

**2. Packed 序列结构**

序列有固定的模态排列顺序，例如：
```
[BOS] [文本tokens] [SEP] [视频tokens] [SEP] [音频tokens] [EOS]
```
模型在训练中学习到序列中不同区段的 token 具有不同的统计特性。

**3. 特殊 Token**

H3-Encoder 的 tokenizer 添加了特殊 token（如 `<d>` 标记对话内容），帮助模型识别特定内容类型和模态边界。

**4. 输入投影的隐式标识**

不同模态经过不同的投影矩阵（W_text, W_video, W_audio），产生的 token 表示**天然携带模态信息**——即使维度相同（d_model），不同模态 token 的表示分布特征不同，模型通过训练学会区分。

**5. AdaLN 的显式模态选择**

AdaLN 分支是模态特定的，这意味着模型需要一种机制来"查询"每个 token 的模态类型，然后选择对应的 AdaLN 参数。这可能通过：
- 序列中的位置索引（已知哪些位置是哪种模态）
- 模态类型 embedding（类似 BERT 的 segment embedding，在输入时添加可学习的模态标识向量）

> **出处**：MiniMax-H3 README — "The model uses three-dimensional Multimodal Rotary Position Embeddings (MM-RoPE) to represent positional relationships across the temporal and two spatial dimensions, (t, h, w)."

#### 8.2.5 设计理念总结

MiniMax-H3 的核心设计理念可以概括为：

> **"思考"是模态无关的，"接口"是模态感知的。**

- **模态无关部分**（共享）：Self-Attention 的 Q/K/V 投影矩阵、FFN 的权重 — 所有模态共享同一组参数
  - 优势：任何模态的 token 可以 attend 到任何其他模态的 token，实现深度跨模态交互
  - 参数效率：不因模态数量增加而膨胀
- **模态感知部分**（特定）：输入投影、输出投影、AdaLN 分支 — 每种模态有独立参数
  - 作用：确保每种模态以适合自身特性的方式进入/离开/被条件化
  - 成本：~13B AdaLN 参数，但可缓存优化

这种设计在保持跨模态交互深度的同时，以最低成本实现模态区分。

---

### 8.3 多流架构的注意力机制与推理流程详解

#### 8.3.1 核心问题回答

**是的，理解正确。** 在多流架构中，不同模态的 token 在各自独立的序列空间中做 self-attention 计算，然后通过交叉注意力（cross-attention）机制关联起来。

以 LTX-2.5 为例：
- 视频 token 在**视频流的序列空间**中做 self-attention（视频 token 之间互相 attend）
- 音频 token 在**音频流的序列空间**中做 self-attention（音频 token 之间互相 attend）
- 两者通过**双向 cross-attention** 在每个 dual-stream block 中交换信息

#### 8.3.2 多流推理流程详解（以 LTX-2.5 为例）

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
阶段 1：编码与准备
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  文本 prompt
      ↓
  Gemma 4 12B 编码（因果注意力，decoder-only LLM）
      ↓
  Multi-Layer Feature Extractor（提取所有层特征 → 投影到统一维度）
      ↓
  Text Connector（双向注意力 Transformer blocks + Thinking Tokens）
      ↓
  文本嵌入序列 T = [t₁, t₂, ..., tₙ]
      ↓                                      ↓
  视频条件（可选）                    音频条件（可选）
      ↓                                      ↓
  Video VAE 编码                     Audio VAE 编码
      ↓                                      ↓
  video latents V₀                   audio latents A₀
      ↓                                      ↓
  投影到视频流宽度                  投影到音频流宽度
      ↓                                      ↓
  V₀ ∈ R^{N_v × d_v}               A₀ ∈ R^{N_a × d_a}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
阶段 2：双流 DiT 去噪（重复 N 步，每步一个 dual-stream block stack）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  对于每个去噪步骤和每个 dual-stream block：

  ┌─── 视频流（14B 参数）──────────┐    ┌─── 音频流（5B 参数）──────────┐
  │                                │    │                                │
  │  ① Self-Attention              │    │  ① Self-Attention              │
  │  视频 token 互相 attend        │    │  音频 token 互相 attend        │
  │  位置编码：3D RoPE (x,y,t)    │    │  位置编码：1D temporal RoPE    │
  │  Q_v · K_v → Attention_v       │    │  Q_a · K_a → Attention_a       │
  │         ↓                      │    │         ↓                      │
  │  ② Text Cross-Attention        │    │  ② Text Cross-Attention        │
  │  Q=视频, K/V=文本嵌入 T         │    │  Q=音频, K/V=文本嵌入 T         │
  │  视频token attend 到文本        │    │  音频token attend 到文本        │
  │         ↓                      │    │         ↓                      │
  │  ③ AV Cross-Attention (V→A)   │←──→│  ③ AV Cross-Attention (A→V)   │
  │  Q=视频, K/V=音频流 hidden      │    │  Q=音频, K/V=视频流 hidden      │
  │  ←─────────────────────────────│────│─────────────────────────────→ │
  │  位置编码：仅 temporal RoPE     │    │  位置编码：仅 temporal RoPE     │
  │  （强制时间对齐，非空间对齐）   │    │  （强制时间对齐，非空间对齐）   │
  │  AdaLN 门控：参数依赖音频       │    │  AdaLN 门控：参数依赖视频       │
  │  的 timestep                    │    │  的 timestep                    │
  │         ↓                      │    │         ↓                      │
  │  ④ FFN                         │    │  ④ FFN                         │
  │  前馈网络精炼                   │    │  前馈网络精炼                   │
  │                                │    │                                │
  └────────────────────────────────┘    └────────────────────────────────┘

  关键细节：
  - 两个流并行计算 self-attention（各自独立的序列空间）
  - Cross-attention 时双向交换信息：V→A 和 A→V 同时发生
  - Cross-modality AdaLN：一个流的调制参数受另一流的 timestep 条件化
  - RMSNorm 在各操作间交错，稳定激活

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
阶段 3：Modality-CFG 引导（推理时）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  对于每个流，计算三种前向传播：
  ① 完整条件：M(x, t, m) — 文本 + 跨模态都有
  ② 无文本条件：M(x, ∅, m) — 去除文本
  ③ 无跨模态条件：M(x, t, ∅) — 去除另一模态

  引导预测：
  M̂(x,t,m) = M(x,t,m)
            + s_t · [M(x,t,m) - M(x,∅,m)]      ← 文本引导项
            + s_m · [M(x,t,m) - M(x,t,∅)]      ← 跨模态引导项

  视频流：s_t=3, s_m=3
  音频流：s_t=7, s_m=3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
阶段 4：解码
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  视频流输出 → 输出投影 → video latents → Video VAE (DiffVAE) → 视频帧
  音频流输出 → 输出投影 → audio latents → Audio VAE + Vocoder → 24kHz 立体声
```

> **出处**：arXiv:2601.03233 — Section 3.1, 3.1.2, 4.1

#### 8.3.3 多流与单流的关键区别

| 维度 | 单流（H3, MAGI-2） | 多流（LTX-2.5） |
|------|-------------------|----------------|
| Self-attention 序列 | 所有模态 token 在同一序列 | 各模态在独立序列 |
| 跨模态交互深度 | 每层每刻都在交互（通过 self-attention） | 仅在 cross-attention 层交互 |
| 模态间信息交换方式 | 隐式（Q·K 点积中自然发生） | 显式（专门的 cross-attention 层） |
| 容量分配 | 所有模态共享同一组参数 | 可按模态差异化分配（视频流 > 音频流） |
| 计算量 | O((N_text+N_video+N_audio)²) | O(N_video² + N_audio²) + cross-attn 开销 |

---

### 8.4 单流架构的 token 融合与解码机制详解

#### 8.4.1 多模态 token 如何融合成一条流

以 MiniMax-H3 为例，融合过程分为三步：

**第一步：各模态独立编码**

```
文本输入                         视频输入                      音频输入
    ↓                               ↓                              ↓
H3-Encoder                     H3-VisualVAE                  H3-AudioVAE
(Qwen3-VL-32B)                 (f16t4d24)                    (32kHz → 40Hz)
    ↓                               ↓                              ↓
第50层隐藏状态                  24通道 latents                 audio latents
维度: d_qwen                   空间压缩16×, 时间压缩4×         时间压缩至40Hz
    ↓                               ↓                              ↓
patchify (1×2×2)
    ↓
flatten: 24×1×2×2 = 96维
```

**第二步：投影到统一维度**

所有模态的编码输出通过**模态特定的线性投影矩阵**映射到 Transformer 的统一隐藏维度 d_model：

```
文本:  h_text  = W_text · hidden_states_50th_layer    ∈ R^{N_text × d_model}
视频:  h_video = W_video · flatten(patchify(vae_latent)) ∈ R^{N_video × d_model}
音频:  h_audio = W_audio · audio_vae_latent            ∈ R^{N_audio × d_model}
```

此时，所有 token 的维度相同（d_model），但**来源不同，表示分布特征不同**。

**第三步：Packed 成统一序列**

所有 token 按特定顺序拼接成一个长序列：

```
序列 = [文本tokens] [视频tokens] [音频tokens]
        ↑             ↑             ↑
     条件输入        待生成         待生成
     (不预测)       (预测目标)     (预测目标)

总长度 = N_text + N_video + N_audio
```

#### 8.4.2 序列中的模态标记机制

**是的，序列中会标记不同模态的 token**，但不是通过显式的"标签"，而是通过以下多重隐式和显式机制：

**1. MM-RoPE 位置编码（主要机制）**

每个 token 被分配一个 3D 位置 `(t, h, w)`，不同模态的位置模式天然不同：

```
文本 token：   位置如 (0, seq_idx, 0) 或特定映射方案
              → 空间维度无真实变异，模型可从位置模式识别为文本

视频 token：   位置如 (frame_t, pixel_h, pixel_w)
              → 空间维度有真实的网格结构，时间维度对应帧序列
              → 例如 768p/32×4 = 6×12 = 72 个空间位置 × N帧
              → 模型从 3D 位置模式识别为视频

音频 token：   位置如 (audio_t, 0, 0) 或 (audio_t, fixed_h, fixed_w)
              → 空间维度无变异（立体声左右声道可能有区分），时间维度对应 40Hz
              → 模型从位置模式识别为音频
```

**2. 输入投影的隐式标识**

不同模态经过不同的投影矩阵（W_text ≠ W_video ≠ W_audio），即使输出维度相同，token 的表示空间分布也不同。模型在训练中学会从表示特征区分模态。

**3. 特殊 Token 标记**

H3 的 tokenizer 包含特殊 token：
- `<d>` 标记对话内容（如 `<d>[English] Follow the wind, live free.</d>`）
- 可能的模态分隔 token 标记不同模态区段的边界

**4. AdaLN 的模态特定选择**

AdaLN 分支是模态特定的，模型需要知道每个 token 的模态来选择对应的 AdaLN 参数。这通过上述位置/投影/特殊 token 机制的综合信息实现。

#### 8.4.3 推理过程中的模态特定处理

在单流 Transformer 的每一层中，处理流程如下：

```
对于 packed 序列中的每个 token：

  1. Self-Attention（模态无关）
     ─────────────────────────────
     所有 token（不分模态）的 Q, K, V 使用相同的投影矩阵
     每个 token attend 到所有其他 token（全双向）
     → 视频token 可以 attend 到音频token（获取声音线索）
     → 音频token 可以 attend 到视频token（获取视觉上下文）
     → 文本token 可以 attend 到所有 token（提供语义条件）
     
     注意：虽然 attention 参数共享，但 MM-RoPE 使得不同模态
     token 的 attention 模式自然不同（位置编码影响 Q·K 点积）

  2. AdaLN 调制（模态特定）
     ─────────────────────────────
     根据 token 的模态类型，选择对应的 AdaLN 参数：
     → 视频token: γ₁ᵛ, β₁ᵛ, α₁ᵛ · LN(Attention输出) + β₁ᵛ
     → 音频token: γ₁ᵃ, β₁ᵃ, α₁ᵃ · LN(Attention输出) + β₁ᵃ
     → 文本token: γ₁ᵗ, β₁ᵗ, α₁ᵗ · LN(Attention输出) + β₁ᵗ
     
     → 不同模态的 attention 输出被不同地缩放和偏移

  3. FFN（模态无关）
     ─────────────────────────────
     所有 token 使用相同的 FFN 权重
     但 FFN 输入已经经过 AdaLN 调制，因此不同模态的实际变换不同
     
     → FFN 输出也经过模态特定的 AdaLN 调制
```

#### 8.4.4 从同一流解码视频和音频

Transformer 处理完成后，输出是所有 token 的统一 hidden states。解码过程基于 **packed 序列的位置信息**分离不同模态：

```
Transformer 输出: [h₁, h₂, ..., h_{N_text}, h_{N_text+1}, ..., h_{N_text+N_video}, ...]
                    ←── 文本 ──→  ←────── 视频 ──────→  ←──── 音频 ────→

  文本 token 的 hidden states:
    → 通常丢弃（文本仅作条件输入，不需要生成输出）
    → 或用于其他下游任务

  视频 token 的 hidden states [h_{N_text+1}, ..., h_{N_text+N_video}]:
    → 模态特定输出投影: W_video_out · h → 24通道 visual latent
    → 反 patchify: (1×2×2) patch 重组为完整的 latent 网格
    → H3-VisualVAE ViT 解码器: latent → 像素空间视频帧
    → 输出: 24 FPS 视频帧序列

  音频 token 的 hidden states [h_{N_text+N_video+1}, ..., h_{end}]:
    → 模态特定输出投影: W_audio_out · h → audio latent
    → H3-AudioVAE 解码器: latent → 波形
    → 左右声道独立解码后重组
    → 输出: 32 kHz 立体声音频
```

**关键设计：晚绑定（Late Binding）**

模态在输入端（编码器+投影）和输出端（投影+解码器）是分离的，但在 Transformer 主体（attention+FFN）中是完全融合的。这意味着：
- **融合发生在中间**：跨模态信息交换通过共享的 self-attention 自然发生
- **分离发生在两端**：输入投影确保每种模态以适合自身的方式进入，输出投影确保以正确格式离开
- **AdaLN 是中间的模态锚点**：在融合的处理过程中，AdaLN 提供模态特定的调制，防止模态信息在深层处理中完全丢失

> **出处**：MiniMax-H3 README — H3-Base Architecture Overview, H3-Encoder, H3-VAE, H3-Omni-Transformer 章节

---

### 8.5 注意力结构与稀疏机制详解

#### 8.5.1 三个模型的注意力结构总览

| 组件 | MiniMax-H3 | MAGI-2 Preview | LTX-2.5 |
|------|-----------|----------------|---------|
| **主 Transformer** | 双向 self-attention | 双向 self-attention | 双向 self-attention + 双向 cross-attention |
| **注意力方向** | 双向（非因果） | 双向（非因果） | 双向（非因果） |
| **稀疏 attention** | 有（训练阶段，未开源推理） | MagiAttention 库支持异构 mask | 无 |
| **文本编码器** | 因果（Qwen3-VL-32B） | 因果（Qwen3.5-27B） | 因果（Gemma 4 12B），用双向 Text Connector 补偿 |
| **Video VAE** | 时间因果 | 时间因果（Wan2.2 VAE） | 时空因果 |
| **Audio VAE** | 时间因果 | 因果（Stable Audio Open） | 因果 |

#### 8.5.2 为什么主 Transformer 使用双向注意力

三个模型的主 Transformer 都使用**双向（非因果）注意力**，原因是它们都是 **Diffusion 模型**：

- **Diffusion 模型的工作方式**：从噪声开始，通过多步去噪逐步生成清晰的视频/音频。每一步去噪需要**同时考虑所有 token**（所有帧、所有音频片段），因为去噪是一个全局优化过程。
- **无因果约束**：与自回归模型（如 GPT）不同，diffusion 模型不需要"后面的 token 不能看到前面的 token"的限制。第 100 帧的去噪可以参考第 1 帧的信息，反之亦然。
- **全局上下文**：视频生成需要全局一致性（角色外观、场景布局、音频节奏），双向注意力确保所有位置都能获取全局信息。

**对比：MAGI-1 vs MAGI-2**

MAGI-1 使用**自回归（因果）**方式——将视频分为时间块，按时间顺序逐块生成（前面的块完全生成后才生成后面的块）。MAGI-2 **明确放弃了这一路线**：

> "We did not directly carry over Magi-1's autoregressive chunking. Autoregressive chunking introduces additional temporal dependencies and a more complex generation schedule."

> **出处**：Sand.ai 技术博客 — Section 1

#### 8.5.3 稀疏注意力机制详解

##### MiniMax-H3 的稀疏注意力

MiniMax-H3 在训练的最后阶段引入了**原生稀疏注意力（native sparse attention）**：

- **目的**：降低长多模态序列的计算成本。当 packed 序列包含文本+视频+音频 token 时，序列长度可达数万，全注意力的 O(N²) 计算量成为瓶颈。
- **状态**：训练阶段已使用，但**推理实现尚未开源**（当前开源版本使用全注意力）。
- **具体机制**：公开资料**未详细披露**具体稀疏模式，但常见的稀疏注意力机制包括：
  - **Local Window Attention**（局部窗口）：每个 token 只 attend 到邻近的 token 窗口
  - **Dilated/Block-Sparse Attention**（膨胀/块稀疏）：按固定模式跳过部分 attention 计算
  - **Learned Sparsity**（可学习稀疏）：模型学习哪些 token 对应该 attend
  - **Modality-Aware Sparsity**（模态感知稀疏）：不同模态使用不同的稀疏模式

> **出处**：MiniMax-H3 README — "During the final stage of training, we introduce native sparse attention to reduce the computational cost of long sequences. The sparse-attention implementation is not included in the initial open-source release."

##### MAGI-2 的 MagiAttention 与异构 Mask

MAGI-2 使用自研的 **MagiAttention** 分布式注意力库，这是一个支持异构 mask 的注意力机制：

**MagiAttention 的核心特性**：
- **异构 Mask 支持**：支持四种 mask 类型（包括 FULL 全注意力、CAUSAL 因果等），且允许任意重叠组合。这意味着**同一序列中不同部分可以使用不同的注意力模式**。
- **Flex-Flash-Attention (FFA)**：通用化的 Flash Attention 内核，性能可比 Flash-Attention 3（Hopper GPU）
- **分布式 Context-Parallel**：线性可扩展的分布式注意力，适合超长上下文
- **计算负载均衡**：细粒度 chunk 级分片策略
- **零冗余通信**：GroupCast 和 GroupReduce 原语
- **可学习 Attention Sink**：可学习的注意力"汇聚点"机制

**MAGI-2 如何使用 MagiAttention**：
- MAGI-2 的技术博客提到使用"self-attention only"的单流设计
- 从 Magi-1（自回归）转向 diffusion（双向），意味着主 Transformer 使用全注意力（FULL mask）
- 但 MagiAttention 的异构 mask 能力可能用于：
  - 文本 token 区域使用因果 mask（保留文本的自回归特性）
  - 视频/音频 token 区域使用全注意力
  - 或者全部使用全注意力，仅利用 MagiAttention 的分布式能力处理长序列

**重要区分：MoE 稀疏 ≠ Attention 稀疏**

MAGI-2 的"稀疏"主要体现在 **FFN 层的 MoE**（每 token 激活 72/3072 个专家单元），而非 attention 层。MagiAttention 的分布式能力用于解决长序列的内存/计算问题，不一定是 attention 模式的稀疏化。

> **出处**：[MagiAttention GitHub](https://github.com/SandAI-org/MagiAttention) — "A Distributed Attention Towards Linear Scalability for Ultra-Long Context, Heterogeneous Mask Training"；Sand.ai 技术博客

##### LTX-2.5 的注意力

LTX-2.5 **不使用稀疏注意力**：
- 主 DiT 使用标准全注意力（self-attention 和 cross-attention）
- 高分辨率通过 **多 tile 推理**实现——将高分辨率 latent 分为重叠的空间和时间 tiles，每个 tile 独立用同一模型精炼
- 这是一种**推理时的工程优化**，不是模型内部的 attention 稀疏化
- 每个 tile 内部仍然使用全注意力

> **出处**：arXiv:2601.03233 — Section 4.2 Multi-scale, Multi-tile Inference

#### 8.5.4 VAE 中的因果注意力

三个模型的 VAE 都使用**因果注意力**（在时间维度上），这与主 Transformer 的双向注意力形成对比：

**为什么 VAE 使用因果注意力**：
- VAE 编码器需要保证：时刻 t 的 latent 表示**只依赖于时刻 t 及之前的输入**，不依赖未来
- 这确保了 latent 表示的**时间因果性**——对于视频，第 10 帧的 latent 不包含第 11 帧的信息
- 这对于流式应用和时序一致性很重要

**各模型 VAE 的因果性**：

| 模型 | Video VAE | Audio VAE |
|------|-----------|-----------|
| MiniMax-H3 | "temporally causal"（时间因果） | 时间因果（左/右声道独立处理） |
| MAGI-2 | Wan2.2 VAE（时间因果） | Stable Audio Open（因果） |
| LTX-2.5 | "spatiotemporal causal"（时空因果） | "causal"（因果） |

#### 8.5.5 文本编码器的因果注意力与补偿

三个模型都使用 **decoder-only LLM**（因果注意力）作为文本编码器：

| 模型 | 文本编码器 | 因果注意力的补偿机制 |
|------|-----------|-------------------|
| MiniMax-H3 | Qwen3-VL-32B | 使用第 50 层隐藏状态；Context-IR 系统级精炼 |
| MAGI-2 | Qwen3.5-27B | 外部 prompt 增强 LLM |
| LTX-2.5 | Gemma 4 12B | Multi-Layer Feature Extractor + 双向 Text Connector + Thinking Tokens |

**LTX-2.5 的补偿最为详细**（论文明确论述）：
- **问题**：decoder-only LLM 使用因果（单向）注意力，限制了全局上下文感知
- **Multi-Layer Feature Extractor**：从所有层提取特征（而非仅最终层），捕获从语音到语义的层次化信息
- **Text Connector**：全双向注意力 Transformer blocks，精炼文本嵌入
- **Thinking Tokens**：可学习 token，作为全局信息载体，聚合和丰富上下文

> **出处**：arXiv:2601.03233 — Section 3.2 "Gemma3-12B decoder-only architecture employs causal (unidirectional) attention rather than full bidirectional context modeling. Therefore, we employ the following two methods to compensate..."

#### 8.5.6 完整的注意力结构对比图

```
MiniMax-H3:
┌─────────────────────────────────────────────────────┐
│ H3-Context-IR (系统级，非模型内部)                    │
│ → 多阶段工作流 + 托管模型                             │
├─────────────────────────────────────────────────────┤
│ H3-Encoder: Qwen3-VL-32B                             │
│ → 因果注意力（decoder-only LLM）                      │
├─────────────────────────────────────────────────────┤
│ H3-Omni-Transformer (33B dense, 单流)                │
│ → 双向 self-attention（全注意力）                     │
│ → 训练阶段: 原生稀疏注意力（未开源）                   │
│ → 推理阶段: 全注意力（开源版本）                       │
├─────────────────────────────────────────────────────┤
│ H3-VisualVAE: 时间因果                                │
│ H3-AudioVAE: 时间因果                                 │
└─────────────────────────────────────────────────────┘

MAGI-2 Preview:
┌─────────────────────────────────────────────────────┐
│ Text Encoder: Qwen3.5-27B                             │
│ → 因果注意力（decoder-only LLM）                      │
├─────────────────────────────────────────────────────┤
│ MAGI-2 Transformer (114B MoE, 单流)                  │
│ → 双向 self-attention                                 │
│ → MagiAttention: 支持异构mask的分布式注意力           │
│   (FULL/CAUSAL/混合, 具体使用未明确)                  │
│ → MoE稀疏在FFN层（非attention层）                     │
├─────────────────────────────────────────────────────┤
│ Video VAE (Wan2.2): 时间因果                          │
│ Audio VAE (Stable Audio Open): 因果                   │
│ Turbo VAE: 蒸馏解码器                                 │
└─────────────────────────────────────────────────────┘

LTX-2.5:
┌─────────────────────────────────────────────────────┐
│ Text Encoder: Gemma 4 12B                             │
│ → 因果注意力（decoder-only LLM）                      │
│ → 补偿: Multi-Layer Feature Extractor (所有层特征)    │
│ → 补偿: Text Connector (双向注意力 + Thinking Tokens) │
├─────────────────────────────────────────────────────┤
│ LTX-2.5 DiT (22B, 非对称双流)                         │
│ → 视频流: 双向 self-attention (3D RoPE)              │
│ → 音频流: 双向 self-attention (1D RoPE)              │
│ → 流间: 双向 cross-attention (仅 temporal RoPE)       │
│ → 文本: cross-attention (各流独立)                    │
│ → 无稀疏注意力                                        │
│ → 高分辨率: 多tile推理 (工程优化, 非attention稀疏)     │
├─────────────────────────────────────────────────────┤
│ Video VAE: 时空因果 (DiffVAE 或 Conv VAE)             │
│ Audio VAE: 因果 + HiFi-GAN Vocoder                   │
└─────────────────────────────────────────────────────┘
```

#### 8.5.7 稀疏注意力的实现方式总结

"稀疏"在三个模型中有不同含义：

| 稀疏类型 | MiniMax-H3 | MAGI-2 | LTX-2.5 |
|---------|-----------|--------|---------|
| **Attention 稀疏** | 有（训练阶段，机制未公开） | MagiAttention 支持异构 mask | 无 |
| **FFN 稀疏（MoE）** | 无（dense） | 有（核心创新，6B/114B） | 无（dense） |
| **推理空间稀疏** | 无 | 无 | 多 tile 分块处理 |
| **参数稀疏（缓存）** | AdaLN 参数缓存（~13B） | 无 | 无 |

**关键结论**：
1. **只有 MiniMax-H3 明确提到了 attention 层面的稀疏**（native sparse attention），但具体机制未公开，且推理版本未开源。
2. **MAGI-2 的"稀疏"主要在 FFN 层**（MoE），而非 attention 层。MagiAttention 提供了异构 mask 的能力，但 MAGI-2 具体使用何种 mask 模式未在公开资料中明确。
3. **LTX-2.5 不使用任何 attention 稀疏**，通过多 tile 推理实现高分辨率，这是推理时的工程策略而非模型架构层面的稀疏。
4. **三个模型的主 Transformer 都是双向注意力**（非因果），这是 diffusion 模型的标准做法。
5. **因果注意力仅出现在 VAE 和文本编码器中**，这是各自组件的设计需求（VAE 需要时间因果性，decoder-only LLM 天然是因果的）。

---

### 8.6 单流架构中文本 token 的注意力机制与模态区分

#### 8.6.1 核心结论

**在单流架构（MiniMax-H3、MAGI-2）中，文本 token 在主 Transformer 中采用全双向注意力，与其他模态 token 在 attention 计算层面不做显式区分。**

MiniMax-H3 README 明确陈述：

> "Neither the attention layers nor the FFN layers contain modality-specific structures."

这意味着：
- Self-attention 的 Q/K/V 投影矩阵是**所有模态共享的**，文本 token 不使用单独的投影
- **没有基于模态的 attention mask**——不存在"文本 token 只能被 attend 而不能 attend 别人"的限制
- 文本 token 同时作为 **Query（主动 attend 视频和音频 token）** 和 **Key/Value（被视频和音频 token attend）**

> **出处**：MiniMax-H3 README — H3-Omni-Transformer 章节

#### 8.6.2 文本 token 在 self-attention 中的角色

在 packed 序列的 self-attention 中，所有 token 完全平等参与：

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V

其中 packed 序列：
  Q = [q_text₁, ..., q_textₙ, q_video₁, ..., q_videoₘ, q_audio₁, ..., q_audioₖ]
  K = [k_text₁, ..., k_textₙ, k_video₁, ..., k_videoₘ, k_audio₁, ..., k_audioₖ]
  V = [v_text₁, ..., v_textₙ, v_video₁, ..., v_videoₘ, v_audio₁, ..., v_audioₖ]

关键特性：
  → 所有 token 的 Q/K/V 使用同一个投影矩阵 W_Q, W_K, W_V（模态无关）
  → 无模态 mask，softmax 覆盖全部 token
  → 文本 token 作为 Query：主动 attend 视频/音频 token
  → 文本 token 作为 Key/Value：被视频/音频 token attend（提供条件信号）
```

**文本 token 作为 Query 的独特价值**：文本 token 不仅"告诉"模型生成什么（作为 K/V 被视频/音频 attend），还能"看到"模型正在生成什么（作为 Q attend 视频/音频）。这意味着文本表示在每层都被视觉/音频上下文**反向精炼**——这是一种迭代式的条件化，文本条件不是静态的，而是随着生成过程动态演化的。

#### 8.6.3 文本 token 与其他模态 token 的区分机制

虽然 attention 计算本身不区分模态（"模态盲"），但文本 token 通过以下**隐式机制**与其他模态 token 区分：

**（1）MM-RoPE 位置编码——最关键的隐式区分**

在 `Q·Kᵀ` 的点积中，RoPE 的旋转编码直接注入到 Q 和 K 中。不同模态 token 的位置模式导致 attention 分数矩阵天然呈现出模态相关的结构：

```
文本 token 位置：(t=0或固定值, h=序列索引映射, w=固定值)
  → 空间维度无真实网格结构，时间维度无帧序列意义
  → RoPE 旋转角度模式与视频 token 完全不同

视频 token 位置：(t=帧索引, h=像素行, w=像素列)
  → 有真实的空间网格和时间序列
  → 相邻视频 token 的 RoPE 角度接近 → attention 分数自然更高
  → 形成空间局部性的 attention 模式

音频 token 位置：(t=音频时间戳, h=占位, w=占位)
  → 只有时间维度有意义
  → attention 模式受时间邻近性主导
```

即使没有任何显式 mask，RoPE 使得：
- 视频 token 之间因空间邻近性而 attention 分数高（空间局部性）
- 文本 token 与视频 token 之间的 attention 分数受位置模式影响，形成不同于视频-视频的交互模式
- 模型在训练中学会利用这些位置模式来区分不同模态的 attention 行为

**（2）输入投影差异**

```
文本: h_text  = W_text · Qwen3VL_50th_layer_output    → d_model
视频: h_video = W_video · patchify(VisualVAE_latent)   → d_model
音频: h_audio = W_audio · AudioVAE_latent              → d_model
```

不同模态经过不同的投影矩阵，虽然输出维度相同（d_model），但表示的**分布特征不同**。模型在训练中学会从表示特征区分模态，即使 attention 参数共享。

**（3）AdaLN 模态特定调制**

在 self-attention 输出后，AdaLN 根据模态应用不同的 scale/shift/gate：

```
文本 token: x_text  = α₁ᵗ · (γ₁ᵗ · LN(Attn_out) + β₁ᵗ) + x_text
视频 token: x_video = α₁ᵛ · (γ₁ᵛ · LN(Attn_out) + β₁ᵛ) + x_video
音频 token: x_audio = α₁ᵃ · (γ₁ᵃ · LN(Attn_out) + β₁ᵃ) + x_audio
```

相同的 attention 输出，经过不同的 AdaLN 调制后，不同模态 token 的更新方向和幅度不同。这确保模态信息在深层处理中不会完全丢失。

**（4）输出端——文本 token 不被解码**

最终输出时：
- 视频 token hidden states → `W_video_out` → 24 通道 visual latent → VisualVAE ViT 解码器 → 视频帧
- 音频 token hidden states → `W_audio_out` → audio latent → AudioVAE 解码器 → 立体声音频
- **文本 token hidden states → 丢弃**（不产生生成输出）

文本 token 的角色是**条件输入**而非生成目标。它们的 hidden states 在每层被更新（通过 self-attention），但最终不被解码——它们"参与思考但不产生输出"。

#### 8.6.4 两个阶段的注意力区分

需要区分文本在两个不同阶段的注意力机制：

```
阶段 1：H3-Encoder（Qwen3-VL-32B）——文本的"预处理"
  ┌─────────────────────────────────────────────┐
  │ 因果注意力（causal，decoder-only LLM）        │
  │ 文本 token 只能 attend 到前面的 token         │
  │ 输出第 50 层隐藏状态                          │
  │ 此阶段的注意力是模态内的（纯文本处理）         │
  └─────────────────────────────────────────────┘
                         ↓ 投影 W_text
                         ↓
阶段 2：H3-Omni-Transformer（主生成模型）——文本的"参与生成"
  ┌─────────────────────────────────────────────┐
  │ 双向注意力（bidirectional，diffusion model）  │
  │ 文本 token 可以 attend 到所有 token           │
  │ 所有 token 可以 attend 到文本 token           │
  │ 此阶段的注意力是跨模态的（packed 序列）        │
  │ 文本 token 作为 Q 和 K/V 完全参与             │
  └─────────────────────────────────────────────┘
```

文本 token 在进入 packed 序列之前已通过因果注意力编码（阶段 1），但一旦进入 Omni-Transformer（阶段 2），它们完全参与双向 self-attention。

#### 8.6.5 单流 vs 多流在文本条件化上的本质差异

文本 token 的注意力机制揭示了单流与多流在文本条件化上的根本区别：

| 维度 | 单流（MiniMax-H3, MAGI-2） | 多流（LTX-2.5） |
|------|---------------------------|----------------|
| 文本条件化方式 | 文本 token 在同一序列中，通过 self-attention 条件化 | 独立的 Text Cross-Attention 层 |
| 文本 token 是否作为 Query | **是**——文本 token 主动 attend 视频/音频 token | **否**——文本仅作为 K/V，视频/音频 token 作为 Q |
| 文本表示是否在层间更新 | **是**——文本 token 的表示随每层 self-attention 动态更新 | **否**——文本嵌入在 Text Connector 后基本固定，各层 cross-attention 读取同一组文本嵌入 |
| 条件化信息流 | **双向**——文本→视频/音频（条件），视频/音频→文本（反向精炼） | **单向**——文本→视频/音频（条件），无反向信息流到文本 |
| 条件化深度 | 全深度——每层 self-attention 都有文本参与 | 每层有专门的 Text Cross-Attention 步骤，但文本嵌入本身不更新 |

**核心差异**：

- **单流**中文本 token 是序列的"一等公民"——既是条件的提供者（K/V），也是条件的接收者（Q），其表示在每层被跨模态上下文反向精炼。这是一种**迭代式动态条件化**。
- **多流**中文本 token 是"外部顾问"——仅通过 cross-attention 被查询（K/V），自身表示不随生成过程更新。这是一种**静态条件化**。

LTX-2.5 论文中的 Text Cross-Attention 描述证实了这一点：

> "Text Cross-Attention for textual-prompt conditioning" — 文本作为 K/V，视频/音频 token 作为 Q。

而 MiniMax-H3 的设计是文本 token 在 packed 序列中完全参与 self-attention，不存在单独的"text cross-attention"步骤。

> **出处**：MiniMax-H3 README — H3-Omni-Transformer；arXiv:2601.03233 — Section 3.1, dual-stream block operations

#### 8.6.6 MAGI-2 的特殊考虑

MAGI-2 使用 MagiAttention 库，该库支持**异构 mask**（heterogeneous mask）——同一序列中不同部分可以使用不同的注意力模式。理论上 MAGI-2 可以对文本 token 使用因果 mask、对视频/音频 token 使用全注意力：

```
理论上的异构 mask 方案（推测，未公开确认）：
              文本token    视频token    音频token
文本token      因果↓        blocked      blocked      ← 文本只 attend 前面文本
视频token      full         full         full         ← 视频 attend 所有
音频token      full         full         full         ← 音频 attend 所有
```

但 MAGI-2 的技术博客明确表示采用了"simpler modeling starting point"和"self-attention only"，且从 Magi-1 的自回归路线转向 diffusion。这更倾向于**所有 token（包括文本）使用全双向注意力**，与 MiniMax-H3 一致。

MagiAttention 的异构 mask 能力更可能用于：
- 分布式训练的 context-parallel 切分（不同 rank 处理不同序列段）
- 不同噪声水平或时间步的 token 使用不同 attention 模式
- 长序列的分段处理

而非简单的模态区分。具体使用方式在公开资料中未明确披露。

> **出处**：MagiAttention GitHub — "Heterogeneous Mask Training"；Sand.ai 技术博客 — Section 1

#### 8.6.7 设计理念总结

单流架构中 attention 的"模态盲"设计是一种深思熟虑的选择：

> **Attention 是"模态盲"的（modality-agnostic），但周围的组件是"模态感知"的（modality-aware）。**

```
                    模态感知                    模态盲                    模态感知
                   ┌──────┐                  ┌──────┐                  ┌──────┐
  文本编码 ──→     │输入投影│ ──→     ┌──→    │Self-  │    ←── ──→     │AdaLN │ ──→     ┌──→    │输出投影│ ──→ 解码
  视频编码 ──→     │(模态  │         │       │Attn   │               │(模态  │         │       │(模态  │
  音频编码 ──→     │ 特定) │         └──→    │(共享) │    ←── ──→     │ 特定) │         └──→    │ 特定) │
                   └──────┘                  │  FFN  │               └──────┘                  └──────┘
                   W_text≠                    │(共享) │                                        W_video_out
                   W_video≠                   └──────┘                                        ≠W_audio_out
                   W_audio                    Q/K/V共享
                                              无模态mask
```

这种设计的优势：
1. **最大跨模态交互深度**：无任何结构性屏障限制模态间的注意力交互，文本可以深度参与视觉/音频的生成过程
2. **参数效率**：attention 和 FFN 参数不因模态数量增加而膨胀，33B 模型中 attention+FFN 约 20B
3. **模态区分的隐式保障**：RoPE 位置模式 + 输入投影差异 + AdaLN 调制，三重隐式机制确保模态信息在"模态盲"的处理中不被抹平
4. **动态条件化**：文本 token 作为 Query 参与每层 attention，实现迭代式动态条件化，而非静态的条件注入

---

## 9. PixelUMM：像素空间统一多模态模型分析

### 9.1 模型概述

PixelUMM 是 NVIDIA TLabs 与滑铁卢大学联合推出的**无编码器（encoder-free）统一多模态模型**，直接在像素空间进行图像和视频的理解与生成。它去掉了传统多模态模型中的 VAE 编码器/解码器和视觉编码器（ViT），仅通过单层线性投影将原始像素连接到共享的多模态骨干网络。

> **出处**：arXiv:2609.38597 — "PixelUMM: Encoder-Free Unified Image and Video Understanding and Generation"；[项目主页](https://nv-tlabs.github.io/PixelUMM/)；[GitHub](https://github.com/nv-tlabs/PixelUMM)；[HuggingFace](https://huggingface.co/nvidia/PixelUMM)

### 9.2 输入输出能力支持

#### 输入

| 模式 | 说明 |
|------|------|
| 文本 | 文本指令/描述 |
| 图像 | 原始像素输入（理解任务），或噪声像素（生成目标） |
| 视频 | 原始像素输入（理解任务），或噪声像素（生成目标） |

#### 输出

| 任务 | 说明 |
|------|------|
| I2T（图像理解） | 图像 → 文本描述/问答 |
| V2T（视频理解） | 视频 → 文本描述/问答 |
| T2I（图像生成） | 文本 → 图像（像素空间 flow matching） |
| T2V（视频生成） | 文本 → 视频（像素空间 flow matching） |
| I2V（图生视频） | 图像条件 → 视频生成 |
| 视频编辑 | 文本指令 + 参考视频 → 编辑后视频 |

**注意**：PixelUMM **不支持音频生成或理解**，这是与前述三个模型（MiniMax-H3、MAGI-2、LTX-2.5）的显著区别。

> **出处**：arXiv:2609.38297 — Section 1, 2.2；GitHub README

### 9.3 模型架构

#### 9.3.1 整体架构

PixelUMM 采用 **Mixture-of-Transformers（MoT）架构**，基于 Qwen3-8B decoder-only Transformer 初始化：

```
                    原始像素输入
                        ↓
              ┌─── Patchify ───┐
              │  图像: 16×16    │
              │  视频: 4×16×16  │
              └───────┬────────┘
                      ↓
            单层线性投影 (linear projection)
                      ↓
              d维 token 序列
                      ↓
    ┌──────────────────────────────────────┐
    │    Mixture-of-Transformers (MoT)     │
    │                                      │
    │  ┌─── 理解专家 ───┐ ┌─── 生成专家 ──┐ │
    │  │ 独立 Norm      │ │ 独立 Norm     │ │
    │  │ 独立 QKV/Proj  │ │ 独立 QKV/Proj│ │
    │  │ 独立 FFN       │ │ 独立 FFN     │ │
    │  └───────┬────────┘ └───────┬──────┘ │
    │          │                    │       │
    │          └──── 共享 Self-Attention ──┘ │
    │             (所有token交互)            │
    └──────────────────────────────────────┘
                      ↓
            RMSNorm + 线性投影
                      ↓
              Unpatchify → 像素输出
```

> **出处**：arXiv:2609.38597 — Section 2.1, Figure 3

#### 9.3.2 像素接口（Native Pixel Interface）

PixelUMM 的核心创新是**完全移除 VAE 和 ViT**，仅使用单层线性投影连接像素和骨干网络：

**图像**：
```
图像 x_img ∈ R^{H×W×3}
  → 2D Patchify (16×16) → R^{N_img × (16·16·3)} = R^{N_img × 768}
  → img_linear_proj (单层线性) → R^{N_img × d}
```

**视频**：
```
视频 x_video ∈ R^{F×H×W×3}
  → 3D Patchify (4×16×16) → R^{N_video × (4·16·16·3)} = R^{N_video × 3072}
  → video_linear_proj (单层线性) → R^{N_video × d}
```

**输出**：
```
R^d → RMSNorm + img_linear_outproj → R^{16×16×3} → Unpatchify → 图像像素
R^d → RMSNorm + video_linear_outproj → R^{4×16×16×3} → Unpatchify → 视频像素
```

**关键特性**：
- 无预训练视觉编码器（VE）、无 VAE、无离散视觉 tokenizer
- 像素和骨干网络之间唯一的变换是单层线性投影 + 确定性 patchify/unpatchify
- 输出投影零初始化（zero-initialized），确保训练初期不干扰骨干网络
- 理解和生成使用不同的投影层（`img_und_linear_proj` vs `img_gen_linear_proj`）

> **出处**：arXiv:2609.38597 — Section 2.1, Equations 1-2

#### 9.3.3 无显式时间步条件化

与 MiniMax-H3（使用 AdaLN）和 LTX-2.5（使用 cross-modality AdaLN）不同，PixelUMM **完全省略显式时间步嵌入和时间步条件化归一化**（遵循 MiniT2I 的方法）：

- 去噪网络接收噪声像素**无单独的时间步输入**
- 模型从噪声像素本身**推断损坏程度**
- 时间步仍用于构建训练目标和引导采样过程，但不作为条件信号输入模型

> **出处**：arXiv:2609.38597 — Section 2.1 "No explicit timestep conditioning"

#### 9.3.4 Mixture-of-Transformers（MoT）骨干

| 属性 | 值 |
|------|-----|
| 初始化来源 | Qwen3-8B（decoder-only Transformer） |
| 架构类型 | Mixture-of-Transformers（MoT） |
| 专家数量 | 2（理解专家 + 生成专家） |
| 路由方式 | token 级硬路由（hard routing） |
| 共享组件 | Self-Attention（所有 token 在每个 block 中通过同一 self-attention 交互） |
| 专家特定组件 | 归一化层、QKV/输出投影、FFN |
| 理解路由 | 文本 token + 干净视觉 token |
| 生成路由 | 噪声视觉 token |

**与 MoE 的区别**：MoT 不是 mixture-of-experts（混合专家），而是 mixture-of-transformers（混合 Transformer）。每个"专家"是一个完整的 Transformer 子网络（有独立的 Norm/QKV/FFN），而非 FFN 中的专家单元。路由是硬路由（每个 token 确定性地分配到一个专家），而非软路由（Top-k 选择）。

> **出处**：arXiv:2609.38597 — Section 2.1 "Mixture-of-Transformers backbone"

#### 9.3.5 注意力机制：广义因果注意力

PixelUMM 使用**广义因果注意力（Generalized Causal Attention）**，这是一种混合注意力模式：

```
注意力模式规则：
  1. 文本块内部：因果（causal）——后面的 token 不能 attend 到前面的
  2. 视觉块内部：双向（bidirectional）——所有视觉 token 互相 attend
  3. 块间：后面的块可以 attend 到前面的块，反之不行
  4. 生成目标：attend 到因果 prompt + 目标块内部双向
  5. 前面的干净上下文：不能 attend 回生成目标
```

**具体场景**：
- **图像理解**：图像形成一个双向注意力"岛"
- **视频理解（稀疏模式）**：每帧形成时间有序的双向"岛"，后面的帧可以 attend 到前面的帧
- **视频理解（密集模式）**：一个 tubelet 形成一个双向块
- **生成**：噪声目标 attend 到因果 prompt（文本）+ 目标块内双向

**实现**：
- 理解任务：FlexAttention 实现广义因果 mask
- 生成任务：两次变长 FlashAttention 调用——一次因果处理文本 prompt，一次非因果让噪声视觉 token attend 到 prompt 和完整目标块

> **出处**：arXiv:2609.38597 — Section 2.2, Figure 5

#### 9.3.6 位置编码

三轴 Native RoPE 设计：
- **时间轴**：50% 的注意力头，θ_T = 10^6（Qwen3 的 RoPE base）
- **高度轴**：25% 的注意力头，θ_H = 10^4
- **宽度轴**：25% 的注意力头，θ_W = 10^4
- 文本 token：仅沿时间轴前进（H=W=0）
- 视觉 token：携带空间网格坐标 + 时间坐标

> **出处**：arXiv:2609.38597 — Section 2.2 "Positional encoding"

#### 9.3.7 视频理解的双模式

| 模式 | 采样率 | 处理方式 | 适用场景 |
|------|--------|---------|---------|
| dense_mode | 44 FPS | 3D patchify → video_und_linear_proj | 高帧率视频（≥44 FPS） |
| sparse_mode | 11 FPS | 每帧独立 img_und_linear_proj | 低帧率视频（<44 FPS） |

实验表明两种模式性能相当（F8 实验：MVBench 差异仅 0.07 分）。

> **出处**：arXiv:2609.38597 — Section 2.1 "Video understanding modes", Section 3.1.8

### 9.4 推理流程

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
图像/视频理解（I2T/V2T）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  输入图像/视频
      ↓
  Patchify (16×16 图像 / 4×16×16 视频)
      ↓
  img_und_linear_proj / video_und_linear_proj（干净像素 → d维）
      ↓
  构建多模态序列（ChatML 格式）
  [system] [user] [视觉tokens] [文本问题] [/user] [assistant]
      ↓
  MoT 处理（理解专家路由）
  - 文本 token → 理解专家（因果注意力）
  - 视觉 token → 理解专家（块内双向注意力）
  - 所有 token 通过共享 self-attention 交互
      ↓
  文本 token 自回归生成回答
      ↓
  输出：文本回答

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
图像/视频生成（T2I/T2V）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  文本 prompt
      ↓
  构建多模态序列（ChatML 格式）
  [system] [user] [文本prompt] [/user] [assistant] [噪声视觉tokens]
      ↓
  噪声像素初始化
  z_t = (1-t)·x + t·ε  （flow matching 噪声方案）
      ↓
  Patchify → img_gen_linear_proj / video_gen_linear_proj（噪声像素 → d维）
      ↓
  MoT 去噪（多步 flow matching）
  - 文本 token → 理解专家（因果注意力）
  - 噪声视觉 token → 生成专家（块内双向注意力）
  - 噪声视觉 token attend 到文本 prompt（共享 self-attention）
  - 无显式时间步输入（模型从噪声推断损坏程度）
      ↓
  RMSNorm + img/video_linear_outproj → Unpatchify → 像素输出
      ↓
  输出：图像/视频像素

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
条件生成（I2V / 视频编辑）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  文本指令 + 参考图像/视频（干净像素）
      ↓
  干净视觉条件 → 理解专家路由
  噪声生成目标 → 生成专家路由
  两者通过共享 self-attention 交互
      ↓
  生成专家去噪生成目标视频
```

> **出处**：arXiv:2609.38597 — Section 2.2, 2.3；GitHub README — Inference

### 9.5 训练流程

#### 9.5.1 六阶段训练

| 阶段 | 训练步数 | 主要任务 | 分辨率 |
|------|---------|---------|--------|
| Joint Stage 1 | 150K | 文本 + 图像理解 + 图像生成 | 256² |
| Gen Stage 1 | 200K | 图像生成 + 视频生成 | 256² |
| Und Stage 1 | 100K | 文本 + 原生分辨率图像理解 | Native |
| Und Stage 2 | 20K | + 视频理解 | 224²/帧 |
| Und Stage 3 | 15K | + 视频理解（更高分辨率） | 448²/帧 |
| Joint Stage 2 | 20K | 所有任务联合训练 | 512² |

**损失权重变化**：
- Joint Stage 1: CE:Image MSE:Video MSE = 1:10:0
- Gen Stage 1: 0:10:30
- Und Stages: 1:0:0
- Joint Stage 2: 1:10:30

> **出处**：arXiv:2609.38597 — Section 2.4, Table 1

#### 9.5.2 联合目标

```
L = λ_CE · L_CE + λ_img · L_img + λ_vid · L_vid

其中：
  L_CE = 文本交叉熵（仅 assistant token，平方根序列长度归一化）
  L_img = 图像像素流匹配速度 MSE
  L_vid = 视频像素流匹配速度 MSE

像素目标（flow matching）：
  干净 patch/tubelet x 被损坏为 z_t = (1-t)·x + t·ε
  模型预测干净像素 x̂_θ
  损失在速度空间计算：v_θ = (z_t - x̂_θ) / t̄
```

> **出处**：arXiv:2609.38597 — Section 2.3

### 9.6 关键技术

1. **无编码器像素空间建模**：完全移除 VAE 和 ViT，仅用单层线性投影连接像素和骨干网络。消除了 VAE 训练、维护和推理的开销。

2. **Mixture-of-Transformers（MoT）**：共享 self-attention + 专家特定 Norm/QKV/FFN。理解专家处理文本和干净视觉，生成专家处理噪声视觉。这是"混合 Transformer"而非"混合专家"——每个专家是完整的 Transformer 子网络。

3. **无显式时间步条件化**：不使用 AdaLN 或时间步嵌入。模型从噪声像素本身推断损坏程度，简化了架构。

4. **广义因果注意力**：混合注意力模式——文本块内因果，视觉块内双向，块间因果。既保留了文本生成的自回归特性，又允许视觉 token 的空间交互。

5. **Clean-Pixel 预测扩展到视频**：将 JiT 的图像像素空间生成方法扩展到视频，使用 3D patchify（4×16×16 tubelets）。

6. **统一视觉接口**：图像和视频使用相同的骨干网络，通过不同的 patchify 方式（2D vs 3D）和不同的线性投影实现统一。

7. **多模态上下文条件化**：干净视觉条件通过理解专家路由，噪声生成目标通过生成专家路由，两者通过共享 self-attention 交互——无需 VAE token 流。

8. **Patch Artifact 研究与卷积解码器**：系统研究了线性解码器的 patch 边界伪影问题，提出 PixelShuffle 卷积解码器作为更好的权衡方案。

> **出处**：arXiv:2609.38597 — Sections 2-3

### 9.7 八组实验发现

| 编号 | 实验主题 | 核心发现 |
|------|---------|---------|
| F1 | 图像 patch 大小 | 16×16 优于 32×32；更强的空间压缩使生成更难学习 |
| F2 | 视频 patch 大小 | 更少时空压缩 → 更低 T2V 损失；最终采用 p16/t4 匹配常见 VAE 压缩率 |
| F3 | Patch 伪影 | 线性头有网格伪影；PixelShuffle 卷积头提供更好的伪影/损失权衡 |
| F4 | 像素空间 vs VAE 空间 | 梯度范数相近；像素空间有偶发损失尖峰；4.3× 损失差不反映学习速度 |
| F5 | 模型规模 | 8B 模型以约 1/3 训练步达到 1.7B 的同等损失 |
| F6 | 计算扩展 | 增加 GPU 对文本 CE 收益（9×）远大于生成 MSE（1.3×） |
| F7 | 多模态上下文条件化 | 干净视觉条件可引导生成，不一定损害理解能力 |
| F8 | 视频理解接口 | 密集模式（4 FPS tubelet）≈ 稀疏模式（1 FPS 帧） |

> **出处**：arXiv:2609.38597 — Section 3.1；[项目主页](https://nv-tlabs.github.io/PixelUMM/)

---

## 10. PixelUMM 与三类多模态生成模型的对比分析

### 10.1 技术路线对比

| 维度 | PixelUMM | MiniMax-H3 | MAGI-2 Preview | LTX-2.5 |
|------|----------|-----------|----------------|---------|
| **核心范式** | 像素空间统一理解+生成 | 潜空间全模态生成 | 潜空间 MoE 视频生成 | 潜空间双流 DiT |
| **视觉表示** | 原始像素（无 VAE/ViT） | VAE latent (f16t4d24) | VAE latent (Wan2.2) | VAE latent (DiffVAE) |
| **编码器** | 单层线性投影 | Qwen3-VL-32B + VisualVAE + AudioVAE | Qwen3.5-27B + Wan2.2 VAE + Stable Audio | Gemma 4 12B + Video/Audio VAE |
| **解码器** | 单层线性投影 + unpatchify | ViT-based VisualVAE + AudioVAE | Wan2.2 VAE + Stable Audio + Turbo VAE | DiffVAE/Conv VAE + Audio VAE + Vocoder |
| **任务范围** | 理解 + 生成 | 仅生成 | 仅生成 | 仅生成 |
| **音频** | 不支持 | 32kHz 立体声 | 支持 | 24kHz 立体声 |

### 10.2 模型架构对比

| 维度 | PixelUMM | MiniMax-H3 | MAGI-2 Preview | LTX-2.5 |
|------|----------|-----------|----------------|---------|
| **骨干网络** | Qwen3-8B (decoder-only) | H3-Omni-Transformer (33B dense) | MagiMoE Transformer (114B MoE) | 非对称双流 DiT (22B) |
| **参数量** | 8B | ~33B (推理 ~20B) | ~114B (激活 ~6B) | ~22B |
| **架构类型** | MoT (混合 Transformer) | 单流 Dense | 单流 MoE | 双流 Dense |
| **模态融合** | 单流 + 共享 self-attention + 专家路由 | 单流 packed 序列 | 单流 packed 序列 | 双流 + cross-attention |
| **时间步条件化** | 无（从噪声推断） | AdaLN（模态特定，~13B） | 未明确 | Cross-modality AdaLN |
| **位置编码** | 3-axis Native RoPE (T/H/W) | MM-RoPE (t,h,w) | 未明确 | 视频 3D RoPE + 音频 1D RoPE |
| **注意力类型** | 广义因果（混合） | 双向（训练有稀疏） | 双向 | 双向 |
| **MoE/MoT** | MoT（2 专家，硬路由） | 无 | MoE（3072 专家单元，Top-72） | 无 |

### 10.3 架构差异深度分析

#### 10.3.1 像素空间 vs 潜空间

这是 PixelUMM 与其他三个模型最根本的技术路线差异：

**PixelUMM（像素空间）**：
```
原始像素 → Patchify → 单层线性投影 → Transformer → 单层线性投影 → Unpatchify → 像素
```

**其他三个模型（潜空间）**：
```
原始像素 → VAE编码器 → latent → 投影 → Transformer → 投影 → latent → VAE解码器 → 像素
```

**像素空间的优势**：
1. **无 VAE 开销**：不需要训练、存储、推理 VAE 编码器/解码器
2. **统一视觉接口**：理解和生成使用相同的像素表示，无需维护 ViT + VAE 双接口
3. **无 VAE 信息损失**：VAE 压缩会丢失高频细节，像素空间保留全部信息
4. **简化流程**：从像素到骨干网络仅一层线性变换

**像素空间的劣势**：
1. **序列长度更长**：VAE 压缩（如 H3 的 32×空间×4×时间）大幅减少 token 数；像素空间无此压缩，序列长度显著增加
2. **计算成本更高**：更长的序列 → 更大的 attention 矩阵 → 更多计算
3. **Patch 伪影**：线性解码器在 patch 边界产生网格伪影（F3 实验）
4. **训练动态不稳定**：偶发损失尖峰（F4 实验）
5. **生成质量**：8B 模型在理解 benchmark 上落后于使用大型 VLM 的模型（如 Qwen3-VL 8B 在 MMMU 上 69.60 vs PixelUMM 41.67）

> **出处**：arXiv:2609.38597 — F3, F4 实验；MiniMax-H3 README；arXiv:2601.03233

#### 10.3.2 MoT vs 单流/双流/MoE

| 架构 | 模型 | 共享组件 | 专家/流特定组件 | 路由方式 |
|------|------|---------|---------------|---------|
| **MoT** | PixelUMM | Self-Attention | Norm + QKV/Proj + FFN | 硬路由（理解 vs 生成） |
| **单流 Dense** | MiniMax-H3 | Self-Attention + FFN | AdaLN（模态特定） | 无路由（全部共享） |
| **单流 MoE** | MAGI-2 | Self-Attention | FFN（专家池） | 软路由（Top-k） |
| **双流** | LTX-2.5 | 无（各流独立） | 全部（各流完整独立） | 无路由（模态分流） |

**PixelUMM MoT 的独特性**：
- MoT 是介于单流和双流之间的设计——attention 共享（像单流），但 QKV/FFN 分专家（像多流）
- 路由基于**任务**（理解 vs 生成）而非**模态**（视频 vs 音频），这是与双流架构的根本区别
- 硬路由意味着每个 token 确定性地分配到一个专家，没有路由器的计算开销和负载均衡问题

#### 10.3.3 注意力机制对比

| 模型 | 主注意力 | 文本块内 | 视觉块内 | 块间 | 时间步条件化 |
|------|---------|---------|---------|------|------------|
| **PixelUMM** | 广义因果 | 因果 | 双向 | 因果（后→前） | 无 |
| **MiniMax-H3** | 双向 | 双向 | 双向 | 双向 | AdaLN（模态特定） |
| **MAGI-2** | 双向 | 双向 | 双向 | 双向 | 未明确 |
| **LTX-2.5** | 双向 | 双向 | 双向 | 双向 | Cross-modality AdaLN |

**PixelUMM 广义因果注意力的独特性**：
- 是唯一使用**混合注意力模式**的模型——文本因果，视觉双向
- 其他三个模型在主 Transformer 中全部使用双向注意力
- 这种设计源于 PixelUMM 需要同时支持**自回归文本生成**（需要因果）和**diffusion 视觉生成**（需要双向），而其他三个模型是纯 diffusion 生成模型（文本仅作条件输入，不需要生成）

**关于时间步条件化**：
- PixelUMM 是唯一**无显式时间步条件化**的模型——模型从噪声像素本身推断损坏程度
- MiniMax-H3 使用模态特定 AdaLN（~13B 参数，可缓存）
- LTX-2.5 使用 cross-modality AdaLN（一个模态的参数由另一模态条件化）
- 这是一种架构简化，但也可能限制模型对不同噪声水平的适应能力

> **出处**：arXiv:2609.38597 — Section 2.1, 2.2；arXiv:2601.03233 — Section 3.1

### 10.4 优缺点对比

#### 10.4.1 PixelUMM 的优势

1. **架构极简**：无 VAE、无 ViT、无时间步条件化、无 cross-attention 层。像素到骨干网络仅一层线性投影。
2. **理解+生成统一**：唯一支持四类任务（I2T/V2T/T2I/T2V）的模型，其他三个仅支持生成。
3. **视觉接口统一**：理解和生成使用相同的像素表示，无需维护双视觉接口（ViT + VAE），视觉上下文长度减半。
4. **无 VAE 依赖**：不依赖外部 VAE 的质量和压缩特性，避免了 VAE 信息损失。
5. **开源完整**：代码 + 权重 + 论文 + 实验全部公开，Apache-2.0 许可。
6. **系统消融研究**：八组实验（F1-F8）提供了像素空间建模的设计洞察，具有方法论价值。

#### 10.4.2 PixelUMM 的劣势

1. **无音频支持**：不支持音频生成或理解，而其他三个模型都生成同步音频。
2. **理解性能落后**：在理解 benchmark 上显著落后于专用 VLM（MMMU 41.67 vs Qwen3-VL 69.60），因为无预训练 ViT 提供语义特征。
3. **序列长度更长**：无 VAE 压缩，像素 token 数远多于 latent token 数。512×512 图像 = 1024 个 patch token，而 VAE 压缩后可能仅 256 个。
4. **Patch 伪影**：线性解码器在 patch 边界产生网格伪影（F3），需要卷积解码器缓解。
5. **模型规模较小**：8B 参数，远小于 H3（33B）、MAGI-2（114B）、LTX-2.5（22B）。
6. **生成分辨率有限**：默认 512²（图像）/ 320×176（视频），远低于 H3 的 2K 和其他模型的 1080p。
7. **训练动态不稳定**：像素空间训练有偶发损失尖峰（F4）。
8. **无高分辨率策略**：缺乏 H3 的 in-context 重生成、MAGI-2 的 Refiner、LTX-2.5 的多 tile 推理等高分辨率方案。

#### 10.4.3 综合对比表

| 维度 | PixelUMM | MiniMax-H3 | MAGI-2 | LTX-2.5 |
|------|----------|-----------|--------|---------|
| 架构复杂度 | 最低 | 高 | 最高 | 中 |
| 参数量 | 8B | ~33B | ~114B | ~22B |
| 视觉编码器 | 无（线性投影） | Qwen3-VL-32B + VisualVAE | Qwen3.5-27B + Wan2.2 VAE | Gemma4-12B + DiffVAE |
| 理解能力 | 有 | 无 | 无 | 无 |
| 音频 | 无 | 32kHz 立体声 | 有 | 24kHz 立体声 |
| 最大分辨率 | 512² | 2K | 1080p | 1080p |
| 最大时长 | 4s (96帧/24fps) | 15s | 10s | 20s |
| 推理效率 | 中（长序列但无 VAE） | 中 | 低（需 8 GPU） | 高（18× 快于同等模型） |
| 开源程度 | 完全（Apache-2.0） | 部分 | 完整 | 完全 |

### 10.5 技术路线本质差异

PixelUMM 与其他三个模型的差异不仅是架构差异，而是**对"视觉表示"的根本不同理解**：

**潜空间路线（H3, MAGI-2, LTX-2.5）**：
- 核心信念：视觉信号需要被**压缩**到低维 latent 空间才能高效处理
- VAE 是"翻译器"：将高维像素压缩为低维 latent，大幅减少 token 数
- 理解和生成使用不同的编码器（ViT for 理解，VAE for 生成），因为它们需要不同类型的特征
- 代价：VAE 训练/维护开销、信息损失、双视觉接口的复杂性

**像素空间路线（PixelUMM）**：
- 核心信念：视觉信号可以**直接**在像素空间处理，无需压缩
- 线性投影是"连接器"：将像素 patch 映射到骨干网络维度
- 理解和生成使用相同的像素表示，因为像素空间天然统一
- 代价：更长的序列、更高的计算成本、patch 伪影

**根本权衡**：
```
潜空间：压缩 → 更短序列 → 更高效计算，但损失信息 + 需要额外 VAE
像素空间：不压缩 → 更长序列 → 更高计算成本，但保留全部信息 + 架构极简
```

PixelUMM 的实验（F5: 8B 模型以 1/3 步数达到 1.7B 同等损失）表明，像素空间建模可以从模型扩展中受益，但当前规模（8B）仍不足以完全补偿无 VAE 压缩带来的序列长度增加。

> **出处**：arXiv:2609.38597 — Section 1, 3.1；arXiv:2601.03233 — Section 1

### 10.6 趋势洞察

1. **像素空间建模的可行性已验证**：PixelUMM 证明了无需 VAE/ViT 的像素空间统一多模态建模是可行的，但当前规模和性能尚不足以挑战潜空间路线的主流地位。

2. **理解+生成统一是独特价值**：PixelUMM 是四个模型中唯一支持理解的，代表了"统一多模态模型"（UMM）的完整愿景——同一模型既能看懂又能创造。其他三个模型是纯生成模型。

3. **MoT 架构的启示**：共享 attention + 专家特定 QKV/FFN 的设计，在"全共享"（单流）和"全分离"（双流）之间提供了中间路线，值得在更大规模上验证。

4. **无时间步条件化的探索**：PixelUMM 证明了模型可以从噪声本身推断损坏程度，无需显式时间步输入。这简化了架构，但是否在大规模/高质量生成中仍有效待验证。

5. **Patch 伪影是像素空间的固有挑战**：线性解码器的网格伪影是 patchify 操作的必然结果。PixelShuffle 卷积解码器提供了缓解方案，但增加了计算成本。

6. **规模扩展的关键性**：F5 实验表明 8B 模型收敛速度是 1.7B 的 3 倍，暗示像素空间建模可能需要更大规模才能达到与潜空间路线可比的效率和质量。未来的像素空间统一模型可能需要 30B+ 规模才能形成真正竞争力。

> **出处**：arXiv:2609.38597 — Sections 1, 3.1, 5

---

## 11. PixelUMM MoT 架构深度解析（源码级）

### 11.1 用户理解确认

**用户的描述是准确的。** 在 PixelUMM 的 Mixture-of-Transformers（MoT）架构中：

- **理解任务和生成任务使用独立的**：Normalization、QKV 投影、Output 投影、Q/K 归一化、FFN
- **但 attention 计算是共享的**：所有 token（无论来自理解路由还是生成路由）在同一个 attention 矩阵中交互

论文原文明确陈述：

> "Each block has route-specific normalization, QKV/output projections, and FFNs, while all tokens participate in the same multimodal self-attention operation."

> **出处**：arXiv:2609.38597 — Section 2.1 "Mixture-of-Transformers backbone"

### 11.2 源码验证

以下分析基于 PixelUMM 官方 GitHub 仓库的源码文件 `modeling/pixelumm/qwen3_navit.py` 和 `modeling/qwen3/configuration_qwen3.py`。

> **出处**：[GitHub nv-tlabs/PixelUMM — modeling/pixelumm/qwen3_navit.py](https://github.com/nv-tlabs/PixelUMM/blob/main/modeling/pixelumm/qwen3_navit.py)

#### 11.2.1 MoT Block 的组件定义

源码中 `Qwen3MoTDecoderLayer` 类明确定义了每个 MoT block 的双专家结构：

```python
class Qwen3MoTDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        # 共享的 Self-Attention（内部有双 QKV 投影）
        self.self_attn = PackedAttentionMoT(config, layer_idx)

        # 理解专家 FFN
        self.mlp = Qwen3MLP(config)
        # 生成专家 FFN
        self.mlp_moe_gen = Qwen3MLP(config)

        # 理解专家 Norm
        self.input_layernorm = Qwen3RMSNorm(...)
        self.post_attention_layernorm = Qwen3RMSNorm(...)
        # 生成专家 Norm
        self.input_layernorm_moe_gen = Qwen3RMSNorm(...)
        self.post_attention_layernorm_moe_gen = Qwen3RMSNorm(...)
```

**关键发现**：每个 MoT block 有 **2 套 FFN** 和 **4 个 Norm 层**（2 个 input + 2 个 post-attention），但只有 **1 个 Self-Attention 模块**。

> **出处**：qwen3_navit.py — `Qwen3MoTDecoderLayer.__init__`

#### 11.2.2 Attention 模块的双 QKV 投影

`PackedAttentionMoT` 类继承自 `Qwen3Attention`（拥有理解专家的 `q_proj`/`k_proj`/`v_proj`/`o_proj`），并额外添加了生成专家的投影：

```python
class PackedAttentionMoT(SenseNovaMRoPEMixin, Qwen3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)  # 继承理解专家的 q_proj, k_proj, v_proj, o_proj

        # 生成专家的独立 QKV/O 投影
        self.q_proj_moe_gen = nn.Linear(hidden_size, num_heads * head_dim, ...)
        self.k_proj_moe_gen = nn.Linear(hidden_size, num_kv_heads * head_dim, ...)
        self.v_proj_moe_gen = nn.Linear(hidden_size, num_kv_heads * head_dim, ...)
        self.o_proj_moe_gen = nn.Linear(num_heads * head_dim, hidden_size, ...)

        # 生成专家的独立 Q/K 归一化
        self.q_norm_t_moe_gen = Qwen3RMSNorm(...)
        self.q_norm_hw_moe_gen = Qwen3RMSNorm(...)
        self.k_norm_t_moe_gen = Qwen3RMSNorm(...)
        self.k_norm_hw_moe_gen = Qwen3RMSNorm(...)
```

**关键发现**：每个 attention 层有 **2 套完整的 QKV 投影矩阵**（`q_proj` vs `q_proj_moe_gen`）和 **2 套 Q/K 归一化**，但 attention 计算本身是共享的。

> **出处**：qwen3_navit.py — `PackedAttentionMoT.__init__`

#### 11.2.3 训练前向传播：分离投影 + 共享 attention

`forward_train` 方法清晰展示了"分离 QKV 投影 → 共享 attention → 分离输出投影"的三步流程：

```python
def forward_train(self, packed_sequence, ..., packed_und_token_indexes, packed_gen_token_indexes):
    # ═══ 第一步：分离的 QKV 投影 ═══
    # 为整个序列创建空的 Q, K, V 张量
    query = packed_sequence.new_zeros(seq_len, num_heads * head_dim)
    key = packed_sequence.new_zeros(seq_len, num_kv_heads * head_dim)
    value = packed_sequence.new_zeros(seq_len, num_kv_heads * head_dim)

    # 理解 token 用理解专家的投影矩阵
    und = packed_sequence[packed_und_token_indexes]
    query[packed_und_token_indexes] = self.q_proj(und)       # ← 理解专家 W_Q
    key[packed_und_token_indexes] = self.k_proj(und)         # ← 理解专家 W_K
    value[packed_und_token_indexes] = self.v_proj(und)       # ← 理解专家 W_V

    # 生成 token 用生成专家的投影矩阵
    gen = packed_sequence[packed_gen_token_indexes]
    query[packed_gen_token_indexes] = self.q_proj_moe_gen(gen)  # ← 生成专家 W_Q
    key[packed_gen_token_indexes] = self.k_proj_moe_gen(gen)    # ← 生成专家 W_K
    value[packed_gen_token_indexes] = self.v_proj_moe_gen(gen)  # ← 生成专家 W_V

    # ═══ 第二步：共享的 Attention 计算 ═══
    # 所有 token（理解 + 生成）的 Q, K, V 拼在一起做 attention
    output = flex_attention(
        query.unsqueeze(0),   # ← 所有 token 的 Q（理解Q + 生成Q 混在一起）
        key.unsqueeze(0),     # ← 所有 token 的 K（理解K + 生成K 混在一起）
        value.unsqueeze(0),   # ← 所有 token 的 V（理解V + 生成V 混在一起）
        block_mask=attention_mask,  # 广义因果 mask
    )
    # 此时：理解 token 的 Q 可以 attend 到生成 token 的 K/V
    #       生成 token 的 Q 可以 attend 到理解 token 的 K/V

    # ═══ 第三步：分离的输出投影 ═══
    projected = output.new_zeros(output.shape[0], hidden_size)
    projected[packed_und_token_indexes] = self.o_proj(output[packed_und_token_indexes])      # ← 理解专家 W_O
    projected[packed_gen_token_indexes] = self.o_proj_moe_gen(output[packed_gen_token_indexes])  # ← 生成专家 W_O

    return projected
```

> **出处**：qwen3_navit.py — `PackedAttentionMoT.forward_train`

#### 11.2.4 MoT Block 的完整前向传播

`Qwen3MoTDecoderLayer.forward_train` 展示了一个完整 MoT block 的处理流程：

```python
def forward_train(self, packed_sequence, ..., packed_und_token_indexes, packed_gen_token_indexes):
    # ═══ Attention 子层 ═══

    # 步骤 1：路由特定的输入归一化
    und_normalized = self.input_layernorm(packed_sequence[packed_und_token_indexes])
    gen_normalized = self.input_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])

    # 步骤 2：共享 Self-Attention（内部分离 QKV + 共享 attention + 分离 O）
    attention_output = self.self_attn(
        packed_sequence=normalized,  # 混合的归一化序列
        packed_und_token_indexes=packed_und_token_indexes,
        packed_gen_token_indexes=packed_gen_token_indexes,
    )

    # 步骤 3：残差连接
    packed_sequence = residual + attention_output

    # ═══ FFN 子层 ═══

    # 步骤 4：路由特定的 post-attention 归一化
    und_normed = self.post_attention_layernorm(packed_sequence[packed_und_token_indexes])
    gen_normed = self.post_attention_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])

    # 步骤 5：路由特定的 FFN
    und_mlp = self.mlp(und_normed)           # ← 理解专家 FFN
    gen_mlp = self.mlp_moe_gen(gen_normed)   # ← 生成专家 FFN

    # 步骤 6：残差连接
    mlp_output[packed_und_token_indexes] = und_mlp
    mlp_output[packed_gen_token_indexes] = gen_mlp
    return residual + mlp_output
```

> **出处**：qwen3_navit.py — `Qwen3MoTDecoderLayer.forward_train`

#### 11.2.5 模型级最终归一化也是双路由

`Qwen3Model` 的最终归一化层同样是双路由的：

```python
class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config):
        self.norm = Qwen3RMSNorm(hidden_size, ...)        # 理解专家最终 Norm
        self.norm_moe_gen = Qwen3RMSNorm(hidden_size, ...) # 生成专家最终 Norm

    def forward_train(self, ...):
        # ... 所有层处理完毕 ...
        output = torch.zeros_like(packed_sequence)
        output[packed_und_token_indexes] = self.norm(packed_sequence[packed_und_token_indexes])
        output[packed_gen_token_indexes] = self.norm_moe_gen(packed_sequence[packed_gen_token_indexes])
        return output
```

> **出处**：qwen3_navit.py — `Qwen3Model.__init__`, `Qwen3Model.forward_train`

### 11.3 具体示例：文本到视频生成的一次前向传播

假设执行 T2V 任务，prompt 为 "A cat runs across a field"，生成 96 帧 176×320 视频。

#### 步骤 1：序列构建

```
Packed 序列 = [文本 tokens] [噪声视频 tokens]
               ↑                ↑
           理解路由          生成路由
           (packed_und_token_indexes)  (packed_gen_token_indexes)

假设：文本 20 tokens，视频 96×(176/16)×(320/16)/(4×16×16) = 96×11×20/4 = 5280 tubelets
总序列长度 = 20 + 5280 = 5300 tokens
```

#### 步骤 2：每个 MoT Block 的处理

```
输入：packed_sequence [5300, d]

┌─── 步骤 1：路由特定归一化 ───────────────────────────────┐
│                                                          │
│  tokens[0:20] (文本) → input_layernorm → und_normalized  │
│  tokens[20:5300] (噪声视频) → input_layernorm_moe_gen → gen_normalized │
└──────────────────────────────────────────────────────────┘
                            ↓
┌─── 步骤 2：分离 QKV 投影 ────────────────────────────────┐
│                                                          │
│  und_normalized → q_proj → Q_und [20, heads×dim]         │
│  und_normalized → k_proj → K_und [20, kv_heads×dim]      │
│  und_normalized → v_proj → V_und [20, kv_heads×dim]      │
│                                                          │
│  gen_normalized → q_proj_moe_gen → Q_gen [5280, ...]     │
│  gen_normalized → k_proj_moe_gen → K_gen [5280, ...]     │
│  gen_normalized → v_proj_moe_gen → V_gen [5280, ...]     │
└──────────────────────────────────────────────────────────┘
                            ↓
┌─── 步骤 3：Q/K 归一化（路由特定）─────────────────────────┐
│                                                          │
│  Q_und, K_und → q_norm_t, q_norm_hw, k_norm_t, k_norm_hw │
│  Q_gen, K_gen → q_norm_t_moe_gen, ...（生成专家的 QK Norm）│
└──────────────────────────────────────────────────────────┘
                            ↓
┌─── 步骤 4：MRoPE 位置编码（共享逻辑，T/H/W 三轴）─────────┐
│                                                          │
│  所有 Q, K 按位置编码旋转                                 │
│  文本 token: T=序列位置, H=0, W=0                        │
│  视频 token: T=帧时间, H=空间行, W=空间列                 │
└──────────────────────────────────────────────────────────┘
                            ↓
┌─── 步骤 5：共享 Attention 计算 ──────────────────────────┐
│                                                          │
│  Q = [Q_und; Q_gen]  → [5300, heads, dim]                │
│  K = [K_und; K_gen]  → [5300, kv_heads, dim]             │
│  V = [V_und; V_gen]  → [5300, kv_heads, dim]             │
│                                                          │
│  Attention = softmax(Q · K^T / √d) · V                   │
│                                                          │
│  ★ 文本 Q 可以 attend 到噪声视频 K/V                     │
│    （文本"看到"正在生成的视频内容）                       │
│  ★ 噪声视频 Q 可以 attend 到文本 K/V                     │
│    （视频从文本获取生成指令）                             │
│  ★ 广义因果 mask 控制可见性                               │
│    （文本块内因果，视频块内双向，块间因果）               │
└──────────────────────────────────────────────────────────┘
                            ↓
┌─── 步骤 6：分离输出投影 ─────────────────────────────────┐
│                                                          │
│  attn_out[0:20] → o_proj → projected_und                 │
│  attn_out[20:5300] → o_proj_moe_gen → projected_gen      │
└──────────────────────────────────────────────────────────┘
                            ↓
                    残差连接
                            ↓
┌─── 步骤 7：路由特定 post-attn Norm + FFN ────────────────┐
│                                                          │
│  projected_und → post_attn_norm → mlp → und_ffn_out      │
│  projected_gen → post_attn_norm_moe_gen → mlp_moe_gen → gen_ffn_out │
└──────────────────────────────────────────────────────────┘
                            ↓
                    残差连接
                            ↓
输出：更新后的 packed_sequence [5300, d]
```

### 11.4 MoT 与标准架构的精确对比

#### 11.4.1 组件共享/分离矩阵

| 组件 | 标准 Transformer | 单流（MiniMax-H3） | MoE（MAGI-2） | MoT（PixelUMM） | 双流（LTX-2.5） |
|------|-----------------|-------------------|--------------|----------------|----------------|
| Self-Attention 计算 | 共享 | 共享 | 共享 | **共享** | 分离 |
| QKV 投影 | 共享 | 共享 | 共享 | **分离** | 分离 |
| Output 投影 | 共享 | 共享 | 共享 | **分离** | 分离 |
| Q/K Norm | 共享 | 共享 | 共享 | **分离** | 分离 |
| FFN | 共享 | 共享 | **分离**（MoE） | **分离** | 分离 |
| Norm (input/post-attn) | 共享 | 共享 | 共享 | **分离** | 分离 |
| 路由方式 | 无路由 | 无路由 | 软路由（Top-k） | **硬路由**（任务） | 无路由（模态） |
| 跨专家/流交互 | N/A | 全深度 self-attn | 全深度 self-attn | **全深度 self-attn** | 仅 cross-attn 层 |

#### 11.4.2 MoT 的独特定位

MoT 位于单流和双流之间的**精确中间位置**：

```
单流（全共享）  ←→  MoT（attention共享，其余分离）  ←→  双流（全分离）
     ↑                        ↑                           ↑
  MiniMax-H3              PixelUMM                    LTX-2.5
  MAGI-2
```

- **与单流的区别**：单流共享 QKV 和 FFN；MoT 分离 QKV 和 FFN，但共享 attention 计算
- **与双流的区别**：双流连 attention 都分离，靠 cross-attention 交互；MoT 的 attention 是共享的，跨专家交互在 self-attention 中**自然发生**
- **与 MoE 的区别**：MoE 在 FFN 内部分离（多个 FFN 专家），QKV 共享；MoT 在 QKV 和 FFN 都分离，但分离维度是**任务**而非**专家池**

### 11.5 为什么 Attention 共享而其余分离

这一设计选择背后的逻辑：

**Attention 共享的原因**：
- Attention 是跨 token 信息交换的核心机制——如果分离 attention，则需要额外的 cross-attention 层来交换信息（如 LTX-2.5）
- 共享 attention 允许理解 token（文本 prompt）和生成 token（噪声视频）在**每一层**都自然交互，无需额外的接口
- 一个文本 token 的 Query（由理解专家投影）可以直接 attend 到噪声视频 token 的 Key（由生成专家投影），实现跨任务的信息流动

**QKV/FFN 分离的原因**：
- 理解 token（文本、干净像素）和生成 token（噪声像素）的**统计分布不同**
- 不同的 QKV 投影允许每种 token 被投影到适合自身任务的注意力子空间
- 不同的 FFN 允许每种 token 接受适合自身任务的非线性变换
- 理解任务需要提取语义信息，生成任务需要预测像素值——两者的计算需求不同

**比喻**：
> MoT 就像一个会议室，理解专家和生成专家各自准备自己的材料（分离的 QKV 投影 = 不同的材料准备方式），但在会议中所有人一起讨论（共享 attention = 同一个讨论空间），最后各自写自己的报告（分离的 FFN = 不同的总结方式）。

> **出处**：arXiv:2609.38597 — Section 2.1；qwen3_navit.py — `PackedAttentionMoT`, `Qwen3MoTDecoderLayer`

### 11.6 硬路由机制的实现细节

路由是**token 级硬路由**（hard routing），意味着每个 token 确定性分配到一个专家，无路由器、无 Top-k 选择、无负载均衡损失：

```
路由规则（确定性）：
  文本 token           → 理解专家（input_layernorm, q_proj, mlp, ...）
  干净视觉 token       → 理解专家（input_layernorm, q_proj, mlp, ...）
  噪声视觉 token       → 生成专家（input_layernorm_moe_gen, q_proj_moe_gen, mlp_moe_gen, ...）
```

源码中通过 `packed_und_token_indexes` 和 `packed_gen_token_indexes` 两个索引数组实现：

```python
# 理解 token 的索引（文本 + 干净视觉）
packed_und_token_indexes = [0, 1, 2, ..., 19]  # 文本 token 位置

# 生成 token 的索引（噪声视觉）
packed_gen_token_indexes = [20, 21, 22, ..., 5299]  # 噪声视频 token 位置

# 每个 token 根据其索引被路由到对应专家的参数
```

**与 MoE 软路由的对比**：
- MoE（MAGI-2）：路由器网络计算每个 token 对每个专家的分数，选 Top-k，有路由计算开销和负载均衡需求
- MoT（PixelUMM）：路由在**数据预处理阶段**确定（哪些 token 是理解、哪些是生成），无运行时路由计算

> **出处**：arXiv:2609.38597 — "token-level hard routing between understanding and generation experts"；qwen3_navit.py — `packed_und_token_indexes`, `packed_gen_token_indexes` 参数

### 11.7 参数量分析

基于 Qwen3-8B 配置（hidden_size=4096, num_layers=36, num_heads=32, head_dim=128, intermediate_size=12288）：

| 组件 | 每层参数 | 理解专家 | 生成专家 | 共享 |
|------|---------|---------|---------|------|
| QKV 投影 | 2 × (n_heads+n_kv+n_kv) × d × head_dim | ✓ | ✓ | |
| O 投影 | n_heads × head_dim × d | ✓ | ✓ | |
| Q/K Norm | 4 × head_dim/2 | ✓ | ✓ | |
| FFN | 3 × d × intermediate_size | ✓ | ✓ | |
| Input Norm | d | ✓ | ✓ | |
| Post-attn Norm | d | ✓ | ✓ | |
| Attention 计算 | 0（无参数） | | | ✓ |
| **每层总计** | | ~100M | ~100M | 0 |

MoT 相比标准 Transformer 的参数开销约为 **~2×**（每个参数都有理解和生成两份，除 attention 计算无参数外）。但 attention 计算虽然共享，QKV 投影的分离意味着 Q/K/V 的生成仍然分别进行，计算量也是约 2×。

> **注**：以上参数量为估算，基于 Qwen3-8B 的典型配置。实际配置可能有所不同。

---

## 12. 采用 MoT 架构的其他多模态模型

### 12.1 概述

Mixture-of-Transformers（MoT）架构并非 PixelUMM 独创。arXiv 上检索 "Mixture-of-Transformers" 关键词返回 **106 篇论文**（截至 2026 年 11 月），表明该架构已成为多模态领域的重要范式。MoT 架构的采用主要集中在两个领域：**统一多模态理解与生成** 和 **世界动作模型（World Action Models / VLA）**。

> **出处**：arXiv 搜索 "Mixture-of-Transformers"，2026-11 检索，返回 106 篇结果

### 12.2 统一多模态理解与生成模型

这些模型与 PixelUMM 类似，旨在用同一模型同时支持视觉理解和生成。

| 模型 | 来源 | arXiv | 架构特点 | 与 PixelUMM 的关系 |
|------|------|-------|---------|-------------------|
| **BAGEL** | ByteDance | [2505.14683](https://arxiv.org/abs/2505.14683) | Decoder-only，MoT 架构（理解专家+生成专家），共享 self-attention | **PixelUMM 的直接前身**。源码基于 BAGEL 的 `qwen2_navit.py` |
| **Show-o** | Show Lab | 提及于 PixelUMM benchmark | 统一理解和生成 | 在视频理解 benchmark 中作为对比 |
| **TUNA / TUNA-2** | — | 提及于 PixelUMM 论文 | 无编码器像素空间，统一理解+生成 | 与 PixelUMM 技术路线相似（像素空间） |
| **SenseNova-U1 / U1.5** | SenseTime | 提及于 PixelUMM 论文 [32][30] | 无编码器像素空间，引入了 MoT 条件化设计 | PixelUMM 继承了其视觉条件化设计和 MRoPE 拓扑 |
| **NEO** | — | 提及于 PixelUMM 论文 [28] | 三轴 Native RoPE 设计 | PixelUMM 继承了其 T/H/W RoPE 分配方案 |
| **Lance** | — | 提及于 PixelUMM benchmark | 3B MoT 架构 | 在视频理解 benchmark 中作为对比 |
| **HBridge** | Adobe | [2511.20520](https://arxiv.org/abs/2511.20520) | **非对称 H 形** MoT 变体，选择性桥接中间层 | 对 MoT 的改进：减少 40%+ attention 共享 |
| **LMFusion** | Meta | [2412.15188](https://arxiv.org/abs/2412.15188) | MoT 范式，适配预训练 LLM 为多模态生成 | 与原始 MoT 论文同属 Meta，BAGEL 引用为 [66] |

**关键发现 — MoT 架构溯源**：

**MoT 架构的原始提出者是 Meta（Facebook AI Research）**，而非 BAGEL。原始论文为：

> Weixin Liang, Lili Yu, Liang Luo, et al. "Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models." arXiv:2411.04996, 2024 年 11 月（TMLR 2025 接收）。

原始 MoT 论文提出"按模态解耦非嵌入参数（包括 FFN、注意力矩阵和层归一化），同时保持全局 self-attention"的架构，在 Chameleon 7B 和 Transfusion 设置中验证了其有效性——用 55.8% 的 FLOPs 匹配密集基线性能。

**BAGEL 的角色**是首个将 MoT 架构**大规模应用于统一多模态理解与生成**的开源模型，并提供了开源代码实现。BAGEL 论文 Section 2.4 明确引用原始 MoT 论文 [40] 和 LlamaFusion [66] 作为架构来源：

> "This type of architecture has been adopted by existing works [66], [40]."

其中 [40] = Liang et al. (原始 MoT 论文, arXiv:2411.04996)，[66] = LlamaFusion (Shi et al., arXiv:2412.15188)，两者均来自 Meta。

**PixelUMM 的直接代码来源是 BAGEL**。PixelUMM 的源码文件 `qwen3_navit.py` 开头明确标注：

> "Qwen3 MoT + NaViT runtime for PixelUMM. Based on Bagel/modeling/bagel/qwen2_navit.py (ByteDance, Apache-2.0)."

因此，MoT 架构的传播路径为：**Meta（原始提出）→ BAGEL（大规模应用+开源）→ PixelUMM（像素空间扩展）**。

> **出处**：arXiv:2411.04996 (原始 MoT 论文)；arXiv:2412.15188 (LlamaFusion)；arXiv:2505.14683 (BAGEL, Section 2.4)；[PixelUMM 源码注释](https://github.com/nv-tlabs/PixelUMM/blob/main/modeling/pixelumm/qwen3_navit.py)

### 12.3 世界动作模型（World Action Models / VLA）

MoT 架构在机器人学和视觉-语言-动作（VLA）领域被**大量采用**，成为 WAM 的事实标准骨干架构。以下是从 arXiv 检索到的代表性模型（均为 2026 年发表）：

| 模型 | arXiv | MoT 应用方式 | 专家数量 |
|------|-------|------------|---------|
| **SparseWAM** | [2611.00044](https://arxiv.org/abs/2611.00044) | 模态解耦 MoT + 因果 mask | 2（action + register） |
| **CoRAct** | [2610.39543](https://arxiv.org/abs/2610.39543) | 统一 MoT 建模 ECoT 推理+动作+未来预测 | 3 |
| **WorldFlow** | [2610.39278](https://arxiv.org/abs/2610.39278) | MoT 联合去噪视频+3D 运动 | 2（video + motion） |
| **Trace-WAM** | [2610.36840](https://arxiv.org/abs/2610.36840) | **Tri-MoT**（三分支 MoT） | 3（video + motion + action） |
| **AW-0** | [2610.30136](https://arxiv.org/abs/2610.30136) | MoT 联合建模 code+video+action | 3（code + video + action） |
| **UniWALM** | [2610.28626](https://arxiv.org/abs/2610.28626) | MoT 统一 video+action+language | 3 |
| **ULIFlow** | [2610.24286](https://arxiv.org/abs/2610.24286) | MoT 自编码器建模异构流 | 多模态 |
| **LAW** | [2610.23334](https://arxiv.org/abs/2610.23334) | MoT Experts 集成模态特定+共享计算 | 2（video + action） |
| **P³-WAM** | [2610.22363](https://arxiv.org/abs/2610.22363) | MoT 共享基础权重+模态特定适应 | 2（video + action） |
| **Ask4NV** | [2610.20610](https://arxiv.org/abs/2610.20610) | MoT 架构 + 模态感知 mask | 2（understanding + generation） |
| **DR-VLA** | [2610.18373](https://arxiv.org/abs/2610.18373) | MoT 框架 + flow-matching action expert | 2 |
| **DriveWLA** | [2610.17903](https://arxiv.org/abs/2610.17903) | MoT 三专家：World+Language+Action | 3 |
| **BridgeWAM** | [2610.14746](https://arxiv.org/abs/2610.14746) | MoT + Latent Bridge Queries 解耦 | 2（video + action） |
| **PLaW-VLA** | [2610.12285](https://arxiv.org/abs/2610.12285) | MoT + 结构化因果 attention | 2 |
| **VPP2** | [2610.10270](https://arxiv.org/abs/2610.10270) | MoT 学习隐式逆动力学 | 2（video + action） |
| **OpenWAM** | [2610.07922](https://arxiv.org/abs/2610.07922) | 开源 WAM 框架，MoT 为标准架构 | 可组合 |

> **出处**：arXiv 搜索 "Mixture-of-Transformers"，各论文摘要

### 12.4 MoT 架构的演化谱系

```
Meta 原始 MoT 论文 (2024.11, arXiv:2411.04996)
  │  ← MoT 架构原始提出：按模态解耦非嵌入参数 + 全局 self-attention
  │
  ├── LlamaFusion (2024.12, Meta, arXiv:2412.15188)
  │     └── 适配预训练 LLM 为多模态生成
  │
  ├── BAGEL (2025.05, ByteDance)
  │     └── 首个大规模开源应用：MoT + 双视觉编码器 + 广义因果注意力
  ├── PixelUMM (2026.09, NVIDIA)
  │     └── 继承 BAGEL MoT，扩展到像素空间+视频
  │
  ├── SenseNova-U1/U1.5 (SenseTime)
  │     └── 无编码器像素空间 + MoT 条件化设计
  │
  ├── TUNA/TUNA-2
  │     └── 无编码器像素空间 + MoT
  │
  ├── NEO
  │     └── 三轴 Native RoPE + MoT
  │
  ├── HBridge (2025.11, Adobe)
  │     └── MoT 变体：非对称 H 形，选择性桥接
  │
  └── WAM/VLA 生态 (2026, 16+ 论文)
        └── MoT 成为标准骨干：
            ├── 双专家（video + action）：LAW, P³-WAM, BridgeWAM, VPP2, ...
            ├── 三专家（video + action + X）：Trace-WAM, AW-0, DriveWLA, ...
            └── 可组合框架：OpenWAM
```

### 12.5 MoT 架构流行的原因分析

MoT 架构之所以被广泛采用，源于其在**架构效率**和**任务统一**之间的独特平衡：

1. **天然的统一性**：共享 self-attention 允许不同任务/模态的 token 在每层自然交互，无需额外的 cross-attention 接口（如 LTX-2.5 的双流架构）
2. **任务特化能力**：分离的 QKV/FFN 允许不同任务使用适合自身的投影和变换，避免"一刀切"的参数共享
3. **简洁的硬路由**：无需 MoE 的路由器网络和负载均衡，路由在数据层面确定，无额外计算开销
4. **兼容 LLM 预训练**：可直接从预训练 decoder-only LLM（如 Qwen3）初始化，理解专家继承 LLM 权重
5. **可扩展到多专家**：从 BAGEL/PixelUMM 的 2 专家扩展到 Trace-WAM 的 3 专家、DriveWLA 的 3 专家，甚至更多
6. **适配 WAM 需求**：WAM 需要联合建模视频预测+动作生成，MoT 的"共享 attention + 分离 FFN"天然适合这种"共享视觉先验 + 任务特定输出"的场景

### 12.6 MoT 架构的变体与改进

| 变体 | 模型 | 改进点 |
|------|------|--------|
| **标准 MoT** | BAGEL, PixelUMM | 共享 attention + 双专家 QKV/FFN |
| **Tri-MoT** | Trace-WAM | 扩展到三专家（video + motion + action） |
| **非对称 MoT** | HBridge, BridgeWAM | 选择性桥接（非全层共享 attention），减少计算 |
| **Cross-Expert Attention** | DriveWLA | 在共享 attention 基础上增加跨专家注意力控制 |
| **Latent Bridge Queries** | BridgeWAM | 用少量 latent query 替代全层 attention 桥接 |
| **MoT + 条件化设计** | SenseNova-U1, PixelUMM | 干净条件走理解专家，噪声目标走生成专家 |

> **出处**：arXiv:2511.20520 (HBridge)；arXiv:2610.36840 (Trace-WAM)；arXiv:2610.17903 (DriveWLA)；arXiv:2610.14746 (BridgeWAM)；arXiv:2609.38597 (PixelUMM)

### 12.7 总结

MoT（Mixture-of-Transformers）架构由 **Meta** 于 2024 年 11 月提出（arXiv:2411.04996），此后在两年内被 **106 篇论文**采用或讨论，已成为多模态统一建模的重要架构范式：

- **架构起源**：Meta 原始 MoT 论文（2024.11）→ LlamaFusion（2024.12）→ BAGEL 首次大规模应用（2025.05）→ PixelUMM/HBridge/WAM 生态广泛采用
- **统一多模态理解+生成领域**：BAGEL → PixelUMM → HBridge 等形成演化链
- **世界动作模型（WAM/VLA）领域**：16+ 篇论文采用 MoT 作为标准骨干，成为该领域的事实标准
- **核心设计理念**："共享 attention 保证跨任务交互，分离 QKV/FFN 允许任务特化"这一平衡设计，在统一建模场景中展现了广泛的适用性
- **持续演化**：从标准双专家 MoT 发展出 Tri-MoT、非对称 MoT、选择性桥接等变体，显示该架构仍有探索空间

---

## 13. BAGEL 模型架构详解

### 13.1 模型概述

BAGEL（ByteDance，2025 年 5 月）是 MoT（Mixture-of-Transformer-Experts）架构的**首个大规模开源应用**。MoT 架构原始由 Meta 提出（arXiv:2411.04996, 2024 年 11 月），BAGEL 首次将其大规模应用于统一多模态理解与生成，并提供了开源代码。PixelUMM 的 MoT 架构直接继承自 BAGEL——其源码文件 `qwen3_navit.py` 明确标注 "Based on Bagel/modeling/bagel/qwen2_navit.py (ByteDance, Apache-2.0)"。

BAGEL 是一个统一的多模态基础模型，原生支持多模态理解和生成，在万亿级交错多模态数据上预训练，展现出复杂多模态推理的涌现能力。

> **出处**：arXiv:2505.14683 — "Emerging Properties in Unified Multimodal Pretraining"；[项目主页](https://bagel-ai.org/)；[PixelUMM 源码注释](https://github.com/nv-tlabs/PixelUMM/blob/main/modeling/pixelumm/qwen3_navit.py)

### 13.2 模型架构

#### 13.2.1 整体架构

BAGEL 采用 **MoT（Mixture-of-Transformer-Experts）架构**，包含两个 Transformer 专家：

```
                    输入
                      ↓
          ┌──── 双视觉编码器 ────┐
          │                      │
    ViT 编码器               VAE 编码器
  (SigLIP2-so400m)         (FLUX VAE)
    理解导向                 生成导向
    语义特征                 像素重建特征
          │                      │
    MLP 连接器             2×2 Patch Embed
          │                      │
          └────── packed 序列 ───┘
                      ↓
    ┌──────────────────────────────────────┐
    │    Mixture-of-Transformer-Experts    │
    │                                      │
    │  ┌── 理解专家 ──┐  ┌── 生成专家 ──┐  │
    │  │ Norm/QKV/FFN │  │ Norm/QKV/FFN│  │
    │  │ (继承Qwen2.5) │  │ (完整复制)   │  │
    │  └──────┬───────┘  └───────┬──────┘  │
    │         └── 共享 Self-Attention ──┘  │
    │            (每层所有token交互)        │
    └──────────────────────────────────────┘
                      ↓
              文本: 自回归预测
              视觉: Rectified Flow 去噪
```

> **出处**：arXiv:2505.14683 — Section 2.2, Figure 2

#### 13.2.2 骨干网络

| 属性 | 值 |
|------|-----|
| 初始化来源 | Qwen2.5 LLM |
| 架构类型 | Decoder-only Transformer |
| 激活参数量 | 7B |
| 总参数量 | 14B（理解专家 7B + 生成专家 7B） |
| 归一化 | RMSNorm |
| 激活函数 | SwiGLU |
| 位置编码 | RoPE |
| KV 缓存 | GQA（Grouped Query Attention） |
| QK-Norm | 每个 attention block 添加（稳定训练） |

> **出处**：arXiv:2505.14683 — Section 2.2

#### 13.2.3 MoT 专家设计

BAGEL 论文详细比较了三种 Transformer 变体，并提供了**消融实验证据**支持 MoT 选择：

| 变体 | 设计 | 参数量 | FLOPs | 实验结果 |
|------|------|--------|-------|---------|
| **Dense** | 标准密集 Transformer，所有 token 共享所有参数 | 1× | 1× | 基线 |
| **MoE** | 仅复制 FFN 作为生成专家，QKV 共享 | ~2× | 1× | 优于 Dense，但不如 MoT |
| **MoT** | 复制**所有可训练参数**（Norm/QKV/FFN）作为生成专家 | ~2× | 1× | **最优**，尤其生成任务 |

**关键消融发现**（Figure 3, 1.5B 模型实验）：
- MoT 在生成 MSE 损失上**始终优于** Dense 和 MoE，收敛最快且最终损失最低
- MoT 在理解 CE 损失上也表现最佳
- **核心结论**：分离理解和生成的参数容量可以缓解两种模态特定学习目标的竞争

**硬路由规则**：
```
文本 token    → 理解专家（原始 Qwen2.5 参数）
ViT token     → 理解专家（原始 Qwen2.5 参数）
VAE token     → 生成专家（复制的参数）
```

> **出处**：arXiv:2505.14683 — Section 2.4, Figure 3

#### 13.2.4 双视觉编码器

与 PixelUMM 的无编码器设计不同，BAGEL 使用**双视觉编码器**：

**理解编码器**：
| 属性 | 值 |
|------|-----|
| 模型 | SigLIP2-so400m/14 |
| 初始分辨率 | 384×384（固定） |
| 最大分辨率 | 980×980（位置插值后） |
| 原生比例 | NaViT 支持 |
| 连接器 | 2 层 MLP（匹配 LLM 隐藏维度） |
| 特征类型 | 语义特征 |

**生成编码器**：
| 属性 | 值 |
|------|-----|
| 模型 | FLUX VAE |
| 下采样率 | 8× |
| Latent 通道 | 16 |
| Patch 嵌入 | 2×2（进一步减少空间尺寸，匹配 LLM 隐藏维度） |
| 训练状态 | 冻结 |
| 特征类型 | 像素重建特征 |

> **出处**：arXiv:2505.14683 — Section 2.2

#### 13.2.5 时间步条件化

BAGEL **不使用 AdaLN**，而是将时间步嵌入**直接添加到 VAE token 的初始隐藏状态**：

> "we follow [17] and add a timestep embedding directly to the initial hidden states of VAE tokens, instead of using AdaLN as in conventional diffusion transformers. This modification preserves performance while yielding a cleaner architecture."

这与 PixelUMM 进一步省略时间步条件化形成对比：
- **BAGEL**：时间步嵌入 → 直接加到 VAE token 隐藏状态
- **PixelUMM**：完全无时间步条件化（模型从噪声本身推断损坏程度）

> **出处**：arXiv:2505.14683 — Section 2.2

#### 13.2.6 广义因果注意力

BAGEL 使用与 PixelUMM 相同的**广义因果注意力**机制（PixelUMM 继承自 BAGEL）：

**注意力规则**：
1. Token 按模态分组为连续的 split（文本 split、ViT split、VAE split）
2. 一个 split 可以 attend 到所有前面的 split
3. 文本 split 内部：因果注意力
4. 视觉 split 内部：双向注意力

**交错生成中的三种视觉 token**：
| Token 类型 | 用途 | 注意力可见性 |
|-----------|------|------------|
| 噪声 VAE token | Rectified Flow 训练目标 | 后续 token 不可 attend |
| 干净 VAE token | 条件（已生成的图像） | 后续 token 可 attend |
| ViT token | 统一输入格式，提升交错生成质量 | 后续 token 可 attend |

**Diffusion Forcing 策略**：多图像生成时，对不同图像添加独立噪声级别，每张图条件于前图的噪声表示。

**实现**：PyTorch FlexAttention，相比朴素 SDPA 实现 ~2× 加速。推理时缓存干净 VAE 和 ViT token 的 KV 对。

> **出处**：arXiv:2505.14683 — Section 2.3

#### 13.2.7 训练目标

| 目标 | 方法 | 损失 | 权重 |
|------|------|------|------|
| 文本 | Next-Token-Prediction（自回归） | Cross-Entropy | 0.25 |
| 视觉 | Rectified Flow（diffusion） | MSE | 1.0 |

**损失权重设计的经验依据**：
- 较大学习率有利于 MSE（生成）收敛，较小学习率有利于 CE（理解）收敛
- 通过损失权重（CE:MSE = 0.25:1）平衡两种目标
- 生成数据采样比例应显著高于理解数据（80% vs 20%）

> **出处**：arXiv:2505.14683 — Section 2.2, 4.1, 4.2, Table 3

### 13.3 训练流程

BAGEL 采用 4 阶段训练：

| 阶段 | 步数 | Token 量 | 关键内容 |
|------|------|---------|---------|
| Alignment | 5K | 4.9B | 训练 MLP 连接器，冻结 ViT 和 LLM，378×378 固定分辨率 |
| Pre-training (PT) | 200K | 2.5T | 除 VAE 外全部可训练，原生分辨率，最大上下文 16K |
| Continued Training (CT) | 100K | 2.6T | 提升分辨率，增加交错数据比例，最大上下文 40K |
| SFT | 15K | 72.7B | 高质量微调子集 |

**数据组成**（PT 阶段采样比例）：
- 文本：5%
- 图文对（T2I）：60%
- 图文对（I2T）：10%
- 交错理解数据：10%
- 交错生成数据（视频+Web）：15%

**总训练数据量**：~5.2T tokens

> **出处**：arXiv:2505.14683 — Section 4, Table 3

### 13.4 BAGEL 与 PixelUMM 的架构对比

| 维度 | BAGEL | PixelUMM |
|------|-------|----------|
| **骨干 LLM** | Qwen2.5 | Qwen3 |
| **MoT 架构** | ✅ 首个大规模开源应用 | ✅ 继承自 BAGEL |
| **视觉编码** | 双编码器（ViT + VAE） | 无编码器（单层线性投影） |
| **理解编码器** | SigLIP2-so400m/14 ViT | 无（像素直接投影） |
| **生成编码器** | FLUX VAE（冻结） | 无（像素直接投影） |
| **时间步条件化** | 嵌入直接加到 VAE token | 完全无 |
| **位置编码** | 2D（ViT/VAE token） | 3-axis Native RoPE（T/H/W） |
| **视觉表示空间** | VAE latent 空间 | 像素空间 |
| **注意力** | 广义因果（FlexAttention） | 广义因果（FlexAttention + FlashAttn） |
| **视频支持** | 通过交错视频数据 | 原生 3D patchify（4×16×16 tubelets） |
| **参数量** | 7B 激活 / 14B 总 | 8B |
| **训练规模** | ~5.2T tokens | 未公开总量 |
| **开源许可** | 开源（代码+权重） | Apache-2.0 |

**继承与传播链**：
```
Meta 原始 MoT 论文 (2024.11, arXiv:2411.04996)
  │  ← MoT 架构原始提出：共享 self-attention + 模态特定参数解耦
  │
  ├── LlamaFusion (2024.12, Meta, arXiv:2412.15188)
  │     └── 适配预训练 LLM 为多模态生成
  │
  ├── BAGEL (2025.05, ByteDance)
  │     └── 首个大规模开源应用：MoT + 双视觉编码器 + 广义因果注意力
  │     │
  │     └── PixelUMM (2026.09, NVIDIA)
  │          ├── 继承：MoT 架构、广义因果注意力、硬路由
  │          ├── 改变：Qwen2.5 → Qwen3，2D → 3D RoPE
  │          ├── 简化：移除 ViT/VAE → 单层线性投影
  │          ├── 简化：移除时间步条件化
  │          └── 扩展：图像 → 图像+视频（3D patchify）
  │
  └── WAM/VLA 生态 (2026, 16+ 论文)
        └── MoT 成为标准骨干
```

> **出处**：arXiv:2505.14683 — 全文；arXiv:2609.38597 — 全文；[PixelUMM 源码注释](https://github.com/nv-tlabs/PixelUMM/blob/main/modeling/pixelumm/qwen3_navit.py)

### 13.5 BAGEL 的设计哲学

BAGEL 论文明确阐述了其架构选择的核心理念：

> "Our primary objective is to maximize the capacity of the model without introducing heuristic bottlenecks or task-specific constraints commonly employed in previous models."

**三种设计空间的比较**（BAGEL 选择第三种）：

| 设计 | 描述 | 优势 | 劣势 |
|------|------|------|------|
| Quantized AR | 离散视觉 token 自回归生成 | 直接复用 LLM 基础设施 | 生成质量差，推理慢 |
| External Diffuser | LLM + 外部扩散模块 | 快速收敛，少量数据 | 瓶颈连接器，信息损失 |
| **Integrated Transformer** | LLM 和 diffusion 统一在单个 Transformer 中 | **无瓶颈，无损交互，可扩展** | 训练计算量大 |

BAGEL 选择 Integrated Transformer（MoT）的核心原因：**无瓶颈的上下文交互**使得理解和生成模块可以在所有 Transformer block 中无损交互，这对长上下文多模态推理至关重要。

> **出处**：arXiv:2505.14683 — Section 2.1

### 13.6 涌现能力

BAGEL 论文报告了随着交错多模态预训练规模扩大而出现的**涌现模式**：

1. **基础能力**：多模态理解和高保真生成首先收敛
2. **中级能力**：复杂编辑和自由形式视觉操控能力出现
3. **高级能力**：长上下文推理开始惠及多模态理解和生成

这些涌现能力包括：
- 自由形式图像操控（free-form image manipulation）
- 未来帧预测（future frame prediction）
- 3D 操控（3D manipulation）
- 世界导航（world navigation）

> **出处**：arXiv:2505.14683 — Section 1, 6

---

## 14. 像素空间生成模型全景与技术成熟度分析

### 14.1 像素空间生成模型全景

#### 14.1.1 纯像素空间生成模型（非统一多模态）

这些模型专注于在像素空间进行图像/视频生成，不涉及多模态理解任务。

| 模型 | 来源 | 时间 | arXiv | 核心特点 |
|------|------|------|-------|---------|
| **JiT** | — | 2025 | — | 首创 clean-pixel prediction，无 VAE。PixelUMM 直接继承其方法并扩展到视频 |
| **PixelDiT** | — | 2025 | — | 像素空间 diffusion transformer，移除 VAE 瓶颈 |
| **MiniT2I** | — | 2025 | — | 极简 T2I，无时间步条件化。PixelUMM 继承其无 AdaLN 设计 |
| **PixRestore** | — | 2026.08 | 2608.16793 | VAE-free 像素空间 DiT 用于图像修复，仅 50M 参数，单步推理 |
| **PixWorld** | — | 2026.07 | 2607.05373 | 像素空间 3D 场景生成与重建统一，移除 VAE/RAE |
| **Observation Operators** | — | 2026.08 | 2608.21885 | 改善像素空间 diffusion 优化，ImageNet-256 FID 1.52 |
| **pDiT 综述** | — | 2026.07 | 2607.17585 | 系统综述像素空间 Diffusion Transformer，总结方法与挑战 |

> **出处**：arXiv:2609.38597 — PixelUMM 论文引用 JiT[66]、PixelDiT[168]、MiniT2I[138]；arXiv:2608.16793, 2607.05373, 2608.21885, 2607.17585

#### 14.1.2 像素空间统一多模态模型（理解+生成）

这些模型在像素空间同时支持视觉理解和生成，是无编码器路线的代表。

| 模型 | 来源 | 特点 | 开源/商业状态 |
|------|------|------|-------------|
| **PixelUMM** | NVIDIA (2026.09) | MoT + 像素空间 + 图像/视频，8B，Qwen3 骨干 | 开源（Apache-2.0，代码+权重） |
| **SenseNova-U1 / U1.5** | SenseTime | 无编码器像素空间，引入 MoT 条件化设计和 T/H/W MRoPE 拓扑 | 商业平台（SenseNova），论文未完全开源 |
| **TUNA / TUNA-2** | — | 无编码器像素空间统一理解+生成 | 论文公开 |
| **NEO** | — | 三轴 Native RoPE + 像素空间统一模型 | 论文公开 |
| **HiDream-O1-Image** | — | 像素级架构，集成图像生成与编辑 | 论文公开 |

> **出处**：arXiv:2609.38597 — PixelUMM 论文 Section 1 引用 TUNA-2[77]、SenseNova-U1[32]、SenseNova-U1.5[30]、NEO[28]；PixelUMM 论文 Related Work 引用 HiDream-O1-Image[8]

#### 14.1.3 相关但非纯像素空间方法

| 方法 | 特点 | 与像素空间的关系 |
|------|------|----------------|
| **RAE (Representation Auto-Encoders)** | 用预训练语义编码器替代 VAE 编码器 + 学习的解码器 | 移除传统 VAE，但仍有编码器（非纯像素空间） |
| **REPA (Representation Alignment)** | 将 DiT 中间特征与预训练表示模型对齐 | 加速训练，但仍依赖 VAE 做 latent diffusion |
| **UniRA** (arXiv:2610.39948) | 统一 REPA + RAE，表示空间 tokenizer 同时作扩散目标和对齐锚点 | 融合表示空间与像素空间优势 |

> **出处**：arXiv:2610.39948 (UniRA)；PixelUMM 论文 Related Work

### 14.2 技术成熟度评估

**结论：不成熟，仍在研究探索阶段。**

以下五个维度的证据支持这一判断：

#### 14.2.1 模型规模受限

| 模型 | 参数量 | 对比（潜空间模型） |
|------|--------|------------------|
| PixelUMM | 8B | MiniMax-H3: 33B, MAGI-2: 114B, LTX-2.5: 22B |
| PixRestore | 50M | SD3: 8B, FLUX: 12B |
| TUNA-2 | ~7B | BAGEL: 14B |

PixelUMM 的 F5 实验表明 8B 模型收敛速度是 1.7B 的 3 倍，暗示像素空间建模需要更大规模才能补偿无 VAE 压缩带来的序列长度增加。当前最大像素空间统一模型仅 8B，远小于潜空间路线的 22B-114B。

> **出处**：arXiv:2609.38597 — Section 3.1.5 (F5)

#### 14.2.2 性能未达领先

| 维度 | PixelUMM | 主流潜空间模型 |
|------|----------|--------------|
| 图像理解 MMMU | 41.67 | Qwen3-VL: 69.60, BAGEL: 55.30 |
| 图像生成分辨率 | 512² | SD3/FLUX: 1024²+, MiniMax-H3: 2K |
| 视频生成分辨率 | 320×176 | MAGI-2: 1080p, LTX-2.5: 1080p |
| 视频时长 | 4s (96帧/24fps) | LTX-2.5: 20s, MiniMax-H3: 15s |
| 音频 | 不支持 | H3/MAGI-2/LTX-2.5 均支持 |

> **出处**：arXiv:2609.38597 — Section 3.2 benchmark 结果

#### 14.2.3 固有技术挑战未解决

**Patch 伪影**（F3 实验）：
- 线性解码器在 16 像素 patch 边界产生网格对齐的强度跳变
- 在低纹理区域（如天空、墙壁）尤为明显
- 高 CFG（~6）下伪影加剧
- PixelShuffle 卷积解码器可缓解，但增加 5.6-9.6× 计算量

**训练动态不稳定**（F4 实验）：
- 像素空间训练出现偶发损失尖峰
- 梯度范数与 VAE 空间相近，但尖峰表明优化景观更粗糙

**序列长度问题**：
- 无 VAE 压缩，512×512 图像 = 1024 个 patch token
- VAE 压缩后同等图像可能仅 256 个 token
- 更长的序列 → 更大的 attention 矩阵 → 更多计算和内存

> **出处**：arXiv:2609.38597 — Section 3.1.3 (F3), 3.1.4 (F4)

#### 14.2.4 无主流商业落地

当前所有主流商业图像/视频生成产品均使用 VAE 潜空间：

| 商业产品 | 架构 | 是否像素空间 |
|---------|------|------------|
| DALL-E 3 (OpenAI) | 潜空间 diffusion | 否 |
| Midjourney | 潜空间 diffusion | 否 |
| Stable Diffusion 3 (Stability AI) | 潜空间 Rectified Flow | 否 |
| FLUX (Black Forest Labs) | 潜空间 Rectified Flow | 否 |
| GPT-4o 图像生成 (OpenAI) | 未公开（推测潜空间） | 否 |
| Gemini 图像生成 (Google) | 未公开（推测潜空间） | 否 |
| MiniMax Hailuo (MiniMax) | 潜空间 Omni-Transformer | 否 |
| SenseNova (SenseTime) | 可能有像素空间组件 | 未确认 |

唯一可能涉及像素空间商业应用的是 SenseTime 的 SenseNova 平台（SenseNova-U1/U1.5 的来源），但其商业部署的具体架构未公开确认是否为纯像素空间。

#### 14.2.5 学术界仍在探索

2026 年 7 月发表的综述论文 "Pixel-Space Diffusion Transformers"（arXiv:2607.17585）系统回顾了该领域，明确指出像素空间建模仍面临以下挑战：

> "high-dimensional modeling, including noise scheduling, loss weighting, token efficiency, and scalable architecture design"

该综述将像素空间建模定位为"有前景的统一多模态系统基础"，但承认其"引入了高维建模的挑战"。

> **出处**：arXiv:2607.17585 — "Pixel-Space Diffusion Transformers" 综述

### 14.3 像素空间 vs 潜空间的根本权衡

```
潜空间路线（主流商业方案）：
  像素 → VAE压缩(8-32×) → 短序列 → 高效计算 → 高质量生成
  优势：成熟、高效、高质量、大规模验证
  代价：VAE训练/维护开销 + 信息损失 + 双视觉接口复杂性

像素空间路线（研究探索中）：
  像素 → Patchify → 单层线性投影 → 长序列 → 架构极简 → 无信息损失
  优势：无VAE依赖、统一视觉接口、无信息损失、架构简洁
  代价：计算成本高 + patch伪影 + 训练不稳定 + 需要更大模型规模
```

### 14.4 像素空间路线的未来发展条件

基于当前证据，像素空间路线要形成真正竞争力，需要以下条件成熟：

1. **模型规模扩展至 30B+**：F5 实验的 scaling 趋势表明，像素空间建模从模型扩展中受益显著（8B 收敛速度 3× 于 1.7B），但当前规模仍不足以补偿无 VAE 压缩的序列长度增加

2. **Patch 伪影的有效解决方案**：当前 PixelShuffle 卷积解码器提供了缓解方案（F3 实验），但增加了 5.6-9.6× 计算量。需要更高效的伪影抑制方案

3. **高效注意力机制**：更长的像素 token 序列需要更高效的注意力（如线性注意力、稀疏注意力）来控制计算成本

4. **训练稳定性改进**：解决偶发损失尖峰（F4 实验），使训练动态更平滑

5. **高分辨率策略**：当前像素空间模型分辨率受限（512²），需要类似潜空间模型的多阶段/多 tile 高分辨率方案

### 14.5 总结

像素空间生成是一个**有学术价值但尚未成熟的技术方向**：

- **学术价值**：消除 VAE 瓶颈、统一理解与生成的视觉接口、无信息损失——这些理论优势使其成为有吸引力的研究方向
- **代表模型**：PixelUMM（NVIDIA，开源）、SenseNova-U1/U1.5（SenseTime，商业平台）、TUNA-2、NEO 等构成了当前像素空间统一多模态模型的核心阵容
- **商业落地**：无主流商业产品采用纯像素空间方案。所有主流商业生成模型（DALL-E 3、Midjourney、SD3、FLUX 等）均使用 VAE 潜空间
- **成熟度判断**：仍在研究探索阶段，需在模型规模、性能、效率、稳定性和高分辨率策略等方面取得突破后才能挑战潜空间路线的主流地位

> **出处**：arXiv:2609.38597 (PixelUMM)；arXiv:2607.17585 (pDiT 综述)；arXiv:2608.16793 (PixRestore)；arXiv:2607.05373 (PixWorld)；arXiv:2608.21885 (Observation Operators)；各商业产品官方文档
