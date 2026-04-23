# OmniVoice LoRA 训练指南

在 ComfyUI 中使用 LoRA（低秩适应）微调 OmniVoice，让模型学习你的声音。单 GPU 即可完成。

## 前置条件

1. **训练依赖** — 通过 ComfyUI-Manager 安装，或手动执行：
   ```bash
   pip install peft safetensors
   ```
2. **GPU 显存 >= 8 GB**（OmniVoice ~0.6B 参数 + LoRA 开销）
3. **语音数据** — 配对的音频文件 + `.txt` 文本

## 准备数据集

### 目录结构

创建一个文件夹，放置配对的音频 + 文本文件。每个音频文件必须有同名 `.txt` 文件：

```
my_voice_data/
├── sample_001.wav
├── sample_001.txt       ← "你好，欢迎收听我的播客。"
├── sample_002.wav
├── sample_002.txt       ← "今天我们来聊聊微调。"
├── sample_003.wav
├── sample_003.txt       ← "让我们开始吧。"
└── ...
```

### 音频要求

- **格式**：`.wav`、`.mp3`、`.flac`、`.ogg` 或 `.m4a`
- **时长**：每条 3–20 秒
- **质量**：干净、无背景噪音、音量一致 — **输入质量越高，输出质量越高**
- **内容**：自然朗读句子、段落或文稿
- **数量**：至少 20–100 条（越多越好）

### 音频总时长

| 总时长 | 质量 | 速度 |
|---|---|---|
| 2–5 分钟 | 基本效果的最低要求 | 快 |
| 6–10 分钟 | 推荐 — 声音克隆效果好 | 适中 |
| 11–30 分钟 | 最佳效果 — 声音复制最强 | 最慢 |

### 文本要求

- 纯文本，每个 `.txt` 文件一句或一段话
- 内容必须与音频中实际说的话一致
- 无需特殊格式

## ComfyUI 工作流

训练工作流使用 **3 个节点** 串联：

### 节点 1：OmniVoice Train Config

配置训练超参数：

| 参数 | 默认值 | 推荐范围 | 说明 |
|---|---|---|---|
| `learning_rate` | 5e-5 | — | 保持默认 — 已测试有效 |
| `lora_rank` | 32 | 16 – 64 | 32 是最佳值。越大容量越大，显存越多 |
| `lora_alpha` | 16 | 8 – 32 | 通常为 rank 的一半 |
| `lora_dropout` | 0.0 | 0.0 – 0.1 | 过拟合时添加 |
| `warmup_steps` | 100 | 50 – 200 | 约为总步数的 10% |
| `grad_accum_steps` | 1 | 1 – 4 | 显存不足时增大 |
| `weight_decay` | 0.01 | 0.0 – 0.05 | 正则化 |
| `target_modules` | q_proj,k_proj,v_proj,o_proj | — | 加 gate_proj,up_proj,down_proj 可更深适配 |
| `sequence_packing` | False | — | 将多个样本打包成一个序列。推荐开启，训练质量最佳 |
| `batch_tokens` | 4096 | 2048 – 8192 | 打包序列的最大 token 长度。4096 约需 24GB 显存 |
| `torch_compile` | False | — | 编译模型以加速训练。强烈建议在启用序列打包时开启。启动时有约 30-60 秒的一次性编译开销 |
| `train_audio_layers` | True | True | 推荐 — 适配音频接口层 |

### 节点 2：OmniVoice Dataset Maker

- `audio_directory` 设置为数据集文件夹路径
- `language_id` 设置语言代码（如 `en`、`zh`、`ja`、`de`）
- 输出：JSONL 清单文件路径

### 节点 3：OmniVoice LoRA Trainer

- 连接节点 1 的 `train_config` 输出
- 设置 `dataset_path` 为节点 2 的输出
- 配置：
  - `max_steps`：250–1000 步效果最佳（见下方训练建议）
  - `save_every_steps`：250（保存检查点用于对比）
  - `output_name`：LoRA 输出的文件夹名称

**注意**：训练过程会阻塞 ComfyUI 界面。使用中断按钮取消。

## 使用训练好的 LoRA

训练完成后，LoRA 保存到 `models/loras/<output_name>/`。所有 4 个 OmniVoice 推理节点都有 `lora_name` 下拉菜单：

1. 打开任意 OmniVoice TTS 节点（长文本、声音克隆、声音设计、多说话人）
2. 从 `lora_name` 下拉菜单选择你的 LoRA
3. 生成语音 — LoRA 声音会自动应用

## 训练建议

### 实测配置（40 条音频，每条 8–20 秒）

使用 40 条音频（每条 8–20 秒）+ 同名 `.txt` 文本进行实测，最佳设置：

- **lora_rank**：32
- **learning_rate**：5e-5（默认）
- **sequence_packing**：True
- **batch_tokens**：4096
- **torch_compile**：True
- **train_audio_layers**：True
- **最佳检查点范围**：250–1000 步

使用以上设置，loss 从约 4.8 平滑下降到约 1.7。训练速度约 0.4s/步（RTX 5090，1250 步约 500 秒）。

**更多步数不一定更好。** 过度训练会导致过拟合 — 声音可能变得生硬、失真或破损。此数据集的最佳范围是 **250–750 步**。超过 1000 步的检查点开始退化。务必测试多个检查点，选择最接近目标声音的那个。

### 实际案例

- **数据集**：40 条音频 + 40 个同名 `.txt` 文本，总时长约 7 分钟
- **设置**：rank=32, LR=5e-5, sequence_packing=True, batch_tokens=4096, torch_compile=True
- **训练**：1250 步，RTX 5090 约 500 秒
- **最佳检查点**：250–750 步

### 何时停止训练

观察 loss 并逐步测试检查点：

| Loss 范围 | 声音质量 |
|---|---|
| ~5–6 | 刚开始，几乎听不出 |
| ~3–4 | 有些相似 |
| ~2–3 | 较好匹配 |
| ~1–2 | 强克隆 |
| <1 | 过拟合风险 — 可能声音生硬 |

每 250 步保存检查点并逐一测试。声音最接近目标时停止。

### 常见问题

- **显存不足 (OOM)**：降低 `lora_rank` 到 16，降低 `batch_tokens` 到 2048，或增大 `grad_accum_steps` 到 2–4
- **训练太慢**：启用 `sequence_packing` + `torch_compile=True` + `batch_tokens=4096`
- **Loss 不下降**：降低 `learning_rate` 到 1e-5，检查音频质量
- **过拟合**（声音生硬）：减少 `max_steps`，添加 `lora_dropout` 0.05–0.1
- **音频乱码**：确保源音频干净、兼容 24kHz

### 训练内容说明

- **Qwen3-0.6B 主干的 LoRA**：作用于注意力投影（q、k、v、o）。学习你的声音的文本→音频 token 映射。
- **音频层**（`train_audio_layers=True` 时）：`audio_embeddings` 和 `audio_heads`。负责音频与 LLM 的接口，适配你的声音 token 分布。

### 序列打包

推荐启用**序列打包**以获得最佳训练质量。它将多个样本拼接成一个固定长度的序列，使用块注意力掩码防止样本间的交叉注意力。这与 OmniVoice 上游训练使用的技术相同。

使用 `batch_tokens` 控制打包序列长度：
- `4096` 约需 24GB 显存（推荐起点）
- `2048` 显存更少（约 16GB）
- `8192` 显存更多（约 32GB+）

**重要**：使用序列打包时，请同时开启 `torch_compile`。没有编译时，PyTorch 的 flex_attention 会展开完整的注意力矩阵，速度极慢。编译后会生成融合核心，高效运行。

### 架构图

```
OmniVoice (Qwen3-0.6B)
├── llm              ← LoRA 作用于此（注意力投影）
├── audio_embeddings ← train_audio_layers=True 时训练
└── audio_heads      ← train_audio_layers=True 时训练
```
