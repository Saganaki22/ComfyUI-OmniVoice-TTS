# ComfyUI-OmniVoice-TTS

**OmniVoice TTS ComfyUI节点** — 零样本多语言语音合成，支持声音克隆和声音设计。支持**600+种语言**，质量一流。

[![OmniVoice Model](https://img.shields.io/badge/%F0%9F%A4%97%20OmniVoice%20Model-k2--fsa/OmniVoice-blue)](https://huggingface.co/k2-fsa/OmniVoice)
[![OmniVoice-bf16](https://img.shields.io/badge/%F0%9F%A4%97%20OmniVoice--bf16-drbaph/OmniVoice--bf16-blue)](https://huggingface.co/drbaph/OmniVoice-bf16)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Demo%20Space-OmniVoice-yellow)](https://huggingface.co/spaces/k2-fsa/OmniVoice)
[![Demo](https://img.shields.io/badge/Demo%20Page-OmniVoice-green)](https://zhu-han.github.io/omnivoice/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.00688-b31b1b)](https://arxiv.org/abs/2604.00688)
[![GitHub](https://img.shields.io/badge/GitHub%20OmniVoice-k2--fsa/OmniVoice-black)](https://github.com/k2-fsa/OmniVoice)

## 特性

- **600+种语言** — 零样本TTS模型中语言覆盖最广
- **声音克隆** — 3-15秒参考音频即可克隆任意声音
- **声音设计** — 通过文字描述创建合成声音（性别、年龄、音调、口音）
- **多说话人对白** — 使用 `[Speaker_N]:` 标签生成多人对话
- **快速推理** — RTF低至0.025（比实时快40倍）
- **非语言表达** — 内联标签如 `[laughter]`、`[sigh]`、`[sniff]`
- **SageAttention支持** — 通过monkey-patch Qwen3Attention实现GPU优化注意力（仅CUDA，SM80+）
- **自动下载** — 首次使用时自动从HuggingFace下载模型
- **Whisper ASR缓存** — 预加载Whisper避免每次重新下载
- **显存高效** — 自动CPU卸载，VBAR/aimdo集成，智能缓存失效
- **LoRA训练** — 使用PEFT LoRA微调OmniVoice学习你的声音，支持序列打包和torch.compile

## 安装

### 方法1：ComfyUI Manager（推荐）
在ComfyUI Manager中搜索"OmniVoice"并点击安装。

### 方法2：手动安装
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/saganaki22/ComfyUI-OmniVoice-TTS.git
cd ComfyUI-OmniVoice-TTS
python install.py
```

### 为什么使用 `--no-deps`？
`omnivoice` pip包指定了 `torch==2.8.*` 作为依赖，这可能会将您的PyTorch降级为CPU版本，导致ComfyUI无法使用GPU加速。我们在 `install.py` 中通过 `--no-deps` 安装 `omnivoice` 来绕过这个问题，然后单独安装ComfyUI未提供的缺失依赖。

### 如果PyTorch被破坏
如果其他包意外降级了您的PyTorch，请参阅 [PyTorch兼容性矩阵](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS/blob/main/pytorch_compatibility_matrix.md) 获取与您环境匹配的恢复命令。

## 节点

<details>
<summary><strong>1. OmniVoice Longform TTS</strong> — 长文本语音合成，智能分句，可选声音克隆</summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | COMBO | (自动) | OmniVoice模型检查点 |
| text | STRING, 多行 | `"你好..."` | 要合成的文本 |
| ref_text | STRING, 多行 | "" | 参考音频转录文本（空=自动识别） |
| steps | INT | 32 | 扩散步数（4-64，16=快，64=最佳） |
| guidance_scale | FLOAT | 2.0 | 分类器自由引导比例（0-10） |
| t_shift | FLOAT | 0.1 | 噪声调度时间步偏移（0-1） |
| speed | FLOAT | 1.0 | 语速（0.5-2.0，>1=加快） |
| duration | FLOAT | 0.0 | 固定时长秒数（0=自动） |
| device | COMBO | auto | `auto`、`cuda`、`cpu`、`mps`、`xpu` |
| dtype | COMBO | auto | `auto`、`bf16`、`fp16`、`fp32` |
| attention | COMBO | auto | `auto`、`eager`、`sage_attention` |
| seed | INT | 0 | 随机种子（0=随机） |
| words_per_chunk | INT | 100 | 每块词数（0=不分块） |
| position_temperature | FLOAT | 5.0 | 掩码位置选择温度（0=贪心，越高越随机） |
| class_temperature | FLOAT | 0.0 | token采样温度（0=贪心） |
| layer_penalty_factor | FLOAT | 5.0 | 深层码本惩罚因子 |
| denoise | BOOLEAN | True | 在输入前添加去噪token以获得更干净输出 |
| preprocess_prompt | BOOLEAN | True | 预处理参考音频（去除静音，添加标点） |
| postprocess_output | BOOLEAN | True | 后处理生成音频（去除长静音） |
| keep_model_loaded | BOOLEAN | True | 保持模型加载（运行间自动卸载到CPU） |
| instruct | STRING | "" | 方言/风格指令，仅支持特定值 — 见[方言/风格指令](#方言风格指令)。应用于每个分块 |

**可选输入：**
- `ref_audio` — 声音克隆参考音频（3-15秒最佳）
- `whisper_model` — 预加载的Whisper ASR模型

</details>

<details>
<summary><strong>2. OmniVoice Voice Clone TTS</strong> — 从参考音频克隆声音</summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | COMBO | (自动) | OmniVoice模型检查点 |
| text | STRING, 多行 | `"你好..."` | 要用克隆声音合成的文本 |
| ref_audio | AUDIO | 必填 | 参考音频（3-15秒） |
| ref_text | STRING, 多行 | "" | 转录文本（空=Whisper自动识别） |
| steps | INT | 32 | 扩散步数（4-64） |
| guidance_scale | FLOAT | 2.0 | 分类器自由引导比例（0-10） |
| t_shift | FLOAT | 0.1 | 噪声调度时间步偏移（0-1） |
| speed | FLOAT | 1.0 | 语速（0.5-2.0） |
| duration | FLOAT | 0.0 | 固定时长秒数（0=自动） |
| device | COMBO | auto | `auto`、`cuda`、`cpu`、`mps`、`xpu` |
| dtype | COMBO | auto | `auto`、`bf16`、`fp16`、`fp32` |
| attention | COMBO | auto | `auto`、`eager`、`sage_attention` |
| seed | INT | 0 | 随机种子（0=随机） |
| position_temperature | FLOAT | 5.0 | 掩码位置选择温度（0=贪心） |
| class_temperature | FLOAT | 0.0 | token采样温度（0=贪心） |
| layer_penalty_factor | FLOAT | 5.0 | 深层码本惩罚因子 |
| denoise | BOOLEAN | True | 在输入前添加去噪token |
| preprocess_prompt | BOOLEAN | True | 预处理参考音频 |
| postprocess_output | BOOLEAN | True | 后处理生成音频 |
| keep_model_loaded | BOOLEAN | True | 保持模型加载 |
| instruct | STRING | "" | 方言/风格指令，仅支持特定值 — 见[方言/风格指令](#方言风格指令) |

**可选输入：**
- `whisper_model` — 预加载的Whisper ASR模型

</details>

<details>
<summary><strong>3. OmniVoice Voice Design TTS</strong> — 通过文字描述设计声音，无需参考音频</summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | COMBO | (自动) | OmniVoice模型检查点 |
| text | STRING, 多行 | `"你好..."` | 要用设计声音合成的文本 |
| voice_instruct | STRING, 多行 | `"female, low pitch..."` | 声音属性描述 |
| steps | INT | 32 | 扩散步数（4-64） |
| guidance_scale | FLOAT | 2.0 | 分类器自由引导比例（0-10） |
| t_shift | FLOAT | 0.1 | 噪声调度时间步偏移（0-1） |
| speed | FLOAT | 1.0 | 语速（0.5-2.0） |
| duration | FLOAT | 0.0 | 固定时长秒数（0=自动） |
| device | COMBO | auto | `auto`、`cuda`、`cpu`、`mps`、`xpu` |
| dtype | COMBO | auto | `auto`、`bf16`、`fp16`、`fp32` |
| attention | COMBO | auto | `auto`、`eager`、`sage_attention` |
| seed | INT | 0 | 随机种子（0=随机） |
| position_temperature | FLOAT | 5.0 | 掩码位置选择温度（0=贪心） |
| class_temperature | FLOAT | 0.0 | token采样温度（0=贪心） |
| layer_penalty_factor | FLOAT | 5.0 | 深层码本惩罚因子 |
| denoise | BOOLEAN | True | 在输入前添加去噪token |
| postprocess_output | BOOLEAN | True | 后处理生成音频 |
| keep_model_loaded | BOOLEAN | True | 保持模型加载 |

</details>

<details>
<summary><strong>4. OmniVoice Multi-Speaker TTS</strong> — 使用 <code>[Speaker_N]:</code> 标签生成多说话人对白</summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | COMBO | (自动) | OmniVoice模型检查点 |
| text | STRING, 多行 | `"[Speaker_1]: 你好..."` | 多说话人文本 |
| num_speakers | 动态 | 2 | 说话人数量（2-10，动态输入） |
| steps | INT | 32 | 每个说话人的扩散步数 |
| guidance_scale | FLOAT | 2.0 | 分类器自由引导比例（0-10） |
| t_shift | FLOAT | 0.1 | 噪声调度时间步偏移（0-1） |
| speed | FLOAT | 1.0 | 所有说话人的语速 |
| pause_between_speakers | FLOAT | 0.3 | 说话人间静音秒数 |
| device | COMBO | auto | `auto`、`cuda`、`cpu`、`mps`、`xpu` |
| dtype | COMBO | auto | `auto`、`bf16`、`fp16`、`fp32` |
| attention | COMBO | auto | `auto`、`eager`、`sage_attention` |
| position_temperature | FLOAT | 5.0 | 掩码位置选择温度（0=贪心） |
| class_temperature | FLOAT | 0.0 | token采样温度（0=贪心） |
| layer_penalty_factor | FLOAT | 5.0 | 深层码本惩罚因子 |
| denoise | BOOLEAN | True | 在输入前添加去噪token |
| preprocess_prompt | BOOLEAN | True | 预处理参考音频 |
| postprocess_output | BOOLEAN | True | 后处理生成音频 |
| seed | INT | 0 | 随机种子（0=随机） |
| keep_model_loaded | BOOLEAN | True | 保持模型加载 |
| speaker_N_audio | AUDIO | 可选 | 说话人N的参考音频（1-10） |
| speaker_N_ref_text | STRING | "" | 说话人N参考音频的转录文本 |
| speaker_N_instruct | STRING | "" | 说话人N的方言/风格指令，仅支持特定值 — 见[方言/风格指令](#方言风格指令) |

说话人输入根据 `num_speakers` 动态显示/隐藏（ComfyUI >= 0.8.1）。

</details>

<details>
<summary><strong>5. OmniVoice Whisper Loader</strong> — 预加载Whisper ASR模型，避免每次重新下载</summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | COMBO | (自动) | Whisper模型选择 |
| device | COMBO | auto | `auto`、`cuda`、`cpu` |
| dtype | COMBO | auto | `auto`、`bf16`、`fp16`、`fp32` |

**自动下载：** 选择带"(auto-download)"后缀的模型可在首次使用时自动下载。

</details>

<details>
<summary><strong>6. OmniVoice Train Config</strong> — 配置LoRA训练超参数</summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| learning_rate | FLOAT | 5e-5 | 优化器学习率 |
| lora_rank | INT | 32 | LoRA适配器的秩（维度） |
| lora_alpha | INT | 16 | LoRA的Alpha缩放因子 |
| lora_dropout | FLOAT | 0.0 | LoRA层的Dropout概率 |
| warmup_steps | INT | 100 | 学习率调度器的预热步数 |
| grad_accum_steps | INT | 1 | 梯度累积步数 |
| weight_decay | FLOAT | 0.01 | 正则化权重衰减 |
| target_modules | STRING | q_proj,k_proj,v_proj,o_proj | Qwen3骨干中的目标模块（逗号分隔） |
| sequence_packing | BOOLEAN | False | 将多个样本打包成一个序列（推荐） |
| batch_tokens | INT | 4096 | 打包序列的最大token长度（4096约需24GB显存） |
| torch_compile | BOOLEAN | False | 编译模型加速训练（推荐开启打包时使用） |
| train_audio_layers | BOOLEAN | True | 同时训练audio_embeddings和audio_heads |

输出：`train_config` 字典 — 连接到LoRA训练器节点。

</details>

<details>
<summary><strong>7. OmniVoice Dataset Maker</strong> — 从配对的音频+文本文件创建训练数据集</summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| audio_directory | STRING | "" | 配对的 .wav + .txt 文件所在文件夹路径 |
| language_id | STRING | "en" | 语言代码（en、zh、ja、de等） |

输出：`dataset_path` — JSONL清单文件路径，用于训练器节点。

</details>

<details>
<summary><strong>8. OmniVoice LoRA Trainer</strong> — 在你的声音数据上训练LoRA适配器</summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_name | COMBO | OmniVoice | 要微调的基础OmniVoice模型 |
| train_config | OMNIVOICE_TRAIN_CONFIG | 必填 | 训练配置节点的输出 |
| dataset_path | STRING | "" | Dataset Maker生成的JSONL清单路径 |
| output_name | STRING | omnivoice_lora_v1 | models/loras/中的子文件夹名称 |
| max_steps | INT | 1000 | 总训练步数 |
| save_every_steps | INT | 200 | 每N步保存检查点 |

输出：`lora_path` — 已保存LoRA的路径。所有4个推理节点都有 `lora_name` 下拉菜单可选择已训练的LoRA。

**注意：** 训练期间会阻塞ComfyUI界面。使用中断按钮取消。

详见 [LoRA训练指南](docs/LORA_TRAINING_zh.md)。

</details>

## 生成参数指南

这些参数控制基于扩散的音频生成过程：

| 参数 | 作用 | 建议 |
|------|------|------|
| `steps` | 迭代去遮蔽步数 | 16=更快，32=平衡，64=最佳质量 |
| `guidance_scale` | 分类器自由引导强度 | 越高越对齐文本；默认2.0 |
| `t_shift` | 噪声调度时间步偏移 | 较小值强调早期解码步骤 |
| `speed` | 语速因子 | >1.0=加快，<1.0=减慢 |
| `duration` | 固定输出长度（秒） | 设定时覆盖speed；0=自动 |
| `position_temperature` | 掩码位置选择随机性 | 0=贪心（确定），越高越随机 |
| `class_temperature` | token采样随机性 | 0=贪心（确定），越高越随机 |
| `layer_penalty_factor` | 深层码本惩罚 | 鼓励低层先解码 |
| `denoise` | 在输入前添加去噪token | 通常可改善输出质量 |
| `preprocess_prompt` | 清理参考音频 | 去除长静音，添加标点 |
| `postprocess_output` | 清理生成音频 | 去除输出中的长静音 |

## 注意力后端

OmniVoice的架构（Qwen3骨干）通过transformers支持的注意力后端有限。`attention`下拉菜单提供以下选项：

| 选项 | 实际行为 |
|------|----------|
| `auto` | OmniVoice默认（eager） |
| `eager` | 标准eager注意力（始终可用） |
| `sage_attention` | **Monkey-patch Qwen3Attention**为SageAttention CUDA内核。仅GPU，需要SM80+（Ampere+）。当存在注意力mask时回退到SDPA。安装：`pip install sageattention` |

### SageAttention GPU兼容性
| GPU架构 | 计算能力 | 使用的内核 |
|---------|----------|-----------|
| Blackwell (RTX 5090) | SM120 | FP8 |
| Hopper (RTX 4090) | SM90 | FP8 |
| Ada Lovelace (RTX 4070) | SM89 | FP8 |
| Ampere (RTX 3090) | SM80 | FP16 |
| SM80以下 | — | 不支持 |

## 多说话人用法

使用 `[Speaker_N]:` 标签分配台词：
```
[Speaker_1]: 你好，我是说话人一。
[Speaker_2]: 我是说话人二！
[Speaker_1]: 很高兴认识你！
```
每个说话人需要连接对应的 `speaker_N_audio` 参考音频输入。

## 方言/风格指令

声音克隆、长文本和多说话人节点提供了 `instruct` 字段，用于指定方言或说话风格。**仅支持以下列出的值** — 模型会验证输入并拒绝不支持的值。

**英文值**（逗号分隔，如 `male, indian accent`）：
| 类别 | 有效值 |
|------|--------|
| **性别** | `male`, `female` |
| **年龄** | `child`, `young adult`, `teenager`, `middle-aged`, `elderly` |
| **口音** | `american accent`, `british accent`, `australian accent`, `canadian accent`, `chinese accent`, `indian accent`, `japanese accent`, `korean accent`, `portuguese accent`, `russian accent` |
| **音调** | `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`, `very high pitch` |
| **风格** | `whisper` |

**中文值**（全角逗号分隔，如 `男，河南话`）：
| 类别 | 有效值 |
|------|--------|
| **性别** | `男`, `女` |
| **年龄** | `儿童`, `少年`, `青年`, `中年`, `老年` |
| **方言** | `四川话`, `东北话`, `陕西话`, `河南话`, `云南话`, `贵州话`, `甘肃话`, `宁夏话`, `石家庄话`, `济南话`, `青岛话`, `桂林话` |
| **音调** | `极低音调`, `低音调`, `中音调`, `高音调`, `极高音调` |
| **风格** | `耳语` |

> **注意：** 每条指令只使用英文或中文值，不要混合使用。

留空则使用默认行为（中文文本默认标准普通话）。

> **注意：** 此字段与声音设计节点的 `voice_instruct` 字段不同，后者用于控制性别、年龄、音调、口音等属性来合成全新的声音。

## 声音设计属性

`voice_instruct` 参数用逗号分隔的属性（有效值与上方 `instruct` 字段相同）：

| 类别 | 选项 |
|------|------|
| **性别** | `male`, `female` |
| **年龄** | `child`, `young adult`, `teenager`, `middle-aged`, `elderly` |
| **口音** | `american accent`, `british accent`, `australian accent`, `canadian accent`, `chinese accent`, `indian accent`, `japanese accent`, `korean accent`, `portuguese accent`, `russian accent` |
| **音调** | `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`, `very high pitch` |
| **风格** | `whisper` |
| **汉语方言** | `四川话`, `东北话`, `陕西话`, `河南话`, `云南话`, `贵州话`, `甘肃话`, `宁夏话`, `石家庄话`, `济南话`, `青岛话`, `桂林话` |

**示例：** `"female, young adult, high pitch, british accent, whisper"`

## 非语言标签

直接在文本中插入：

| 标签 | 效果 |
|------|------|
| `[laughter]` | 笑声 |
| `[sigh]` | 叹气 |
| `[sniff]` | 吸鼻子 |
| `[question-en]`、`[question-ah]`、`[question-oh]` | 疑问语气 |
| `[surprise-ah]`、`[surprise-oh]`、`[surprise-wa]`、`[surprise-yo]` | 惊讶语气 |
| `[dissatisfaction-hnn]` | 不满 |
| `[confirmation-en]` | 确认 |

**示例：**
```
[laughter] 你真是把我逗乐了！[sigh] 我完全没想到会这样。
```

## 模型存储

```
ComfyUI/models/
  omnivoice/
    OmniVoice/          (~4GB, fp32)
    OmniVoice-bf16/     (~2GB, bf16)
  audio_encoders/
    openai_whisper-large-v3-turbo/
    openai_whisper-large-v3/
    openai_whisper-medium/
```

### OmniVoice模型
| 模型 | 大小 | 说明 |
|------|------|------|
| `OmniVoice` | ~4GB | 完整fp32模型 - 600+语言 |
| `OmniVoice-bf16` | ~2GB | Bfloat16量化 - 显存更低 |

### Whisper模型
| 模型 | 显存 | 下载 |
|------|------|------|
| whisper-large-v3-turbo | ~1.5GB | [下载](https://huggingface.co/openai/whisper-large-v3-turbo) |
| whisper-large-v3 | ~3GB | [下载](https://huggingface.co/openai/whisper-large-v3) |
| whisper-medium | ~1GB | [下载](https://huggingface.co/openai/whisper-medium) |
| whisper-small | ~0.5GB | [下载](https://huggingface.co/openai/whisper-small) |
| whisper-tiny | ~0.4GB | [下载](https://huggingface.co/openai/whisper-tiny) |

模型首次使用时自动从HuggingFace下载。

## 显存需求

| 精度 | 显存（约） |
|------|-----------|
| fp32 | ~8-12 GB |
| bf16/fp16 | ~4-6 GB |
| CPU卸载 | ~2-4 GB |

## 模型缓存

节点会缓存已加载的模型以供复用。更改以下任何参数都会**强制完全清除缓存**（模型卸载 + GC + CUDA缓存刷新），即使 `keep_model_loaded` 为 `True`：

- 模型选择
- 设备
- 精度（dtype）
- 注意力后端

## 故障排除

详细的故障排除指南请参阅 [docs/TROUBLESHOOTING_zh.md](docs/TROUBLESHOOTING_zh.md)（[English](docs/TROUBLESHOOTING.md)）。

<details>
<summary>常见问题快速修复</summary>

### 模型下载失败（中国）
启动ComfyUI前设置HuggingFace镜像：
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

### Whisper每次都重新下载
将 `OmniVoice Whisper Loader` 连接到 Voice Clone TTS 的 `whisper_model` 输入以缓存模型。

### CUDA显存不足
- 设置 `keep_model_loaded = False`
- 使用 `dtype = fp16` 或 `bf16`
- 使用 `device = cpu`（较慢但可用）

### 安装后出现导入错误
完全重启ComfyUI以重新加载Python模块。

### Transformers版本

OmniVoice需要 `transformers>=5.3.0`。如果您在ComfyUI日志中看到 `omnivoice import failed` 或 `cannot import name 'HiggsAudioV2TokenizerModel'` 等错误，可能是transformers版本过旧。

> ⚠️ **请仅在了解风险的情况下操作。** 升级transformers可能会导致依赖旧版本的其他自定义节点出现问题。升级后请测试其他节点。

升级方法：
```
path\to\ComfyUI\venv\Scripts\python.exe -m pip install "transformers>=5.3.0"
```

### Windows保存音频时FFmpeg错误
在ComfyUI启动 `.bat` 文件中将FFmpeg的 `bin/` 文件夹添加到 `PATH`，或使用WAV音频保存节点。

</details>

## 致谢

- **OmniVoice** — [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) by k2-fsa — 原始fp32模型
- **OmniVoice-bf16** — [drbaph/OmniVoice-bf16](https://huggingface.co/drbaph/OmniVoice-bf16) by drbaph — Bfloat16量化模型
- **ComfyUI节点** — [saganaki22/ComfyUI-OmniVoice-TTS](https://github.com/saganaki22/ComfyUI-OmniVoice-TTS) — 本自定义节点

## 引用

```bibtex
@article{zhu2026omnivoice,
      title={OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models},
      author={Zhu, Han and Ye, Lingxuan and Kang, Wei and Yao, Zengwei and Guo, Liyong and Kuang, Fangjun and Han, Zhifeng and Zhuang, Weiji and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2604.00688},
      year={2026}
}
```

## 许可证

本自定义节点采用Apache 2.0许可证发布。OmniVoice模型有自己的许可证 — 详见 [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)。
