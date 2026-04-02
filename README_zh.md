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
- **自动下载** — 首次使用时自动从HuggingFace下载模型
- **Whisper ASR缓存** — 预加载Whisper避免每次重新下载
- **显存高效** — 自动CPU卸载，VBAR/aimdo集成

## 安装

> ⚠️ **安装前：请记录您的PyTorch版本！**
>
> 运行此命令并保存输出 - 如果出现问题您会需要它：
> ```bash
> python -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda}')"
> ```
> 示例输出：`torch=2.10.0+cu128 cuda=12.8`
>
> 如果安装破坏了您的PyTorch，使用以下命令恢复：
> ```bash
> # 使用pip：
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/<您的CUDA版本>
>
> # 使用uv：
> uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/<您的CUDA版本>
> ```
> 将`<您的CUDA版本>`替换为您的版本（如`cu128`、`cu124`、`cu121`、`cu118`）

### 方法1：ComfyUI Manager（推荐）
在ComfyUI Manager中搜索"OmniVoice"并点击安装。

### 方法2：手动安装
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/saganaki22/ComfyUI-OmniVoice-TTS.git
cd ComfyUI-OmniVoice-TTS
pip install -r requirements.txt
```

### 方法3：使用uv
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/saganaki22/ComfyUI-OmniVoice-TTS.git
cd ComfyUI-OmniVoice-TTS
uv pip install -r requirements.txt
```

### ⚠️ PyTorch CUDA降级警告
`omnivoice` pip包可能在安装时**将PyTorch降级为CPU版本**，导致ComfyUI无法使用GPU。

**此节点的自动处理：**
首次加载时如果缺少 `omnivoice`，节点会尝试：
1. 通过pip安装 `omnivoice`
2. 重新安装原来的CUDA版PyTorch以恢复GPU支持

**手动修复：**
```bash
# 检查CUDA版本（如cu118, cu121, cu128）
python -c "import torch; print(torch.version.cuda)"

# 重新安装对应CUDA版本的PyTorch（cu128示例）：
pip install torch==2.9.0+cu128 torchaudio==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

**建议：** 安装此节点后，**重启ComfyUI**以确保CUDA PyTorch正确恢复。

## 节点

### 1. OmniVoice Longform TTS
长文本语音合成，智能分句，可选声音克隆。

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| model | DROPDOWN | OmniVoice-bf16 | OmniVoice模型检查点 |
| text | STRING | `"你好..."` | 要合成的文本 |
| ref_text | STRING | "" | 参考音频转录文本（空=自动识别） |
| steps | INT | 32 | 扩散步数（16=快，64=最佳） |
| speed | FLOAT | 1.0 | 语速（>1=加快） |
| duration | FLOAT | 0.0 | 固定时长秒数（0=自动） |
| device | DROPDOWN | auto | 计算设备 |
| dtype | DROPDOWN | auto | 模型精度 |
| attention | DROPDOWN | auto | 注意力实现 |
| seed | INT | 0 | 随机种子（0=随机） |
| words_per_chunk | INT | 100 | 每块词数（0=不分块） |
| keep_model_loaded | BOOLEAN | True | 保持模型加载 |

**可选输入：**
- `ref_audio` — 声音克隆参考音频（3-15秒最佳）
- `whisper_model` — 预加载的Whisper ASR模型

### 2. OmniVoice Voice Clone TTS
从参考音频克隆声音。

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| model | DROPDOWN | OmniVoice-bf16 | OmniVoice模型检查点 |
| text | STRING | `"你好..."` | 要用克隆声音合成的文本 |
| ref_audio | AUDIO | 必填 | 参考音频（3-15秒） |
| ref_text | STRING | "" | 转录文本（空=Whisper自动识别） |
| steps | INT | 32 | 扩散步数（16=快，64=最佳） |
| speed | FLOAT | 1.0 | 语速（>1=加快） |
| duration | FLOAT | 0.0 | 固定时长秒数（0=自动） |
| device | DROPDOWN | auto | 计算设备 |
| dtype | DROPDOWN | auto | 模型精度 |
| attention | DROPDOWN | auto | 注意力实现 |
| seed | INT | 0 | 随机种子（0=随机） |
| keep_model_loaded | BOOLEAN | True | 保持模型加载 |

**可选输入：**
- `whisper_model` — 预加载的Whisper ASR模型

### 3. OmniVoice Voice Design TTS
通过文字描述设计声音。

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| model | DROPDOWN | OmniVoice-bf16 | OmniVoice模型检查点 |
| text | STRING | `"你好..."` | 要用设计声音合成的文本 |
| voice_instruct | STRING | `"female, low pitch..."` | 声音属性描述 |
| steps | INT | 32 | 扩散步数（16=快，64=最佳） |
| speed | FLOAT | 1.0 | 语速（>1=加快） |
| duration | FLOAT | 0.0 | 固定时长秒数（0=自动） |
| device | DROPDOWN | auto | 计算设备 |
| dtype | DROPDOWN | auto | 模型精度 |
| attention | DROPDOWN | auto | 注意力实现 |
| seed | INT | 0 | 随机种子（0=随机） |
| keep_model_loaded | BOOLEAN | True | 保持模型加载 |

### 4. OmniVoice Multi-Speaker TTS
使用 `[Speaker_N]:` 标签生成多说话人对白。

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| model | DROPDOWN | OmniVoice-bf16 | OmniVoice模型检查点 |
| text | STRING | `"[Speaker_1]: 你好..."` | 多说话人文本 |
| num_speakers | INT | 2 | 说话人数量（2-10） |
| steps | INT | 32 | 每个说话人的扩散步数 |
| speed | FLOAT | 1.0 | 所有说话人的语速 |
| pause_between_speakers | FLOAT | 0.3 | 说话人间静音秒数 |
| device | DROPDOWN | auto | 计算设备 |
| dtype | DROPDOWN | auto | 模型精度 |
| attention | DROPDOWN | auto | 注意力实现 |
| seed | INT | 0 | 随机种子（0=随机） |
| keep_model_loaded | BOOLEAN | True | 保持模型加载 |
| speaker_1_audio | AUDIO | 必填 | 说话人1的参考音频 |
| speaker_1_ref_text | STRING | "" | 说话人1参考音频的转录文本 |
| ... | ... | ... | （说话人输入根据num_speakers自动显示） |

**注意：** 在V2（旧版ComfyUI）中，还有`whisper_model`输入可用于预加载Whisper ASR。

### 5. OmniVoice Whisper Loader
预加载Whisper ASR模型，避免每次重新下载。

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| model | DROPDOWN | whisper-large-v3-turbo | Whisper模型选择 |
| device | DROPDOWN | auto | 设备：auto/cuda/cpu |
| dtype | DROPDOWN | auto | 精度：auto/bf16/fp16/fp32 |

**dtype选项：**
- `auto` — Ampere+显卡用bf16，旧显卡用fp16，CPU用fp32
- `bf16` — bfloat16（需要Ampere+显卡）
- `fp16` — float16
- `fp32` — float32（最高精度，显存占用最大）

**自动下载：** 选择"whisper-large-v3-turbo (auto-download)"首次使用时自动下载。

## 模型存储

```
📂 ComfyUI/models/
├── 📂 omnivoice/
│   ├── 📂 OmniVoice/          (~4GB, fp32)
│   └── 📂 OmniVoice-bf16/     (~2GB, bf16)
└── 📂 audio_encoders/
    ├── 📂 openai_whisper-large-v3-turbo/
    ├── 📂 openai_whisper-large-v3/
    └── 📂 openai_whisper-medium/
```

### OmniVoice模型
| 模型 | 大小 | 说明 |
|-------|------|-------------|
| `OmniVoice` | ~4GB | 完整fp32模型 - 600+语言 |
| `OmniVoice-bf16` | ~2GB | Bfloat16量化 - 显存更低 |

### Whisper模型
| 模型 | 显存 | 下载 |
|-------|------|------|
| whisper-large-v3-turbo | ~1.5GB | [下载](https://huggingface.co/openai/whisper-large-v3-turbo) |
| whisper-large-v3 | ~3GB | [下载](https://huggingface.co/openai/whisper-large-v3) |
| whisper-medium | ~1GB | [下载](https://huggingface.co/openai/whisper-medium) |
| whisper-small | ~0.5GB | [下载](https://huggingface.co/openai/whisper-small) |
| whisper-tiny | ~0.4GB | [下载](https://huggingface.co/openai/whisper-tiny) |

模型首次使用时自动从HuggingFace下载。

## 多说话人用法

使用 `[Speaker_N]:` 标签分配台词：

```
[Speaker_1]: 你好，我是说话人一。
[Speaker_2]: 我是说话人二！
[Speaker_1]: 很高兴认识你！
```

每个说话人需要连接对应的 `speaker_N_audio` 参考音频输入。

## 声音设计属性

`voice_instruct` 参数用逗号分隔的属性：

| 类别 | 选项 |
|----------|---------|
| **性别** | `male`（男）, `female`（女） |
| **年龄** | `child`（儿童）, `young`（青年）, `middle-aged`（中年）, `elderly`（老年） |
| **音调** | `very low pitch`, `low pitch`, `medium pitch`, `high pitch`, `very high pitch` |
| **风格** | `whisper`（耳语） |
| **英语口音** | `american accent`, `british accent`, `australian accent` 等 |
| **汉语方言** | `四川话`, `陕西话`, `广东话`, `东北话` 等 |

**示例：** `"female, young, high pitch, british accent, whisper"`

## 非语言标签

直接在文本中插入：
- `[laughter]` — 笑声
- `[sigh]` — 叹气
- `[sniff]` — 吸鼻子
- `[question-en]`, `[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]` — 疑问语气
- `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]` — 惊讶语气
- `[dissatisfaction-hnn]` — 不满
- `[confirmation-en]` — 确认

**示例：**
```
[laughter] 你真是把我逗乐了！[sigh] 我完全没想到会这样。
```

## 显存需求

| 精度 | 显存（约） |
|-----------|---------------|
| fp32 | ~8-12 GB |
| bf16/fp16 | ~4-6 GB |
| CPU卸载 | ~2-4 GB |

## 故障排除

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
