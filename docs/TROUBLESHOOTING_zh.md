# 故障排除 — ComfyUI-OmniVoice-TTS

---

## 目录

- [从这里开始 — 只有 Whisper 节点加载，其他节点全部缺失](#从这里开始--只有-whisper-节点加载其他节点全部缺失)
- [从 omnivoice 0.1.1 升级到 0.1.2](#从-omnivoice-011-升级到-012)
- [No module named pip](#no-module-named-pip)
- [安装后 PyTorch CUDA 损坏](#安装后-pytorch-cuda-损坏)
- [Transformers 版本冲突](#transformers-版本冲突)
- [模型下载失败（中国大陆 / HuggingFace 被墙）](#模型下载失败中国大陆--huggingface-被墙)
- [Whisper 每次运行都重新下载](#whisper-每次运行都重新下载)
- [CUDA 内存不足](#cuda-内存不足)
- [Windows 上的 FFmpeg 错误](#windows-上的-ffmpeg-错误)
- [安装后出现导入错误](#安装后出现导入错误)

---

## 从这里开始 — 只有 Whisper 节点加载，其他节点全部缺失

这是最常见的问题。如果你在 ComfyUI 中只能看到 `OmniVoice Whisper Loader`，而其他所有节点（`OmniVoiceLongformTTS`、`OmniVoiceVoiceCloneTTS`、`OmniVoiceVoiceDesignTTS`、`OmniVoiceMultiSpeakerTTS`）缺失或显示为红色，请按照本指南操作。

**原因：** 其他节点只有在 `omnivoice` 启动时成功导入后才会注册。任何导入失败都会导致它们全部被跳过。Whisper Loader 因为不依赖 `omnivoice`，所以始终会注册。

---

### 第一步 — 从启动日志中读取实际错误

打开你的 ComfyUI 启动日志，找到 `[OmniVoice]` 部分。你会看到类似以下的行：

```
[OmniVoice] omnivoice import failed: <此处为错误信息>
```

将你的错误信息与下方的情况进行对照。

---

### 情况 A — `cannot import name 'HiggsAudioV2TokenizerModel' from 'transformers'`

> ⚠️ **尽管这个错误看起来像 `transformers` 太旧，但实际上并不是。** 此错误是由缺少 `soxr` 包引起的，而不是 `transformers` 中缺少某个类。这个令人困惑的错误信息是已知的误导性提示。

**原因：** `transformers 5.4+` 在其音频工具模块中新增了 `soxr` 作为必需导入。当 `omnivoice` 加载 `HiggsAudioV2TokenizerModel` 时，`transformers` 内部会尝试导入 `soxr` — 如果未安装，整条链路就会崩溃并显示此误导性错误。

**修复方法 — 安装 `soxr`：**

Windows（venv）：
```bash
C:\Users\<你>\Documents\ComfyUI\venv\Scripts\pip install soxr
```

Windows（便携版 / 内嵌 Python）：
```bash
C:\ComfyUI\python_embeded\python.exe -m pip install soxr
```

Linux / macOS：
```bash
path/to/ComfyUI/venv/bin/pip install soxr
```

使用 uv：
```bash
uv pip install soxr
```

然后**重启 ComfyUI**。所有节点应该正常出现。

> **新安装用户注意：** 从节点版本 `0.2.7` 开始，`soxr` 已包含在 `install.py` 中，会自动安装。如果你在此版本之前安装的，请使用上述命令手动安装。

---

### 情况 B — `No module named 'omnivoice'`

`omnivoice` 包未安装。请手动运行 `install.py`：

Windows（venv）：
```bash
C:\Users\<你>\Documents\ComfyUI\venv\Scripts\python install.py
```

Windows（便携版 / 内嵌 Python）：
```bash
C:\ComfyUI\python_embeded\python.exe install.py
```

Linux / macOS：
```bash
path/to/ComfyUI/venv/bin/python install.py
```

然后**重启 ComfyUI**。

> ⚠️ 永远不要在没有 `--no-deps` 的情况下运行 `pip install omnivoice`。omnivoice 包固定了 `torch==2.8.*`，这会将你的 PyTorch 降级为仅 CPU 版本并破坏 ComfyUI 的 GPU 加速。`install.py` 会为你安全地处理这个问题。

---

### 情况 C — 其他任何错误信息

如果你的错误信息与情况 A 或 B 均不匹配：

1. 复制启动日志中完整的 `[OmniVoice]` 部分
2. 运行 `pip show omnivoice transformers torch` 并复制输出
3. 携带上述信息在 [github.com/Saganaki22/ComfyUI-OmniVoice-TTS/issues](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS/issues) 提交 Issue

---

## 从 omnivoice 0.1.1 升级到 0.1.2

omnivoice `0.1.1` 于 2026 年 4 月 2 日发布。版本 `0.1.2` 于 2026 年 4 月 4 日发布，是当前推荐的版本。

**如果通过 ComfyUI Manager 安装：** 在 Manager 中对该节点运行"Update"，然后重启 ComfyUI。

**如果手动安装或想自行更新 omnivoice 包：**

> ⚠️ **必须使用 `--no-deps`**。omnivoice pip 包声明了 `torch==2.8.*` 和 `transformers==5.3.0` 作为硬依赖。不使用 `--no-deps` 安装会将 PyTorch 降级为仅 CPU 版本，破坏 ComfyUI 的 GPU 加速。

Windows（venv）：
```bash
C:\Users\<你>\Documents\ComfyUI\venv\Scripts\pip install omnivoice==0.1.2 --no-deps
```

Windows（便携版 / 内嵌 Python）：
```bash
C:\ComfyUI\python_embeded\python.exe -m pip install omnivoice==0.1.2 --no-deps
```

Linux / macOS：
```bash
path/to/ComfyUI/venv/bin/pip install omnivoice==0.1.2 --no-deps
```

使用 uv：
```bash
uv pip install omnivoice==0.1.2 --no-deps
```

升级后，如果你有 `transformers 5.4+`，还需要安装 `soxr`：
```bash
pip install soxr
```

---

## No module named pip

**症状：**
```
[OmniVoice] Failed to install omnivoice: ...\python.exe: No module named pip
```

**修复 — 先引导安装 pip：**
```bash
python -m ensurepip --upgrade
```

然后重试安装命令。如果你使用 `uv`，所有命令改用 `uv pip install` — `uv` 不需要安装 pip。

---

## 安装后 PyTorch CUDA 损坏

**症状：** ComfyUI 之前使用 GPU，现在只在 CPU 上运行。或者你看到：
```
UserWarning: CUDA initialization: CUDA unknown error
```

**原因：** 另一个包（可能是未使用 `--no-deps` 安装的 `omnivoice`，或其他自定义节点）将你的 PyTorch 降级到了仅 CPU 版本。

**修复：** 查看 [PyTorch 兼容性矩阵](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS/blob/main/pytorch_compatibility_matrix.md) 获取与你的 CUDA 版本匹配的恢复命令。

CUDA 12.8 的通用恢复命令：
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

其他 CUDA 版本访问：https://pytorch.org/get-started/locally/

> ⚠️ 这就是我们始终使用 `--no-deps` 安装 omnivoice 的原因。永远不要在没有它的情况下运行普通的 `pip install omnivoice`。

---

## Transformers 版本冲突

**症状：** 安装 OmniVoice 后其他自定义节点损坏，或出现提到 `transformers` 版本不兼容的错误。

**背景：** 上游 `omnivoice` 包固定了 `transformers==5.3.0`，但 `install.py` 通过 `--no-deps` 故意忽略了这个固定版本，以避免破坏你现有的设置。OmniVoice 可以与任何较新的 `transformers` 版本配合使用，但有一个注意事项：`transformers 5.4+` 需要 `soxr`（见上方[情况 A](#情况-a--cannot-import-name-higgsaudiov2tokenizermodel-from-transformers)）。

**检查 transformers 版本：**
```bash
pip show transformers
```

**降级 transformers**（只在确定不会破坏其他节点时才这样做）：
```bash
pip install transformers==5.3.0
```

**升级 transformers**（然后也需要安装 soxr）：
```bash
pip install transformers --upgrade
pip install soxr
```

---

## 模型下载失败（中国大陆 / HuggingFace 被墙）

在启动 ComfyUI 之前设置 HuggingFace 镜像：

Linux / macOS：
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

Windows（命令提示符）：
```cmd
set HF_ENDPOINT=https://hf-mirror.com
```

Windows（PowerShell）：
```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

将此添加到你的 ComfyUI 启动脚本中，以便在重启后保持生效。

---

## Whisper 每次运行都重新下载

将 `OmniVoice Whisper Loader` 节点连接到 TTS 节点上的 `whisper_model` 输入。这会将 Whisper 模型缓存在内存中，避免每次运行时从磁盘重新加载或重新下载。

---

## CUDA 内存不足

按以下顺序尝试：

1. 设置 `keep_model_loaded = False` — 在运行之间卸载模型
2. 将 `dtype` 切换为 `bf16` 或 `fp16` — 显存使用减半（约 4-6GB 而非 8-12GB）
3. 使用 `OmniVoice-bf16` 模型代替 `OmniVoice` — 磁盘上 2GB vs 4GB
4. 设置 `device = cpu` — 速度慢但可在任何系统上运行

---

## Windows 上的 FFmpeg 错误

**症状：** 音频保存节点因 FFmpeg 相关错误而失败。

**修复：** 在你的 ComfyUI 启动 `.bat` 文件中将 FFmpeg 的 `bin/` 文件夹添加到 `PATH`：
```bat
set PATH=C:\path\to\ffmpeg\bin;%PATH%
```

或者使用 WAV 音频保存节点代替 MP3/AAC 格式，WAV 不需要 FFmpeg。

---

## 安装后出现导入错误

**症状：** 全新安装或更新后 ComfyUI 启动时出现通用 Python 导入错误。

**修复：** 完全重启 ComfyUI。Python 模块在每个进程中只加载一次 — 需要完全重启才能使新安装的包生效。不要使用 ComfyUI 界面中的"Reload"按钮，因为它不会重新加载 Python 模块。

> **Google Colab 用户注意：** 断开连接后环境会重置，每次会话都需要重新安装。如果在安装过程中使用 `--cpu` 运行，可能导致 PyTorch CUDA 检测失败 — 请尽量在 GPU 可用的情况下安装包。

---

## 仍然无法解决？

在 [github.com/Saganaki22/ComfyUI-OmniVoice-TTS/issues](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS/issues) 提交 Issue，并附上：

1. 完整的 ComfyUI 启动日志（包含 `[OmniVoice]` 行的部分）
2. `pip show omnivoice transformers torch` 的输出
3. 你的操作系统和 Python 版本
4. 安装方式（Manager / 手动 / git clone）
