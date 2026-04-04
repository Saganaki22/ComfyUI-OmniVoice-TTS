# 故障排除 — ComfyUI-OmniVoice-TTS

---

## 目录

- [安装后节点缺失（最常见）](#安装后节点缺失最常见)
- [从 omnivoice 0.1.1 升级到 0.1.2](#从-omnivoice-011-升级到-012)
- [soxr 缺失错误](#soxr-缺失错误)
- [omnivoice 导入失败 / HiggsAudioV2TokenizerModel](#omnivoice-导入失败--higgsaudiov2tokenizermodel)
- [No module named pip](#no-module-named-pip)
- [安装后 PyTorch CUDA 损坏](#安装后-pytorch-cuda-损坏)
- [Transformers 版本冲突](#transformers-版本冲突)
- [模型下载失败（中国大陆 / HuggingFace 被墙）](#模型下载失败中国大陆--huggingface-被墙)
- [Whisper 每次运行都重新下载](#whisper-每次运行都重新下载)
- [CUDA 内存不足](#cuda-内存不足)
- [Windows 上的 FFmpeg 错误](#windows-上的-ffmpeg-错误)
- [安装后出现导入错误](#安装后出现导入错误)

---

## 安装后节点缺失（最常见）

**症状：** ComfyUI 加载了节点包，但只出现 `OmniVoice Whisper Loader`。其他所有节点（`OmniVoiceLongformTTS`、`OmniVoiceVoiceCloneTTS`、`OmniVoiceVoiceDesignTTS`、`OmniVoiceMultiSpeakerTTS`）都缺失。

**原因：** 其他节点只有在 `omnivoice` 成功导入时才会注册。如果 `omnivoice` 因任何原因导入失败，`__init__.py` 会跳过它们。Whisper Loader 不依赖 `omnivoice`，所以始终会注册。

**检查 ComfyUI 启动日志中是否有类似以下的行：**
```
[OmniVoice] omnivoice import failed: ...
```

冒号后面的错误信息才是真正的原因。最常见的原因和修复方法见下方。

---

## 从 omnivoice 0.1.1 升级到 0.1.2

omnivoice `0.1.1` 于 2026 年 4 月 2 日发布。版本 `0.1.2` 于 2026 年 4 月 4 日发布，是当前推荐的版本。

**如果通过 ComfyUI Manager 安装：** 在 Manager 中对该节点运行"Update"，然后重启 ComfyUI。

**如果手动安装或想自行更新 omnivoice 包：**

> ⚠️ **必须使用 `--no-deps`**。omnivoice pip 包声明了 `torch==2.8.*` 和 `transformers==5.3.0` 作为硬依赖。不使用 `--no-deps` 安装会将 PyTorch 降级为仅 CPU 版本，破坏 ComfyUI 的 GPU 加速。

### Windows（venv）
```bash
C:\Users\<你>\Documents\ComfyUI\venv\Scripts\pip install omnivoice==0.1.2 --no-deps
```

### Windows（便携版 / 内嵌 Python）
```bash
C:\ComfyUI\python_embeded\python.exe -m pip install omnivoice==0.1.2 --no-deps
```

### Linux / macOS
```bash
path/to/ComfyUI/venv/bin/pip install omnivoice==0.1.2 --no-deps
```

### 使用 uv（如果你的环境使用 uv）
```bash
uv pip install omnivoice==0.1.2 --no-deps
```

升级后，如果你有 `transformers 5.4+`，还需要安装 `soxr` — 见下一节。

---

## soxr 缺失错误

**症状：** ComfyUI 日志显示：
```
ModuleNotFoundError: No module named 'soxr'
```
或者
```
[OmniVoice] omnivoice import failed: Could not import module 'HiggsAudioV2TokenizerModel'. Are this object's requirements defined correctly?
```

**原因：** `transformers 5.4+` 在 `audio_utils.py` 中新增了 `soxr` 作为 HiggsAudio tokenizer 路径的必需导入。当 `omnivoice` 尝试从 transformers 加载 `HiggsAudioV2TokenizerModel` 时，transformers 内部导入 `audio_utils`，后者导入 `soxr` — 如果 `soxr` 未安装，整条链路就会崩溃。

令人困惑的 `HiggsAudioV2TokenizerModel` 错误信息是误导性的 — 它是 `soxr` 导入失败的下游后果，而不是缺少模型类。

这只影响使用 `transformers 5.4+` 的用户。使用 `transformers 5.3.0` 的用户不会遇到此问题。

**修复方法：**

### Windows（venv）
```bash
C:\Users\<你>\Documents\ComfyUI\venv\Scripts\pip install soxr
```

### Windows（便携版 / 内嵌 Python）
```bash
C:\ComfyUI\python_embeded\python.exe -m pip install soxr
```

### Linux / macOS
```bash
path/to/ComfyUI/venv/bin/pip install soxr
```

### 使用 uv
```bash
uv pip install soxr
```

然后**重启 ComfyUI**。所有节点应该正常出现。

> **新安装用户注意：** 从节点版本 `0.2.7` 开始，`soxr` 已包含在 `install.py` 中，会自动安装。如果你在此版本之前安装的，请使用上述命令手动安装。

---

## omnivoice 导入失败 / HiggsAudioV2TokenizerModel

见上方的 [soxr 缺失错误](#soxr-缺失错误) 部分 — 这是同一个问题。`HiggsAudioV2TokenizerModel` 错误是由 `soxr` 未安装引起的。

如果安装 `soxr` 后仍未解决，请检查完整 ComfyUI 日志中 `[OmniVoice]` 下的其他错误行，并在 [Issues](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS/issues) 中报告。

---

## No module named pip

**症状：**
```
[OmniVoice] Failed to install omnivoice: ...\python.exe: No module named pip
```

**原因：** 你的 Python 环境完全缺少 `pip`，这不太常见，但可能在某些 venv 设置或 `uv` 管理的环境中发生。

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

**修复：** 查看 [PyTorch 兼容性矩阵](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS/blob/main/pytorch_compatibility_matrix.md)获取与你的 CUDA 版本匹配的恢复命令。

CUDA 12.8 的通用恢复命令：
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

其他 CUDA 版本访问：https://pytorch.org/get-started/locally/

> ⚠️ 这就是我们始终使用 `--no-deps` 安装 omnivoice 的原因。永远不要在没有它的情况下运行普通的 `pip install omnivoice`。

---

## Transformers 版本冲突

**症状：** 提到 transformers 版本的错误，或安装 OmniVoice 后其他自定义节点损坏。

**背景：** 上游 `omnivoice` 包固定了 `transformers==5.3.0`。我们的 `install.py` 通过 `--no-deps` 故意忽略了这个固定版本，以避免破坏你现有的设置。这意味着 OmniVoice 可以与更新的 transformers 版本一起工作，但有一个注意事项：`transformers 5.4+` 需要 `soxr`（见上方）。

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

**Linux / macOS：**
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

**Windows（命令提示符）：**
```cmd
set HF_ENDPOINT=https://hf-mirror.com
```

**Windows（PowerShell）：**
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

---

## 仍然无法解决？

在 [github.com/Saganaki22/ComfyUI-OmniVoice-TTS/issues](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS/issues) 提交 Issue，并附上：

1. 完整的 ComfyUI 启动日志（包含 `[OmniVoice]` 行的部分）
2. `pip show omnivoice transformers torch` 的输出
3. 你的操作系统和 Python 版本
4. 安装方式（Manager / 手动 / git clone）
