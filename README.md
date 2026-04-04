# ComfyUI-OmniVoice-TTS

**OmniVoice TTS nodes for ComfyUI** — Zero-shot multilingual text-to-speech with voice cloning and voice design. Supports **600+ languages** with state-of-the-art quality.

[![OmniVoice Model](https://img.shields.io/badge/%F0%9F%A4%97%20OmniVoice%20Model-k2--fsa/OmniVoice-blue)](https://huggingface.co/k2-fsa/OmniVoice)
[![OmniVoice-bf16](https://img.shields.io/badge/%F0%9F%A4%97%20OmniVoice--bf16-drbaph/OmniVoice--bf16-blue)](https://huggingface.co/drbaph/OmniVoice-bf16)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Demo%20Space-OmniVoice-yellow)](https://huggingface.co/spaces/k2-fsa/OmniVoice)
[![Demo](https://img.shields.io/badge/Demo%20Page-OmniVoice-green)](https://zhu-han.github.io/omnivoice/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.00688-b31b1b)](https://arxiv.org/abs/2604.00688)
[![GitHub](https://img.shields.io/badge/GitHub%20OmniVoice-k2--fsa/OmniVoice-black)](https://github.com/k2-fsa/OmniVoice)


<img width="1611" height="1260" alt="Screenshot 2026-04-03 124716" src="https://github.com/user-attachments/assets/d30440cf-2121-40f1-9b22-22c89fe7540c" />


## Features

- **600+ Languages** — Broadest language coverage among zero-shot TTS models
- **Voice Cloning** — Clone any voice from 3-15 seconds of reference audio
- **Voice Design** — Create synthetic voices from text descriptions (gender, age, pitch, accent)
- **Multi-Speaker Dialogue** — Generate conversations between multiple speakers using `[Speaker_N]:` tags
- **Fast Inference** — RTF as low as 0.025 (40x faster than real-time)
- **Non-Verbal Expressions** — Inline tags like `[laughter]`, `[sigh]`, `[sniff]`
- **SageAttention Support** — GPU-optimized attention via monkey-patching Qwen3Attention (CUDA, SM80+)
- **Auto-Download** — Models download automatically from HuggingFace on first use
- **Whisper ASR Caching** — Pre-load Whisper to avoid re-downloading on each run
- **VRAM Efficient** — Automatic CPU offload, VBAR/aimdo integration, smart cache invalidation

https://github.com/user-attachments/assets/b9c75048-915a-4993-9169-ddb1d2b28f41

## Installation

### Method 1: ComfyUI Manager (Recommended)
Search for "OmniVoice" in ComfyUI Manager and click Install.

### Method 2: Manual Install
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/saganaki22/ComfyUI-OmniVoice-TTS.git
cd ComfyUI-OmniVoice-TTS
python install.py
```

### Why `--no-deps`?
The `omnivoice` pip package specifies `torch==2.8.*` as a dependency, which can downgrade your PyTorch to a CPU-only version and break ComfyUI's GPU acceleration. We work around this by installing `omnivoice` with `--no-deps` in `install.py`, then separately installing only the missing dependencies that ComfyUI doesn't already provide.

### If PyTorch Gets Broken
If another package accidentally downgrades your PyTorch, see the [PyTorch Compatibility Matrix](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS/blob/main/pytorch_compatibility_matrix.md) for restore commands matching your setup.

## Nodes

### 1. OmniVoice Longform TTS
Long-form text-to-speech with smart chunking and optional voice cloning.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | COMBO | (auto) | OmniVoice model checkpoint |
| text | STRING, multiline | `"Hello!..."` | Text to synthesize |
| ref_text | STRING, multiline | "" | Reference audio transcript (empty=auto-detect) |
| steps | INT | 32 | Diffusion steps (4-64, 16=faster, 64=best) |
| guidance_scale | FLOAT | 2.0 | Classifier-free guidance scale (0-10) |
| t_shift | FLOAT | 0.1 | Time-step shift for noise schedule (0-1) |
| speed | FLOAT | 1.0 | Speaking speed (0.5-2.0, >1=faster) |
| duration | FLOAT | 0.0 | Fixed duration in seconds (0=auto) |
| device | COMBO | auto | `auto`, `cuda`, `cpu`, `mps` |
| dtype | COMBO | auto | `auto`, `bf16`, `fp16`, `fp32` |
| attention | COMBO | auto | `auto`, `eager`, `sage_attention` |
| seed | INT | 0 | Random seed (0=random) |
| words_per_chunk | INT | 100 | Words per chunk (0=no chunking) |
| position_temperature | FLOAT | 5.0 | Mask-position temperature (0=greedy, higher=more random) |
| class_temperature | FLOAT | 0.0 | Token sampling temperature (0=greedy) |
| layer_penalty_factor | FLOAT | 5.0 | Penalty on deeper codebook layers |
| denoise | BOOLEAN | True | Prepend denoise token for cleaner output |
| preprocess_prompt | BOOLEAN | True | Preprocess reference audio (remove silences) |
| postprocess_output | BOOLEAN | True | Post-process generated audio (remove long silences) |
| keep_model_loaded | BOOLEAN | True | Keep model in memory (offloads to CPU between runs) |

**Optional Inputs:**
- `ref_audio` — Reference audio for voice cloning (3-15s optimal)
- `whisper_model` — Pre-loaded Whisper ASR model

### 2. OmniVoice Voice Clone TTS
Clone a voice from reference audio.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | COMBO | (auto) | OmniVoice model checkpoint |
| text | STRING, multiline | `"Hello!..."` | Text to synthesize in cloned voice |
| ref_audio | AUDIO | required | Reference audio (3-15s) |
| ref_text | STRING, multiline | "" | Transcript (empty=auto-transcribe with Whisper) |
| steps | INT | 32 | Diffusion steps (4-64) |
| guidance_scale | FLOAT | 2.0 | Classifier-free guidance scale (0-10) |
| t_shift | FLOAT | 0.1 | Time-step shift for noise schedule (0-1) |
| speed | FLOAT | 1.0 | Speaking speed (0.5-2.0) |
| duration | FLOAT | 0.0 | Fixed duration in seconds (0=auto) |
| device | COMBO | auto | `auto`, `cuda`, `cpu`, `mps` |
| dtype | COMBO | auto | `auto`, `bf16`, `fp16`, `fp32` |
| attention | COMBO | auto | `auto`, `eager`, `sage_attention` |
| seed | INT | 0 | Random seed (0=random) |
| position_temperature | FLOAT | 5.0 | Mask-position temperature (0=greedy) |
| class_temperature | FLOAT | 0.0 | Token sampling temperature (0=greedy) |
| layer_penalty_factor | FLOAT | 5.0 | Penalty on deeper codebook layers |
| denoise | BOOLEAN | True | Prepend denoise token for cleaner output |
| preprocess_prompt | BOOLEAN | True | Preprocess reference audio (remove silences) |
| postprocess_output | BOOLEAN | True | Post-process generated audio (remove long silences) |
| keep_model_loaded | BOOLEAN | True | Keep model in memory |

**Optional Input:**
- `whisper_model` — Pre-loaded Whisper from OmniVoice Whisper Loader

### 3. OmniVoice Voice Design TTS
Design voices from text descriptions. No reference audio needed.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | COMBO | (auto) | OmniVoice model checkpoint |
| text | STRING, multiline | `"Hello!..."` | Text to synthesize in designed voice |
| voice_instruct | STRING, multiline | `"female, low pitch..."` | Voice attributes (comma-separated) |
| steps | INT | 32 | Diffusion steps (4-64) |
| guidance_scale | FLOAT | 2.0 | Classifier-free guidance scale (0-10) |
| t_shift | FLOAT | 0.1 | Time-step shift for noise schedule (0-1) |
| speed | FLOAT | 1.0 | Speaking speed (0.5-2.0) |
| duration | FLOAT | 0.0 | Fixed duration in seconds (0=auto) |
| device | COMBO | auto | `auto`, `cuda`, `cpu`, `mps` |
| dtype | COMBO | auto | `auto`, `bf16`, `fp16`, `fp32` |
| attention | COMBO | auto | `auto`, `eager`, `sage_attention` |
| seed | INT | 0 | Random seed (0=random) |
| position_temperature | FLOAT | 5.0 | Mask-position temperature (0=greedy) |
| class_temperature | FLOAT | 0.0 | Token sampling temperature (0=greedy) |
| layer_penalty_factor | FLOAT | 5.0 | Penalty on deeper codebook layers |
| denoise | BOOLEAN | True | Prepend denoise token for cleaner output |
| postprocess_output | BOOLEAN | True | Post-process generated audio (remove long silences) |
| keep_model_loaded | BOOLEAN | True | Keep model in memory |

### 4. OmniVoice Multi-Speaker TTS
Generate dialogue between multiple speakers using `[Speaker_N]:` tags.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | COMBO | (auto) | OmniVoice model checkpoint |
| text | STRING, multiline | `"[Speaker_1]: Hello..."` | Multi-speaker text |
| num_speakers | DYNAMIC | 2 | Number of speakers (2-10, dynamic inputs) |
| steps | INT | 32 | Diffusion steps per speaker |
| guidance_scale | FLOAT | 2.0 | Classifier-free guidance scale (0-10) |
| t_shift | FLOAT | 0.1 | Time-step shift for noise schedule (0-1) |
| speed | FLOAT | 1.0 | Speaking speed for all speakers |
| pause_between_speakers | FLOAT | 0.3 | Silence between speakers (seconds) |
| device | COMBO | auto | `auto`, `cuda`, `cpu`, `mps` |
| dtype | COMBO | auto | `auto`, `bf16`, `fp16`, `fp32` |
| attention | COMBO | auto | `auto`, `eager`, `sage_attention` |
| position_temperature | FLOAT | 5.0 | Mask-position temperature (0=greedy) |
| class_temperature | FLOAT | 0.0 | Token sampling temperature (0=greedy) |
| layer_penalty_factor | FLOAT | 5.0 | Penalty on deeper codebook layers |
| denoise | BOOLEAN | True | Prepend denoise token for cleaner output |
| preprocess_prompt | BOOLEAN | True | Preprocess reference audio |
| postprocess_output | BOOLEAN | True | Post-process generated audio |
| seed | INT | 0 | Random seed (0=random) |
| keep_model_loaded | BOOLEAN | True | Keep model in memory |
| speaker_N_audio | AUDIO | optional | Reference audio for speaker N (1-10) |
| speaker_N_ref_text | STRING | "" | Transcript for speaker N's ref audio |

Speaker inputs dynamically show/hide based on `num_speakers` (ComfyUI >= 0.8.1).

### 5. OmniVoice Whisper Loader
Pre-load Whisper ASR model for auto-transcription. Avoid re-downloading on each run.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | COMBO | (auto) | Whisper model selection |
| device | COMBO | auto | `auto`, `cuda`, `cpu` |
| dtype | COMBO | auto | `auto`, `bf16`, `fp16`, `fp32` |

**Auto-download:** Select models with "(auto-download)" suffix to download on first use.

## Generation Parameters Guide

These parameters control the diffusion-based audio generation process:

| Parameter | What it does | Tips |
|-----------|-------------|------|
| `steps` | Number of iterative unmasking steps | 16 = faster, 32 = balanced, 64 = best quality |
| `guidance_scale` | Classifier-free guidance strength | Higher = more text-aligned; 2.0 is default |
| `t_shift` | Time-step shift for noise schedule | Smaller values emphasise earlier decoding steps |
| `speed` | Speaking speed factor | >1.0 = faster, <1.0 = slower |
| `duration` | Fixed output length in seconds | Overrides speed when set; 0 = automatic |
| `position_temperature` | Randomness in mask-position selection | 0 = greedy (deterministic), higher = more random |
| `class_temperature` | Randomness in token sampling | 0 = greedy (deterministic), higher = more random |
| `layer_penalty_factor` | Penalty on deeper codebook layers | Encourages lower layers to unmask first |
| `denoise` | Prepend denoise token to input | Generally improves output quality |
| `preprocess_prompt` | Clean reference audio | Removes long silences, adds punctuation |
| `postprocess_output` | Clean generated audio | Removes long silences from output |

## Attention Backends

OmniVoice's architecture (Qwen3 backbone) has limited attention support through transformers. The `attention` dropdown offers these options:

| Option | What Actually Happens |
|--------|----------------------|
| `auto` | OmniVoice's default (eager) |
| `eager` | Standard eager attention (always works) |
| `sage_attention` | **Monkey-patches Qwen3Attention** with SageAttention CUDA kernels. GPU-only, requires SM80+ (Ampere+). Falls back to SDPA when attention masks are present. Install: `pip install sageattention` |

### SageAttention GPU Compatibility
| GPU Architecture | Compute Capability | Kernel Used |
|-----------------|-------------------|-------------|
| Blackwell (RTX 5090) | SM120 | FP8 |
| Hopper (RTX 4090) | SM90 | FP8 |
| Ada Lovelace (RTX 4070) | SM89 | FP8 |
| Ampere (RTX 3090) | SM80 | FP16 |
| Below SM80 | — | Not supported |

## Multi-Speaker Usage

Use `[Speaker_N]:` tags in text to assign lines to different speakers:
```
[Speaker_1]: Hello, I'm speaker one.
[Speaker_2]: And I'm speaker two!
[Speaker_1]: Nice to meet you!
```
Each speaker needs reference audio connected to the corresponding `speaker_N_audio` input.

## Voice Design Attributes

Comma-separated attributes for `voice_instruct`:

| Category | Options |
|----------|---------|
| **Gender** | `male`, `female` |
| **Age** | `child`, `young`, `middle-aged`, `elderly` |
| **Pitch** | `very low pitch`, `low pitch`, `medium pitch`, `high pitch`, `very high pitch` |
| **Style** | `whisper` |
| **English Accent** | `american accent`, `british accent`, `australian accent`, etc. |
| **Chinese Dialect** | `四川话`, `陕西话`, `广东话`, `东北话`, `山东话`, etc. |

**Example:** `"female, young, high pitch, british accent, whisper"`

## Non-Verbal Tags

Insert these directly in your text:

| Tag | Effect |
|-----|--------|
| `[laughter]` | Natural laughter |
| `[sigh]` | Expressive sigh |
| `[sniff]` | Sniffing sound |
| `[question-en]`, `[question-ah]`, `[question-oh]` | Question intonations |
| `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]` | Surprise expressions |
| `[dissatisfaction-hnn]` | Dissatisfaction sound |
| `[confirmation-en]` | Confirmation grunt |

**Example:**
```
[laughter] You really got me! [sigh] I didn't see that coming at all.
```

## Model Storage

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

### Available OmniVoice Models
| Model | Size | Description |
|-------|------|-------------|
| `OmniVoice` | ~4GB | Full fp32 model - 600+ languages |
| `OmniVoice-bf16` | ~2GB | Bfloat16 quantized - lower VRAM |

### Whisper Models
| Model | VRAM | Link |
|-------|------|------|
| whisper-large-v3-turbo | ~1.5GB | [Download](https://huggingface.co/openai/whisper-large-v3-turbo) |
| whisper-large-v3 | ~3GB | [Download](https://huggingface.co/openai/whisper-large-v3) |
| whisper-medium | ~1GB | [Download](https://huggingface.co/openai/whisper-medium) |
| whisper-small | ~0.5GB | [Download](https://huggingface.co/openai/whisper-small) |
| whisper-tiny | ~0.4GB | [Download](https://huggingface.co/openai/whisper-tiny) |

Models auto-download from HuggingFace on first use.

## VRAM Requirements

| Precision | VRAM (Approx) |
|-----------|---------------|
| fp32 | ~8-12 GB |
| bf16/fp16 | ~4-6 GB |
| With CPU offload | ~2-4 GB |

## Model Caching

The node caches loaded models for reuse. Changing any of these parameters **forces a full cache clear** (model unload + GC + CUDA cache flush), even when `keep_model_loaded` is `True`:

- Model selection
- Device
- Precision (dtype)
- Attention backend

## Troubleshooting

### Model download fails (China)
Set the HuggingFace mirror before starting ComfyUI:
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

### Whisper re-downloads every run
Connect `OmniVoice Whisper Loader` to `whisper_model` input on Voice Clone TTS to cache the model.

### CUDA out of memory
- Set `keep_model_loaded = False`
- Use `dtype = fp16` or `bf16`
- Use `device = cpu` (slower but works)

### Import errors after install
Restart ComfyUI completely to reload Python modules.

### Transformers version

OmniVoice requires `transformers>=5.3.0`. If you see an error like `omnivoice import failed` or `cannot import name 'HiggsAudioV2TokenizerModel'` in your ComfyUI logs, your transformers version may be too old.

> ⚠️ **Only do this if you know what you are doing.** Upgrading transformers may break other custom nodes that depend on an older version. Test your other nodes after upgrading.

To upgrade:
```
path\to\ComfyUI\venv\Scripts\python.exe -m pip install "transformers>=5.3.0"
```

### FFmpeg error on Windows when saving audio
Add your FFmpeg `bin/` folder to `PATH` in your ComfyUI launch `.bat` file, or use a WAV audio save node instead.

## Credits

- **OmniVoice** — [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) by k2-fsa — Original fp32 model
- **OmniVoice-bf16** — [drbaph/OmniVoice-bf16](https://huggingface.co/drbaph/OmniVoice-bf16) by drbaph — Bfloat16 quantized model
- **ComfyUI Node** — [saganaki22/ComfyUI-OmniVoice-TTS](https://github.com/saganaki22/ComfyUI-OmniVoice-TTS) — This custom node

## Citation

```bibtex
@article{zhu2026omnivoice,
      title={OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models},
      author={Zhu, Han and Ye, Lingxuan and Kang, Wei and Yao, Zengwei and Guo, Liyong and Kuang, Fangjun and Han, Zhifeng and Zhuang, Weiji and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2604.00688},
      year={2026}
}
```

## License

This custom node is released under the Apache 2.0 License. The OmniVoice model has its own license — see [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) for details.
