# OmniVoice LoRA Training Guide

Fine-tune OmniVoice on your own voice data using LoRA (Low-Rank Adaptation). This runs entirely on a single GPU inside ComfyUI.

## Prerequisites

1. **Training dependencies** — install via ComfyUI-Manager or manually:
   ```bash
   pip install peft safetensors
   ```
2. **GPU with >= 8 GB VRAM** (OmniVoice ~0.6B params + LoRA overhead)
3. **Audio data** — paired audio files + `.txt` transcripts

## Preparing Your Dataset

### Directory Structure

Create a folder with paired audio + text files. Each audio file must have a matching `.txt` file with the **same base name**:

```
my_voice_data/
├── sample_001.wav
├── sample_001.txt       ← "Hello, welcome to my podcast."
├── sample_002.wav
├── sample_002.txt       ← "Today we'll discuss fine-tuning."
├── sample_003.wav
├── sample_003.txt       ← "Let's get started."
└── ...
```

### Audio Guidelines

- **Format**: `.wav`, `.mp3`, `.flac`, `.ogg`, or `.m4a`
- **Duration**: 3–20 seconds per file works well
- **Quality**: Clean, no background noise, consistent volume
- **Content**: Natural speech — read sentences, paragraphs, or scripts
- **Amount**: 20–100 files minimum for basic voice cloning (more = better)

### Text Guidelines

- Plain text, one sentence or phrase per `.txt` file
- Must match what was actually spoken in the audio
- No special formatting needed

## ComfyUI Workflow

The training workflow uses **3 nodes** connected in sequence:

### Node 1: OmniVoice Train Config

Configure training hyperparameters:

| Parameter | Default | Recommended Range | Notes |
|---|---|---|---|
| `learning_rate` | 5e-5 | — | Keep default — tested and works well |
| `lora_rank` | 32 | 16 – 64 | 32 is the sweet spot. Higher = more capacity, more VRAM |
| `lora_alpha` | 16 | 8 – 32 | Usually 0.5× rank |
| `lora_dropout` | 0.0 | 0.0 – 0.1 | Add if overfitting |
| `warmup_steps` | 100 | 50 – 200 | ~10% of total steps |
| `grad_accum_steps` | 1 | 1 – 4 | Increase if OOM |
| `weight_decay` | 0.01 | 0.0 – 0.05 | Regularization |
| `target_modules` | q_proj,k_proj,v_proj,o_proj | — | Add gate_proj,up_proj,down_proj for deeper adaptation |
| `sequence_packing` | False | — | Pack multiple samples into one sequence. Recommended for best training quality |
| `batch_tokens` | 4096 | 2048 – 8192 | Max token length per packed sequence. 4096 uses ~24GB VRAM |
| `torch_compile` | False | — | Compile model for faster training. Strongly recommended with Sequence Packing. Adds ~30-60s one-time compile |
| `train_audio_layers` | True | True | Recommended — adapts audio interface layers |

### Node 2: OmniVoice Dataset Maker

- Set `audio_directory` to your dataset folder path
- Set `language_id` to your language code (e.g., `en`, `zh`, `ja`, `de`)
- Output: path to the JSONL manifest

### Node 3: OmniVoice LoRA Trainer

- Connect `train_config` from Node 1
- Set `dataset_path` from Node 2 output
- Configure:
  - `max_steps`: 250–1000 for best results (see Training Tips below)
  - `save_every_steps`: 250 (save checkpoints to compare)
  - `output_name`: folder name for the LoRA output

**Warning**: Training blocks the ComfyUI UI. Use the interrupt button to cancel.

## Using Your Trained LoRA

After training, the LoRA is saved to `models/loras/<output_name>/`. All 4 OmniVoice inference nodes have a `lora_name` dropdown:

1. Open any OmniVoice TTS node (Longform, Voice Clone, Voice Design, Multi-Speaker)
2. Select your LoRA from the `lora_name` dropdown
3. Generate speech — the LoRA voice will be applied automatically

## Training Tips

### Tested Configuration (40 clips, 8–20 seconds each)

Tested with 40 audio clips (8–20 seconds each) with matching `.txt` transcripts. Best results:

- **lora_rank**: 32
- **learning_rate**: 5e-5 (default)
- **sequence_packing**: True
- **batch_tokens**: 4096
- **torch_compile**: True
- **train_audio_layers**: True
- **Best checkpoint range**: 250–1000 steps

With these settings, loss converges smoothly from ~4.8 down to ~1.7. The sweet spot for voice similarity was between steps 250–1000. Going beyond 1000 steps with 40 clips risks overfitting.

### When to Stop Training

Watch the loss and test checkpoints along the way:

| Loss Range | Voice Quality |
|---|---|
| ~5–6 | Starting out, barely noticeable |
| ~3–4 | Some similarity |
| ~2–3 | Good voice match |
| ~1–2 | Strong clone |
| <1 | Overfitting risk — may sound robotic |

Save checkpoints every 250 steps and test each one. Stop when the voice sounds closest to your target.

### Troubleshooting

- **OOM (Out of Memory)**: Reduce `lora_rank` to 16, reduce `batch_tokens` to 2048, or increase `grad_accum_steps` to 2–4
- **Training too slow**: Enable `sequence_packing` with `torch_compile=True` and `batch_tokens=4096`
- **Loss not decreasing**: Lower `learning_rate` to 1e-5, check audio quality
- **Overfitting** (sounds robotic): Reduce `max_steps`, add `lora_dropout` of 0.05–0.1
- **Audio garbled**: Ensure source audio is clean, 24kHz-compatible

### What Gets Trained

- **LoRA on Qwen3-0.6B backbone**: Targets attention projections (q, k, v, o). Learns the text→audio-token patterns for your voice.
- **Audio layers** (when `train_audio_layers=True`): `audio_embeddings` and `audio_heads`. These handle the audio↔LLM interface and adapt to your voice's token distribution.

### Sequence Packing

**Sequence packing** is recommended for best training quality. It concatenates multiple samples into one fixed-length sequence and uses block attention masking to prevent cross-sample attention. This is the same technique the upstream OmniVoice training uses.

Use `batch_tokens` to control the packed sequence length:
- `4096` uses ~24GB VRAM (good starting point)
- `2048` uses less VRAM (~16GB)
- `8192` uses more VRAM (~32GB+)

**Important**: When using sequence packing, enable `torch_compile` as well. Without compile, PyTorch's flex_attention materializes the full attention matrix which is extremely slow. With compile, it generates a fused kernel that runs efficiently.

### Architecture

```
OmniVoice (Qwen3-0.6B)
├── llm              ← LoRA applied here (attention projections)
├── audio_embeddings ← Trained if train_audio_layers=True
└── audio_heads      ← Trained if train_audio_layers=True
```
