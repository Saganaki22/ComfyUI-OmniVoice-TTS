# OmniVoice LoRA Training Guide

Fine-tune OmniVoice on your own voice data using LoRA (Low-Rank Adaptation). This runs entirely on a single GPU inside ComfyUI.

## Prerequisites

1. **Training dependencies** — install via ComfyUI-Manager or manually:
   ```bash
   pip install peft safetensors
   ```
2. **GPU with >= 8 GB VRAM** (OmniVoice ~0.6B params + LoRA overhead)
3. **Audio data** — paired `.wav` files + `.txt` transcripts

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
- **Duration**: 3–15 seconds per file is ideal
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
| `learning_rate` | 5e-5 | 1e-5 – 1e-4 | Lower for more data |
| `lora_rank` | 32 | 16 – 64 | Higher = more capacity, more VRAM |
| `lora_alpha` | 16 | 8 – 32 | Usually 0.5× rank |
| `lora_dropout` | 0.0 | 0.0 – 0.1 | Add if overfitting |
| `warmup_steps` | 100 | 50 – 200 | ~10% of total steps |
| `grad_accum_steps` | 1 | 1 – 4 | Increase if OOM |
| `weight_decay` | 0.01 | 0.0 – 0.05 | Regularization |
| `target_modules` | q_proj,k_proj,v_proj,o_proj | — | Add gate_proj,up_proj,down_proj for deeper adaptation |
| `batch_size` | 1 | 1 – 8 | Samples per step. batch=2 ~8GB, batch=4 ~10-12GB VRAM. Ignored when Sequence Packing is on |
| `sequence_packing` | False | — | Pack multiple short samples into one sequence. Overrides batch_size. Most GPU-efficient option |
| `batch_tokens` | 4096 | 2048 – 8192 | Max token length per packed sequence (only used when Sequence Packing is on) |
| `torch_compile` | False | — | Compile model for faster training. Strongly recommended with Sequence Packing — without it, flex_attention is very slow. Adds ~30-60s one-time compile at start |
| `train_audio_layers` | True | True | Recommended — adapts audio interface layers |

### Node 2: OmniVoice Dataset Maker

- Set `audio_directory` to your dataset folder path
- Set `language_id` to your language code (e.g., `en`, `zh`, `ja`, `de`)
- Output: path to the JSONL manifest

### Node 3: OmniVoice LoRA Trainer

- Connect `train_config` from Node 1
- Set `dataset_path` from Node 2 output
- Configure:
  - `max_steps`: 500–2000 for small datasets, 2000–5000 for large
  - `save_every_steps`: 200–500 (saves intermediate checkpoints)
  - `output_name`: folder name for the LoRA output

**Warning**: Training blocks the ComfyUI UI. Use the interrupt button to cancel.

## Using Your Trained LoRA

After training, the LoRA is saved to `models/loras/<output_name>/`. All 4 OmniVoice inference nodes now have a `lora_name` dropdown:

1. Open any OmniVoice TTS node (Longform, Voice Clone, Voice Design, Multi-Speaker)
2. Select your LoRA from the `lora_name` dropdown
3. Generate speech — the LoRA voice will be applied automatically

## Training Tips

### How Many Steps?

| Dataset Size | Recommended Steps | Expected Result |
|---|---|---|
| 20–50 files | 500–1000 | Basic voice similarity |
| 50–200 files | 1000–3000 | Good voice cloning |
| 200+ files | 3000–5000 | High-quality voice replication |

### Troubleshooting

- **OOM (Out of Memory)**: Reduce `lora_rank` to 16, increase `grad_accum_steps` to 2–4, or reduce `batch_size` / `batch_tokens`
- **Training too slow**: Enable `sequence_packing` with `batch_tokens=4096` for best GPU utilization, or increase `batch_size` to 2–4
- **Unused GPU VRAM**: Increase `batch_size` (e.g. 2–4) to fill more of the GPU. Or enable `sequence_packing` for maximum efficiency
- **Loss not decreasing**: Lower `learning_rate` to 1e-5, check audio quality
- **Overfitting** (sounds robotic): Reduce `max_steps`, add `lora_dropout` of 0.05–0.1
- **Audio garbled**: Ensure source audio is clean, 24kHz-compatible, 3–15s per file

### What Gets Trained

- **LoRA on Qwen3-0.6B backbone**: Targets attention projections (q, k, v, o). Learns the text→audio-token patterns for your voice.
- **Audio layers** (when `train_audio_layers=True`): `audio_embeddings` and `audio_heads`. These handle the audio↔LLM interface and adapt to your voice's token distribution.

### Speeding Up Training

The default settings (batch_size=1, no packing) only use ~4 GB VRAM. If your GPU has more headroom, you can train much faster:

| Method | How | VRAM Impact | Speed Gain |
|---|---|---|---|
| **Increase batch_size** | Set `batch_size` to 2–4 | ~2-3× more VRAM per sample | Near-linear speedup |
| **Enable sequence packing** | Set `sequence_packing=True` + `batch_tokens=4096` | Scales with batch_tokens | Most efficient — packs multiple samples into one forward pass |

**Sequence packing** is the most GPU-efficient option. It concatenates multiple short samples into one fixed-length sequence and uses block attention masking to prevent cross-sample attention. This is the same technique the upstream OmniVoice training uses.

When `sequence_packing` is enabled, `batch_size` is ignored (packing handles multi-sample internally). Use `batch_tokens` to control the packed sequence length — higher values fit more samples per step but use more VRAM.

**Important**: When using sequence packing, enable `torch_compile` as well. Without compile, PyTorch's flex_attention materializes the full attention matrix which is extremely slow. With compile, it generates a fused kernel that runs efficiently.

### Architecture

```
OmniVoice (Qwen3-0.6B)
├── llm              ← LoRA applied here (attention projections)
├── audio_embeddings ← Trained if train_audio_layers=True
└── audio_heads      ← Trained if train_audio_layers=True
```
