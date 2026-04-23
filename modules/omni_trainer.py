"""Core LoRA training loop for OmniVoice.

Implements the full training pipeline:
  1. Audio tokenization (HiggsAudio v2)
  2. Model loading with PEFT LoRA on Qwen3-0.6B backbone
  3. Training with weighted cross-entropy loss across 8 codebooks
  4. Checkpoint saving in safetensors format

Designed for single-GPU training within ComfyUI.
"""

import gc
import json
import logging
import os
import contextlib
from typing import Optional

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

logger = logging.getLogger("OmniVoice")

# Check optional deps
PEFT_AVAILABLE = False
SAFETENSORS_AVAILABLE = False
TRAINING_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    pass

try:
    from safetensors.torch import save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    pass

try:
    from omnivoice.data.processor import OmniVoiceSampleProcessor
    TRAINING_AVAILABLE = True
except ImportError:
    pass


def _check_deps():
    """Verify all training dependencies are available."""
    missing = []
    if not PEFT_AVAILABLE:
        missing.append("peft")
    if not SAFETENSORS_AVAILABLE:
        missing.append("safetensors")
    if not TRAINING_AVAILABLE:
        missing.append("omnivoice (with training support)")
    if missing:
        raise RuntimeError(
            f"Missing required packages for training: {', '.join(missing)}. "
            f"Run: pip install {' '.join(missing)}"
        )


def _load_text_tokenizer(model_path: str) -> AutoTokenizer:
    """Load the text tokenizer with OmniVoice special tokens.

    Mirrors omnivoice.training.builder.build_model_and_tokenizer().
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    new_tokens = [
        "<|denoise|>",
        "<|lang_start|>",
        "<|lang_end|>",
        "<|instruct_start|>",
        "<|instruct_end|>",
        "<|text_start|>",
        "<|text_end|>",
    ]
    tokens_to_add = [t for t in new_tokens if t not in tokenizer.get_vocab()]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    return tokenizer


def _resolve_model_path(model_name: str) -> str:
    """Resolve a model name to a local path for training.

    Checks ComfyUI's omnivoice models folder and HF cache.
    """
    from ..nodes.loader import _get_models_base, _strip_auto_download_suffix, HF_MODELS

    model_name = _strip_auto_download_suffix(model_name)
    base = _get_models_base()

    # Check if it's a known HF model
    if model_name in HF_MODELS:
        local_path = base / model_name
        if local_path.is_dir():
            return str(local_path)

    # Check if it's a local folder name
    local_path = base / model_name
    if local_path.is_dir():
        return str(local_path)

    # Try as-is (might be an absolute path or HF repo)
    if os.path.isdir(model_name):
        return model_name

    raise ValueError(
        f"Cannot find model '{model_name}'. Checked: {base}, and direct path."
    )


def _apply_lora(model, config: dict):
    """Apply PEFT LoRA to the Qwen3 backbone inside OmniVoice.

    Targets attention projections by default. Optionally targets MLP
    projections if specified in config.
    """
    target_modules = config.get("target_modules", "q_proj,k_proj,v_proj,o_proj")
    if isinstance(target_modules, str):
        target_modules = [m.strip() for m in target_modules.split(",")]

    lora_config = LoraConfig(
        r=config.get("lora_rank", 32),
        lora_alpha=config.get("lora_alpha", 16),
        target_modules=target_modules,
        lora_dropout=config.get("lora_dropout", 0.0),
        bias="none",
        task_type=None,  # Qwen3Model is AutoModel, not AutoModelForCausalLM
    )

    model.llm = get_peft_model(model.llm, lora_config)
    return model


def _print_trainable_params(model):
    """Log the number of trainable parameters."""
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    logger.info(
        f"Trainable: {trainable:,} / {total:,} params "
        f"({100 * trainable / total:.2f}%)"
    )


@torch.inference_mode(False)
def run_lora_training(
    model_name: str,
    train_config: dict,
    dataset_path: str,
    output_dir: str,
    max_steps: int,
    save_every_steps: int,
):
    """Execute the full LoRA training pipeline.

    Steps:
        1. Tokenize audio files (HiggsAudio v2)
        2. Load OmniVoice model + text tokenizer
        3. Apply PEFT LoRA to Qwen3 backbone
        4. Train with weighted CE loss + data augmentation
        5. Save LoRA checkpoints as safetensors

    Args:
        model_name: OmniVoice model identifier (from dropdown).
        train_config: Dict of hyperparams from TrainConfig node.
        dataset_path: Path to JSONL manifest from DatasetMaker.
        output_dir: Directory to save LoRA outputs.
        max_steps: Total training steps.
        save_every_steps: Checkpoint interval.

    Returns:
        output_dir path on success.
    """
    _check_deps()
    torch.set_grad_enabled(True)

    import comfy.model_management as mm
    from comfy.utils import ProgressBar

    from .omni_tokenizer import tokenize_dataset
    from .omni_dataset import (
        TokenizedAudioDataset,
        build_processor,
        collate_processed_sample,
        collate_multi_sample,
        collate_packed_samples,
    )

    # ------------------------------------------------------------------
    # 1. Resolve model path
    # ------------------------------------------------------------------
    model_path = _resolve_model_path(model_name)
    logger.info(f"Model path: {model_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Save training config
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump({**train_config, "base_model": model_name, "max_steps": max_steps}, f, indent=2)

    # ------------------------------------------------------------------
    # 2. Tokenize audio
    # ------------------------------------------------------------------
    # Free all models before tokenization to maximize VRAM for HiggsAudio
    mm.unload_all_models()
    mm.soft_empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Step 1/4: Tokenizing audio files...")
    tokenized_manifest = tokenize_dataset(
        jsonl_path=dataset_path,
        device=device,
    )

    # Free tokenization memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. Load model + tokenizer + apply LoRA
    # ------------------------------------------------------------------
    logger.info("Step 2/4: Loading model and applying LoRA...")
    mm.unload_all_models()
    mm.soft_empty_cache()

    from omnivoice import OmniVoice

    text_tokenizer = _load_text_tokenizer(model_path)

    # Attention backend selection:
    #   - SDPA: best for non-packed training (fused Flash Attention kernels, fast, low VRAM)
    #   - flex_attention: required for sequence packing (custom mask_mod for document
    #     boundaries via BlockMask). Needs torch_compile for speed.
    sequence_packing = train_config.get("sequence_packing", False)
    attn_impl = "flex_attention" if sequence_packing else "sdpa"

    model = OmniVoice.from_pretrained(
        model_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
        train=True,
    )
    logger.info(f"Attention backend: {attn_impl}")

    # Resize embeddings if tokenizer added tokens
    if len(text_tokenizer) != model.config.llm_config.vocab_size:
        model.llm.resize_token_embeddings(len(text_tokenizer))
        model.config.llm_config.vocab_size = len(text_tokenizer)

    # Apply LoRA to the LLM backbone
    model = _apply_lora(model, train_config)

    # Configure audio layers
    train_audio_layers = train_config.get("train_audio_layers", True)
    # Freeze all OmniVoice-level params first (LLM handled by PEFT)
    for name, param in model.named_parameters():
        if "llm" not in name:
            param.requires_grad = bool(train_audio_layers)

    model = model.to(device)
    model.train()

    # Optional torch.compile for faster training (critical for flex_attention/packing)
    torch_compile = train_config.get("torch_compile", False)
    if torch_compile:
        logger.info("Compiling model with torch.compile() — one-time overhead (~30-60s)...")
        model = torch.compile(model)

    _print_trainable_params(model)

    # Verify gradients flow
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters! Check LoRA config.")
    logger.info(f"Trainable param groups: {len(trainable_params)}")

    # ------------------------------------------------------------------
    # 4. Build dataset + processor
    # ------------------------------------------------------------------
    logger.info("Step 3/4: Building dataset...")
    processor = build_processor(text_tokenizer)

    batch_size = train_config.get("batch_size", 1)
    batch_tokens = train_config.get("batch_tokens", 4096)

    # When packing is enabled, force batch_size=1 (packing handles multi-sample internally)
    if sequence_packing:
        effective_batch_size = 1
        logger.info(
            f"Sequence packing enabled: batch_tokens={batch_tokens}, "
            f"batch_size overridden to 1 (packing handles multi-sample internally)"
        )
    else:
        effective_batch_size = batch_size
        if batch_size > 1:
            logger.info(f"Batch mode: batch_size={batch_size}")

    dataset = TokenizedAudioDataset(tokenized_manifest, shuffle=True, seed=42)
    data_iter = iter(dataset)

    def get_next_sample():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            dataset.set_epoch(dataset.epoch + 1)
            data_iter = iter(dataset)
            return next(data_iter)

    def get_next_batch():
        """Get and process samples, returning a collated batch dict.

        Handles three modes:
          - Packing: accumulate samples up to batch_tokens, collate with document_ids
          - batch>1: collect N samples, pad to longest
          - batch=1: single sample (original behavior)
        """
        if sequence_packing:
            # Pack samples up to batch_tokens total length
            packed = []
            total_len = 0
            while total_len < batch_tokens:
                raw = get_next_sample()
                try:
                    processed = processor(raw)
                except Exception as e:
                    logger.warning(f"Skipping sample: {e}")
                    continue

                sample_len = processed["length"]
                if sample_len > batch_tokens:
                    continue  # Skip samples longer than batch_tokens
                if total_len + sample_len > batch_tokens and packed:
                    break  # Would exceed, yield what we have

                packed.append(processed)
                total_len += sample_len

            if not packed:
                return None

            logger.debug(
                f"Packed {len(packed)} samples into {total_len}/{batch_tokens} tokens"
            )

            return collate_packed_samples(
                packed,
                batch_tokens=batch_tokens,
                pad_token_id=text_tokenizer.pad_token_id,
                device=device,
            )

        elif effective_batch_size > 1:
            # Collect batch_size samples and pad to longest
            processed_list = []
            for _ in range(effective_batch_size):
                raw = get_next_sample()
                try:
                    processed_list.append(processor(raw))
                except Exception as e:
                    logger.warning(f"Skipping sample: {e}")
                    continue

            if not processed_list:
                return None

            return collate_multi_sample(
                processed_list,
                pad_token_id=text_tokenizer.pad_token_id,
                device=device,
            )

        else:
            # Original single-sample mode
            raw = get_next_sample()
            try:
                processed = processor(raw)
            except Exception as e:
                logger.warning(f"Skipping sample: {e}")
                return None

            return collate_processed_sample(processed, device=device)

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    logger.info("Step 4/4: Starting training...")

    grad_accum_steps = train_config.get("grad_accum_steps", 1)
    learning_rate = train_config.get("learning_rate", 5e-5)
    warmup_steps = train_config.get("warmup_steps", 100)
    weight_decay = train_config.get("weight_decay", 0.01)

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    pbar = ProgressBar(max_steps)
    pbar.update(0)

    try:
        with torch.enable_grad():
            for step in range(max_steps):
                mm.throw_exception_if_processing_interrupted()

                optimizer.zero_grad(set_to_none=True)

                total_loss_val = 0.0
                did_backward = False

                for micro_step in range(grad_accum_steps):
                    # Get next batch (handles packing / multi-sample / single)
                    batch = get_next_batch()
                    if batch is None:
                        continue

                    seq_len = batch["input_ids"].shape[-1]
                    batch_dim = batch["input_ids"].shape[0]
                    if step == 0 and micro_step == 0:
                        mode = "packed" if "document_ids" in batch else f"batch={batch_dim}"
                        logger.info(
                            f"First batch: seq_len={seq_len}, {mode}"
                        )

                    # Forward pass — no autocast needed since model is already bf16.
                    # autocast(dtype=bfloat16) conflicts with flex_attention's
                    # backward pass (dtype mismatch in sdpa_dense_backward).
                    forward_kwargs = {
                        "input_ids": batch["input_ids"],
                        "audio_mask": batch["audio_mask"],
                        "labels": batch["labels"],
                        "position_ids": batch["position_ids"],
                    }
                    # SDPA multi-sample: pass 4D attention_mask for proper padding handling
                    if "attention_mask" in batch:
                        forward_kwargs["attention_mask"] = batch["attention_mask"]
                    # Packed sequences: pass document_ids for BlockMask (flex_attention)
                    if "document_ids" in batch:
                        forward_kwargs["document_ids"] = batch["document_ids"]

                    outputs = model(**forward_kwargs)

                    loss = outputs.loss
                    if loss is None:
                        logger.warning(f"Step {step}, micro {micro_step}: no loss returned")
                        continue

                    loss_scaled = loss / grad_accum_steps

                    if loss_scaled.grad_fn is not None:
                        loss_scaled.backward()
                        total_loss_val += loss.item()
                        did_backward = True
                    else:
                        if step == 0 and micro_step == 0:
                            raise RuntimeError(
                                "Loss has no grad_fn at step 0 — LoRA params "
                                "not receiving gradients. Check lora_config."
                            )

                if did_backward:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad),
                        max_norm=1.0,
                    )
                    optimizer.step()
                    scheduler.step()
                    pbar.update(1)

                if step == 0 or step % 10 == 0:
                    lr_val = optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"Step {step}/{max_steps} | "
                        f"Loss: {total_loss_val:.4f} | "
                        f"LR: {lr_val:.2e}"
                    )

                # Save checkpoint
                if (step + 1) % save_every_steps == 0 or (step + 1) == max_steps:
                    _save_lora_checkpoint(model, output_dir, step + 1)
                    logger.info(f"Saved checkpoint at step {step + 1}")

    finally:
        del model, optimizer, scheduler, data_iter
        gc.collect()
        mm.soft_empty_cache()

    logger.info(f"Training complete. Output: {output_dir}")
    return output_dir


def _save_lora_checkpoint(model, output_dir: str, step: int):
    """Save LoRA weights and config in PEFT-compatible format.

    Uses PEFT's own save_pretrained() for correct key format, then saves
    audio layers separately.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Save per-step checkpoint in subfolder ---
    step_dir = os.path.join(output_dir, f"checkpoints", f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)

    _save_adapter(model, step_dir)

    # --- Also save as latest (overwrite) at top level ---
    _save_adapter(model, output_dir)

    # --- Latest marker ---
    latest_path = os.path.join(output_dir, "latest.txt")
    with open(latest_path, "w") as f:
        f.write(step_dir)


def _save_adapter(model, directory):
    """Save adapter files using PEFT's save_pretrained for correct key format."""
    os.makedirs(directory, exist_ok=True)

    # Use PEFT's own save — guarantees correct key names
    model.llm.save_pretrained(directory)

    # Collect and save audio layer weights if trained
    audio_state = {}
    has_audio_layers = False
    for name, param in model.named_parameters():
        if "llm" not in name and param.requires_grad:
            audio_state[name] = param.data.cpu()
            has_audio_layers = True

    if has_audio_layers and audio_state:
        audio_path = os.path.join(directory, "audio_layers.safetensors")
        if SAFETENSORS_AVAILABLE:
            save_file(audio_state, audio_path)
        else:
            torch.save(
                {"state_dict": audio_state},
                audio_path.replace(".safetensors", ".bin"),
            )

    # Add custom fields to the config PEFT already wrote
    config_path = os.path.join(directory, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            adapter_config = json.load(f)
    else:
        adapter_config = {}

    adapter_config["train_audio_layers"] = has_audio_layers

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2)
