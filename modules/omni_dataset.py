"""Dataset classes for OmniVoice LoRA training.

Loads pre-tokenized audio data and applies OmniVoice's data augmentation
(prompt ratio, mask ratio, condition dropout) using the upstream
OmniVoiceSampleProcessor.

Supports three batching modes:
  - batch_size=1 (default): single sample per step, no padding needed
  - batch_size>1: multiple samples padded to the longest in the batch
  - sequence_packing: multiple samples concatenated into one fixed-length
    sequence with document_ids for block attention masking
"""

import json
import logging
import os
import random
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger("OmniVoice")


class TokenizedAudioDataset(IterableDataset):
    """Iterable dataset that loads pre-tokenized audio from a JSONL manifest.

    Each entry in the manifest should have:
        - "audio_tokens_path": path to .npy file with shape [8, T] int16 tokens
        - "text": transcript
        - "id": unique identifier
        - "language_id" (optional): language code (e.g. "en")
        - "instruct" (optional): voice design instruction
        - "text_pinyin" (optional): pinyin for Chinese text

    Yields dicts matching the format expected by OmniVoiceSampleProcessor:
        {"audio_tokens": Tensor[C, T], "label": dict}
    """

    def __init__(
        self,
        manifest_path: str,
        shuffle: bool = True,
        seed: int = 42,
        epoch: int = 0,
    ):
        self.manifest_path = manifest_path
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch
        self.entries = self._load_manifest()

    def _load_manifest(self) -> List[Dict]:
        entries = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        logger.info(f"Loaded {len(entries)} entries from {self.manifest_path}")
        return entries

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        indices = list(range(len(self.entries)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)

        for idx in indices:
            entry = self.entries[idx]
            npy_path = entry.get("audio_tokens_path")
            if not npy_path or not os.path.exists(npy_path):
                continue

            try:
                audio_tokens = np.load(npy_path)
                audio_tokens = torch.from_numpy(audio_tokens).long()
            except Exception as e:
                logger.warning(f"Failed to load tokens for {entry.get('id', '?')}: {e}")
                continue

            if audio_tokens.dim() != 2 or audio_tokens.size(0) != 8:
                logger.warning(
                    f"Unexpected token shape {audio_tokens.shape} for {entry.get('id', '?')}"
                )
                continue

            yield {
                "audio_tokens": audio_tokens,  # [8, T]
                "label": entry,
            }


def build_processor(text_tokenizer, language_id: str = "en"):
    """Build an OmniVoiceSampleProcessor for training data augmentation.

    Uses the upstream processor from omnivoice.data.processor.
    For single-speaker fine-tuning, we simplify the augmentation:
    - Always include the language tag
    - No pinyin (not relevant for most languages)
    - No instruct tokens (not relevant for voice cloning)
    - Moderate prompt ratio for voice conditioning
    - High mask ratio for effective training

    Args:
        text_tokenizer: HuggingFace text tokenizer.
        language_id: Default language ID for the dataset.

    Returns:
        OmniVoiceSampleProcessor instance.
    """
    from omnivoice.data.processor import OmniVoiceSampleProcessor

    return OmniVoiceSampleProcessor(
        text_tokenizer=text_tokenizer,
        num_channels=8,
        audio_mask_id=1024,
        prompt_ratio_range=(0.0, 0.3),
        mask_ratio_range=(0.0, 1.0),
        drop_cond_ratio=0.1,
        language_ratio=0.8,
        use_pinyin_ratio=0.0,
        instruct_ratio=0.0,
        only_instruct_ratio=0.0,
    )


def collate_processed_sample(processed: Dict[str, Any], device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Convert a single processed sample into batch format for model forward.

    The processor returns:
        input_ids: [C, L]
        labels: [C, L]
        audio_mask: [L]
        length: int

    The model expects batch dimension:
        input_ids: [1, C, L]
        audio_mask: [1, L]
        labels: [1, C, L]
        position_ids: [1, L]

    Args:
        processed: Output from OmniVoiceSampleProcessor.
        device: Target device.

    Returns:
        Dict of tensors ready for model.forward().
    """
    input_ids = processed["input_ids"].unsqueeze(0).to(device)
    labels = processed["labels"].unsqueeze(0).to(device)
    audio_mask = processed["audio_mask"].unsqueeze(0).to(device)
    length = processed["length"]
    position_ids = torch.arange(length, dtype=torch.long, device=device).unsqueeze(0)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "audio_mask": audio_mask,
        "position_ids": position_ids,
    }


def collate_multi_sample(
    processed_list: List[Dict[str, Any]],
    pad_token_id: int,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Collate multiple processed samples into one batch with padding.

    Pads all samples to the length of the longest sample in the batch.
    Uses right-side padding with:
      - input_ids: pad_token_id
      - labels: -100 (ignored in loss)
      - audio_mask: False
      - position_ids: 0

    Args:
        processed_list: List of outputs from OmniVoiceSampleProcessor.
        pad_token_id: Token ID used for padding input_ids.
        device: Target device.

    Returns:
        Dict of batched tensors [B, C, L] / [B, L].
    """
    if not processed_list:
        raise ValueError("Empty processed_list passed to collate_multi_sample")

    if len(processed_list) == 1:
        return collate_processed_sample(processed_list[0], device=device)

    # Find max length
    max_len = max(s["length"] for s in processed_list)
    batch_size = len(processed_list)
    num_channels = processed_list[0]["input_ids"].shape[0]

    # Pre-allocate padded tensors
    input_ids = torch.full(
        (batch_size, num_channels, max_len), pad_token_id, dtype=torch.long
    )
    labels = torch.full(
        (batch_size, num_channels, max_len), -100, dtype=torch.long
    )
    audio_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    position_ids = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, sample in enumerate(processed_list):
        length = sample["length"]
        input_ids[i, :, :length] = sample["input_ids"]
        labels[i, :, :length] = sample["labels"]
        audio_mask[i, :length] = sample["audio_mask"]
        position_ids[i, :length] = torch.arange(length, dtype=torch.long)

    # 4D bidirectional attention mask for SDPA — matches upstream PaddingDataCollator.
    # mask[b, 0, i, j] = True if position j is a real (non-padding) token for sample b.
    # All query positions attend to all non-padding key positions.
    valid = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, sample in enumerate(processed_list):
        valid[i, :sample["length"]] = True
    attention_mask = valid[:, None, None, :].expand(batch_size, 1, max_len, max_len).contiguous()

    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "audio_mask": audio_mask.to(device),
        "position_ids": position_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


def collate_packed_samples(
    processed_list: List[Dict[str, Any]],
    batch_tokens: int,
    pad_token_id: int,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Concatenate multiple samples into one packed sequence with document_ids.

    Mirrors the upstream PackingDataCollator logic:
      1. Concatenate input_ids/labels/audio_mask along sequence dim
      2. Pad to batch_tokens length
      3. Build document_ids for block attention masking

    The model forward() uses document_ids to create a block attention mask
    that prevents cross-sample attention.

    Args:
        processed_list: List of outputs from OmniVoiceSampleProcessor.
        batch_tokens: Target sequence length to pad to.
        pad_token_id: Token ID used for padding input_ids.
        device: Target device.

    Returns:
        Dict with [1, C, L] / [1, L] tensors including document_ids.
    """
    if not processed_list:
        raise ValueError("Empty processed_list passed to collate_packed_samples")

    # Concatenate along sequence dimension
    input_ids = torch.cat(
        [s["input_ids"] for s in processed_list], dim=1
    )  # [C, Total_Len]
    labels_cat = torch.cat(
        [s["labels"] for s in processed_list], dim=1
    )  # [C, Total_Len]
    audio_mask = torch.cat(
        [s["audio_mask"] for s in processed_list], dim=0
    )  # [Total_Len]

    # Position IDs: reset per-sample (each sample starts at 0)
    position_ids = torch.cat(
        [torch.arange(s["length"], dtype=torch.long) for s in processed_list],
        dim=0,
    )  # [Total_Len]

    # Pad to target length
    pad_length = batch_tokens - input_ids.shape[1]
    if pad_length < 0:
        # Truncate if somehow too long (shouldn't happen with PackingIterableDataset)
        input_ids = input_ids[:, :batch_tokens]
        labels_cat = labels_cat[:, :batch_tokens]
        audio_mask = audio_mask[:batch_tokens]
        position_ids = position_ids[:batch_tokens]
        pad_length = 0

    input_ids = torch.nn.functional.pad(
        input_ids, pad=(0, pad_length), value=pad_token_id
    )
    labels_cat = torch.nn.functional.pad(labels_cat, pad=(0, pad_length), value=-100)
    audio_mask = torch.nn.functional.pad(
        audio_mask, pad=(0, pad_length), value=False
    )
    position_ids = torch.nn.functional.pad(
        position_ids, pad=(0, pad_length), value=0
    )

    # Build document_ids: each sample's tokens get a unique ID, padding gets -1
    document_ids_list = []
    for i, s in enumerate(processed_list):
        document_ids_list.append(torch.full((s["length"],), i, dtype=torch.int32))
    document_ids = torch.cat(document_ids_list, dim=0)
    document_ids = torch.nn.functional.pad(
        document_ids, pad=(0, pad_length), value=-1
    )

    return {
        "input_ids": input_ids.unsqueeze(0).to(device),      # [1, C, L]
        "labels": labels_cat.unsqueeze(0).to(device),         # [1, C, L]
        "audio_mask": audio_mask.unsqueeze(0).to(device),     # [1, L]
        "position_ids": position_ids.unsqueeze(0).to(device), # [1, L]
        "document_ids": document_ids.unsqueeze(0).to(device), # [1, L]
    }
