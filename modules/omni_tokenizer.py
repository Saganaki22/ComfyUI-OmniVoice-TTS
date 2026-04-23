"""Audio tokenization utilities for OmniVoice training.

Handles loading the HiggsAudio v2 tokenizer and converting raw audio files
into 8-layer discrete tokens suitable for OmniVoice training.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("OmniVoice")

# HiggsAudio expects 24 kHz input
HIGGS_SAMPLE_RATE = 24_000
TOKENIZER_HF_PATH = "eustlb/higgs-audio-v2-tokenizer"


def _load_higgs_tokenizer(device: str = "cpu"):
    """Load the HiggsAudio v2 tokenizer and feature extractor.

    Returns:
        (tokenizer, feature_extractor) tuple
    """
    from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel

    logger.info("Loading HiggsAudio v2 tokenizer...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(TOKENIZER_HF_PATH)
    tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
        TOKENIZER_HF_PATH, device_map=device
    )
    logger.info("HiggsAudio v2 tokenizer loaded.")
    return tokenizer, feature_extractor


def tokenize_audio_file(
    audio_path: str,
    tokenizer,
    feature_extractor,
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """Tokenize a single audio file into 8-layer discrete tokens.

    Args:
        audio_path: Path to audio file (wav/mp3/flac).
        tokenizer: HiggsAudioV2TokenizerModel instance.
        feature_extractor: Corresponding feature extractor.
        device: Device to run tokenization on.

    Returns:
        numpy array of shape [8, T] with int16 token IDs,
        or None if tokenization fails.
    """
    try:
        from omnivoice.utils.audio import load_audio

        waveform = load_audio(audio_path, HIGGS_SAMPLE_RATE)
        waveform_tensor = torch.from_numpy(waveform)

        with torch.inference_mode():
            inputs = feature_extractor(
                raw_audio=waveform_tensor.squeeze(0).numpy(),
                sampling_rate=HIGGS_SAMPLE_RATE,
                return_tensors="pt",
            ).to(tokenizer.device)

            audio_codes = tokenizer.encode(inputs["input_values"]).audio_codes.squeeze(0)
            # Shape: [8, T] — verify
            assert audio_codes.dim() == 2 and audio_codes.size(0) == 8

            return audio_codes.to(torch.int16).cpu().numpy()

    except Exception as e:
        logger.warning(f"Failed to tokenize {audio_path}: {e}")
        return None


def tokenize_dataset(
    jsonl_path: str,
    cache_dir: Optional[str] = None,
    device: str = "cpu",
    progress_callback=None,
) -> str:
    """Tokenize all audio files in a JSONL dataset manifest.

    For each entry in the JSONL, tokenizes the audio and saves tokens as
    a .npy file. Creates a companion JSONL with token metadata.

    Args:
        jsonl_path: Path to input JSONL manifest.
            Each line: {"id": "...", "audio_path": "...", "text": "...", ...}
        cache_dir: Directory for cached token files. Defaults to
            jsonl_path parent + "_tokenized/".
        device: Device for tokenization ("cpu", "cuda", "xpu").
        progress_callback: Optional callable(current, total) for progress.

    Returns:
        Path to the output JSONL manifest (with audio_tokens_path fields added).
    """
    if cache_dir is None:
        cache_dir = str(Path(jsonl_path).parent / "_tokenized")

    os.makedirs(cache_dir, exist_ok=True)

    # Read input JSONL
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        raise ValueError(f"No entries found in {jsonl_path}")

    logger.info(f"Tokenizing {len(entries)} audio files...")

    # Load tokenizer
    higgs_tokenizer, feature_extractor = _load_higgs_tokenizer(device=device)

    output_path = os.path.join(cache_dir, "tokenized_manifest.jsonl")
    success_count = 0

    try:
        with open(output_path, "w", encoding="utf-8") as out_f:
            for i, entry in enumerate(entries):
                audio_path = entry.get("audio_path", "")
                sample_id = entry.get("id", f"sample_{i:06d}")

                if not audio_path or not os.path.exists(audio_path):
                    logger.warning(f"Skipping {sample_id}: audio not found at {audio_path}")
                    if progress_callback:
                        progress_callback(i + 1, len(entries))
                    continue

                tokens = tokenize_audio_file(audio_path, higgs_tokenizer, feature_extractor, device)
                if tokens is None:
                    if progress_callback:
                        progress_callback(i + 1, len(entries))
                    continue

                # Save tokenized audio as .npy
                npy_path = os.path.join(cache_dir, f"{sample_id}.npy")
                np.save(npy_path, tokens)

                # Build output entry
                out_entry = dict(entry)
                out_entry["audio_tokens_path"] = npy_path
                out_entry["num_tokens"] = tokens.shape[1]
                out_f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
                success_count += 1

                if progress_callback:
                    progress_callback(i + 1, len(entries))

                if (i + 1) % 10 == 0:
                    logger.info(f"Tokenized {i + 1}/{len(entries)} files ({success_count} successful)")

    finally:
        # Free tokenizer from memory
        del higgs_tokenizer, feature_extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

    logger.info(f"Tokenization complete: {success_count}/{len(entries)} files successful.")
    logger.info(f"Tokenized manifest saved to: {output_path}")

    return output_path
