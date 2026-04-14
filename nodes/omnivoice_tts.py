"""OmniVoice Longform TTS - Text-to-speech with smart chunking and optional voice cloning.

This node handles long text by automatically chunking at sentence boundaries.
Supports optional voice cloning from reference audio.
"""

import logging
import re
from typing import Tuple

import numpy as np
import torch

from .loader import (
    get_model_names,
    numpy_audio_to_comfy,
    comfy_audio_to_numpy,
    to_numpy_audio,
)
from .whisper_loader import find_local_whisper_model, load_whisper_pipeline
from .model_cache import (
    cancel_event,
    get_or_cache_whisper,
    get_or_load_model,
    offload_model_to_cpu,
    offload_whisper_to_cpu,
    unload_model,
    unload_whisper,
)

try:
    from comfy.utils import ProgressBar
    _PBAR = True
except ImportError:
    _PBAR = False

try:
    import comfy.model_management as mm
    _MM = True
except ImportError:
    _MM = False

logger = logging.getLogger("OmniVoice")

OMNIVOICE_SAMPLE_RATE = 24000


def _is_cjk(char: str) -> bool:
    """Check if a character belongs to a script that doesn't use spaces between words."""
    cp = ord(char)
    return (
        0x4E00 <= cp <= 0x9FFF       # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF     # CJK Extension A
        or 0x20000 <= cp <= 0x2A6DF   # CJK Extension B
        or 0xF900 <= cp <= 0xFAFF     # CJK Compat Ideographs
        or 0x3040 <= cp <= 0x309F     # Hiragana
        or 0x30A0 <= cp <= 0x30FF     # Katakana
        or 0xAC00 <= cp <= 0xD7AF     # Hangul Syllables
        or 0x0E00 <= cp <= 0x0E7F     # Thai
        or 0x0E80 <= cp <= 0x0EFF     # Lao
        or 0x1000 <= cp <= 0x109F     # Myanmar
        or 0x1780 <= cp <= 0x17FF     # Khmer
    )


def _chunk_by_characters(text: str, chars_per_chunk: int, sentence_end) -> list:
    """Chunk non-space-separated text by character count at sentence boundaries.

    Uses string slicing to preserve original spacing without corruption.
    """
    if len(text) <= chars_per_chunk:
        return [text]

    chunks = []
    pos = 0
    text_len = len(text)

    while pos < text_len:
        while pos < text_len and text[pos].isspace():
            pos += 1
        if pos >= text_len:
            break

        target_end = min(pos + chars_per_chunk, text_len)

        if target_end >= text_len:
            remaining = text[pos:].strip()
            if remaining:
                chunks.append(remaining)
            break

        segment = text[pos:target_end]
        matches = list(sentence_end.finditer(segment))

        if matches:
            last_match = matches[-1]
            split_at = pos + last_match.end()
            chunk = text[pos:split_at].strip()
            if chunk:
                chunks.append(chunk)
            pos = split_at
        else:
            chunk = text[pos:target_end].strip()
            if chunk:
                chunks.append(chunk)
            pos = target_end

    return chunks if chunks else [text]


def _smart_chunk_text(text: str, words_per_chunk: int = 100) -> list:
    """Split text into chunks at sentence boundaries, not cutting words.

    Works for space-separated languages (English, Arabic, Hindi, etc.) and
    non-space-separated languages (Chinese, Japanese, Korean, Thai, etc.).
    Sentence boundary detection supports Latin (.!?), CJK (。？！),
    Devanagari (।॥), Arabic (؟), Myanmar (။), and Tibetan (།).
    """
    if words_per_chunk <= 0:
        return [text]

    # Latin punctuation requires trailing space/end to avoid matching abbreviations.
    # Non-Latin punctuation is only used at true sentence boundaries, so no space needed.
    sentence_end = re.compile(
        r'(?:[.!?]+(?:\s|$)|[。？！\u0964\u0965\u061F\u104B\u0F0D]+)'
    )

    cjk_count = sum(1 for ch in text if _is_cjk(ch))
    alpha_count = sum(1 for ch in text if ch.isalpha() or _is_cjk(ch))
    is_cjk_dominant = alpha_count > 0 and cjk_count / alpha_count > 0.3

    if is_cjk_dominant:
        return _chunk_by_characters(text, words_per_chunk, sentence_end)

    # Space-separated language path
    words = text.split()
    if len(words) <= words_per_chunk:
        return [text]

    chunks = []
    current_chunk = []
    current_word_count = 0

    for word in words:
        current_chunk.append(word)
        current_word_count += 1

        if current_word_count >= words_per_chunk:
            chunk_text = ' '.join(current_chunk)
            matches = list(sentence_end.finditer(chunk_text))

            if matches:
                last_match = matches[-1]
                split_pos = last_match.end()
                final_chunk = chunk_text[:split_pos].strip()
                remaining = chunk_text[split_pos:].strip()

                if final_chunk:
                    chunks.append(final_chunk)

                current_chunk = remaining.split() if remaining else []
                current_word_count = len(current_chunk)
            else:
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())
                current_chunk = []
                current_word_count = 0

    if current_chunk:
        remaining = ' '.join(current_chunk).strip()
        if remaining:
            chunks.append(remaining)

    return chunks


class OmniVoiceLongformTTS:
    """OmniVoice Longform TTS with smart chunking and optional voice cloning."""

    @classmethod
    def INPUT_TYPES(cls):
        model_names = get_model_names()
        return {
            "required": {
                "model": (
                    model_names,
                    {
                        "tooltip": (
                            "OmniVoice model checkpoint. "
                            "Models are stored in ComfyUI/models/omnivoice/"
                        ),
                    },
                ),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello! This is a test of OmniVoice text to speech synthesis.",
                        "tooltip": (
                            "Text to synthesize. Supports inline non-verbal tags like "
                            "[laughter], [sigh], [sniff], [question-en], etc. "
                            "Long text will be automatically chunked at sentence boundaries."
                        ),
                    },
                ),
                "ref_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Transcript of the reference audio. "
                            "Leave empty to auto-transcribe with Whisper. "
                            "Only used if 'ref_audio' is connected."
                        ),
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 32,
                        "min": 4,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Number of diffusion steps. "
                            "16 = faster, 32 = balanced, 64 = best quality."
                        ),
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Classifier-free guidance scale. Higher = more aligned with text.",
                    },
                ),
                "t_shift": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Time-step shift for noise schedule. Smaller = emphasis on earlier steps.",
                    },
                ),
                "speed": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": (
                            "Speaking speed factor. "
                            ">1.0 = faster, <1.0 = slower."
                        ),
                    },
                ),
                "duration": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.5,
                        "tooltip": (
                            "Fixed output duration in seconds. "
                            "0 = automatic (uses speed). Overrides speed if set."
                        ),
                    },
                ),
                "device": (
                    ["auto", "cuda", "cpu", "mps"],
                    {
                        "default": "auto",
                        "tooltip": "Compute device. 'auto' picks CUDA > MPS > CPU.",
                    },
                ),
                "dtype": (
                    ["auto", "bf16", "fp16", "fp32"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Model precision. 'auto' picks bf16 for CUDA (Ampere+), "
                            "fp16 for older CUDA/MPS, fp32 for CPU."
                        ),
                    },
                ),
                "attention": (
                    ["auto", "eager", "sage_attention"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Attention implementation. "
                            "'auto' uses model default (eager). "
                            "'sage_attention' uses SageAttention CUDA kernels (requires SM80+ GPU)."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "tooltip": "Random seed. 0 = random.",
                    },
                ),
                "words_per_chunk": (
                    "INT",
                    {
                        "default": 100,
                        "min": 0,
                        "max": 500,
                        "step": 10,
                        "tooltip": (
                            "Words per chunk for long text. 0 = no chunking. "
                            "Chunks split at sentence boundaries, not mid-word."
                        ),
                    },
                ),
                "position_temperature": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.5,
                        "tooltip": "Temperature for mask-position selection. 0 = greedy, higher = more random.",
                    },
                ),
                "class_temperature": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Temperature for token sampling. 0 = greedy, higher = more random.",
                    },
                ),
                "layer_penalty_factor": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.5,
                        "tooltip": "Penalty on deeper codebook layers, encouraging lower layers to unmask first.",
                    },
                ),
                "denoise": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Prepend denoise token to input for cleaner output.",
                    },
                ),
                "preprocess_prompt": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Preprocess voice-clone prompt audio (remove silences, add punctuation).",
                    },
                ),
                "postprocess_output": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Post-process generated audio (remove long silences).",
                    },
                ),
                "keep_model_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Keep model loaded between runs. "
                            "Model is automatically offloaded to CPU after generation to free VRAM, "
                            "then resumed to GPU on the next run."
                        ),
                    },
                ),
                "instruct": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": (
                            "Dialect/style instruction from the model's supported values. "
                            "English: american/british/australian/canadian/chinese/indian/"
                            "japanese/korean/portuguese/russian accent, "
                            "male/female, child/young adult/teenager/middle-aged/elderly, "
                            "very low pitch/low pitch/moderate pitch/high pitch/very high pitch, whisper. "
                            "Chinese: 四川话/东北话/陕西话/河南话/云南话/贵州话/甘肃话/"
                            "宁夏话/石家庄话/济南话/青岛话/桂林话, "
                            "男/女, 儿童/少年/青年/中年/老年, "
                            "极低音调/低音调/中音调/高音调/极高音调, 耳语. "
                            "Use comma-separated (English) or full-width comma (Chinese). "
                            "Applied to every chunk. Leave empty for default."
                        ),
                    },
                ),
            },
            "optional": {
                "ref_audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Optional reference audio for voice cloning. "
                            "3-15 seconds of clear speech works best. "
                            "If not connected, uses automatic voice selection."
                        ),
                    },
                ),
                "whisper_model": (
                    "WHISPER_ASR",
                    {
                        "tooltip": (
                            "Optional pre-loaded Whisper ASR model for auto-transcription. "
                            "Connect from OmniVoice Whisper Loader to avoid re-downloading."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "OmniVoice"
    DESCRIPTION = (
        "OmniVoice Longform TTS - Zero-shot multilingual TTS with smart chunking. "
        "Handles long text by splitting at sentence boundaries. "
        "Optional voice cloning from reference audio."
    )

    def generate(
        self,
        model: str,
        text: str,
        ref_text: str,
        steps: int,
        guidance_scale: float,
        t_shift: float,
        speed: float,
        duration: float,
        device: str,
        dtype: str,
        attention: str,
        seed: int,
        words_per_chunk: int,
        position_temperature: float,
        class_temperature: float,
        layer_penalty_factor: float,
        denoise: bool,
        preprocess_prompt: bool,
        postprocess_output: bool,
        keep_model_loaded: bool,
        instruct: str,
        ref_audio: dict = None,
        whisper_model: dict = None,
    ) -> Tuple[dict]:
        cancel_event.clear()
        self._check_interrupt()

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        omnivoice_model, _ = get_or_load_model(
            model, device, dtype, attention, keep_model_loaded
        )

        # Set random seed early so Whisper transcription is also seeded
        actual_seed = seed if seed != 0 else torch.randint(0, 2**31, (1,)).item()
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(actual_seed)

        use_voice_clone = ref_audio is not None

        ref_audio_tensor = None
        if use_voice_clone:
            logger.info("Processing reference audio for voice cloning...")
            ref_audio_np, _ = comfy_audio_to_numpy(ref_audio, target_sr=OMNIVOICE_SAMPLE_RATE)
            ref_audio_tensor = torch.from_numpy(ref_audio_np).float()

            ref_duration = len(ref_audio_np) / OMNIVOICE_SAMPLE_RATE
            if ref_duration < 1:
                logger.warning(
                    f"Reference audio is only {ref_duration:.1f}s — "
                    "recommend 3-15s for best quality."
                )
            elif ref_duration > 30:
                logger.warning(
                    f"Reference audio is {ref_duration:.1f}s — "
                    "longer than recommended 15s may cause issues."
                )

            if not ref_text.strip() and whisper_model is not None:
                whisper_pipe = get_or_cache_whisper(whisper_model, model, device, dtype)
                if whisper_pipe is not None:
                    logger.info("Using pre-loaded Whisper ASR for voice transcription")
                    omnivoice_model._asr_pipe = whisper_pipe
            elif not ref_text.strip():
                # Check for locally downloaded Whisper before letting OmniVoice download
                local_name = find_local_whisper_model()
                if local_name is not None:
                    logger.info(
                        f"No ref_text — auto-detected local Whisper "
                        f"({local_name}) for transcription"
                    )
                    try:
                        pipe = load_whisper_pipeline(local_name, device, dtype)
                        get_or_cache_whisper(
                            {"pipeline": pipe, "model_name": local_name},
                            model, device, dtype,
                        )
                        omnivoice_model._asr_pipe = pipe
                    except Exception as e:
                        logger.warning(f"Failed to load local Whisper: {e}")
                        logger.info("No ref_text — Whisper will auto-transcribe (downloads if not cached)")
                else:
                    logger.info("No ref_text — Whisper will auto-transcribe (downloads if not cached)")

        chunks = _smart_chunk_text(text, words_per_chunk)

        if len(chunks) > 1:
            logger.info(f"Long text detected — splitting into {len(chunks)} chunks at sentence boundaries")

        total_chunks = len(chunks)
        pbar = ProgressBar(total_chunks + 1) if _PBAR else None

        preview = text[:80] + "..." if len(text) > 80 else text
        mode = "voice clone" if use_voice_clone else "auto voice"
        logger.info(f"Longform TTS ({mode}): {preview}")

        if pbar:
            pbar.update_absolute(1, total_chunks + 1)

        self._check_interrupt()

        audio_chunks = []
        result = None
        auto_ref_audio_tensor = None
        first_chunk_text = ""

        try:
            for chunk_idx, chunk_text in enumerate(chunks):
                self._check_interrupt()

                if len(chunks) > 1:
                    logger.info(f"  Chunk {chunk_idx + 1}/{len(chunks)}: {chunk_text[:50]}{'...' if len(chunk_text) > 50 else ''}")

                gen_kwargs = {
                    "text": chunk_text,
                    "num_step": steps,
                    "guidance_scale": guidance_scale,
                    "t_shift": t_shift,
                    "speed": speed,
                    "position_temperature": position_temperature,
                    "class_temperature": class_temperature,
                    "layer_penalty_factor": layer_penalty_factor,
                    "denoise": denoise,
                    "preprocess_prompt": preprocess_prompt,
                    "postprocess_output": postprocess_output,
                }

                # Auto-voice consistency: use first chunk's output as
                # reference for all subsequent chunks to keep the same voice.
                if not use_voice_clone and chunk_idx > 0 and auto_ref_audio_tensor is not None:
                    gen_kwargs["ref_audio"] = (auto_ref_audio_tensor, OMNIVOICE_SAMPLE_RATE)
                    gen_kwargs["ref_text"] = first_chunk_text
                elif use_voice_clone:
                    gen_kwargs["ref_audio"] = (ref_audio_tensor, OMNIVOICE_SAMPLE_RATE)
                    if ref_text.strip():
                        gen_kwargs["ref_text"] = ref_text.strip()

                if instruct and instruct.strip():
                    gen_kwargs["instruct"] = instruct.strip()

                if duration > 0:
                    gen_kwargs["duration"] = duration

                with torch.no_grad():
                    try:
                        audio_list = omnivoice_model.generate(**gen_kwargs)
                    except ValueError as e:
                        if "instruct" in str(e).lower() or "unsupported" in str(e).lower():
                            raise RuntimeError(
                                f"Invalid instruct value '{gen_kwargs.get('instruct')}'. "
                                f"The model only accepts specific values. Original error:\n{e}"
                            ) from e
                        raise

                audio_np = to_numpy_audio(audio_list[0])
                audio_chunks.append(audio_np)

                # Capture first chunk audio as reference for auto-voice consistency
                if not use_voice_clone and chunk_idx == 0 and len(chunks) > 1:
                    max_ref_samples = 25 * OMNIVOICE_SAMPLE_RATE
                    if len(audio_np) > max_ref_samples:
                        auto_ref_audio_np = audio_np[:max_ref_samples]
                        logger.info("  Cropped auto-reference audio to 25s for voice consistency")
                    else:
                        auto_ref_audio_np = audio_np
                    auto_ref_audio_tensor = torch.from_numpy(auto_ref_audio_np).float()
                    first_chunk_text = chunk_text
                    ref_dur = len(auto_ref_audio_np) / OMNIVOICE_SAMPLE_RATE
                    logger.info(f"  Using first chunk ({ref_dur:.1f}s) as voice reference for remaining chunks")

                if pbar:
                    pbar.update_absolute(chunk_idx + 2, total_chunks + 1)

            if len(audio_chunks) == 1:
                audio_out = audio_chunks[0]
            else:
                audio_out = np.concatenate(audio_chunks, axis=0)

            result = numpy_audio_to_comfy(audio_out, OMNIVOICE_SAMPLE_RATE)

            logger.info(
                f"Generated {len(audio_out) / OMNIVOICE_SAMPLE_RATE:.2f}s of audio "
                f"at {OMNIVOICE_SAMPLE_RATE}Hz"
            )

        finally:
            if not keep_model_loaded:
                unload_model()
                unload_whisper()
            else:
                offload_model_to_cpu()
                offload_whisper_to_cpu()

        if result is None:
            raise RuntimeError("Generation failed — see logs above.")
        return (result,)

    def _check_interrupt(self):
        """Check if processing was interrupted."""
        if _MM:
            try:
                mm.throw_exception_if_processing_interrupted()
            except Exception:
                cancel_event.set()
                raise
