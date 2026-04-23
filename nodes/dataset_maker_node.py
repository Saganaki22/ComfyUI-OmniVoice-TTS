"""OmniVoice Dataset Maker node.

Scans a directory of .wav + .txt file pairs and creates a JSONL manifest
for training. Each .wav must have a matching .txt file with the same base name.

Output format (one JSON object per line):
    {"id": "basename", "audio_path": "/path/to/file.wav", "text": "transcript", "language_id": "en"}
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("OmniVoice")


def create_jsonl_dataset(audio_directory: str, output_filename: str, language_id: str = "en") -> str:
    """Create a JSONL manifest from paired .wav + .txt files.

    Args:
        audio_directory: Directory containing .wav and .txt file pairs.
        output_filename: Name of the output JSONL file (saved in same directory).
        language_id: Language code to tag each entry with.

    Returns:
        Absolute path to the created JSONL file.

    Raises:
        ValueError: If directory doesn't exist or no valid pairs found.
    """
    audio_dir = Path(audio_directory).resolve()
    if not audio_dir.is_dir():
        raise ValueError(f"Directory not found: {audio_dir}")

    # Collect .wav files (also accept .mp3, .flac)
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = sorted(
        f for f in audio_dir.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    )

    if not audio_files:
        raise ValueError(f"No audio files found in {audio_dir}")

    output_path = audio_dir / output_filename
    valid_count = 0
    skipped = []

    with open(output_path, "w", encoding="utf-8") as out_f:
        for audio_file in audio_files:
            # Look for matching .txt file
            txt_file = audio_file.with_suffix(".txt")
            if not txt_file.is_file():
                skipped.append(f"{audio_file.name} (no matching .txt)")
                continue

            try:
                text = txt_file.read_text(encoding="utf-8").strip()
            except Exception as e:
                skipped.append(f"{audio_file.name} (txt read error: {e})")
                continue

            if not text:
                skipped.append(f"{audio_file.name} (empty transcript)")
                continue

            entry = {
                "id": audio_file.stem,
                "audio_path": str(audio_file),
                "text": text,
                "language_id": language_id,
            }
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            valid_count += 1

    if skipped:
        logger.warning(
            f"Skipped {len(skipped)} files: {skipped[:5]}"
            + ("..." if len(skipped) > 5 else "")
        )

    if valid_count == 0:
        raise ValueError(
            f"No valid audio+text pairs found in {audio_dir}. "
            f"Ensure each audio file has a matching .txt file with the same name."
        )

    logger.info(f"Created dataset: {valid_count} samples -> {output_path}")
    return str(output_path)


class OmniVoice_DatasetMaker:
    """Creates a JSONL training dataset from a folder of audio + text pairs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_directory": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to directory containing .wav/.mp3/.flac and .txt file pairs. "
                        "Each audio file must have a matching .txt with the same base name.",
                    },
                ),
                "output_filename": (
                    "STRING",
                    {
                        "default": "train.jsonl",
                        "tooltip": "Name of the output JSONL manifest file.",
                    },
                ),
                "language_id": (
                    "STRING",
                    {
                        "default": "en",
                        "tooltip": "Language code for the dataset (e.g. 'en', 'zh', 'ja', 'de').",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dataset_path",)
    FUNCTION = "execute"
    CATEGORY = "audio/tts/training"
    OUTPUT_NODE = True
    DESCRIPTION = "Create a JSONL training manifest from a folder of .wav + .txt file pairs."

    def execute(self, audio_directory: str, output_filename: str, language_id: str):
        dataset_path = create_jsonl_dataset(audio_directory, output_filename, language_id)
        return (dataset_path,)
