# Troubleshooting — ComfyUI-OmniVoice-TTS

---

## Table of Contents

- [Nodes missing after install (most common)](#nodes-missing-after-install-most-common)
- [Upgrading from omnivoice 0.1.1 to 0.1.2](#upgrading-from-omnivoice-011-to-012)
- [soxr missing error](#soxr-missing-error)
- [omnivoice import failed / HiggsAudioV2TokenizerModel](#omnivoice-import-failed--higgsaudiov2tokenizermodel)
- [No module named pip](#no-module-named-pip)
- [PyTorch CUDA broken after install](#pytorch-cuda-broken-after-install)
- [Transformers version conflicts](#transformers-version-conflicts)
- [Model download fails (China / HuggingFace blocked)](#model-download-fails-china--huggingface-blocked)
- [Whisper re-downloads every run](#whisper-re-downloads-every-run)
- [CUDA out of memory](#cuda-out-of-memory)
- [FFmpeg error on Windows](#ffmpeg-error-on-windows)
- [Import errors after install](#import-errors-after-install)

---

## Nodes missing after install (most common)

**Symptom:** ComfyUI loads the node pack but only `OmniVoice Whisper Loader` appears. All other nodes (`OmniVoiceLongformTTS`, `OmniVoiceVoiceCloneTTS`, `OmniVoiceVoiceDesignTTS`, `OmniVoiceMultiSpeakerTTS`) are missing.

**Why this happens:** The other nodes only register if `omnivoice` imports cleanly. If `omnivoice` fails to import for any reason, `__init__.py` skips them entirely. The Whisper loader always registers because it has no dependency on `omnivoice`.

**Check your ComfyUI startup log for a line like:**
```
[OmniVoice] omnivoice import failed: ...
```

The error message after the colon is the actual cause. The most common causes and their fixes are listed below.

---

## Upgrading from omnivoice 0.1.1 to 0.1.2

omnivoice `0.1.1` was released on April 2, 2026. Version `0.1.2` was released on April 4, 2026 and is the current recommended version.

**If you installed via ComfyUI Manager:** Run "Update" on the node in Manager, then restart ComfyUI.

**If you installed manually or want to update the underlying omnivoice package yourself:**

> ⚠️ **You must use `--no-deps`**. The omnivoice pip package declares `torch==2.8.*` and `transformers==5.3.0` as hard dependencies. Installing without `--no-deps` will downgrade your PyTorch to a CPU-only version and break ComfyUI's GPU acceleration.

### Windows (venv)
```bash
C:\Users\<you>\Documents\ComfyUI\venv\Scripts\pip install omnivoice==0.1.2 --no-deps
```

### Windows (portable / embedded Python)
```bash
C:\ComfyUI\python_embeded\python.exe -m pip install omnivoice==0.1.2 --no-deps
```

### Linux / macOS
```bash
path/to/ComfyUI/venv/bin/pip install omnivoice==0.1.2 --no-deps
```

### Using uv (if your environment uses uv)
```bash
uv pip install omnivoice==0.1.2 --no-deps
```

After upgrading, also install `soxr` if you have `transformers 5.4+` — see the next section.

---

## soxr missing error

**Symptom:** ComfyUI log shows:
```
ModuleNotFoundError: No module named 'soxr'
```
or
```
[OmniVoice] omnivoice import failed: Could not import module 'HiggsAudioV2TokenizerModel'. Are this object's requirements defined correctly?
```

**Why this happens:** `transformers 5.4+` added `soxr` as a required import inside `audio_utils.py` for the HiggsAudio tokenizer path. When `omnivoice` tries to load `HiggsAudioV2TokenizerModel` from transformers, transformers internally imports `audio_utils`, which imports `soxr` — and if `soxr` isn't installed the whole chain crashes.

The confusing `HiggsAudioV2TokenizerModel` error message is a red herring — it's a downstream consequence of the `soxr` import failure, not a missing model class.

This only affects users on `transformers 5.4+`. Users on `transformers 5.3.0` do not hit this issue.

**Fix:**

### Windows (venv)
```bash
C:\Users\<you>\Documents\ComfyUI\venv\Scripts\pip install soxr
```

### Windows (portable / embedded Python)
```bash
C:\ComfyUI\python_embeded\python.exe -m pip install soxr
```

### Linux / macOS
```bash
path/to/ComfyUI/venv/bin/pip install soxr
```

### Using uv
```bash
uv pip install soxr
```

Then **restart ComfyUI**. All nodes should appear.

> **Note for fresh installs:** As of node version `0.2.7`, `soxr` is included in `install.py` and will be installed automatically. If you installed before this version, install it manually using the commands above.

---

## omnivoice import failed / HiggsAudioV2TokenizerModel

See the [soxr missing error](#soxr-missing-error) section above — this is the same issue. The `HiggsAudioV2TokenizerModel` error is caused by `soxr` not being installed.

If installing `soxr` does not fix it, check your full ComfyUI log for any other error lines under `[OmniVoice]` and report them in the [issues tracker](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS/issues).

---

## No module named pip

**Symptom:**
```
[OmniVoice] Failed to install omnivoice: ...\python.exe: No module named pip
```

**Why this happens:** Your Python environment is missing `pip` entirely, which is unusual but can happen with certain venv setups or `uv`-managed environments.

**Fix — bootstrap pip first:**
```bash
python -m ensurepip --upgrade
```

Then retry the install commands. If you're using `uv`, use `uv pip install` instead of `pip install` for all commands — `uv` does not require pip to be installed.

---

## PyTorch CUDA broken after install

**Symptom:** ComfyUI was using your GPU before, now it runs on CPU only. Or you see:
```
UserWarning: CUDA initialization: CUDA unknown error
```

**Why this happens:** Another package (possibly `omnivoice` if installed without `--no-deps`, or another custom node) downgraded your PyTorch to a CPU-only version.

**Fix:** Check the [PyTorch Compatibility Matrix](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS/blob/main/pytorch_compatibility_matrix.md) for the restore command that matches your CUDA version.

A general restore for CUDA 12.8:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

For other CUDA versions visit: https://pytorch.org/get-started/locally/

> ⚠️ This is why we always install omnivoice with `--no-deps`. Never run a plain `pip install omnivoice` without it.

---

## Transformers version conflicts

**Symptom:** Errors mentioning transformers version, or other custom nodes breaking after installing OmniVoice.

**Background:** The upstream `omnivoice` package pins `transformers==5.3.0`. Our `install.py` deliberately ignores this pin (via `--no-deps`) to avoid breaking your existing setup. This means OmniVoice will work with newer transformers versions, with one caveat: `transformers 5.4+` requires `soxr` (see above).

**If you need to check your transformers version:**
```bash
pip show transformers
```

**If you need to downgrade transformers** (only do this if you know it won't break your other nodes):
```bash
pip install transformers==5.3.0
```

**If you need to upgrade transformers** (and then also install soxr):
```bash
pip install transformers --upgrade
pip install soxr
```

---

## Model download fails (China / HuggingFace blocked)

Set the HuggingFace mirror before starting ComfyUI:

**Linux / macOS:**
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

**Windows (Command Prompt):**
```cmd
set HF_ENDPOINT=https://hf-mirror.com
```

**Windows (PowerShell):**
```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

Add this to your ComfyUI launch script so it persists between restarts.

---

## Whisper re-downloads every run

Connect the `OmniVoice Whisper Loader` node to the `whisper_model` input on your TTS node. This caches the Whisper model in memory so it doesn't reload from disk or re-download on each run.

---

## CUDA out of memory

Try the following in order:

1. Set `keep_model_loaded = False` — unloads the model between runs
2. Switch `dtype` to `bf16` or `fp16` — halves VRAM usage (~4-6GB instead of ~8-12GB)
3. Use `OmniVoice-bf16` model instead of `OmniVoice` — 2GB vs 4GB on disk
4. Set `device = cpu` — slow but works on any system

---

## FFmpeg error on Windows

**Symptom:** Audio save node fails with an FFmpeg-related error.

**Fix:** Add your FFmpeg `bin/` folder to `PATH` in your ComfyUI launch `.bat` file:
```bat
set PATH=C:\path\to\ffmpeg\bin;%PATH%
```

Or use a WAV audio save node instead of MP3/AAC formats, which don't require FFmpeg.

---

## Import errors after install

**Symptom:** Generic Python import errors on ComfyUI startup after a fresh install or update.

**Fix:** Restart ComfyUI completely. Python modules are only loaded once per process — a full restart is required for any newly installed packages to take effect. Do not use the "Reload" button in the ComfyUI UI as this does not reload Python modules.

---

## Still stuck?

Open an issue at [github.com/Saganaki22/ComfyUI-OmniVoice-TTS/issues](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS/issues) and include:

1. Your full ComfyUI startup log (the section with `[OmniVoice]` lines)
2. Output of `pip show omnivoice transformers torch`
3. Your OS and Python version
4. How you installed (Manager / manual / git clone)
