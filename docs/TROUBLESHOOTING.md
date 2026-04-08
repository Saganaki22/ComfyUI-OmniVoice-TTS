# Troubleshooting — ComfyUI-OmniVoice-TTS

---

## Table of Contents

- [START HERE — Only the Whisper node loads, all other nodes are missing](#start-here--only-the-whisper-node-loads-all-other-nodes-are-missing)
- [Upgrading from omnivoice 0.1.1 to 0.1.2](#upgrading-from-omnivoice-011-to-012)
- [Upgrading to omnivoice 0.1.3](#upgrading-to-omnivoice-013)
- [No module named pip](#no-module-named-pip)
- [PyTorch CUDA broken after install](#pytorch-cuda-broken-after-install)
- [Transformers version conflicts](#transformers-version-conflicts)
- [Model download fails (China / HuggingFace blocked)](#model-download-fails-china--huggingface-blocked)
- [Whisper re-downloads every run](#whisper-re-downloads-every-run)
- [CUDA out of memory](#cuda-out-of-memory)
- [FFmpeg error on Windows](#ffmpeg-error-on-windows)
- [Import errors after install](#import-errors-after-install)

---

## START HERE — Only the Whisper node loads, all other nodes are missing

This is the most common issue. If you can see `OmniVoice Whisper Loader` in ComfyUI but all other nodes (`OmniVoiceLongformTTS`, `OmniVoiceVoiceCloneTTS`, `OmniVoiceVoiceDesignTTS`, `OmniVoiceMultiSpeakerTTS`) are missing or red, follow this guide.

**Why it happens:** The other nodes only register if `omnivoice` imports cleanly at startup. Any import failure causes them all to be skipped. The Whisper loader always registers because it has no `omnivoice` dependency.

---

### Step 1 — Read the actual error from your startup log

Open your ComfyUI startup log and find the `[OmniVoice]` section. You will see a line like:

```
[OmniVoice] omnivoice import failed: <error message here>
```

Match your error message to one of the cases below.

---

### Case A — `cannot import name 'HiggsAudioV2TokenizerModel' from 'transformers'`

**What's happening:** Either your `transformers` is too old and genuinely doesn't have this class yet, or your `transformers` is new enough (`5.4+`) but is missing `soxr`, which it now requires internally. Both causes produce the exact same error message.

**Fix — run both of these, then restart ComfyUI:**

Windows (venv):
```bash
C:\Users\<you>\Documents\ComfyUI\venv\Scripts\pip install transformers --upgrade
C:\Users\<you>\Documents\ComfyUI\venv\Scripts\pip install soxr
```

Windows (portable / embedded Python):
```bash
C:\ComfyUI\python_embeded\python.exe -m pip install transformers --upgrade
C:\ComfyUI\python_embeded\python.exe -m pip install soxr
```

Linux / macOS:
```bash
path/to/ComfyUI/venv/bin/pip install transformers --upgrade
path/to/ComfyUI/venv/bin/pip install soxr
```

Using uv:
```bash
uv pip install transformers --upgrade
uv pip install soxr
```

Then **restart ComfyUI**. All nodes should appear.

> **Note:** `soxr` is included automatically in node version `0.2.7+` fresh installs. If you installed before this version, install it manually with the commands above.
> 
> **If upgrading transformers breaks other nodes:** You have a dependency conflict in your environment. See [Transformers version conflicts](#transformers-version-conflicts) below.

---

### Case B — `No module named 'omnivoice'`

The `omnivoice` package isn't installed. Run `install.py` manually:

Windows (venv):
```bash
C:\Users\<you>\Documents\ComfyUI\venv\Scripts\python install.py
```

Windows (portable / embedded Python):
```bash
C:\ComfyUI\python_embeded\python.exe install.py
```

Linux / macOS:
```bash
path/to/ComfyUI/venv/bin/python install.py
```

Then **restart ComfyUI**.

> ⚠️ Never run a plain `pip install omnivoice` without `--no-deps`. The omnivoice package pins `torch==2.8.*`, which will downgrade your PyTorch to CPU-only and break ComfyUI's GPU acceleration. `install.py` handles this safely for you.

---

### Case C — Any other error message

If your error message doesn't match A or B above:

1. Copy the full `[OmniVoice]` section from your startup log
2. Run `pip show omnivoice transformers torch` and copy the output
3. Open an issue at [github.com/Saganaki22/ComfyUI-OmniVoice-TTS/issues](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS/issues) with that info

---

## Upgrading from omnivoice 0.1.1 to 0.1.2

omnivoice `0.1.1` was released on April 2, 2026. Version `0.1.2` was released on April 4, 2026 and is the current recommended version.

**If you installed via ComfyUI Manager:** Run "Update" on the node in Manager, then restart ComfyUI.

**If you installed manually or want to update the underlying omnivoice package yourself:**

> ⚠️ **You must use `--no-deps`**. The omnivoice pip package declares `torch==2.8.*` and `transformers==5.3.0` as hard dependencies. Installing without `--no-deps` will downgrade your PyTorch to a CPU-only version and break ComfyUI's GPU acceleration.

Windows (venv):
```bash
C:\Users\<you>\Documents\ComfyUI\venv\Scripts\pip install omnivoice==0.1.2 --no-deps
```

Windows (portable / embedded Python):
```bash
C:\ComfyUI\python_embeded\python.exe -m pip install omnivoice==0.1.2 --no-deps
```

Linux / macOS:
```bash
path/to/ComfyUI/venv/bin/pip install omnivoice==0.1.2 --no-deps
```

Using uv:
```bash
uv pip install omnivoice==0.1.2 --no-deps
```

After upgrading, also install `soxr` if you have `transformers 5.4+`:
```bash
pip install soxr
```

---

## Upgrading to omnivoice 0.1.3

omnivoice `0.1.3` was released on April 7, 2026. If you installed the ComfyUI-OmniVoice-TTS node before this date, your `omnivoice` package is on `0.1.2` or earlier and you should upgrade to get the latest fixes and features.

**If you installed via ComfyUI Manager:** Run "Update" on the node in Manager, then restart ComfyUI.

**If you installed manually or want to update the underlying omnivoice package yourself:**

> ⚠️ **You must use `--no-deps`**. The omnivoice pip package declares `torch==2.8.*` as a hard dependency. Installing without `--no-deps` will downgrade your PyTorch to a CPU-only version and break ComfyUI's GPU acceleration.

Windows (venv):
```bash
C:\Users\<you>\Documents\ComfyUI\venv\Scripts\pip install omnivoice==0.1.3 --no-deps
```

Windows (portable / embedded Python):
```bash
C:\ComfyUI\python_embeded\python.exe -m pip install omnivoice==0.1.3 --no-deps
```

Linux / macOS:
```bash
path/to/ComfyUI/venv/bin/pip install omnivoice==0.1.3 --no-deps
```

Using uv:
```bash
uv pip install omnivoice==0.1.3 --no-deps
```

Then **restart ComfyUI**.

**Symptom:**
```
[OmniVoice] Failed to install omnivoice: ...python.exe: No module named pip
```

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

**Why it happens:** Another package (possibly `omnivoice` if installed without `--no-deps`, or another custom node) downgraded your PyTorch to a CPU-only version.

**Fix:** Check the [PyTorch Compatibility Matrix](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS/blob/main/pytorch_compatibility_matrix.md) for the restore command matching your CUDA version.

General restore for CUDA 12.8:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

For other CUDA versions visit: https://pytorch.org/get-started/locally/

> ⚠️ This is why we always install omnivoice with `--no-deps`. Never run a plain `pip install omnivoice` without it.

---

## Transformers version conflicts

**Symptom:** Other custom nodes break after installing OmniVoice, or errors mentioning `transformers` version incompatibility.

**Background:** The upstream `omnivoice` package pins `transformers==5.3.0`, but `install.py` deliberately ignores this pin via `--no-deps` to avoid breaking your existing setup. OmniVoice works with any recent `transformers` version, with one caveat: `transformers 5.4+` requires `soxr` (see [Case A](#case-a--cannot-import-name-higgsaudiov2tokenizermodel-from-transformers) above).

**Check your transformers version:**
```bash
pip show transformers
```

**If you need to downgrade** (only if you know it won't break your other nodes):
```bash
pip install transformers==5.3.0
```

**If you need to upgrade** (then also install soxr):
```bash
pip install transformers --upgrade
pip install soxr
```

---

## Model download fails (China / HuggingFace blocked)

Set the HuggingFace mirror before starting ComfyUI:

Linux / macOS:
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

Windows (Command Prompt):
```cmd
set HF_ENDPOINT=https://hf-mirror.com
```

Windows (PowerShell):
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

> **Google Colab users:** Your environment resets on disconnect. You will need to reinstall each session. Running with `--cpu` during install can cause PyTorch CUDA detection to fail — install packages with your GPU available if possible.

---

## Still stuck?

Open an issue at [github.com/Saganaki22/ComfyUI-OmniVoice-TTS/issues](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS/issues) and include:

1. Your full ComfyUI startup log (the section with `[OmniVoice]` lines)
2. Output of `pip show omnivoice transformers torch`
3. Your OS and Python version
4. How you installed (Manager / manual / git clone)
