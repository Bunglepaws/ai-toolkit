# Ostris AI Toolkit

AI Toolkit is an easy to use all in one training suite for diffusion models. I try to support all the latest models on consumer grade hardware. Image and video models. It can be run as a GUI or CLI. It is designed to be easy to use but still have every feature imaginable. Free and open source.

---

## Fork Additions (MarcBate)

This is a personal fork of [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) with the following additions on top of upstream.

### Training / Backend

- **Save before pause** — checkpoint is always saved before stopping; `saveAndPauseJob()` sets `save` + `stop` atomically
- **Save and Stop Queue** — saves checkpoint, stops job, and re-queues it for later resumption
- **On-demand save/sample** — trigger a save or sample generation mid-training from the UI without stopping
- **Stop during quantization** — `JobStoppedException(BaseException)` propagates through quantization loops via `maybe_stop()` hooks so Stop/Pause works even during slow model loading
- **Abort active sampling** — "Return to Training" button aborts in-progress sample generation and resumes training immediately (`SampleAbortedException` + `stop_sample` DB flag)
- **Persistent quantized model cache** — `run_ui.py` keeps the quantized model in system RAM between consecutive same-model queued jobs; only LoRA weights reload, skipping the multi-minute re-quantization (works for LTX-2.3, Qwen, Flux, etc.)
- **Automagic v3 backward compat** — resumes from checkpoints saved by any prior v3 variant (per-row lr tensors, missing `dir_ema`/`prev_sign` keys all handled)
- **LightX2V for WAN 2.2** — 4-step distilled samples (~40s vs ~6 min); PEFT adapter reuse fix
- **LTX-2.3 distilled LoRA** — 8-step samples instead of 30
- **Gemma API for LTX-2.3** — use free Gemma API instead of loading 12B text encoder locally
- **Sampling LoRA (Krea 2 & Qwen Image)** — apply a LoRA only during sample generation, not training; useful for filter-bypass or style LoRAs (see [Krea 2 Training](#krea-2-training)); fixed crash on quantized Qwen Image models where `QModuleMixin._load_from_state_dict` would raise `KeyError` on `_data` keys belonging to unrelated modules
- **Corrupt/truncated JSON captions** — graceful fallback with warning instead of crashing the job
- **Optimizer archiving** — option to archive optimizer state on each save
- **AceStep 1.5 XL audio LM (`audio_lm_path`)** — set `audio_lm_path` in your model config to a Qwen3 ACE15 safetensors file (e.g. `qwen_4b_ace15.safetensors`) to enable proper `lm_hints` context generation at sample time. Without this the DiT uses silence context and output quality is poor. The FSQ quantizer and AudioTokenDetokenizer are extracted automatically from the AIO base model file. Supports the XL AIO format (`ostris/ace_step_1.5_ComfyUI_files`); non-XL AIO untested.
- **Combine Datasets for Bucketing** — `combine_datasets: true` in `train` config (or checkbox in UI when 2+ datasets) merges all dataset file lists (after `num_repeats` expansion) into one pool and runs bucket assignment once globally, identical to having all images in a single folder; each item retains its own per-dataset settings (caption dropout, trigger words, etc.); requires all datasets to share the same `resolution`, `buckets`, and `square_crop` settings.
- **Video-only files auto-fixed** — dataset videos with no audio stream at all (common with some video generators) used to crash the whole job on `torchaudio.load()`; a silent stereo AAC track is now muxed in automatically the first time the file is loaded, in place, via ffmpeg
- **`torch.compile` + CPU/GPU layer-offloading stream fix** — the offloading autograd functions (`_BouncingLinearFn`, `_BouncingConv2dFn`) manage raw CUDA streams/events directly; newer PyTorch Dynamo's stream tracing mis-codegenned `torch.ops.streams.record_event` on them under `compile: true` + `low_vram: true`, crashing with `RuntimeError: expected event to be a torch.Event object`. Their `forward`/`backward` are now marked `@torch._dynamo.disable` so Dynamo treats them as an opaque call instead of tracing into them
- **`block_compile` + torchao guard** — block-level `torch.compile` is automatically disabled (with a warning) when the model is torchao-quantized, avoiding an infinite-recursion crash in `torchao.utils._dispatch__torch_function__` under PyTorch 2.9+'s AOT autograd path

### UI — Queue & Job Management

- **Drag-to-reorder queue** — drag jobs to reorder training queue; move-to-top button
- **Queue filter** — filter jobs list by name, model path, or job ref with AND/OR/quoted search
- **Save and Stop Queue** button in stop modal — saves checkpoint and re-queues the job
- **Return to Training** button — aborts current sample batch and resumes training
- **Resume From Checkpoint** — gear-menu action on any job that can be (re)started; lists saved checkpoint/optimizer pairs, rolls back to the selected step by deleting newer safetensors and optimizer archives, restores the matching optimizer state, and prunes `loss_log.db` past that step so the graph is accurate immediately; handles WAN 2.2's `_high_noise`/`_low_noise` split checkpoints
- **Checkpoint delete cleans up optimizer archive** — deleting a `.safetensors` checkpoint from the Checkpoints panel also removes its matching `optimizer_{step}.pt` archive (previously left 4GB+ orphan files behind)

### UI — Settings

- **AI Config Check settings** — API URL, key, model, and web-search toggle stored in DB and editable from the Settings page; no server file access or `.env.local` required; works with any OpenAI-compatible endpoint (Claude, Ollama, etc.)

### UI — Model Config

- **Compile options** — Compile Model checkbox exposes Block Compile, Compile Mode (default/max-autotune/fastest), and Full Graph toggle
- **Cache quantized model** — skip re-quantization on subsequent runs
- **Negative Prompt field** — exposed in job config UI
- **Automagic v3 in optimizer dropdown** — was missing from upstream UI

### UI — Loss Graph

- **Settings persistence** — display settings (smooth/raw/log/clip) saved per job across sessions
- **Training time grid** — expandable panel below the loss graph shows per-session timing broken into three components: startup (model load), sampling (inference), and pure training time, plus a total column. Subtotals row at top with grand total. Columns: Start, End (time-only when same calendar day), Start Step, Startup, Sampling, Training, Total.
  - **Startup time persisted** — `startup_seconds` is written to `training_sessions` in `loss_log.db` once the first training step completes; backfilled from `logs/N_log.txt` files for up to 30 days of history via `scripts/backfill_startup_times.py`
  - **Sampling time from `sampling_periods`** — already tracked; now surfaced per-session instead of just subtracted from training
  - **Pure training time** — computed as step-span minus sampling time so the three components sum correctly to wall-clock elapsed
  - **Manual override** — click any Training cell to type a corrected value (`2h 15m`, `135m`, etc.); Enter/blur saves via `PUT /api/jobs/[jobID]/sessions/[sessionId]`; overridden cells shown in blue with `*`; blank input clears the override; stored in `training_seconds_override` column

### UI — Samples

- **Placeholder grid cells** — placeholders for unsampled slots keep grid aligned
- **Prompt from file metadata** — reads prompt from PNG/MP4 metadata; falls back to job config
- **On-demand sampling when idle** — generate samples for a completed job
- **Step counter on Samples tab** — "Step X of Y" progress shown left of the Generate Samples Now button, updating live
- **Sample button blocked during startup** — Generate Samples disabled while loading model, quantizing, encoding dataset, etc.; only active once in the training loop
- **Toolbar sample button hidden on Samples tab** — avoids duplicate camera buttons when already on the Samples page

### UI — Datasets / Captions

- **Find & replace honors caption ext** — find/replace works correctly for JSON captions and respects the selected caption extension type
- **Find & replace in JSON** — updates the `caption` field inside the JSON structure, preserving other fields
- **Find & replace captions** — bulk find-and-replace with AND/OR/quoted search
- **Caption filtering** — filter dataset images by caption content

### UI — Training Alerts

Real-time anomaly detection that writes to the DB and surfaces in the UI without ever pausing training.

- **Loss spike detection** — rolling 50-step deque; flags when current loss > 3× average and > 0.4 absolute floor; 10-step debounce prevents alert storms during sustained divergence
- **White-noise sample detection** — compares JPEG/PNG file sizes of new samples against the step-0 baseline; flags when current batch avg exceeds baseline by 1.8× (empirically confirmed signal for mode collapse / LR divergence)
- **OOM crash detection** — `on_error()` catches CUDA out-of-memory errors, collects VRAM stats via `nvidia-smi`, and writes an `oom` alert type with memory details
- **Dataset stats persistence** — image count and bucket distribution written to DB after each latent-caching phase so the AI Config Check has context without a running trainer
- **Safe checkpoint snapshot** — on any alert, the most recent checkpoint's `.safetensors` + `.pt` files are copied to `{save_root}/_safe_snapshots/{reason}_step_{N}/` with a `README.txt` before scheduled cleanup can remove them; WAN 2.2 copies both high-noise and low-noise files
- **Amber alert chip** — job card shows ⚠ N when alerts exist; clicking expands an inline panel with timestamped entries (step, type, message); Clear button resets the list
- **5-second toast** — job detail page fires an amber toast within one poll cycle when a new alert arrives
- **OOM auto-trigger** — if Check Config is configured, an OOM alert automatically opens the AI reviewer with `autoRun=true` and the OOM message shown as a red context banner at the top of results

### UI — AI Config Check

"Check Config ✦" button on every job form (new and edit). Assembles training context and asks an LLM for structured, source-cited findings.

**What it sends to the LLM:**
- Full job config JSON
- Dataset stats (image count, bucket distribution) from DB
- Last 200 steps of loss curve from `loss_log.db`
- Real-time GPU stats (VRAM used/free, utilization, temperature, power) via `nvidia-smi`
- System RAM stats
- Optional: recent sample images/frames and training dataset images (vision analysis)

**Findings format:** `{field, current_value, suggested_value, reason, confidence, references, applyable, severity}`

- **Confidence tiers** — High (vendor docs / maintainer posts), Medium (Reddit/Discord/forum), Low (LLM reasoning from analogous models); Low-confidence Apply requires an explicit "I understand this is speculative" checkbox
- **Per-finding Apply** — patches the config field via dot-path and re-saves the job; disabled on running jobs
- **Model-type-aware** — three distinct prompt branches: image models (white noise, mode collapse, rank/LR/optimizer), video models (temporal consistency, LightX2V dual-LoRA config, frame-based dataset thresholds), audio models (sample rate, BPM consistency, `audio_lm_path`, no image metrics)
- **Visual analysis** — optional; sends up to 4 recent sample images (or 4 frames extracted from the latest MP4 via ffmpeg) and up to 6 training dataset images; falls back gracefully to text-only when images are unavailable or ffmpeg is missing
- **Performance optimization** — uses live GPU/RAM stats to recommend batch size increases when VRAM headroom > 4 GB, always paired with the linear LR scaling rule (batch×2 → LR×2); flags VRAM pressure, thermal throttling, and system RAM shortage
- **Automagic3 + Qwen Image rule** — hardcoded critical rule: if optimizer is `automagic3` on a Qwen Image job, flags as `severity=error` (confirmed incident: LR ramped 100× by step 1500 producing white-noise outputs; `automagic3` has no `max_lr` clamp)
- **OOM context** — when triggered by an OOM alert, analysis fires immediately (`autoRun` mode) with the OOM message and VRAM stats included for targeted recommendations
- **Configurable via Settings page** — API URL, API key, model name, and web-search toggle stored in DB (no server file access required); works with Claude (via Anthropic's OpenAI-compatible endpoint) and Ollama (local models); vision analysis requires a vision-capable model
- **Ollama web search** — when the API endpoint is an Ollama server and "Enable Ollama Web Search" is toggled on, a `POST /api/web_search` call is made against the Ollama host before the LLM call; results are prepended to the training context so the model can reference current best-practice sources; the same API key (if set) is sent as `Authorization: Bearer` for both the web search and the LLM call

### Infrastructure

- **Claude-assisted merge** — `run_ai_toolkit.sh` offers Claude-assisted upstream merge at startup
- **CivitAI metadata in PNG/MP4** — A1111-format `parameters` in PNG tEXt chunk; ffmpeg FFMETADATA1 for MP4

---



## Supported Models

### Image
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) (FLUX.1)
- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) (FLUX.2)
- [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) (FLUX.2-klein-base-4B)
- [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) (FLUX.2-klein-base-9B)
- [ostris/Flex.1-alpha](https://huggingface.co/ostris/Flex.1-alpha) (Flex.1)
- [ostris/Flex.2-preview](https://huggingface.co/ostris/Flex.2-preview) (Flex.2)
- [lodestones/Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base) (Chroma)
- [Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) (Lumina2)
- [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) (Qwen-Image)
- [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) (Qwen-Image-2512)
- [HiDream-ai/HiDream-I1-Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full) (HiDream I1)
- [OmniGen2/OmniGen2](https://huggingface.co/OmniGen2/OmniGen2) (OmniGen2)
- [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Z-Image Turbo)
- [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (Z-Image)
- [ostris/Z-Image-De-Turbo](https://huggingface.co/ostris/Z-Image-De-Turbo) (Z-Image De-Turbo)
- [zhen-nan/L2P](https://huggingface.co/zhen-nan/L2P) (Z-Image L2P)
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (SDXL)
- [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) (SD 1.5)
- [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) (ERNIE-Image)
- [NucleusAI/Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) (Nucleus-Image)
- [Boogu/Boogu-Image-0.1-Base](https://huggingface.co/Boogu/Boogu-Image-0.1-Base) (Boogu Image 0.1)
- [HiDream-ai/HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image) (HiDream O1)
- [ideogram-ai/ideogram-4-fp8](https://huggingface.co/ideogram-ai/ideogram-4-fp8) (Ideogram 4 FP8)
- [Photoroom/prxpixel-t2i](https://huggingface.co/Photoroom/prxpixel-t2i) (PRXPixel)
- [circlestone-labs/Anima-Base-v1.0-Diffusers](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers) (Anima)
- [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) (Krea 2)
- [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) (Krea 2 Turbo)

### Instruction / Edit
- [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) (FLUX.1-Kontext-dev)
- [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) (Qwen-Image-Edit)
- [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) (Qwen-Image-Edit-2509)
- [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) (Qwen-Image-Edit-2511)
- [HiDream-ai/HiDream-E1-1](https://huggingface.co/HiDream-ai/HiDream-E1-1) (HiDream E1)
- [Boogu/Boogu-Image-0.1-Edit](https://huggingface.co/Boogu/Boogu-Image-0.1-Edit) (Boogu Image Edit)
- [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) (Krea 2 Edit Training)
- [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) (Krea 2 Turbo Edit Training)

### Video
- [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) (Wan 2.1 1.3B)
- [Wan-AI/Wan2.1-I2V-14B-480P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) (Wan 2.1 I2V 14B-480P)
- [Wan-AI/Wan2.1-I2V-14B-720P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers) (Wan 2.1 I2V 14B-720P)
- [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) (Wan 2.1 14B)
- [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) (Wan 2.2 14B)
- [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) (Wan 2.2 I2V 14B)
- [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (Wan 2.2 TI2V 5B)
- [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) (LTX-2)
- [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (LTX-2.3)

### Audio
- [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) (Ace Step 1.5)
- [ACE-Step/acestep-v15-xl-base](https://huggingface.co/ACE-Step/acestep-v15-xl-base) (Ace Step 1.5 XL)

### Experimental
- [lodestones/Zeta-Chroma](https://huggingface.co/lodestones/Zeta-Chroma) (Zeta Chroma)

## Installation

Requirements:
- python >=3.10 (3.12 recommended)
- Nvidia GPU with enough ram to do what you need
- python venv
- git


Linux:
```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python3 -m venv venv
source venv/bin/activate
# install torch first
pip3 install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip3 install -r requirements.txt
```

For devices running **DGX OS** (including DGX Spark), follow [these](dgx_instructions.md) instructions.


Windows:

If you are having issues with Windows. I recommend using the easy install script at [https://github.com/Tavris1/AI-Toolkit-Easy-Install](https://github.com/Tavris1/AI-Toolkit-Easy-Install)

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python -m venv venv
.\venv\Scripts\activate
pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

MacOS:

Experimental support for Silicon Macs is available. I do not have a Mac with enough RAM to fully test this
so please let me know if there are issues. There is a convience script to install and run on MacOS 
locates at `./run_mac.zsh` that will install the dependencies locally and run the UI. To run this, 
do the following:

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
chmod +x run_mac.zsh
./run_mac.zsh
```


# AI Toolkit UI

<img src="https://ostris.com/wp-content/uploads/2025/02/toolkit-ui.jpg" alt="AI Toolkit UI" width="100%">

The AI Toolkit UI is a web interface for the AI Toolkit. It allows you to easily start, stop, and monitor jobs. It also allows you to easily train models with a few clicks. It also allows you to set a token for the UI to prevent unauthorized access so it is mostly safe to run on an exposed server.

## Running the UI

Requirements:
- Node.js > 20

The UI does not need to be kept running for the jobs to run. It is only needed to start/stop/monitor jobs. The commands below
will install / update the UI and it's dependencies and start the UI. 

```bash
cd ui
npm run build_and_start
```

You can now access the UI at `http://localhost:8675` or `http://<your-ip>:8675` if you are running it on a server.

## Securing the UI

If you are hosting the UI on a cloud provider or any network that is not secure, I highly recommend securing it with an auth token. 
You can do this by setting the environment variable `AI_TOOLKIT_AUTH` to super secure password. This token will be required to access
the UI. You can set this when starting the UI like so:

```bash
# Linux
AI_TOOLKIT_AUTH=super_secure_password npm run build_and_start

# Windows
set AI_TOOLKIT_AUTH=super_secure_password && npm run build_and_start

# Windows Powershell
$env:AI_TOOLKIT_AUTH="super_secure_password"; npm run build_and_start
```

### Training
1. Copy the example config file located at `config/examples/train_lora_flux_24gb.yaml` (`config/examples/train_lora_flux_schnell_24gb.yaml` for schnell) to the `config` folder and rename it to `whatever_you_want.yml`
2. Edit the file following the comments in the file
3. Run the file like so `python run.py config/whatever_you_want.yml`

A folder with the name and the training folder from the config file will be created when you start. It will have all 
checkpoints and images in it. You can stop the training at any time using ctrl+c and when you resume, it will pick back up
from the last checkpoint.

IMPORTANT. If you press crtl+c while it is saving, it will likely corrupt that checkpoint. So wait until it is done saving

### Need help?

Please do not open a bug report unless it is a bug in the code. You are welcome to [Join my Discord](https://discord.gg/VXmU2f5WEU)
and ask for help there. However, please refrain from PMing me directly with general question or support. Ask in the discord
and I will answer when I can.

## Ostris Cloud

You can use many cloud providers to rent GPUs. If you want to help support this project in the largest way possible, please consider using [Ostris Cloud](https://cloud.ostris.com). Ostris Cloud is owned and operated by me, Ostris, and every dollar earned goes directly back into funding the development of this project.

<a href="https://cloud.ostris.com" target="_blank"><img src="https://cloud.ostris.com/api/og" alt="Ostris Cloud" style="max-width:100%;width:600px;height:auto;"></a>


## Training in RunPod
If you would like to use Runpod, but have not signed up yet, please consider using [my Runpod affiliate link](https://runpod.io?ref=h0y9jyr2) to help support this project.


I maintain an official Runpod Pod template here which can be accessed [here](https://console.runpod.io/deploy?template=0fqzfjy6f3&ref=h0y9jyr2).

I have also created a short video showing how to get started using AI Toolkit with Runpod [here](https://youtu.be/HBNeS-F6Zz8).

## Training in Modal

### 1. Setup
#### ai-toolkit:
```
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
pip install torch
pip install -r requirements.txt
pip install --upgrade accelerate transformers diffusers huggingface_hub #Optional, run it if you run into issues
```
#### Modal:
- Run `pip install modal` to install the modal Python package.
- Run `modal setup` to authenticate (if this doesn’t work, try `python -m modal setup`).

#### Hugging Face:
- Get a READ token from [here](https://huggingface.co/settings/tokens) and request access to Flux.1-dev model from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev).
- Run `huggingface-cli login` and paste your token.

### 2. Upload your dataset
- Drag and drop your dataset folder containing the .jpg, .jpeg, or .png images and .txt files in `ai-toolkit`.

### 3. Configs
- Copy an example config file located at ```config/examples/modal``` to the `config` folder and rename it to ```whatever_you_want.yml```.
- Edit the config following the comments in the file, **<ins>be careful and follow the example `/root/ai-toolkit` paths</ins>**.

### 4. Edit run_modal.py
- Set your entire local `ai-toolkit` path at `code_mount = modal.Mount.from_local_dir` like:
  
   ```
   code_mount = modal.Mount.from_local_dir("/Users/username/ai-toolkit", remote_path="/root/ai-toolkit")
   ```
- Choose a `GPU` and `Timeout` in `@app.function` _(default is A100 40GB and 2 hour timeout)_.

### 5. Training
- Run the config file in your terminal: `modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/whatever_you_want.yml`.
- You can monitor your training in your local terminal, or on [modal.com](https://modal.com/).
- Models, samples and optimizer will be stored in `Storage > flux-lora-models`.

### 6. Saving the model
- Check contents of the volume by running `modal volume ls flux-lora-models`. 
- Download the content by running `modal volume get flux-lora-models your-model-name`.
- Example: `modal volume get flux-lora-models my_first_flux_lora_v1`.

### Screenshot from Modal

<img width="1728" alt="Modal Traning Screenshot" src="https://github.com/user-attachments/assets/7497eb38-0090-49d6-8ad9-9c8ea7b5388b">

---

## Dataset Preparation

Datasets generally need to be a folder containing images and associated text files. Currently, the only supported
formats are jpg, jpeg, and png. Webp currently has issues. The text files should be named the same as the images
but with a `.txt` extension. For example `image2.jpg` and `image2.txt`. The text file should contain only the caption.
You can add the word `[trigger]` in the caption file and if you have `trigger_word` in your config, it will be automatically
replaced. 

Images are never upscaled but they are downscaled and placed in buckets for batching. **You do not need to crop/resize your images**.
The loader will automatically resize them and can handle varying aspect ratios. 


## Training Specific Layers

To train specific layers with LoRA, you can use the `only_if_contains` network kwargs. For instance, if you want to train only the 2 layers
used by The Last Ben, [mentioned in this post](https://x.com/__TheBen/status/1829554120270987740), you can adjust your
network kwargs like so:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks.7.proj_out"
            - "transformer.single_transformer_blocks.20.proj_out"
```

The naming conventions of the layers are in diffusers format, so checking the state dict of a model will reveal 
the suffix of the name of the layers you want to train. You can also use this method to only train specific groups of weights.
For instance to only train the `single_transformer` for FLUX.1, you can use the following:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks."
```

You can also exclude layers by their names by using `ignore_if_contains` network kwarg. So to exclude all the single transformer blocks,


```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          ignore_if_contains:
            - "transformer.single_transformer_blocks."
```

`ignore_if_contains` takes priority over `only_if_contains`. So if a weight is covered by both,
if will be ignored.

## LoKr Training

To learn more about LoKr, read more about it at [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Guidelines.md). To train a LoKr model, you can adjust the network type in the config file like so:

```yaml
      network:
        type: "lokr"
        lokr_full_rank: true
        lokr_factor: 8
```

Everything else should work the same including layer targeting.


## Krea 2 Training

Krea 2 is a high-quality image model available in two variants:

- **Krea 2 Raw** (`krea2_raw`) — the base model; straightforward LoRA training with no extra config needed.
- **Krea 2 Turbo with Adapter** (`krea2_turbo_with_adapter`) — requires a pre-trained turbo adapter. Set `turbo_model_path` in your model config to the path of the adapter safetensors file.

### Sampling LoRA (filter bypass / style)

Some community LoRAs are designed to be applied only during inference, not training — for example to bypass Krea's content filter or apply a look. You can inject one of these into every sample generation without it affecting your training weights.

In your config's `sample` block:

```yaml
      sample:
        sampler: euler
        sample_every: 500
        width: 1024
        height: 1024
        prompts:
          - "your prompt here"
        neg: ""
        seed: 42
        steps: 20
        cfg_scale: 1
        sample_lora_path: /path/to/krea2filterbypass3.safetensors
        sample_lora_strength: 4
```

The LoRA is loaded before each sample batch and removed immediately after, so it never influences the training gradient. `sample_lora_strength` can be tuned — higher values push the bypass harder; typical range is 1–8.

Example community LoRA: [Krea2FilterBypass](https://civitai.red/models/2728234/krea2filterbypass) — apply at strength 4 to reliably bypass the built-in content filter during sampling.

> **UI**: the "Apply LoRA during sampling" checkbox and path/strength fields appear in the Sample card for both Krea 2 variants when creating or editing a job.


## Support My Work

If you enjoy my projects or use them commercially, please consider sponsoring me. Every bit helps! 💖

<a href="https://ostris.com/sponsors" target="_blank"><img src="https://ostris.com/wp-content/uploads/2025/05/support-banner2.png" alt="Support my work" style="max-width:100%;height:auto;"></a>

### Current Sponsors

All of these people / organizations are the ones who selflessly make this project possible. Thank you!!

<a href="https://ostris.com/sponsors"><img src="https://ostris.com/sponsors.svg" alt="Sponsors" style="width:100%;height:auto;"></a>
