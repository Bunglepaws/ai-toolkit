#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Start AI Toolkit in WSL
# - optional git pull
# - optional pip install -r requirements.txt if repo updated
# - optional npm install if missing
# - optional python environment checks
# - start UI
# - wait for http://localhost:8675
# - open in Windows browser
# ------------------------------------------------------------

# -----------------------------
# Settings
# -----------------------------
ROOT="/mnt/c/Data/git/AIToolkitWSL"
REPO_DIR="${ROOT}/ai-toolkit"
UI_DIR="${REPO_DIR}/ui"

# The venv lives on the distro's ext4, NOT under /mnt/c. Importing torch +
# transformers + diffusers costs ~67s across the drvfs bridge versus ~2s native —
# the venv is 56k small files, which is drvfs's worst case.
#
# It must be invoked by this real path. <repo>/venv is a symlink here for
# convenience, but going through it does NOT help: Python keeps the invocation
# path as sys.prefix, so every site-packages read is still translated across the
# bridge (measured 30s vs 2s).
VENV_DIR="/home/marcbate/venvs/ai-toolkit"

UI_PORT="8675"
UI_URL="http://localhost:${UI_PORT}"
LOG_FILE="${ROOT}/ai-toolkit-ui.log"

DO_GIT_PULL="1"
DO_PIP_INSTALL_ON_UPDATE="1"
DO_NPM_INSTALL_IF_MISSING="1"
DO_NPM_INSTALL_ON_UPDATE="0"

# Set to 1 to run Python environment checks before launch.
# Default is 0 to skip them.
RUN_ENV_CHECKS="0"

# Hugging Face cache on a dedicated ext4 VHD (C:\Data\WSL\hf-cache.vhdx).
# It lives on a real Linux filesystem rather than under /mnt/c because the WSL
# drvfs bridge caps at ~225 MB/s regardless of how fast the underlying Windows
# drive is (measured: 223 MB/s on /mnt/c, 228 MB/s on a second NVMe, 13.3 GB/s
# here). Loading a 25GB transformer went from ~112s to a couple of seconds.
HF_CACHE_MOUNT="/mnt/wsl/hfcache"
export HF_HOME="${HF_CACHE_MOUNT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

# Non-diffusers (ComfyUI-style) checkpoint downloads, e.g. MiniMax H3's
# separate DiT/text-encoder/VAE safetensors, default to <repo>/models on the
# NTFS-backed repo dir if left unset. Keep them on the same ext4 disk as the
# HF cache above to avoid the drvfs bottleneck.
export MODELS_PATH="${HF_CACHE_MOUNT}/aitk-models"

# The UI worker spawns training processes via resolvePythonPath(), which would
# otherwise find the <repo>/venv symlink and pay the drvfs import cost on every
# job start. Point it at the real interpreter.
export AITK_PYTHON="${VENV_DIR}/bin/python3"

# Optional stability knobs
export GIT_LFS_SKIP_SMUDGE=1
export PYTHONUNBUFFERED=1

# Claude Code CLI (installed to user-local npm prefix, not on system PATH)
export PATH="$HOME/.npm-global/bin:$PATH"

# -----------------------------
# Helpers
# -----------------------------
die() {
  echo
  echo "ERROR: $*" >&2
  exit 1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

wait_for_url() {
  local url="$1"
  local max_sec="$2"
  local start now
  start="$(date +%s)"

  while true; do
    if have_cmd curl; then
      if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
        return 0
      fi
    else
      if python - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("${url}", timeout=2).read(1)
PY
      then
        return 0
      fi
    fi

    sleep 2
    now="$(date +%s)"
    if (( now - start > max_sec )); then
      return 1
    fi
  done
}

open_in_windows() {
  local url="$1"

  if have_cmd powershell.exe; then
    powershell.exe -NoProfile -Command "Start-Process '${url}'" >/dev/null 2>&1 || true
  elif have_cmd cmd.exe; then
    cmd.exe /c start "$url" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  local exit_code=$?

  if [[ -n "${UI_PID:-}" ]]; then
    if kill -0 "$UI_PID" >/dev/null 2>&1; then
      echo
      echo "Stopping AI Toolkit UI..."
      kill "$UI_PID" >/dev/null 2>&1 || true
      wait "$UI_PID" >/dev/null 2>&1 || true
    fi
  fi

  exit "$exit_code"
}

# -----------------------------
# Main
# -----------------------------
trap cleanup EXIT INT TERM

echo
echo "========== AI Toolkit (WSL) =========="
echo "Repo: ${REPO_DIR}"
echo "Venv: ${VENV_DIR}"
echo "UI:   ${UI_URL}"
echo "Log:  ${LOG_FILE}"
echo

have_cmd git || die "git not found. Run: sudo apt-get update && sudo apt-get install -y git"
have_cmd python3 || die "python3 not found in WSL."
have_cmd npm || die "npm not found in WSL. Run: sudo apt-get update && sudo apt-get install -y nodejs npm"
have_cmd ffmpeg || die "ffmpeg not found in WSL. Run: sudo apt-get update && sudo apt-get install -y ffmpeg"

[[ -d "${REPO_DIR}/.git" ]] || die "Repo not found at ${REPO_DIR}"
[[ -f "${VENV_DIR}/bin/activate" ]] || die "venv not found at ${VENV_DIR}. Create/fix it first."

# WSL does not re-attach the HF cache VHD after a reboot. Fail loudly rather than
# starting, because HF_HOME would silently point at an empty directory and every
# model would re-download (~325GB) instead of erroring.
grep -q " ${HF_CACHE_MOUNT} " /proc/mounts || die "HF cache disk not mounted at ${HF_CACHE_MOUNT}.
Attach it from an elevated Windows prompt:
  wsl --mount --vhd \"C:\\Data\\WSL\\hf-cache.vhdx\" --name hfcache"

# Training processes are spawned detached (startJob.ts uses detached + unref), so
# they keep running after this script exits. Restarting the stack while one is
# alive hands a fresh worker a database full of rows it did not create, alongside
# a live trainer it never spawned — which is how two jobs ended up sharing one
# GPU (90s/it instead of <5s/it, no OOM, just CUDA spilling to shared memory).
LIVE_TRAINERS="$(pgrep -af 'run_ui\.py' 2>/dev/null || true)"
if [[ -n "${LIVE_TRAINERS}" ]]; then
  echo
  echo "WARNING: training is already running. Stopping this script does not stop it."
  echo
  echo "${LIVE_TRAINERS}"
  echo
  echo "Restarting the UI on top of a live trainer risks a second job on the same GPU."
  echo "Safe if you only need the UI back (a crashed UI does not kill training)."
  read -r -p "Continue anyway? [y/N] " _trainer_answer
  case "${_trainer_answer}" in
    [Yy]|[Yy][Ee][Ss])
      echo "---- Continuing with a trainer still running."
      ;;
    *)
      die "Aborted. Stop the job from the UI, or wait for it to finish, then re-run."
      ;;
  esac
fi

cd "${REPO_DIR}"

UPDATED="0"

if [[ "${DO_GIT_PULL}" == "1" ]]; then
  echo "---- Checking for updates from origin/main..."
  git config pull.rebase false
  git fetch origin main

  LOCAL_HASH="$(git rev-parse HEAD)"
  REMOTE_HASH="$(git rev-parse origin/main)"
  BASE_HASH="$(git merge-base HEAD origin/main)"

  if [[ "${LOCAL_HASH}" == "${REMOTE_HASH}" ]]; then
    echo "---- Already up to date."
  elif [[ "${BASE_HASH}" == "${REMOTE_HASH}" ]]; then
    echo "---- Your local branch is ahead of origin/main. No updates to pull."
  else
    # origin/main has new commits not yet in this branch
    NEW_COUNT="$(git rev-list HEAD..origin/main --count)"
    echo
    echo "---- ${NEW_COUNT} update(s) available from origin/main:"
    git log --oneline HEAD..origin/main
    echo

    if have_cmd claude; then
      read -r -p "Merge with [C]laude (recommended) / [m]anual pull / [s]kip? " _answer
      case "${_answer}" in
        [Mm]*)
          echo "---- Pulling updates manually..."
          if git pull origin main; then
            echo "---- Successfully pulled updates."
            UPDATED="1"
          else
            echo "---- Pull failed! Please resolve manually."
            exit 1
          fi
          ;;
        [Ss]*)
          echo "---- Skipping pull. Continuing with current code."
          ;;
        *)
          echo "---- Running Claude merge..."
          if claude --dangerously-skip-permissions -p "
Merge the latest upstream changes from origin/main into this fork's main branch.
Run: git merge origin/main

This is MarcBate's fork of ostris/ai-toolkit. Remote 'origin' = ostris upstream.
Our branch has many customizations — if there are conflicts, keep ALL of ours:
- toolkit/dataloader_mixins.py: keep our read_text_file + JSON parsing
  (caption/caption_short/extra_values) and our PoiFileItemDTOMixin class
- ui/src/app/jobs/new/SimpleJob.tsx: keep our MRU LoRA input, neg-prompt
  hidden for ideogram4, LoraPathInput component
- ui/src/app/jobs/new/options.ts: keep 'sample.neg' in DisableableSections
  for ideogram4
- ui/src/app/layout.tsx: keep StopJobModal and StripAudioModal alongside
  any new Ostris modals
- ui/src/helpers/defaultSamples.ts: keep guidance_scale: 7 in
  defaultIdeogramSamplesConfig
- ui/src/components/SampleImageViewer.tsx: keep both our promptExpanded
  state and Ostris showBoxes state
If ostris merged something we already built, STOP and ask the user which
implementation to keep before proceeding.
After resolving all conflicts, commit the merge.
Report what was merged and any decisions made.
"; then
            echo "---- Claude merge complete."
            UPDATED="1"
          else
            echo "---- Claude merge failed or was cancelled. Please resolve manually."
            exit 1
          fi
          ;;
      esac
    else
      # claude CLI not available — fall back to plain pull
      read -r -p "Pull updates now? [y/N] " _answer
      case "${_answer}" in
        [Yy]|[Yy][Ee][Ss])
          echo "---- Pulling updates..."
          if git pull origin main; then
            echo "---- Successfully pulled updates."
            UPDATED="1"
          else
            echo "---- Pull failed! Please resolve manually."
            exit 1
          fi
          ;;
        *)
          echo "---- Skipping pull. Continuing with current code."
          ;;
      esac
    fi
  fi
else
  echo "---- Skipping git pull"
fi

echo
echo "---- Activating venv..."
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -c "import sys; print('python', sys.executable)" || die "python failed after activating venv"

if [[ "${RUN_ENV_CHECKS}" == "1" ]]; then
  echo
  echo "---- Verifying Python environment..."
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || die "torch import failed"

  python - <<'PY' || die "torchaudio import failed"
import torchaudio
print("torchaudio", torchaudio.__version__)
PY

  python - <<'PY' || die "torchcodec import failed"
import torchcodec
print("torchcodec import ok")
PY

  python - <<'PY' || die "LTX-2.3 vocoder import failed"
import diffusers.pipelines.ltx2.vocoder as v
ok = "LTX2VocoderWithBWE" in dir(v)
print("LTX2VocoderWithBWE:", ok)
if not ok:
    raise SystemExit(1)
PY
else
  echo
  echo "---- Skipping Python environment checks (RUN_ENV_CHECKS=0)"
fi

if [[ "${DO_PIP_INSTALL_ON_UPDATE}" == "1" && "${UPDATED}" == "1" ]]; then
  echo
  echo "---- Repo updated. Syncing Python requirements..."
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --no-cache-dir -r requirements.txt
fi

echo
echo "---- Checking Node/npm dependencies..."
if [[ "${DO_NPM_INSTALL_IF_MISSING}" == "1" && ! -e "${UI_DIR}/node_modules" ]]; then
  echo "---- node_modules missing. Running npm install..."
  cd "${UI_DIR}"
  npm install
  cd "${REPO_DIR}"
elif [[ "${DO_NPM_INSTALL_ON_UPDATE}" == "1" && "${UPDATED}" == "1" ]]; then
  echo "---- Repo updated. Running npm install..."
  cd "${UI_DIR}"
  npm install
  cd "${REPO_DIR}"
fi

if wait_for_url "${UI_URL}" 2; then
  echo
  echo "UI already responding at ${UI_URL}"
  open_in_windows "${UI_URL}"
  exit 0
fi

echo
echo "---- Starting UI server..."
echo "Logs will also be appended to: ${LOG_FILE}"
echo

mkdir -p "$(dirname "${LOG_FILE}")"

cd "${UI_DIR}"

# Only rebuild if source files changed since the last build.
# Checks src/, cron/, public/, package.json, next.config.*, and tsconfig files.
#
# cron/ matters as much as src/: the background worker runs from the compiled
# dist/cron/*.js, so edits to pythonPath.ts / processQueue.ts / startJob.ts do
# nothing until `npm run build` recompiles them. Leaving cron/ out of this check
# meant worker fixes silently ran as stale builds.
NEXT_BUILD="${UI_DIR}/.next/BUILD_ID"
UI_NEEDS_BUILD=0

if [[ ! -f "${NEXT_BUILD}" ]]; then
  echo "---- No existing build found, will build."
  UI_NEEDS_BUILD=1
elif find "${UI_DIR}/src" "${UI_DIR}/cron" "${UI_DIR}/public" \
         "${UI_DIR}/package.json" "${UI_DIR}/tsconfig.json" \
         "${UI_DIR}/tsconfig.worker.json" \
         -newer "${NEXT_BUILD}" -print -quit 2>/dev/null | grep -q .; then
  echo "---- UI source files changed since last build, will rebuild."
  UI_NEEDS_BUILD=1
elif [[ -n "$(find "${UI_DIR}" -maxdepth 1 -name "next.config.*" -newer "${NEXT_BUILD}" 2>/dev/null)" ]]; then
  echo "---- next.config changed since last build, will rebuild."
  UI_NEEDS_BUILD=1
else
  echo "---- UI build is up to date, skipping rebuild."
fi

(
  if [[ "${UI_NEEDS_BUILD}" == "1" ]]; then
    npm install && npm run update_db && npm run build && npm run start
  else
    npm run update_db && npm run start
  fi
) 2>&1 | tee -a "${LOG_FILE}" &
UI_PID=$!


echo "UI PID: ${UI_PID}"
echo "Waiting for ${UI_URL} ..."

if wait_for_url "${UI_URL}" 600; then
  echo "UI is up: ${UI_URL}"
  open_in_windows "${UI_URL}"
else
  echo
  echo "UI did not come up within 600 seconds."
  echo "Check the log: ${LOG_FILE}"
  exit 1
fi

wait "${UI_PID}"
