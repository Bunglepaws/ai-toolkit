#!/usr/bin/env bash
# Push a local dataset (and optionally a warm-start LoRA) to the pod.
# Uses tar-over-ssh so it needs nothing but ssh - no rsync on Windows.
#
#   scripts/runpod/sync_up.sh <ssh_host> <ssh_port> <dataset_name> [lora_file]
#
# Example:
#   scripts/runpod/sync_up.sh 213.173.108.4 22065 uncut_images
set -euo pipefail

HOST="${1:?ssh host}"; PORT="${2:?ssh port}"; DS="${3:?dataset folder name}"; LORA="${4:-}"
LOCAL_ROOT="${DATASETS_ROOT:-C:/Data/AIToolkit-StagingArea/datasets}"
SRC="$LOCAL_ROOT/$DS"
REMOTE=/app/ai-toolkit/datasets

[ -d "$SRC" ] || { echo "no such dataset: $SRC"; exit 1; }

# Caches are large and machine-specific - never ship them.
echo "==> Uploading $DS -> $HOST:$REMOTE/$DS"
tar -cz -C "$LOCAL_ROOT" \
    --exclude='_latent_cache' --exclude='_t_e_cache' --exclude='.thumbs' \
    "$DS" \
  | ssh -p "$PORT" -o StrictHostKeyChecking=accept-new "root@$HOST" \
    "mkdir -p $REMOTE && tar -xz -C $REMOTE && echo '   items:' \$(ls $REMOTE/$DS | wc -l)"

if [ -n "$LORA" ]; then
  echo "==> Uploading warm-start LoRA $(basename "$LORA")"
  ssh -p "$PORT" "root@$HOST" "mkdir -p /app/ai-toolkit/output/_warmstart"
  scp -P "$PORT" "$LORA" "root@$HOST:/app/ai-toolkit/output/_warmstart/"
fi
echo "==> done"
