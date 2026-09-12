#!/usr/bin/env bash
# Pull a finished job's checkpoints + samples back down from the pod.
#
#   scripts/runpod/fetch_output.sh <ssh_host> <ssh_port> <job_name> [dest_dir]
set -euo pipefail

HOST="${1:?ssh host}"; PORT="${2:?ssh port}"; JOB="${3:?job name}"
DEST="${4:-C:/Data/AIToolkit-StagingArea/output/aitoolkit}"
REMOTE="/app/ai-toolkit/output/$JOB"

mkdir -p "$DEST"
echo "==> Fetching $JOB -> $DEST/$JOB"
# Skip optimizer.pt by default: it is ~1.4GB and only needed to resume ON the pod.
ssh -p "$PORT" -o StrictHostKeyChecking=accept-new "root@$HOST" \
    "tar -cz -C /app/ai-toolkit/output --exclude='optimizer.pt' --exclude='_latent_cache' '$JOB'" \
  | tar -xz -C "$DEST"
echo "==> got: $(ls "$DEST/$JOB"/*.safetensors 2>/dev/null | wc -l) checkpoints"
