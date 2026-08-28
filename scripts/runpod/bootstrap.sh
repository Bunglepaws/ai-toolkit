#!/usr/bin/env bash
# Run this ON the pod, once, right after it boots.
#
#   curl -sL https://raw.githubusercontent.com/MarcBate/ai-toolkit/main/scripts/runpod/bootstrap.sh | bash
#
# Assumes a pod started from the ostris/ai-toolkit image (torch, flash-attn,
# natten, torchcodec and ui/node_modules are already baked in). This only
# swaps the ostris source for OUR fork and rebuilds the UI, which is the
# difference between a generic pod and one that has our trainers, our
# dataloader caption handling, and our UI.
set -euo pipefail

FORK_URL="${FORK_URL:-https://github.com/MarcBate/ai-toolkit.git}"
FORK_REF="${FORK_REF:-main}"
APP=/app/ai-toolkit

echo "==> Overlaying fork: $FORK_URL @ $FORK_REF"
rm -rf /tmp/fork && git clone --depth 1 --branch "$FORK_REF" "$FORK_URL" /tmp/fork

# Keep the image's prebuilt dependency dirs; replace only source.
rsync -a --delete \
  --exclude 'ui/node_modules' \
  --exclude 'ui/.next' \
  --exclude 'output' \
  --exclude 'datasets' \
  --exclude 'aitk_db.db' \
  /tmp/fork/ "$APP/"
rm -rf /tmp/fork

echo "==> Fork commit: $(cd "$APP" && git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"

# Our requirements may have drifted from the image's baked layer.
echo "==> Syncing python deps (delta only)"
pip install --no-cache-dir --break-system-packages -r "$APP/requirements.txt" -q || \
  echo "!! requirements install had errors - check above before training"

echo "==> Rebuilding UI (prisma client + next build)"
cd "$APP/ui" && npm run update_db >/dev/null && npm run build >/dev/null

mkdir -p "$APP/datasets" "$APP/output"

cat <<'NOTE'

==> Ready.

  Start the UI:   cd /app/ai-toolkit/ui && npm run start     (port 8675)
  Train direct:   cd /app/ai-toolkit && python run.py config/<job>.yaml

  Set before training if the model is gated:
    export HF_TOKEN=...

NOTE
