# RunPod deploy kit — day-one training on a new base model

Built for the first-to-market case: a new model drops, and the local box is
either busy or too small. Goal is minutes-to-training, not a perfect pipeline.

## Why this shape

The `ostris/ai-toolkit` image already contains torch 2.13/cu130, flash-attn,
natten, torchcodec and `ui/node_modules` — the slow parts. Building our own
image would cost ~30+ minutes on the day it matters. So we boot **their** image
and overlay **our** fork's source at pod start (~1 min). That gets our trainers,
our caption/JSON dataloader handling, and our UI onto a machine that already has
the heavy deps compiled.

## Prerequisites (do these NOW, not on launch day)

1. **Push the fork.** `bootstrap.sh` clones `MarcBate/ai-toolkit` @ `main`.
   Local commits that were never pushed do not exist to the pod.
   Check: `git log --oneline fork/main..main` should be empty.
2. **HF token** — needed for gated weights. `export HF_TOKEN=...` on the pod.
3. **SSH key** in the RunPod account, so the pod comes up reachable.

## Launch day

```bash
# 1. Start a pod from the ostris/ai-toolkit image, note its ssh host/port.

# 2. On the pod - overlay our fork (~1 min)
curl -sL https://raw.githubusercontent.com/MarcBate/ai-toolkit/main/scripts/runpod/bootstrap.sh | bash

# 3. From here - ship the dataset up
scripts/runpod/sync_up.sh <host> <port> uncut_images

# 4. Train (UI on :8675, or direct)
#    ssh -p <port> root@<host>
#    cd /app/ai-toolkit && python run.py config/<job>.yaml

# 5. Pull results back
scripts/runpod/fetch_output.sh <host> <port> <job_name>
```

`sync_up.sh` strips `_latent_cache` / `_t_e_cache` (machine-specific, and the
pod will rebuild them). `fetch_output.sh` skips `optimizer.pt` (~1.4GB, only
needed to resume on the pod itself) — pass it manually if you intend to resume.

## Flux 3 Dev notes

As of 2026-08-28 FLUX 3 is **API/early-access only**; the open-weight
**FLUX 3 Dev** has no announced date ("later this year"), no license terms and
no published size. There is no `black-forest-labs/FLUX.3*` repo on HF yet.

It is multimodal (image + ~20s video with audio + robotics), so it will need
ai-toolkit support from ostris before any of this is useful — this kit gets us
onto a GPU fast, it does not make an unsupported arch trainable. On drop day the
order is: ostris adds the arch -> merge upstream -> push fork -> bootstrap a pod.

Keep the merge cadence tight so step 2 is small.

## Alternative: Ostris Cloud

`ui/src/app/api/ostris_cloud/route.ts` already talks to an Ostris Cloud machine
(`OSTRIS_CLOUD_APP_URL` + `OSTRIS_CLOUD_API_KEY`, shows account balance).
Likely lower friction than RunPod, but confirm whether it will run **our fork**
rather than stock ai-toolkit — our dataloader caption handling and per-dataset
trigger words live in the fork, and datasets captioned for them may not train
correctly on a stock build.
