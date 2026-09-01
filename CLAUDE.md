# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This is a **personal fork** of [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) with significant additions. The upstream remote is `origin` (ostris); the fork remote is `fork` (MarcBate). Working branch is `main`.

**Rule:** When ostris merges a feature that already exists in this fork, ask the user which implementation to keep before making any changes.

**Merging origin/main:** `git fetch origin main` then `git merge origin/main --no-edit`. This repo is heavily
diverged (hundreds of fork-only commits), so `git diff HEAD..origin/main` looks enormous (thousands of lines,
mostly our fork-only files showing as pure deletions) — that's just total divergence, not what the merge will
actually touch. Trust the real merge conflicts instead, and resolve each by keeping our implementation unless
ostris's side is clearly a superset (verify by reading both sides, don't assume).

After every merge — conflicted or not, since files can auto-merge cleanly and still lose our changes — verify
these known fork customizations are still present before considering the merge done:
- `toolkit/dataloader_mixins.py`: our `read_text_file` + JSON caption parsing (`caption`/`caption_short`/`extra_values`) and `PoiFileItemDTOMixin`
- `ui/src/app/jobs/new/SimpleJob.tsx`: MRU LoRA input (`LoraPathInput`/`MruTextInput`), neg-prompt hidden for ideogram4, per-dataset Trigger Word field
- `ui/src/app/jobs/new/options.tsx`: `sample.neg` in `DisableableSections` for ideogram4
- `ui/src/app/layout.tsx`: `StopJobModal` and `StripAudioModal` alongside any new Ostris modals
- `ui/src/helpers/defaultSamples.ts`: `guidance_scale: 7` in `defaultIdeogramSamplesConfig`
- `ui/src/components/SampleImageViewer.tsx`: both our `promptExpanded` state and Ostris's `showBoxes` state
- `ui/src/components/SampleImageViewer.tsx` and `ui/src/components/DatasetImageViewer.tsx`: `videoMute` wired onto the `<video>` (sticky mute); `ui/src/hooks/useVideoMute.ts` is fork-only and shouldn't be touched by a merge at all
- `ui/src/components/StopJobModal.tsx`, `SaveSnapshotModal.tsx`, `QueueStatusWidget.tsx`, `JobTrainingSessions.tsx`, and the `/api/jobs/[jobID]/save_and_pause`, `/save_and_requeue`, `/save_now` routes: fork-only, shouldn't be touched by a merge at all — confirm they still exist

A one-line `grep -c` per marker (see chat history for the exact commands) is enough — report which markers you
checked and that they survived, don't just say "looks fine." Also `py_compile` any touched `.py` files and
`npx tsc --noEmit` (filtering out pre-existing `.next/types` noise) on touched `.ts`/`.tsx` files before calling
the merge done. Report the *actual* merge diff (`git diff <pre-merge-sha>..HEAD --stat`), not the misleading
two-dot origin comparison, so the user sees what really changed.

**`py_compile` is not enough, and neither is `grep -c`.** Both have already let real bugs through:

- `py_compile` only proves a file *parses*. `toolkit/unloader.py` passed it while calling an undefined
  `_detach_and_cpu` and an unimported `flush`, and raised `NameError` the first time a text encoder unloaded.
- `grep -c` asks "is the marker present?" A merge resolved by keeping **both** sides makes a marker *more*
  present, so the check passes while the file now defines the same method twice. Python accepts that silently:
  the later definition wins, the earlier becomes dead code. This has happened twice — `should_save`, then
  `should_sample`, where the live and dead copies differed in whether the DB read was retried.

So also run, on every touched `.py`:

```bash
.venv/Scripts/python.exe -m pyflakes <files>   # undefined names AND redefinitions
bash check_customizations.sh                    # includes a generic duplicate-method check
```

Treat `redefinition of unused '<name>'` as a merge that kept both sides — go read both and pick one, don't
just delete the one that looks redundant.

**Verify a ported feature actually reaches the class that runs.** `UITrainer` and `DiffusionTrainer` are
*siblings* under `SDTrainer`, not parent and child. Jobs run whichever the config's `type` names — currently
`diffusion_trainer` for everything. A feature added only to `UITrainer` is dead code for every real job, with
no error to show for it: that is how `stop_sample` ("Return to Training") sat broken from June to August while
the button appeared to work. When touching either trainer, check whether the other needs the same change, and
say which class the running jobs actually use.

---

## Running the stack

Everything runs on **native Windows** (migrated off WSL 2026-08-09). The Python venv is
`.venv/` in the repo, managed by `uv`.

> **Never launch anything through WSL.** The Ubuntu-22.04 distro and its venv are gone. Any
> `wsl -d Ubuntu-22.04 ...` command in old notes, scripts, or settings is dead — it will fail or,
> worse, run a half-present stack. `run_ai_toolkit.sh` is legacy and should not be used.

```powershell
# Start the full stack (deps sync + UI). Opens http://localhost:8675
run_windows.bat
```

`run_windows.bat` is a thin bootstrap: it ensures `uv` + a Python exist, reports whether
origin/main has commits to merge (it never merges — bring that to a Claude Code session), then
runs `python -m manager sync` and `python -m manager launch`. `manager launch` runs
`npm run db_build_start` in `ui/` and serves on port 8675. If the UI is already up it refuses to
rebuild over the live stack and offers open/restart/leave.

```bash
# Run a training job directly
.venv/Scripts/python.exe run.py config/my_config.yaml
```

Use `.venv/Scripts/python.exe` explicitly for every Python invocation (py_compile, pyflakes,
one-off scripts). The repo-root `venv/` is a **stale symlink** to the deleted WSL venv — do not
use it, and do not trust `source venv/bin/activate`.

**UI commands** (run from `ui/`):
```bash
npm run dev            # dev mode with hot reload (Next.js + cron worker)
npm run build_and_start  # production build + start
npm run db_build_start   # what `manager launch` actually runs
npm run update_db      # regenerate Prisma client + push schema to SQLite
npm run lint           # ESLint
npx tsc --noEmit       # TypeScript type-check without emitting
```

After any change to `ui/prisma/schema.prisma`, run `npm run update_db`.

---

## Architecture

### Python backend

- **Entry point:** `run.py` — loads a YAML config, resolves the job type, instantiates the right process class, runs it.
- **Job config:** YAML files under `config/`. Each config names an extension (e.g. `sd_trainer.DiffusionTrainer`) and a `process` list.
- **Extensions:** `extensions_built_in/` contains all model-specific trainers. Key ones:
  - `sd_trainer/DiffusionTrainer.py` — main trainer for Flux, SDXL, etc. Extends `SDTrainer`.
  - `sd_trainer/UITrainer.py` — adds SQLite-backed UI control (stop/save/sample flags) on top of `SDTrainer`.
  - `diffusion_models/ltx2/ltx2.py` — LTX-2.3 video model; overrides `save_image` to call `encode_video` then embed MP4 metadata.
  - `diffusion_models/wan22/` — WAN 2.2 with LightX2V fast sampling.
- **`toolkit/config_modules.py`** — `GenerateImageConfig` is the central class for sample generation; owns `save_image()`, `_embed_mp4_metadata()`, and `_build_civitai_parameters()`.
- **`toolkit/ui_utils.py`** — defines `JobStoppedException(BaseException)` used to interrupt training cleanly at any point including quantization.

### UI ↔ Python communication

Jobs communicate entirely through a **SQLite database** (`aitk_db.db` in the repo root). The Python trainer polls these columns on the `Job` table at the end of each step:

| Column | Meaning |
|--------|---------|
| `stop` | Stop training after saving if `save` is also set |
| `save` | Save a checkpoint on the next step, then continue |
| `sample` | Generate samples on the next step |
| `return_to_queue` | Stop and re-queue |

`save` and `stop` are set together by the "Save and Pause" action — the trainer always saves before stopping. `save_now` exists in the schema (added upstream) but is **not used** by the trainer; `save` is the canonical flag.

### UI (Next.js + Prisma)

- **UI server:** Next.js 15 app in `ui/src/app/`. API routes are under `ui/src/app/api/`.
- **Background worker:** `ui/cron/worker.ts` — manages the training queue, spawns Python processes via `startJob.ts`.
- **Job actions** flow: UI button → `ui/src/utils/jobs.ts` function → API route → Prisma update to `aitk_db.db` → Python trainer polls and acts.
- **Key components:** `JobActionBar.tsx` (per-job buttons), `SaveSnapshotModal.tsx` (save/pause dialog), `SampleImageViewer.tsx` (reads prompt from file metadata via `/api/img/metadata`).

### Changing stop / pause / save-checkpoint — treat as production

**Rule:** anything touching how a job stops, pauses, or writes a checkpoint gets handled with
care, spelled-out reasoning, and the user's agreement before it lands. A lost checkpoint is
hours of GPU time and, since the LoRAs are now a real revenue stream, real money. "It compiled
and the tests pass" is not the bar.

This is the single most damaged area of the codebase, and every failure looked fine in review:

- `stop` carried two incompatible meanings ("interrupt now" for ostris' watcher, "stop after
  saving" for save-and-pause). The watcher polls every 2s and on a 12s/it job almost always won
  the race — `KeyboardInterrupt` mid-step, **checkpoint simply lost**. Fixed by giving the
  cooperative case its own `stop_after_save` column.
- `maybe_save()` and `maybe_sample()` both saved, so one step wrote the same checkpoint twice and
  the second archived the optimizer the first had just written under the wrong step number.
- `stop_sample` was implemented on the wrong class and silently did nothing for two months.

So, when changing this path:

1. State explicitly **what each flag means** and who writes and clears it. Most bugs here are one
   flag carrying two meanings, or a flag nothing clears.
2. State what the row looks like **between** two writes, and what the next poll concludes from it.
   Every bug here so far has been a window between writes, not bad logic.
3. Order matters: `maybe_save()` runs before `maybe_stop()` so a pending save always lands before
   the stop is raised. Do not reorder `end_step_hook` without saying why it is still safe.
4. Never let a *preview* concern (sampling, sample LoRAs) be able to abort or skip a save. Saving
   comes first and failures downstream of it must be contained.
5. Prefer self-healing: a stale flag should be cleared by the next operation that owns it rather
   than persisting to break every future attempt.

### Changing the queue or job lifecycle — `ui/cron/**`

**Rule:** a change to `processQueue.ts`, `startJob.ts`, or anything else that decides
whether a job is alive, dead, or startable is **not done when it compiles**. It is done when
it has been watched surviving a real launch. This code has broken the queue twice now (see
"Two job dispatchers" and the launch-race reconcile), and both times it looked correct in
review and failed on timing.

Before calling such a change complete:

1. `cd ui && npx tsc -p tsconfig.worker.json` — the worker runs **compiled `dist/cron`**;
   edits to `ui/cron/*.ts` are inert until this runs. Prefer this over `npm run build`
   while the UI is serving: a full `next build` rewrites `.next/` under the running server.
2. Restart the worker so it picks up the new `dist`. Under `npm run start`, `concurrently`
   respawns it, so killing the worker pid alone is enough — it does not touch the Next UI or
   any detached trainer.
3. Run the launch smoke test against a real job start and paste its verdict:
   ```bash
   python scripts/queue_launch_smoke.py --watch
   ```
   It watches the DB through a launch and fails on the things that actually go wrong: a
   healthy job being reconciled away as "trainer process gone", a row stuck at `running`
   with no pid, or two live trainers landing on the same GPU. See its `--help` for
   `--start <job name>`, which queues the job for you instead of waiting for a UI click.

**Ask before restarting anything while a job is training.** Trainers are launched detached
and survive a worker restart, but this is the user's call, not an assumption to make.

**Reasoning about this code:** every bug here so far has been a window between two writes,
not bad logic. The worker ticks every second; a launch takes longer than that. Whenever a
change adds or moves a DB write in the launch/stop path, state explicitly what the row looks
like *between* the writes and what the next tick will conclude from it.

**Debugging a "hung" job:** the DB row is bookkeeping and can lie. The trainer's
`<training_folder>/<job_name>/log.txt` is ground truth — if its tail is advancing, the job
is loading, not hung. H3 in particular takes 1–4 minutes to get through quantization and the
text encoder before the first step.

### CivitAI metadata

Sample outputs embed A1111-format metadata for CivitAI compatibility:
- **PNG:** `parameters` tEXt chunk via `PIL.PngImagePlugin.PngInfo`
- **MP4:** ffmpeg FFMETADATA1 re-mux with `-movflags use_metadata_tags` writing `parameters=` and `comment=` keys. Falls back to mutagen `©cmt` if ffmpeg fails.
- **Legacy WSL/NTFS fallback:** `os.replace()` used to fail on `/mnt/c/` paths under WSL, so the code catches `OSError` and copies bytes in-place instead (`toolkit/config_modules.py:1489`). Harmless on native Windows — kept as a fallback, no longer a live concern.

---

## Voice datasets (H3 audio training)

**Keep non-word TTS tags out of voice-clip captions unless they are genuinely rare.**
Sighs, breaths, and similar bracketed noises are learned as part of the character's
speech pattern, so their frequency in the captions becomes the character's
behavioural frequency. The original `brock_voice` dataset used them across many
clips and the trained character sighed before *everything* it said. It was recut as
`brock_voice_audio` rather than caption-edited, and `brock_voice` no longer exists.

Check tag frequency across captions *before* training, not after — the failure only
shows up at sample time, and it looks like a model problem rather than a data one.

`testing/test_voice_dataloader.py` runs against `brock_voice_audio` (12 clips on disk
plus a synthetic `handcut.wav` = the 13 items it asserts). It uses a `StubSD`, so it
needs no model weights and no GPU — it is the fastest way to confirm the audio-only
item path still works after a merge:

```bash
.venv/Scripts/python.exe testing/test_voice_dataloader.py
```

It covers the fork-only audio-only feature (`is_audio_only`,
`make_audio_only_placeholder_latent`, VAE bypass) *and* that audio still reaches the
batch through Ostris's `DTO` latent.

---

## Fork additions summary

See `README.md` "Fork additions" section for the full list. Key areas:

- **Save-before-pause** — `saveAndPauseJob()` sets both `save` and `stop` atomically; Python saves then stops in the same step hook.
- **On-demand save/sample** — trigger mid-training from the UI without stopping.
- **Stop during quantization** — `JobStoppedException` propagates through quantization loops via `maybe_stop()` hooks.
- **LightX2V** — fast WAN 2.2 sampling via two-stage PEFT adapter approach.
- **Gemma API** — LTX-2.3 training without loading the 12B text encoder locally.
- **UI additions** — drag-to-reorder queue, queue filter, negative prompt field, find/replace captions, loss graph persistence, sample grid placeholders.

---

## Key paths

Runtime paths come from the UI **Settings** table in `aitk_db.db`, not from this file — check
there if something looks wrong. Current values:

| Purpose | Path |
|---------|------|
| Training output | `C:\Data\AIToolkit-StagingArea\output\aitoolkit` (`TRAINING_FOLDER`) |
| Datasets | `C:\Data\AIToolkit-StagingArea\datasets` (`DATASETS_FOLDER`) |
| Models folder | `M:\models` (`MODELS_PATH`) |
| Quantization cache | `C:\Data\AIToolkit-StagingArea\quant` (`QUANTIZATION_CACHE_DIR`) |
| HuggingFace cache | `C:\Users\marc.bate\.cache\huggingface` (~213 GB) |
| UI runs on | `http://localhost:8675` |
| Python venv | `.venv\Scripts\python.exe` (Python 3.12, uv-managed) |

The Settings table also holds live secrets (HF token, Gemma API key, Check-Config API key).
Never copy those into this file, a config, or a commit.

### The `venv/` symlink is dead

`<repo>/venv` is still a symlink to `/home/marcbate/venvs/ai-toolkit` — a WSL path that no longer
exists. It is leftover clutter. Always use `.venv\Scripts\python.exe`; `source venv/bin/activate`
will not work.

`resolvePythonPath()` in `ui/cron/pythonPath.ts` is what the worker uses to find the interpreter
when spawning trainers.

### HuggingFace cache

Lives at `C:\Users\marc.bate\.cache\huggingface` on native NTFS. The old 400GB ext4 VHD
(`C:\Data\WSL\hf-cache.vhdx`, mounted at `/mnt/wsl/hfcache`) existed only to dodge the WSL drvfs
bandwidth cap (~225 MB/s across the bridge vs 13.3 GB/s on native ext4). That whole problem
disappeared with the move off WSL, and the VHD has been deleted — ignore any note about mounting
it or a logon scheduled task that attaches it.

The captioner project shares this same cache, along with the HF auth token.
