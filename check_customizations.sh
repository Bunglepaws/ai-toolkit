#!/usr/bin/env bash
# check_customizations.sh
# Run after any upstream merge to verify our fork customizations are intact.
# Usage: bash check_customizations.sh
# Exit code 0 = all good, 1 = something is missing.

REPO="$(cd "$(dirname "$0")" && pwd)"
UI="$REPO/ui/src"
EXT="$REPO/extensions_built_in"
TK="$REPO/toolkit"

PASS=0
FAIL=0
FAILURES=()

check() {
  local desc="$1"
  local file="$2"
  local pattern="$3"

  if grep -q "$pattern" "$REPO/$file" 2>/dev/null; then
    echo "  ✓  $desc"
    ((PASS++))
  else
    echo "  ✗  MISSING: $desc"
    echo "       file: $file"
    echo "       grep: $pattern"
    ((FAIL++))
    FAILURES+=("$desc")
  fi
}

echo
echo "========== Fork customization check =========="
echo

echo "── UI: JobOverview ──────────────────────────"
check "Log panel fill-height (85vh)" \
  "ui/src/components/JobOverview.tsx" \
  "fill-height"

echo
echo "── UI: SampleImageCard ──────────────────────"
check "containerType:size for step label cqmin sizing" \
  "ui/src/components/SampleImageCard.tsx" \
  "containerType.*size"

echo
echo "── UI: SampleImages ─────────────────────────"
check "sampleSlots sparse array logic" \
  "ui/src/components/SampleImages.tsx" \
  "sampleSlots"
check "NOT SAMPLED placeholder cells" \
  "ui/src/components/SampleImages.tsx" \
  "NOT SAMPLED"
check "steps array for row step labels" \
  "ui/src/components/SampleImages.tsx" \
  "rowStepLabel"
check "stepLabel passed to SampleImageCard" \
  "ui/src/components/SampleImages.tsx" \
  "stepLabel="

echo
echo "── UI: SampleImageViewer ────────────────────"
check "Expandable prompt (promptExpanded state)" \
  "ui/src/components/SampleImageViewer.tsx" \
  "promptExpanded"
check "Reads prompt from file metadata" \
  "ui/src/components/SampleImageViewer.tsx" \
  "metadataPrompt"

check "Sticky video mute hook wired in"   "ui/src/components/SampleImageViewer.tsx"   "videoMute"

echo
echo "── UI: DatasetImageViewer ───────────────────"
check "Sticky video mute hook wired in"   "ui/src/components/DatasetImageViewer.tsx"   "videoMute"

echo
echo "── UI: useVideoMute ─────────────────────────"
check "Global mute preference storage key"   "ui/src/hooks/useVideoMute.ts"   "aitk_video_muted"

echo
echo "── UI: JobActionBar ─────────────────────────"
check "SaveSnapshotModal import" \
  "ui/src/components/JobActionBar.tsx" \
  "openSaveSnapshotModal"
check "Save snapshot button (canSave)" \
  "ui/src/components/JobActionBar.tsx" \
  "canSave"
check "Generate samples button (canSample)" \
  "ui/src/components/JobActionBar.tsx" \
  "canSample"
check "Edit sample prompts while running (canEditSample)" \
  "ui/src/components/JobActionBar.tsx" \
  "canEditSample"
check "Save snapshot modal wired in JobActionBar" \
  "ui/src/components/JobActionBar.tsx" \
  "openSaveSnapshotModal"
check "sampleJob imported" \
  "ui/src/components/JobActionBar.tsx" \
  "sampleJob"

echo
echo "── UI: JobsTable ────────────────────────────"
check "Drag-to-reorder (draggedJobId)" \
  "ui/src/components/JobsTable.tsx" \
  "draggedJobId"
check "filter prop accepted" \
  "ui/src/components/JobsTable.tsx" \
  "filter.*string"

echo
echo "── UI: Jobs queue page ──────────────────────"
check "Filter textbox on queue page" \
  "ui/src/app/jobs/page.tsx" \
  "Filter by name"

echo
echo "── UI: Dashboard page ───────────────────────"
check "Filter textbox on dashboard page" \
  "ui/src/app/dashboard/page.tsx" \
  "Filter by name"

echo
echo "── UI: Job detail page ──────────────────────"
check "useJobsList called with object (not positional args)" \
  "ui/src/app/jobs/[jobID]/page.tsx" \
  "useJobsList.*onlyActive"
check "menuItem type includes hasSamples" \
  "ui/src/app/jobs/[jobID]/page.tsx" \
  "hasSamples.*boolean"

echo
echo "── UI: Job new/edit page ────────────────────"
check "Empty prompt validation before save" \
  "ui/src/app/jobs/new/page.tsx" \
  "emptyIdx"
check "Negative prompt field" \
  "ui/src/app/jobs/new/SimpleJob.tsx" \
  "[Nn]egative.*[Pp]rompt"

echo
echo "── UI: Types ────────────────────────────────"
check "use_gemma_api in ModelConfig" \
  "ui/src/types.ts" \
  "use_gemma_api"
check "gemma_api_key in ModelConfig" \
  "ui/src/types.ts" \
  "gemma_api_key"

echo
echo "── UI: formInputs ───────────────────────────"
check "onKeyDown prop in TextInput" \
  "ui/src/components/formInputs.tsx" \
  "onKeyDown"

echo
echo "── Python: Stop/Pause integrity ─────────────"
check "JobStoppedException defined" \
  "toolkit/ui_utils.py" \
  "class JobStoppedException"
check "maybe_stop in UITrainer" \
  "extensions_built_in/sd_trainer/UITrainer.py" \
  "def maybe_stop"
check "should_save reads 'save_now' column (ostris canonical schema)" \
  "extensions_built_in/sd_trainer/DiffusionTrainer.py" \
  "SELECT save_now FROM Job"
# Duplicate method definitions.
#
# This is the signature of a merge resolved by keeping BOTH sides. Python
# accepts it silently -- the later definition wins and the earlier one becomes
# dead code -- so py_compile passes and the marker greps above pass too (a
# duplicate makes a marker MORE present, not less). It has happened twice:
# should_save, and then should_sample, where the live and dead copies differed
# in whether the DB read was retried.
#
# Checked generically rather than per symbol: the point is to catch the next
# one without having been bitten by it first.
echo
echo "── Python: no duplicate method definitions ──"
for f in \
  "extensions_built_in/sd_trainer/DiffusionTrainer.py" \
  "extensions_built_in/sd_trainer/UITrainer.py" \
  "extensions_built_in/sd_trainer/SDTrainer.py" \
  "jobs/process/BaseSDTrainProcess.py"
do
  [[ -f "$REPO/$f" ]] || continue
  # method defs only (4-space indent), name captured, duplicates reported
  DUPES=$(grep -oE "^    def [a-zA-Z_][a-zA-Z0-9_]*" "$REPO/$f" 2>/dev/null \
            | sort | uniq -d | sed 's/^    def //' | tr '\n' ' ')
  if [[ -z "$DUPES" ]]; then
    echo "  ✓  $(basename "$f"): no duplicated methods"
    ((PASS++))
  else
    echo "  ✗  MISSING: $(basename "$f") defines these twice: $DUPES"
    echo "       a later definition silently shadows the earlier one"
    ((FAIL++))
    FAILURES+=("Duplicate methods in $(basename "$f"): $DUPES")
  fi
done

# Undefined names / redefinitions across the files a merge is most likely to
# touch. py_compile only proves a file PARSES -- it passed on the unloader
# calling an undefined _detach_and_cpu and an unimported flush, which then
# raised NameError at runtime. pyflakes catches both that and the duplicates
# above. Skipped (not failed) when pyflakes isn't installed.
echo
echo "── Python: pyflakes (undefined names) ───────"
PY="$REPO/.venv/Scripts/python.exe"
[[ -x "$PY" ]] || PY="python"
if "$PY" -m pyflakes --version >/dev/null 2>&1; then
  PF=$("$PY" -m pyflakes \
        "$REPO/extensions_built_in/sd_trainer/DiffusionTrainer.py" \
        "$REPO/extensions_built_in/sd_trainer/UITrainer.py" \
        "$REPO/toolkit/unloader.py" \
        2>/dev/null | grep -E "undefined name|redefinition of" || true)
  if [[ -z "$PF" ]]; then
    echo "  ✓  No undefined names or redefinitions"
    ((PASS++))
  else
    echo "  ✗  MISSING: pyflakes findings:"
    echo "$PF" | sed 's/^/       /'
    ((FAIL++))
    FAILURES+=("pyflakes: undefined names or redefinitions")
  fi
else
  echo "  ~  pyflakes not installed, skipping (pip install pyflakes)"
fi

echo
echo "── Python: CivitAI metadata ─────────────────"
check "_build_civitai_parameters in config_modules" \
  "toolkit/config_modules.py" \
  "_build_civitai_parameters"
check "_embed_mp4_metadata in config_modules" \
  "toolkit/config_modules.py" \
  "_embed_mp4_metadata"
check "WSL/NTFS os.replace fallback" \
  "toolkit/config_modules.py" \
  "copyfileobj"

echo
echo "── Python: Stop during quantization ─────────"
check "maybe_stop hooks in DiffusionTrainer" \
  "extensions_built_in/sd_trainer/DiffusionTrainer.py" \
  "maybe_stop"
check "JobStoppedException caught in run.py" \
  "run.py" \
  "JobStoppedException"

echo
echo "── Python: LightX2V ─────────────────────────"
check "_teardown_lightx2v_loras (LightX2V LoRA lifecycle management)" \
  "extensions_built_in/diffusion_models/wan22/wan22_14b_model.py" \
  "_teardown_lightx2v_loras"
check "_ensure_adapter_absent (safe LightX2V adapter cleanup)" \
  "extensions_built_in/diffusion_models/wan22/wan22_14b_model.py" \
  "_ensure_adapter_absent"

echo
echo "── Python: LTX-2.3 distilled LoRA ──────────"
check "distill_lora_path support in ltx2.py" \
  "extensions_built_in/diffusion_models/ltx2/ltx2.py" \
  "distill_lora"
# Checks the files that actually CALL it. It was previously asserted against
# ltx2.py, where the name only ever appeared on an import line and was never
# called - so the check passed on dead code while the real call sites went
# unguarded. The 2026-08-29 ostris merge dropped the import from qwen_image.py
# and that landed as a runtime NameError this check did not catch.
check "filter_lora_state_dict_for_quantized_model called in qwen_image.py (prevents weight corruption on quantized layers)" \
  "extensions_built_in/diffusion_models/qwen_image/qwen_image.py" \
  "lora_state_dict = filter_lora_state_dict_for_quantized_model"
check "filter_lora_state_dict_for_quantized_model called in wan22_14b_model.py (prevents weight corruption on quantized layers)" \
  "extensions_built_in/diffusion_models/wan22/wan22_14b_model.py" \
  "= filter_lora_state_dict_for_quantized_model"

echo
echo "── Python: MiniMax-H3 turbo LoRA ────────────"
check "turbo sampling LoRA in minimax_h3.py" \
  "extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py" \
  "_build_turbo_lora"
check "_detach_turbo_lora (unwraps forwards on file change)" \
  "extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py" \
  "_detach_turbo_lora"
check "sample.minimax_h3_turbo_lora section wired to minimax_h3" \
  "ui/src/app/jobs/new/options.tsx" \
  "sample.minimax_h3_turbo_lora"

echo
echo "── Python: Gemma API ────────────────────────"
check "gemma_api_key in ltx2.py" \
  "extensions_built_in/diffusion_models/ltx2/ltx2.py" \
  "gemma_api_key"

echo
echo "── UI: Datasets ─────────────────────────────"
check "Find & replace captions" \
  "ui/src/app/datasets/[datasetName]/page.tsx" \
  "findText\|replaceText"
check "Caption filtering" \
  "ui/src/app/datasets/[datasetName]/page.tsx" \
  "filteredImgList\|filterText"
check "Abort off-screen caption requests" \
  "ui/src/components/DatasetImageCard.tsx" \
  "AbortController"

echo
echo "── UI: Loss Graph ───────────────────────────"
check "Settings persistence (localStorage per job)" \
  "ui/src/components/JobLossGraph.tsx" \
  "localStorage\|STORAGE_PREFIX"

echo
echo "── Python: Sample image format ──────────────"
check "Default sample image format is jpg (not png)" \
  "toolkit/config_modules.py" \
  "default_ext.*jpg\|jpg.*default_ext"
check "JPEG CivitAI metadata embedding (UserComment EXIF)" \
  "toolkit/config_modules.py" \
  "UNICODE.*utf-16\|0x9286"

echo
echo "────────────────────────────────────────────"
echo
echo "Results: $PASS passed, $FAIL failed"
echo

if [[ $FAIL -gt 0 ]]; then
  echo "FAILED checks:"
  for f in "${FAILURES[@]}"; do
    echo "  - $f"
  done
  echo
  exit 1
else
  echo "All customizations intact."
  echo
  exit 0
fi
