import type { ModelFamily } from './checkConfigMedia';

const FINDINGS_SCHEMA = `Return ONLY a JSON array of findings (no markdown fences, no prose outside the array). Each finding object:
{
  "field": string | null,           // dot-path into job_config, e.g. "config.process[0].train.lr" — null for non-config advice
  "current_value": string | null,   // stringified current value, or null
  "suggested_value": string | null, // stringified suggested value, or null
  "reason": string,                 // concise explanation
  "confidence": "high" | "medium" | "low",
  "references": string[],           // URLs or short source labels; may be empty for low confidence
  "applyable": boolean,             // true only when field + suggested_value are concrete and safe to patch
  "severity": "info" | "warning" | "error"
}

Confidence rules:
- high: vendor docs, maintainer posts, or hard rules below
- medium: Reddit / Discord / forum consensus
- low: reasoning by analogy — mark applyable only if the change is still concrete

Do not invent URLs. Prefer the web-search context when present.`;

const SHARED_RULES = `Shared rules for all model families:
1. CRITICAL — automagic3 + Qwen Image: if optimizer is "automagic3" AND model arch / name_or_path indicates Qwen Image (qwen_image*), emit severity=error, confidence=high, applyable=true suggesting adamw8bit (or adamw). Reason: automagic3 has no max_lr clamp; confirmed incident where LR ramped ~100× by step 1500 produced white-noise outputs.
2. Performance: if a GPU has memory_free_mb > 4096 and batch_size is low, suggest raising batch_size AND scaling LR linearly (batch×2 → LR×2). Flag VRAM pressure (memory_free_mb < 1024 while training), thermal throttling (temperature_c >= 85), and system RAM shortage (free_mb < 2048).
3. Loss curve: flag sustained spikes, NaN-like jumps, or flatlining near zero after early steps when loss_curve_last_200_steps is present.
4. Dataset stats: flag severe bucket fragmentation (many buckets with very few images) when dataset_stats is present.
5. Only suggest changes that match the JSON schema paths that exist in job_config. Prefer config.process[0].* paths.`;

const IMAGE_RULES = `Image-model focus:
- Watch for white-noise / mode-collapse signals in sample images when visual analysis is available (uniform noise, total loss of subject, identical outputs).
- Rank (network.linear), learning rate, optimizer choice, quantize/low_vram tradeoffs, resolution buckets, caption dropout.
- Do not recommend video-only or audio-only settings.`;

const VIDEO_RULES = `Video-model focus:
- Temporal consistency, num_frames / fps alignment between datasets and sample, LightX2V / distill_lora dual-LoRA setups when present.
- Frame-based dataset thresholds (datasets[*].num_frames, do_i2v, auto_frame_count).
- When visual analysis shows extracted frames, comment on temporal/subject consistency across frames.
- Do not recommend audio-only ACE-Step settings unless the job also trains audio (e.g. LTX do_audio).`;

const AUDIO_RULES = `Audio-model focus:
- Sample rate / BPM / keyscale / duration consistency in sample prompts and dataset captions.
- audio_lm_path and related audio model paths when present.
- Do NOT evaluate image white-noise metrics, VRAM image-batch heuristics tied to resolution buckets, or vision-only findings.
- visual_analysis is not applicable — ignore image fields.`;

const VISUAL_ADDENDUM = `
Visual analysis is ENABLED. Inspect attached sample/dataset images for:
- white noise / pure static
- mode collapse (near-identical outputs)
- severe artifacts, washed-out color, or total subject loss
- obvious dataset quality issues (corruption, tiny subjects, extreme crops)
Emit findings with field=null when the issue is diagnostic rather than a single config knob.`;

const NO_VISUAL_ADDENDUM = `
Visual analysis is NOT available for this request. Base findings only on job_config, dataset_stats, loss curve, and system_stats.`;

export function buildCheckConfigSystemPrompt(modelFamily: ModelFamily, hasImages: boolean): string {
  const familyRules =
    modelFamily === 'video' ? VIDEO_RULES : modelFamily === 'audio' ? AUDIO_RULES : IMAGE_RULES;

  const visual =
    modelFamily === 'audio'
      ? '\nAudio family: do not request or assume images.'
      : hasImages
        ? VISUAL_ADDENDUM
        : NO_VISUAL_ADDENDUM;

  return [
    'You are an expert LoRA / fine-tuning config reviewer for ostris/ai-toolkit (and this fork).',
    `Model family for this job: ${modelFamily}.`,
    FINDINGS_SCHEMA,
    SHARED_RULES,
    familyRules,
    visual,
    'Be concise. Prefer fewer high-signal findings over a long laundry list.',
  ].join('\n\n');
}
