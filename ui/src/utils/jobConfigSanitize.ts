// Settings that have no effect while their gating flag is off. Left alone, a
// saved config can read as "this is active" when it isn't - we hit this for
// real with layer_offloading_transformer_percent surviving at a stale
// non-zero value while the master layer_offloading flag was actually false.
//
// Each entry: when `gate` on `model` is strictly `false`, reset every path in
// `fields` (if currently set to something other than `resetTo`) to `resetTo`.
// `resetTo` matches the backend's own default for that field (see
// toolkit/config_modules.py ModelConfig.__init__) so a re-enabled toggle
// starts from the same value the trainer would have used anyway - never an
// arbitrary sentinel that could confuse a later read.
type MootRule = {
  gate: string;
  fields: string[];
  resetTo: unknown;
};

const MOOT_MODEL_RULES: MootRule[] = [
  {
    gate: 'layer_offloading',
    fields: ['layer_offloading_transformer_percent', 'layer_offloading_text_encoder_percent'],
    resetTo: 0,
  },
  {
    gate: 'quantize',
    fields: ['qtype'],
    resetTo: 'qfloat8',
  },
  {
    gate: 'quantize_te',
    fields: ['qtype_te'],
    resetTo: 'qfloat8',
  },
  // All five only get read inside `if self.model_config.compile:`
  // (jobs/process/BaseSDTrainProcess.py:2272) - dead otherwise.
  { gate: 'compile', fields: ['block_compile'], resetTo: false },
  { gate: 'compile', fields: ['compile_mode'], resetTo: 'default' },
  { gate: 'compile', fields: ['compile_fullgraph'], resetTo: false },
  { gate: 'compile', fields: ['compile_dynamic'], resetTo: true },
  { gate: 'compile', fields: ['cache_size_limit'], resetTo: null },
];

/**
 * Mutates (and returns) jobConfig, zeroing/resetting model-level settings that
 * are inert given their gating flag's current value. Safe to call on any
 * shape - missing config/process/model/fields are all no-ops, never throws.
 */
export function stripMootModelSettings(jobConfig: any): any {
  const model = jobConfig?.config?.process?.[0]?.model;
  if (!model || typeof model !== 'object') {
    return jobConfig;
  }

  for (const rule of MOOT_MODEL_RULES) {
    if (model[rule.gate] !== false) {
      continue;
    }
    for (const field of rule.fields) {
      if (model[field] !== undefined && model[field] !== rule.resetTo) {
        model[field] = rule.resetTo;
      }
    }
  }

  return jobConfig;
}

// dataset.num_frames is overwritten at runtime whenever auto_frame_count is
// on (toolkit/dataloader_mixins.py:578-579), so a stale explicit value there
// is misleading the same way the model settings above were - but UNLIKE
// those, there's no single safe "off" value: a frame count is a real,
// architecture-specific preference the user wants back the moment they
// disable auto_frame_count, so this can't reset to an arbitrary placeholder
// without risking real data loss. Resetting to the architecture's own
// documented default (not some generic constant) is the one value that's
// both meaningful and safe to land on.
const ARCH_DEFAULT_NUM_FRAMES: Array<{ prefix: string; num_frames: number }> = [
  { prefix: 'wan', num_frames: 81 },
  { prefix: 'ltx2', num_frames: 121 }, // covers ltx2, ltx2.3, ltx2.5
  { prefix: 'minimax_h3', num_frames: 124 },
];

function archDefaultNumFrames(arch: unknown): number | undefined {
  if (typeof arch !== 'string') return undefined;
  const match = ARCH_DEFAULT_NUM_FRAMES.find(entry => arch.startsWith(entry.prefix));
  return match?.num_frames;
}

const MOOT_TRAIN_RULES: MootRule[] = [
  {
    gate: 'diff_output_preservation',
    fields: ['diff_output_preservation_class'],
    resetTo: 'person',
  },
  {
    gate: 'diff_output_preservation',
    fields: ['diff_output_preservation_multiplier'],
    resetTo: 1.0,
  },
];

/**
 * Mutates (and returns) jobConfig, resetting train-level settings that are
 * inert while diff_output_preservation is off, back to the same defaults a
 * fresh job would start with (jobConfig.ts's own scaffold) - not an
 * arbitrary placeholder, so re-enabling the toggle lands somewhere sane.
 */
export function stripMootTrainSettings(jobConfig: any): any {
  const train = jobConfig?.config?.process?.[0]?.train;
  if (!train || typeof train !== 'object') {
    return jobConfig;
  }

  for (const rule of MOOT_TRAIN_RULES) {
    if (train[rule.gate] !== false) {
      continue;
    }
    for (const field of rule.fields) {
      if (train[field] !== undefined && train[field] !== rule.resetTo) {
        train[field] = rule.resetTo;
      }
    }
  }

  return jobConfig;
}

/**
 * Mutates (and returns) jobConfig, resetting each dataset's num_frames to the
 * model architecture's documented default whenever that dataset has
 * auto_frame_count on (making the explicit value inert). No-op if the arch
 * isn't one we have a known default for, or the value already matches.
 */
export function stripMootDatasetSettings(jobConfig: any): any {
  const process = jobConfig?.config?.process?.[0];
  const datasets = process?.datasets;
  if (!Array.isArray(datasets)) {
    return jobConfig;
  }

  const defaultFrames = archDefaultNumFrames(process?.model?.arch);
  if (defaultFrames === undefined) {
    return jobConfig;
  }

  for (const dataset of datasets) {
    if (!dataset || typeof dataset !== 'object') continue;
    if (dataset.auto_frame_count !== true) continue;
    if (dataset.num_frames !== undefined && dataset.num_frames !== defaultFrames) {
      dataset.num_frames = defaultFrames;
    }
  }

  return jobConfig;
}
