import { describe, it, expect } from 'vitest';
import {
  stripMootModelSettings,
  stripMootTrainSettings,
  stripMootDatasetSettings,
} from './jobConfigSanitize';

function makeJobConfig(overrides: {
  model?: Record<string, any>;
  train?: Record<string, any>;
  datasets?: Array<Record<string, any>>;
}): { config: { process: [{ model: Record<string, any>; train: Record<string, any>; datasets: Record<string, any>[] }] } } {
  return {
    config: {
      process: [
        {
          model: { arch: 'ltx2.5', ...overrides.model } as Record<string, any>,
          train: { ...overrides.train } as Record<string, any>,
          datasets: overrides.datasets ?? [],
        },
      ],
    },
  };
}

describe('stripMootModelSettings', () => {
  it('zeroes layer_offloading percents when layer_offloading is false', () => {
    const cfg = makeJobConfig({
      model: {
        layer_offloading: false,
        layer_offloading_transformer_percent: 0.45,
        layer_offloading_text_encoder_percent: 1,
      },
    });
    stripMootModelSettings(cfg);
    expect(cfg.config.process[0].model.layer_offloading_transformer_percent).toBe(0);
    expect(cfg.config.process[0].model.layer_offloading_text_encoder_percent).toBe(0);
  });

  it('leaves layer_offloading percents alone when layer_offloading is true', () => {
    const cfg = makeJobConfig({
      model: {
        layer_offloading: true,
        layer_offloading_transformer_percent: 0.45,
        layer_offloading_text_encoder_percent: 1,
      },
    });
    stripMootModelSettings(cfg);
    expect(cfg.config.process[0].model.layer_offloading_transformer_percent).toBe(0.45);
    expect(cfg.config.process[0].model.layer_offloading_text_encoder_percent).toBe(1);
  });

  it('leaves layer_offloading percents alone when layer_offloading is undefined', () => {
    // undefined is not the same as explicitly false - don't guess intent
    const cfg = makeJobConfig({
      model: { layer_offloading_transformer_percent: 0.45 },
    });
    stripMootModelSettings(cfg);
    expect(cfg.config.process[0].model.layer_offloading_transformer_percent).toBe(0.45);
  });

  it('does not touch percents that are already 0', () => {
    const cfg = makeJobConfig({
      model: {
        layer_offloading: false,
        layer_offloading_transformer_percent: 0,
      },
    });
    const before = cfg.config.process[0].model.layer_offloading_transformer_percent;
    stripMootModelSettings(cfg);
    expect(cfg.config.process[0].model.layer_offloading_transformer_percent).toBe(before);
  });

  it('does not add percent fields that were never present', () => {
    const cfg = makeJobConfig({ model: { layer_offloading: false } });
    stripMootModelSettings(cfg);
    expect('layer_offloading_transformer_percent' in cfg.config.process[0].model).toBe(false);
    expect('layer_offloading_text_encoder_percent' in cfg.config.process[0].model).toBe(false);
  });

  it('resets qtype to qfloat8 when quantize is false', () => {
    const cfg = makeJobConfig({ model: { quantize: false, qtype: 'convrot8' } });
    stripMootModelSettings(cfg);
    expect(cfg.config.process[0].model.qtype).toBe('qfloat8');
  });

  it('leaves qtype alone when quantize is true', () => {
    const cfg = makeJobConfig({ model: { quantize: true, qtype: 'convrot8' } });
    stripMootModelSettings(cfg);
    expect(cfg.config.process[0].model.qtype).toBe('convrot8');
  });

  it('resets qtype_te to qfloat8 when quantize_te is false, independent of quantize/qtype', () => {
    const cfg = makeJobConfig({
      model: { quantize: true, qtype: 'convrot8', quantize_te: false, qtype_te: 'nvfp4' },
    });
    stripMootModelSettings(cfg);
    expect(cfg.config.process[0].model.qtype).toBe('convrot8'); // untouched
    expect(cfg.config.process[0].model.qtype_te).toBe('qfloat8'); // reset
  });

  it('applies multiple independent rules in the same pass', () => {
    const cfg = makeJobConfig({
      model: {
        layer_offloading: false,
        layer_offloading_transformer_percent: 0.5,
        quantize: false,
        qtype: 'convrot8',
      },
    });
    stripMootModelSettings(cfg);
    expect(cfg.config.process[0].model.layer_offloading_transformer_percent).toBe(0);
    expect(cfg.config.process[0].model.qtype).toBe('qfloat8');
  });

  it('resets all five compile-related fields when compile is false', () => {
    const cfg = makeJobConfig({
      model: {
        compile: false,
        block_compile: true,
        compile_mode: 'reduce-overhead',
        compile_fullgraph: true,
        compile_dynamic: false,
        cache_size_limit: 256,
      },
    });
    stripMootModelSettings(cfg);
    const m = cfg.config.process[0].model;
    expect(m.block_compile).toBe(false);
    expect(m.compile_mode).toBe('default');
    expect(m.compile_fullgraph).toBe(false);
    expect(m.compile_dynamic).toBe(true);
    expect(m.cache_size_limit).toBe(null);
  });

  it('leaves compile-related fields alone when compile is true', () => {
    const cfg = makeJobConfig({
      model: {
        compile: true,
        block_compile: true,
        compile_mode: 'reduce-overhead',
        compile_fullgraph: true,
        compile_dynamic: false,
        cache_size_limit: 256,
      },
    });
    stripMootModelSettings(cfg);
    const m = cfg.config.process[0].model;
    expect(m.block_compile).toBe(true);
    expect(m.compile_mode).toBe('reduce-overhead');
    expect(m.compile_fullgraph).toBe(true);
    expect(m.compile_dynamic).toBe(false);
    expect(m.cache_size_limit).toBe(256);
  });

  it('does not add compile fields that were never present', () => {
    const cfg = makeJobConfig({ model: { compile: false } });
    stripMootModelSettings(cfg);
    const m = cfg.config.process[0].model;
    for (const key of ['block_compile', 'compile_mode', 'compile_fullgraph', 'compile_dynamic', 'cache_size_limit']) {
      expect(key in m).toBe(false);
    }
  });

  it('does not re-reset a cache_size_limit that is already null', () => {
    const cfg = makeJobConfig({ model: { compile: false, cache_size_limit: null } });
    stripMootModelSettings(cfg);
    expect(cfg.config.process[0].model.cache_size_limit).toBe(null);
  });

  it('is a no-op and does not throw on a missing model section', () => {
    const cfg: any = { config: { process: [{}] } };
    expect(() => stripMootModelSettings(cfg)).not.toThrow();
  });

  it('is a no-op and does not throw on a completely empty object', () => {
    expect(() => stripMootModelSettings({})).not.toThrow();
    expect(() => stripMootModelSettings(null)).not.toThrow();
    expect(() => stripMootModelSettings(undefined)).not.toThrow();
  });

  it('returns the same object it was given (mutates in place)', () => {
    const cfg = makeJobConfig({ model: { layer_offloading: false, layer_offloading_transformer_percent: 1 } });
    const result = stripMootModelSettings(cfg);
    expect(result).toBe(cfg);
  });
});

describe('stripMootTrainSettings', () => {
  it('resets diff_output_preservation_class and _multiplier when the toggle is off', () => {
    const cfg = makeJobConfig({
      train: {
        diff_output_preservation: false,
        diff_output_preservation_class: 'a specific dog breed',
        diff_output_preservation_multiplier: 2.5,
      },
    });
    stripMootTrainSettings(cfg);
    expect(cfg.config.process[0].train.diff_output_preservation_class).toBe('person');
    expect(cfg.config.process[0].train.diff_output_preservation_multiplier).toBe(1.0);
  });

  it('leaves them alone when diff_output_preservation is on', () => {
    const cfg = makeJobConfig({
      train: {
        diff_output_preservation: true,
        diff_output_preservation_class: 'a specific dog breed',
        diff_output_preservation_multiplier: 2.5,
      },
    });
    stripMootTrainSettings(cfg);
    expect(cfg.config.process[0].train.diff_output_preservation_class).toBe('a specific dog breed');
    expect(cfg.config.process[0].train.diff_output_preservation_multiplier).toBe(2.5);
  });

  it('does not throw on a missing train section', () => {
    const cfg: any = { config: { process: [{}] } };
    expect(() => stripMootTrainSettings(cfg)).not.toThrow();
  });
});

describe('stripMootDatasetSettings', () => {
  it('resets num_frames to the ltx2.5 default (121) when auto_frame_count is on', () => {
    const cfg = makeJobConfig({
      model: { arch: 'ltx2.5' },
      datasets: [{ auto_frame_count: true, num_frames: 39 }],
    });
    stripMootDatasetSettings(cfg);
    expect(cfg.config.process[0].datasets[0].num_frames).toBe(121);
  });

  it('resets to 81 for wan21', () => {
    const cfg = makeJobConfig({
      model: { arch: 'wan21' },
      datasets: [{ auto_frame_count: true, num_frames: 16 }],
    });
    stripMootDatasetSettings(cfg);
    expect(cfg.config.process[0].datasets[0].num_frames).toBe(81);
  });

  it('resets to 81 for wan22 variants via prefix match', () => {
    for (const arch of ['wan22_14b', 'wan22_14b_i2v', 'wan22_5b']) {
      const cfg = makeJobConfig({
        model: { arch },
        datasets: [{ auto_frame_count: true, num_frames: 16 }],
      });
      stripMootDatasetSettings(cfg);
      expect(cfg.config.process[0].datasets[0].num_frames).toBe(81);
    }
  });

  it('resets to 121 for ltx2 and ltx2.3 too, not just ltx2.5', () => {
    for (const arch of ['ltx2', 'ltx2.3', 'ltx2.5']) {
      const cfg = makeJobConfig({
        model: { arch },
        datasets: [{ auto_frame_count: true, num_frames: 39 }],
      });
      stripMootDatasetSettings(cfg);
      expect(cfg.config.process[0].datasets[0].num_frames).toBe(121);
    }
  });

  it('resets to 124 for minimax_h3', () => {
    const cfg = makeJobConfig({
      model: { arch: 'minimax_h3' },
      datasets: [{ auto_frame_count: true, num_frames: 39 }],
    });
    stripMootDatasetSettings(cfg);
    expect(cfg.config.process[0].datasets[0].num_frames).toBe(124);
  });

  it('leaves num_frames alone when auto_frame_count is off', () => {
    const cfg = makeJobConfig({
      model: { arch: 'ltx2.5' },
      datasets: [{ auto_frame_count: false, num_frames: 39 }],
    });
    stripMootDatasetSettings(cfg);
    expect(cfg.config.process[0].datasets[0].num_frames).toBe(39);
  });

  it('leaves num_frames alone for an unknown architecture', () => {
    const cfg = makeJobConfig({
      model: { arch: 'some_future_model' },
      datasets: [{ auto_frame_count: true, num_frames: 39 }],
    });
    stripMootDatasetSettings(cfg);
    expect(cfg.config.process[0].datasets[0].num_frames).toBe(39);
  });

  it('handles multiple datasets independently, only touching the ones with auto_frame_count on', () => {
    const cfg = makeJobConfig({
      model: { arch: 'ltx2.5' },
      datasets: [
        { auto_frame_count: true, num_frames: 39 },
        { auto_frame_count: false, num_frames: 39 },
        { auto_frame_count: true, num_frames: 121 }, // already at the default: untouched, not just "same result"
      ],
    });
    stripMootDatasetSettings(cfg);
    const [a, b, c] = cfg.config.process[0].datasets;
    expect(a.num_frames).toBe(121);
    expect(b.num_frames).toBe(39);
    expect(c.num_frames).toBe(121);
  });

  it('does not throw when datasets is missing or not an array', () => {
    expect(() => stripMootDatasetSettings(makeJobConfig({}))).not.toThrow();
    const cfg: any = { config: { process: [{ model: { arch: 'ltx2.5' }, datasets: 'not-an-array' }] } };
    expect(() => stripMootDatasetSettings(cfg)).not.toThrow();
  });

  it('does not throw on malformed entries inside the datasets array', () => {
    const cfg = makeJobConfig({
      model: { arch: 'ltx2.5' },
      datasets: [null as any, undefined as any, { auto_frame_count: true, num_frames: 39 }],
    });
    expect(() => stripMootDatasetSettings(cfg)).not.toThrow();
    expect(cfg.config.process[0].datasets[2].num_frames).toBe(121);
  });
});
