'use client';
import { isMac } from '@/helpers/basic';
import { defaultSampleConfig } from '@/helpers/defaultSamples';
import { JobConfig, SampleConfig, DatasetConfig, SliderConfig, VoiceCloneConfig } from '@/types';

export const defaultDatasetConfig: DatasetConfig = {
  folder_path: '/path/to/images/folder',
  mask_path: null,
  mask_min_value: 0.1,
  default_caption: '',
  trigger_word: null,
  caption_ext: 'txt',
  caption_dropout_rate: 0.05,
  cache_latents_to_disk: false,
  is_reg: false,
  network_weight: 1,
  resolution: [512, 768, 1024],
  controls: [],
  shrink_video_to_frames: true,
  num_frames: 1,
  flip_x: false,
  flip_y: false,
  num_repeats: 1,
};

export const defaultSliderConfig: SliderConfig = {
  guidance_strength: 3.0,
  anchor_strength: 1.0,
  positive_prompt: 'person who is happy',
  negative_prompt: 'person who is sad',
  target_class: 'person',
  anchor_class: '',
};

export const defaultCompileOptions = {
  block_compile: true,
  compile_mode: 'default',
  compile_fullgraph: false,
};

// Mirrors DEFAULT_DIALOGUE in toolkit/voice_clone/sentences.py, which stays authoritative
// for configs written by hand or by the CLI (an empty dialogue there falls back to it).
// Prefilled here so the box shows what will actually be said rather than sitting blank.
// ~13-16 words each, which fills a 5.167s clip without sounding rushed or slowed.
export const defaultVoiceDialogue: string[] = [
  'Alright boys, hit the showers. You\'ve earned every bit of that one out there today.',
  'Nice work, big guy. I honestly didn\'t think you had that last set in you.',
  'Come here a second, let me get a look at that shoulder before you take off.',
  'Yeah, that\'s it. Slow, controlled, all the way down. Just like that, perfect.',
  'You\'ve been holding out on me. Where\'s all of this been hiding lately, huh?',
  'You\'re going to be the death of me one day, you know that, right?',
  'Towel off and come meet me in my office, we\'ve got some things to talk about.',
  'Well now, somebody\'s been putting in the extra work over the summer, haven\'t they?',
  'Chin up, chest out, own the room. That\'s how a champion walks in here.',
  'Lock the door behind you, I don\'t want anybody walking in on this tonight.',
  'You sure you can handle another round, or do you need a minute first?',
  'Good boy. That\'s exactly what I\'ve been wanting to see out of you.',
];

export const defaultVoiceCloneConfig: VoiceCloneConfig = {
  enabled: false,
  mode: 'clone',
  reference_path: '',
  reference_text: '',
  instruct: '',
  voice_seed_path: '',
  target_dataset: '',
  // 60s -> 12 clips at 5.167s. Just above the smallest measured-sufficient amount
  // (~45s across 9 clips on LTX-2.3); 120 matches the CoachBate recipe.
  target_seconds: 60,
  duration_mix: 'long',
  voice_description: 'a man speaking calmly, low pitch',
  trigger_word: '',
  dialogue: defaultVoiceDialogue,
  backend: 'omnivoice',
  seed: 42,
  regenerate_token: '',
};

export const defaultJobConfig: JobConfig = {
  job: 'extension',
  config: {
    name: 'my_first_lora_v1',
    process: [
      {
        type: 'diffusion_trainer',
        training_folder: 'output',
        sqlite_db_path: './aitk_db.db',
        device: 'cuda',
        trigger_word: null,
        performance_log_every: 10,
        network: {
          type: 'lora',
          linear: 32,
          linear_alpha: 32,
          conv: 16,
          conv_alpha: 16,
          lokr_full_rank: true,
          lokr_factor: -1,
          network_kwargs: {
            ignore_if_contains: [],
          },
        },
        save: {
          dtype: 'bf16',
          save_every: 250,
          max_step_saves_to_keep: 4,
          save_format: 'diffusers',
          push_to_hub: false,
          archive_optimizer: false,
        },
        datasets: [defaultDatasetConfig],
        train: {
          batch_size: 1,
          bypass_guidance_embedding: true,
          steps: 3000,
          gradient_accumulation: 1,
          train_unet: true,
          train_text_encoder: false,
          gradient_checkpointing: true,
          noise_scheduler: 'flowmatch',
          optimizer: 'adamw8bit',
          timestep_type: 'sigmoid',
          content_or_style: 'balanced',
          optimizer_params: {
            weight_decay: 1e-4,
          },
          unload_text_encoder: false,
          cache_text_embeddings: false,
          lr: 0.0001,
          ema_config: {
            use_ema: false,
            ema_decay: 0.99,
          },
          skip_first_sample: false,
          force_first_sample: false,
          disable_sampling: false,
          dtype: 'bf16',
          diff_output_preservation: false,
          diff_output_preservation_multiplier: 1.0,
          diff_output_preservation_class: 'person',
          switch_boundary_every: 1,
          loss_type: 'mse',
        },
        logging: {
          log_every: 1,
          use_ui_logger: true,
        },
        model: {
          name_or_path: 'ostris/Flex.1-alpha',
          quantize: true,
          qtype: 'qfloat8',
          quantize_te: true,
          qtype_te: 'qfloat8',
          cache_quantized_model: false,
          arch: 'flex1',
          low_vram: false,
          model_kwargs: {},
          compile: false,
        },
        sample: defaultSampleConfig,
        voice_clone: defaultVoiceCloneConfig,
      },
    ],
  },
  meta: {
    name: '[name]',
    version: '1.0',
  },
};

export const migrateJobConfig = (jobConfig: JobConfig): JobConfig => {
  // upgrade prompt strings to samples
  if (
    jobConfig?.config?.process &&
    jobConfig.config.process[0]?.sample &&
    Array.isArray(jobConfig.config.process[0].sample.prompts) &&
    jobConfig.config.process[0].sample.prompts.length > 0
  ) {
    let newSamples = [];
    for (const prompt of jobConfig.config.process[0].sample.prompts) {
      newSamples.push({
        prompt: prompt,
      });
    }
    jobConfig.config.process[0].sample.samples = newSamples;
    delete jobConfig.config.process[0].sample.prompts;
  }

  // upgrade job from ui_trainer to diffusion_trainer
  if (jobConfig?.config?.process && jobConfig.config.process[0]?.type === 'ui_trainer') {
    jobConfig.config.process[0].type = 'diffusion_trainer';
  }

  if ('auto_memory' in jobConfig.config.process[0].model) {
    jobConfig.config.process[0].model.layer_offloading = (jobConfig.config.process[0].model.auto_memory ||
      false) as boolean;
    delete jobConfig.config.process[0].model.auto_memory;
  }

  if (!('logging' in jobConfig.config.process[0])) {
    //@ts-ignore
    jobConfig.config.process[0].logging = {
      log_every: 1,
      use_ui_logger: true,
    };
  }
  if (isMac()) {
    jobConfig.config.process[0].device = 'mps';
  }

  return jobConfig;
};
