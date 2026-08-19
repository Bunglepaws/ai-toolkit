'use client';
import React, { useMemo, useState, useEffect, useRef, useCallback } from 'react';
import {
  modelArchs,
  ModelArch,
  groupedModelOptions,
  quantizationOptions,
  defaultQtype,
  jobTypeOptions,
  SampleTags,
  isOstrisBackedQtype,
  parseQtypeAra,
} from './options';
import { defaultCompileOptions, defaultDatasetConfig } from './jobConfig';
import { GroupedSelectOption, JobConfig, SelectOption } from '@/types';
import { objectCopy, tagsToObj, objToTags } from '@/utils/basic';
import {
  TextInput,
  TextAreaInput,
  SelectInput,
  Checkbox,
  FormGroup,
  NumberInput,
  SliderInput,
  CreatableSelectInput,
} from '@/components/formInputs';
import Card from '@/components/Card';
import { X, Copy, Wand2, SquareDashed, Info } from 'lucide-react';
import { openDoc } from '@/components/DocModal';
import { openUpsamplePromptsModal, toAspectRatio } from '@/components/UpsamplePromptsModal';
import { openPromptBoxEditor } from '@/components/PromptBoxEditorModal';
import AddSingleImageModal, { openAddImageModal } from '@/components/AddSingleImageModal';
import SampleControlImage from '@/components/SampleControlImage';
import { FlipHorizontal2, FlipVertical2 } from 'lucide-react';
import { handleModelArchChange } from './utils';
import { IoFlaskSharp } from 'react-icons/io5';
import { isMac } from '@/helpers/basic';
import {
  defaultFrameGrid,
  durationToFrameCount,
  frameCountToDuration,
  formatDuration,
  isUntestedFps,
} from '@/helpers/videoFrames';
import useSettings from '@/hooks/useSettings';

const MRU_LORA_KEY = 'aitk_mru_lora_paths';
const MRU_MAX = 8;

function getMruLoraList(): string[] {
  try {
    const raw = localStorage.getItem(MRU_LORA_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function addMruLoraPath(path: string) {
  if (!path?.trim()) return;
  const list = getMruLoraList().filter(p => p !== path);
  list.unshift(path.trim());
  localStorage.setItem(MRU_LORA_KEY, JSON.stringify(list.slice(0, MRU_MAX)));
}

function LoraPathInput({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  const [open, setOpen] = useState(false);
  const [mru, setMru] = useState<string[]>([]);
  const [dropdownRect, setDropdownRect] = useState<{ top: number; left: number; width: number } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setMru(getMruLoraList());
  }, []);

  const handleFocus = useCallback(() => {
    if (inputRef.current) {
      const r = inputRef.current.getBoundingClientRect();
      setDropdownRect({ top: r.bottom + 2, left: r.left, width: r.width });
    }
    setMru(getMruLoraList());
    setOpen(true);
  }, []);

  const handleBlur = useCallback(() => {
    setTimeout(() => {
      setOpen(false);
      if (value?.trim()) {
        addMruLoraPath(value);
        setMru(getMruLoraList());
      }
    }, 150);
  }, [value]);

  return (
    <div className="relative">
      <label className="block text-sm font-medium text-gray-300 mb-1">{label}</label>
      <input
        ref={inputRef}
        type="text"
        className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
        placeholder={placeholder ?? 'Path or HuggingFace repo/file'}
        value={value}
        onChange={e => onChange(e.target.value)}
        onFocus={handleFocus}
        onBlur={handleBlur}
      />
      {open && mru.length > 0 && dropdownRect && (
        <div
          style={{ position: 'fixed', top: dropdownRect.top, left: dropdownRect.left, width: dropdownRect.width, zIndex: 9999 }}
          className="bg-gray-800 border border-gray-600 rounded shadow-xl max-h-48 overflow-y-auto"
        >
          {mru.map(p => (
            <div
              key={p}
              className="px-3 py-2 text-sm text-gray-200 hover:bg-gray-700 cursor-pointer truncate"
              title={p}
              onMouseDown={e => {
                e.preventDefault();
                onChange(p);
                setOpen(false);
              }}
            >
              {p}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

type Props = {
  jobConfig: JobConfig;
  setJobConfig: (value: any, key: string) => void;
  status: 'idle' | 'saving' | 'success' | 'error';
  handleSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
  runId: string | null;
  gpuIDs: string | null;
  setGpuIDs: (value: string | null) => void;
  gpuList: any;
  datasetOptions: any;
  isLoading?: boolean;
  sampleOnly?: boolean;
};

const isDev = process.env.NODE_ENV === 'development';

export default function SimpleJob({
  jobConfig,
  setJobConfig,
  handleSubmit,
  status,
  runId,
  gpuIDs,
  setGpuIDs,
  gpuList,
  datasetOptions,
  isLoading,
  sampleOnly = false,
}: Props) {
  const modelArch = useMemo(() => {
    return modelArchs.find(a => a.name === jobConfig.config.process[0].model.arch) as ModelArch;
  }, [jobConfig.config.process[0].model.arch]);

  const { settings: globalSettings } = useSettings();
  const gemmaApiKeyConfigured = !!(globalSettings.GEMMA_API_KEY && globalSettings.GEMMA_API_KEY.trim() !== '');
  const useGemmaApi = !!(jobConfig.config.process[0].model.use_gemma_api);

  const jobType = useMemo(() => {
    return jobTypeOptions.find(j => j.value === jobConfig.config.process[0].type);
  }, [jobConfig.config.process[0].type]);

  const disableSections = useMemo(() => {
    let sections: string[] = [];
    if (modelArch?.disableSections) {
      sections = sections.concat(modelArch.disableSections);
    }
    if (jobType?.disableSections) {
      sections = sections.concat(jobType.disableSections);
    }
    return sections;
  }, [modelArch, jobType]);

  const isVideoModel = !!(modelArch?.group === 'video');

  // Video samples are configured in seconds; the frame count is derived from
  // duration x fps and snapped onto the architecture's frame grid (WAN n+1,
  // LTX 8n+1, H3 17n+5). num_frames stays in the config as the value the
  // trainer reads. Jobs saved before `duration` existed back-fill from it.
  const frameGrid = modelArch?.frameGrid ?? defaultFrameGrid;
  const sampleFps = jobConfig.config.process[0].sample.fps;
  const sampleDuration = useMemo(() => {
    const stored = jobConfig.config.process[0].sample.duration;
    if (stored !== undefined && stored !== null) return stored;
    return frameCountToDuration(jobConfig.config.process[0].sample.num_frames, sampleFps);
  }, [
    jobConfig.config.process[0].sample.duration,
    jobConfig.config.process[0].sample.num_frames,
    sampleFps,
  ]);
  const sampleFrameCount = useMemo(
    () => durationToFrameCount(sampleDuration, sampleFps, frameGrid),
    [sampleDuration, sampleFps, frameGrid],
  );
  const fpsUntested = isUntestedFps(sampleFps, modelArch?.supportedFps);

  const setSampleDuration = useCallback(
    (duration: number | null, fps: number = sampleFps) => {
      const seconds = duration ?? 0;
      setJobConfig(seconds, 'config.process[0].sample.duration');
      setJobConfig(durationToFrameCount(seconds, fps, frameGrid), 'config.process[0].sample.num_frames');
    },
    [setJobConfig, sampleFps, frameGrid],
  );
  // Clone Voice card. Clip count mirrors the backend: ceil(seconds / 5.167), the longest
  // legal H3 slot -- shown live so the seconds field is not abstract.
  // MERGE with defaults, don't only fall back when absent. Toggling the enable checkbox
  // writes a nested path, so a config that never had a voice_clone block ends up with a
  // PARTIAL one -- and a missing `dialogue` array throws on .join(), which the page's
  // ErrorBoundary surfaces as the misleading "Advanced job detected".
  const vc: VoiceCloneConfig = {
    ...defaultVoiceCloneConfig,
    ...(jobConfig.config.process[0].voice_clone ?? {}),
  };
  const vcDialogue = Array.isArray(vc.dialogue) ? vc.dialogue : [];
  const voiceClipCount = Math.max(1, Math.ceil((vc.target_seconds || 0) / 5.167));
  const voiceDialogueRef = useRef<HTMLTextAreaElement>(null);
  // A voice dataset holds audio-only items: their video side is a zeros placeholder pinned to
  // voice_placeholder_size, never bucketed and never trained on. So resolution, frame count,
  // crop/flip and control pickers are all inert for it -- hide them rather than leave settings
  // that look live and do nothing.
  const isVoiceDataset = (d: { folder_path?: string }) =>
    !!vc.enabled && !!vc.target_dataset && d.folder_path === vc.target_dataset;

  // Picking a target folder also wires it in as a dataset. Generating clips into a folder
  // no dataset entry references is a silent no-op -- the run trains as if the voice were
  // never made -- and this card is the only place with the context to prevent it.
  // Add/ensure only: never removes a dataset, since the user may have put images there.
  const setVoiceTargetDataset = (folderPath: string) => {
    setJobConfig(folderPath, 'config.process[0].voice_clone.target_dataset');
    if (!folderPath) return;
    const datasets = objectCopy(jobConfig.config.process[0].datasets) as any[];
    const idx = datasets.findIndex(d => d.folder_path === folderPath);
    if (idx === -1) {
      const newDataset = objectCopy(defaultDatasetConfig) as any;
      newDataset.folder_path = folderPath;
      newDataset.controls = modelArch?.controls ?? [];
      // voice items are audio-only; both are required or the dataset refuses to load
      newDataset.do_audio = true;
      newDataset.cache_latents_to_disk = true;
      datasets.push(newDataset);
    } else {
      datasets[idx].do_audio = true;
      datasets[idx].cache_latents_to_disk = true;
    }
    setJobConfig(datasets, 'config.process[0].datasets');
  };
  const isAudioModel = !!(modelArch?.group === 'audio');

  const taggedSampleArr: Record<string, any>[] | null = useMemo(() => {
    if (!modelArch) return null;
    if (!modelArch.sampleTags) return null;
    if (!jobConfig.config.process[0].sample.samples) return null;
    let sampleArr: any[] = [];
    for (let i = 0; i < jobConfig.config.process[0].sample.samples.length; i++) {
      const taggedPrompt = jobConfig.config.process[0].sample.samples[i].prompt;
      const tagsObj = tagsToObj(taggedPrompt);
      sampleArr.push(tagsObj);
    }
    return sampleArr;
  }, [modelArch, jobConfig.config.process[0].sample.samples]);

  const modelArchTagSections: SampleTags[] | null = useMemo(() => {
    if (!modelArch?.sampleTags) return null;
    const maxPerGroup = 5;
    let sections: SampleTags[] = [];
    let subSection: SampleTags = {};
    for (const [tagKey, tag] of Object.entries(modelArch.sampleTags)) {
      if ((tag.full && Object.keys(subSection).length > 0) || Object.keys(subSection).length >= maxPerGroup) {
        // reset the sub section build if the next tag is full or max per group is reached
        sections.push(subSection);
        subSection = {};
      }
      subSection[tagKey] = tag;
      if (tag.full) {
        // if the tag is full, push the section immediately and reset the sub section build
        sections.push(subSection);
        subSection = {};
      }
    }
    if (Object.keys(subSection).length > 0) {
      sections.push(subSection);
    }
    return sections.length > 0 ? sections : null;
  }, [modelArch]);

  const numTopCards = useMemo(() => {
    let count = 4; // job settings, model config, target config, save config
    if (modelArch?.additionalSections?.includes('model.multistage')) {
      count += 1; // add multistage card
    }
    if (!disableSections.includes('model.quantize')) {
      count += 1; // add quantization card
    }
    if (!disableSections.includes('slider')) {
      count += 1; // add slider card
    }
    return count;
  }, [modelArch, disableSections]);

  let topBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-4 gap-6';

  if (numTopCards == 5) {
    topBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6';
  }
  if (numTopCards == 6) {
    topBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3 2xl:grid-cols-6 gap-6';
  }

  const numTrainingCols = useMemo(() => {
    let count = 4;
    if (!disableSections.includes('train.diff_output_preservation')) {
      count += 1;
    }
    return count;
  }, [disableSections]);

  let trainingBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6';

  if (numTrainingCols == 5) {
    trainingBarClass = 'grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6';
  }

  const transformerQuantizationOptions: GroupedSelectOption[] | SelectOption[] = useMemo(() => {
    const hasARA = modelArch?.accuracyRecoveryAdapters && Object.keys(modelArch.accuracyRecoveryAdapters).length > 0;
    if (!hasARA) {
      return quantizationOptions;
    }
    let newQuantizationOptions = [
      {
        label: 'Standard',
        options: [quantizationOptions[0], quantizationOptions[1]],
      },
    ];

    // add ARAs if they exist for the model
    let ARAs: SelectOption[] = [];
    if (modelArch.accuracyRecoveryAdapters) {
      for (const [label, value] of Object.entries(modelArch.accuracyRecoveryAdapters)) {
        ARAs.push({ value, label });
      }
    }
    if (ARAs.length > 0) {
      newQuantizationOptions.push({
        label: 'Accuracy Recovery Adapters',
        options: ARAs,
      });
    }

    let additionalQuantizationOptions: SelectOption[] = [];
    // add the quantization options if they are not already included
    for (let i = 2; i < quantizationOptions.length; i++) {
      const option = quantizationOptions[i];
      additionalQuantizationOptions.push(option);
    }
    if (additionalQuantizationOptions.length > 0) {
      newQuantizationOptions.push({
        label: 'Additional Quantization Options',
        options: additionalQuantizationOptions,
      });
    }
    return newQuantizationOptions;
  }, [modelArch]);

  const showGPUSelect = !isMac();

  const validationConfig = jobConfig.config.process[0].train.validation_config;

  let numDatasetCols = 4;
  let numSampleTopCols = 4;
  let datasetStyleClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6';
  let sampleTopStyleClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6';
  if (isVideoModel) {
    numSampleTopCols += 1;
  }
  if (isAudioModel) {
    numDatasetCols -= 1;
    numSampleTopCols -= 1;
  }
  if (numDatasetCols == 3) {
    datasetStyleClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6';
  }
  if (numSampleTopCols == 5) {
    sampleTopStyleClass = 'grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6';
  }
  if (numSampleTopCols == 3) {
    sampleTopStyleClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6';
  }
  return (
    <>
      <form
        onSubmit={handleSubmit}
        className={`space-y-8 relative ${isLoading ? 'pointer-events-none opacity-50' : ''}`}
      >
        {!sampleOnly && isLoading && (
          <div className="absolute inset-0 z-50 flex items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-gray-400 border-t-blue-500" />
              <span className="text-sm text-gray-400">Loading...</span>
            </div>
          </div>
        )}
        <div className={`${topBarClass} ${sampleOnly ? 'hidden' : ''}`}>
          <Card title="Job">
            <TextInput
              label="Training Name"
              value={jobConfig.config.name}
              docKey="config.name"
              onChange={value => setJobConfig(value, 'config.name')}
              placeholder="Enter training name"
              disabled={runId !== null}
              required
            />
            {showGPUSelect && (
              <SelectInput
                label="GPU ID"
                value={`${gpuIDs}`}
                docKey="gpuids"
                onChange={value => setGpuIDs(value)}
                options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
              />
            )}
            {disableSections.includes('trigger_word') ? null : (
              <TextInput
                label="Trigger Word"
                value={jobConfig.config.process[0].trigger_word || ''}
                docKey="config.process[0].trigger_word"
                onChange={(value: string | null) => {
                  if (value?.trim() === '') {
                    value = null;
                  }
                  setJobConfig(value, 'config.process[0].trigger_word');
                }}
                placeholder=""
                required
              />
            )}
          </Card>

          {/* Model Configuration Section */}
          <Card title="Model">
            <SelectInput
              label="Model Architecture"
              value={jobConfig.config.process[0].model.arch}
              onChange={value => {
                handleModelArchChange(jobConfig.config.process[0].model.arch, value, jobConfig, setJobConfig);
              }}
              options={groupedModelOptions}
            />
            <TextInput
              label="Name or Path"
              value={jobConfig.config.process[0].model.name_or_path}
              docKey="config.process[0].model.name_or_path"
              onChange={(value: string | null) => {
                if (value?.trim() === '') {
                  value = null;
                }
                setJobConfig(value, 'config.process[0].model.name_or_path');
              }}
              placeholder=""
              required
            />
            {modelArch?.additionalSections?.includes('model.assistant_lora_path') && (
              <TextInput
                label="Training Adapter Path"
                value={jobConfig.config.process[0].model.assistant_lora_path ?? ''}
                docKey="config.process[0].model.assistant_lora_path"
                onChange={(value: string | undefined) => {
                  if (value?.trim() === '') {
                    value = undefined;
                  }
                  setJobConfig(value, 'config.process[0].model.assistant_lora_path');
                }}
                placeholder=""
              />
            )}
            {modelArch?.additionalSections?.includes('model.unconditional_lora_path') && (
              <TextInput
                label="Unconditional Adapter Path"
                value={jobConfig.config.process[0].model.unconditional_lora_path ?? ''}
                docKey="config.process[0].model.unconditional_lora_path"
                onChange={(value: string | undefined) => {
                  if (value?.trim() === '') {
                    value = undefined;
                  }
                  setJobConfig(value, 'config.process[0].model.unconditional_lora_path');
                }}
                placeholder=""
              />
            )}
            {modelArch?.customModelSelectOptions?.map(customOption => (
              <SelectInput
                key={customOption.label}
                label={customOption.label}
                value={customOption.getValue(jobConfig) ?? ''}
                doc={customOption.doc}
                onChange={value => customOption.onChange(value, jobConfig, setJobConfig)}
                options={customOption.options}
              />
            ))}
            {modelArch?.modelNotes && (
              <div className="pt-2">
                <button
                  type="button"
                  onClick={() => {
                    const gateUrl = modelArch.gateUrl as string;
                    openDoc({
                      title: `Notes - ${modelArch.label}`,
                      description: <div className="space-y-3">{modelArch.modelNotes}</div>,
                    });
                  }}
                  className="w-full flex items-center gap-2 rounded-md bg-blue-950/60 border border-blue-800 px-3 py-2 text-sm text-blue-200 hover:bg-blue-900/60 text-left"
                >
                  <Info className="w-4 h-4 shrink-0 text-blue-400" />
                  <span>Model notes</span>
                </button>
              </div>
            )}
            {modelArch?.gateUrl && (
              <div className="pt-2">
                <button
                  type="button"
                  onClick={() => {
                    const gateUrl = modelArch.gateUrl as string;
                    openDoc({
                      title: 'Gated Model',
                      description: (
                        <div className="space-y-3">
                          <p>
                            This model is gated on Huggingface. Before you can use it, you will need to accept the model
                            terms on the model page:
                          </p>
                          <p>
                            <a
                              href={gateUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-400 hover:text-blue-300 underline"
                            >
                              {gateUrl}
                            </a>
                          </p>
                          <p>
                            You will also need to create a Huggingface{' '}
                            <a
                              href="https://huggingface.co/settings/tokens"
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-400 hover:text-blue-300 underline"
                            >
                              read token
                            </a>{' '}
                            and add it on the{' '}
                            <a href="/settings" className="text-blue-400 hover:text-blue-300 underline">
                              settings page
                            </a>
                            .
                          </p>
                        </div>
                      ),
                    });
                  }}
                  className="w-full flex items-center gap-2 rounded-md bg-yellow-950/60 border border-yellow-800 px-3 py-2 text-sm text-yellow-200 hover:bg-yellow-900/60 text-left"
                >
                  <Info className="w-4 h-4 shrink-0 text-yellow-400" />
                  <span>
                    Gated model. <span className="underline">Learn more.</span>
                  </span>
                </button>
              </div>
            )}
            {modelArch?.additionalSections?.includes('model.low_vram') && (
              <FormGroup label="Options">
                <Checkbox
                  label="Low VRAM"
                  checked={jobConfig.config.process[0].model.low_vram}
                  onChange={value => setJobConfig(value, 'config.process[0].model.low_vram')}
                />
                {modelArch?.additionalSections?.includes('model.gemma_api') && (
                  <Checkbox
                    label={
                      gemmaApiKeyConfigured
                        ? 'Use Gemma API (saves ~24 GB VRAM)'
                        : 'Use Gemma API (set key in Settings)'
                    }
                    checked={jobConfig.config.process[0].model.use_gemma_api || false}
                    docKey="model.gemma_api"
                    disabled={!gemmaApiKeyConfigured}
                    onChange={value => {
                      setJobConfig(value, 'config.process[0].model.use_gemma_api');
                      if (value) {
                        setJobConfig(true, 'config.process[0].train.cache_text_embeddings');
                        setJobConfig(false, 'config.process[0].train.unload_text_encoder');
                      }
                    }}
                  />
                )}
              </FormGroup>
            )}
            {modelArch?.additionalSections?.includes('model.model_kwargs.kv_cache') && (
              <Checkbox
                label="KV Cache"
                docKey="model.model_kwargs.kv_cache"
                checked={jobConfig.config.process[0].model.model_kwargs.kv_cache || false}
                onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.kv_cache')}
              />
            )}
            {modelArch?.additionalSections?.includes('model.qie.match_target_res') && (
              <Checkbox
                label="Match Target Res"
                docKey="model.qie.match_target_res"
                checked={jobConfig.config.process[0].model.model_kwargs.match_target_res}
                onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.match_target_res')}
              />
            )}
            {modelArch?.additionalSections?.includes('model.spatial_upscaler_path') && (
              <TextInput
                label="Spatial Upscaler Path"
                className="md:col-span-2"
                value={jobConfig.config.process[0].model.spatial_upscaler_path ?? ''}
                docKey="config.process[0].model.spatial_upscaler_path"
                onChange={(value: string | undefined) => {
                  if (value?.trim() === '') {
                    value = undefined;
                  }
                  setJobConfig(value, 'config.process[0].model.spatial_upscaler_path');
                }}
                placeholder="Path to .safetensors upscaler model (optional)"
              />
            )}
            {modelArch?.additionalSections?.includes('model.te_name_or_path') && !useGemmaApi && (
              <TextInput
                label="Text Encoder Path Override"
                className="md:col-span-2"
                value={jobConfig.config.process[0].model.te_name_or_path ?? ''}
                docKey="config.process[0].model.te_name_or_path"
                onChange={(value: string | undefined) => {
                  if (value?.trim() === '') {
                    value = undefined;
                  }
                  setJobConfig(value, 'config.process[0].model.te_name_or_path');
                }}
                placeholder="Path to .safetensors text encoder (optional — defaults to the arch's own resolution)"
              />
            )}
            {modelArch?.additionalSections?.includes('model.layer_offloading') && !isMac() && !useGemmaApi && (
              <>
                <Checkbox
                  label={
                    <>
                      Layer Offloading <IoFlaskSharp className="inline text-yellow-500" name="Experimental" />{' '}
                    </>
                  }
                  checked={jobConfig.config.process[0].model.layer_offloading || false}
                  onChange={value => setJobConfig(value, 'config.process[0].model.layer_offloading')}
                  docKey="model.layer_offloading"
                />
                {jobConfig.config.process[0].model.layer_offloading && (
                  <div className="pt-2">
                    <SliderInput
                      label="Transformer Offload %"
                      value={Math.round(
                        (jobConfig.config.process[0].model.layer_offloading_transformer_percent ?? 1) * 100,
                      )}
                      onChange={value =>
                        setJobConfig(value * 0.01, 'config.process[0].model.layer_offloading_transformer_percent')
                      }
                      min={0}
                      max={100}
                      step={1}
                    />
                    <SliderInput
                      label="Text Encoder Offload %"
                      value={Math.round(
                        (jobConfig.config.process[0].model.layer_offloading_text_encoder_percent ?? 1) * 100,
                      )}
                      onChange={value =>
                        setJobConfig(value * 0.01, 'config.process[0].model.layer_offloading_text_encoder_percent')
                      }
                      min={0}
                      max={100}
                      step={1}
                    />
                  </div>
                )}
              </>
            )}
          </Card>
          {disableSections.includes('model.quantize') ? null : (
            <Card title="Quantize / Compile">
              <SelectInput
                label="Transformer"
                value={jobConfig.config.process[0].model.quantize ? jobConfig.config.process[0].model.qtype : ''}
                onChange={value => {
                  if (value === '') {
                    setJobConfig(false, 'config.process[0].model.quantize');
                    value = defaultQtype;
                  } else {
                    setJobConfig(true, 'config.process[0].model.quantize');
                  }
                  setJobConfig(value, 'config.process[0].model.qtype');
                }}
                options={transformerQuantizationOptions}
              />
              {!disableSections.includes('model.quantize_te') && !useGemmaApi && (
                <SelectInput
                  label="Text Encoder"
                  value={
                    jobConfig.config.process[0].model.quantize_te ? jobConfig.config.process[0].model.qtype_te : ''
                  }
                  onChange={value => {
                    if (value === '') {
                      setJobConfig(false, 'config.process[0].model.quantize_te');
                      value = defaultQtype;
                    } else {
                      setJobConfig(true, 'config.process[0].model.quantize_te');
                    }
                    setJobConfig(value, 'config.process[0].model.qtype_te');
                  }}
                  options={quantizationOptions}
                />
              )}
              {jobConfig.config.process[0].model.quantize &&
                (() => {
                  const { base, ara } = parseQtypeAra(jobConfig.config.process[0].model.qtype || '');
                  // An accuracy recovery adapter's own quantization only round-trips
                  // through the ostris backends (uintN / convrot*) -- see
                  // quantize_model's ARA branch in toolkit/util/quantize.py. qfloat8
                  // and float8 go through a different save path that has nothing to
                  // persist when an ARA is attached, so the cache would silently do
                  // nothing.
                  const cacheUnsupported = !!ara && !isOstrisBackedQtype(base);
                  if (cacheUnsupported && jobConfig.config.process[0].model.cache_quantized_model) {
                    // don't let a combination that can't work linger in a saved config
                    setJobConfig(false, 'config.process[0].model.cache_quantized_model');
                  }
                  return (
                    <>
                      <Checkbox
                        label="Cache Quantized Model"
                        docKey="model.cache_quantized_model"
                        checked={
                          !cacheUnsupported && (jobConfig.config.process[0].model.cache_quantized_model ?? false)
                        }
                        disabled={cacheUnsupported}
                        onChange={value => setJobConfig(value, 'config.process[0].model.cache_quantized_model')}
                      />
                      {cacheUnsupported && (
                        <p className="text-xs text-yellow-500 mt-1">
                          Caching isn't supported for this accuracy recovery adapter with a {base} base quantization —
                          only 4/8-bit and convrot quantization can be cached with an adapter attached.
                        </p>
                      )}
                    </>
                  );
                })()}
              <FormGroup label="Compile Options">
                <></>
              </FormGroup>
              <Checkbox
                label="Compile Model"
                checked={jobConfig.config.process[0].model.compile || false}
                onChange={value => {
                  setJobConfig(value, 'config.process[0].model.compile');
                  if (value) {
                    for (const key in defaultCompileOptions) {
                      setJobConfig((defaultCompileOptions as any)[key], `config.process[0].model.${key}`);
                    }
                  } else {
                    for (const key in defaultCompileOptions) {
                      setJobConfig(undefined, `config.process[0].model.${key}`);
                    }
                  }
                }}
              />
              {jobConfig.config.process[0].model.compile && (
                <>
                  <Checkbox
                    label="Block Compile"
                    docKey="model.block_compile"
                    checked={jobConfig.config.process[0].model.block_compile ?? true}
                    onChange={value => setJobConfig(value, 'config.process[0].model.block_compile')}
                  />
                  <SelectInput
                    label="Compile Mode"
                    value={jobConfig.config.process[0].model.compile_mode || 'default'}
                    onChange={value => setJobConfig(value, 'config.process[0].model.compile_mode')}
                    options={[
                      { value: 'default', label: 'Default' },
                      { value: 'max-autotune', label: 'Max Autotune (slower first step, faster training)' },
                      { value: 'fastest', label: 'Fastest' },
                    ]}
                  />
                  <Checkbox
                    label="Full Graph"
                    docKey="model.compile_fullgraph"
                    checked={jobConfig.config.process[0].model.compile_fullgraph ?? false}
                    onChange={value => setJobConfig(value, 'config.process[0].model.compile_fullgraph')}
                  />
                </>
              )}
            </Card>
          )}
          {modelArch?.additionalSections?.includes('model.multistage') && (
            <Card title="Multistage">
              <FormGroup label="Stages to Train" docKey={'model.multistage'}>
                <Checkbox
                  label="High Noise"
                  checked={jobConfig.config.process[0].model.model_kwargs?.train_high_noise || false}
                  onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.train_high_noise')}
                />
                <Checkbox
                  label="Low Noise"
                  checked={jobConfig.config.process[0].model.model_kwargs?.train_low_noise || false}
                  onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.train_low_noise')}
                />
              </FormGroup>
              <NumberInput
                label="Switch Every"
                value={jobConfig.config.process[0].train.switch_boundary_every}
                onChange={value => setJobConfig(value, 'config.process[0].train.switch_boundary_every')}
                placeholder="eg. 1"
                docKey={'train.switch_boundary_every'}
                min={1}
                required
              />
            </Card>
          )}
          <Card title="Target">
            <SelectInput
              label="Target Type"
              value={jobConfig.config.process[0].network?.type ?? 'lora'}
              onChange={value => setJobConfig(value, 'config.process[0].network.type')}
              options={[
                { value: 'lora', label: 'LoRA' },
                { value: 'lokr', label: 'LoKr' },
              ]}
            />
            {jobConfig.config.process[0].network?.type == 'lokr' && (
              <SelectInput
                label="LoKr Factor"
                value={`${jobConfig.config.process[0].network?.lokr_factor ?? -1}`}
                onChange={value => setJobConfig(parseInt(value), 'config.process[0].network.lokr_factor')}
                options={[
                  { value: '-1', label: 'Auto' },
                  { value: '4', label: '4' },
                  { value: '8', label: '8' },
                  { value: '16', label: '16' },
                  { value: '32', label: '32' },
                ]}
              />
            )}
            {jobConfig.config.process[0].network?.type == 'lora' && (
              <>
                <NumberInput
                  label="Linear Rank"
                  value={jobConfig.config.process[0].network.linear}
                  onChange={value => {
                    console.log('onChange', value);
                    setJobConfig(value, 'config.process[0].network.linear');
                    setJobConfig(value, 'config.process[0].network.linear_alpha');
                  }}
                  placeholder="eg. 16"
                  min={0}
                  max={1024}
                  required
                />
                {disableSections.includes('network.conv') ? null : (
                  <NumberInput
                    label="Conv Rank"
                    value={jobConfig.config.process[0].network.conv}
                    onChange={value => {
                      console.log('onChange', value);
                      setJobConfig(value, 'config.process[0].network.conv');
                      setJobConfig(value, 'config.process[0].network.conv_alpha');
                    }}
                    placeholder="eg. 16"
                    min={0}
                    max={1024}
                  />
                )}
              </>
            )}
          </Card>
          {!disableSections.includes('slider') && (
            <Card title="Slider">
              <TextInput
                label="Target Class"
                className=""
                value={jobConfig.config.process[0].slider?.target_class ?? ''}
                onChange={value => setJobConfig(value, 'config.process[0].slider.target_class')}
                placeholder="eg. person"
              />
              <TextInput
                label="Positive Prompt"
                className=""
                value={jobConfig.config.process[0].slider?.positive_prompt ?? ''}
                onChange={value => setJobConfig(value, 'config.process[0].slider.positive_prompt')}
                placeholder="eg. person who is happy"
              />
              <TextInput
                label="Negative Prompt"
                className=""
                value={jobConfig.config.process[0].slider?.negative_prompt ?? ''}
                onChange={value => setJobConfig(value, 'config.process[0].slider.negative_prompt')}
                placeholder="eg. person who is sad"
              />
              <TextInput
                label="Anchor Class"
                className=""
                value={jobConfig.config.process[0].slider?.anchor_class ?? ''}
                onChange={value => setJobConfig(value, 'config.process[0].slider.anchor_class')}
                placeholder=""
              />
            </Card>
          )}
          <Card title="Save">
            <SelectInput
              label="Data Type"
              value={jobConfig.config.process[0].save.dtype}
              onChange={value => setJobConfig(value, 'config.process[0].save.dtype')}
              options={[
                { value: 'bf16', label: 'BF16' },
                { value: 'fp16', label: 'FP16' },
                { value: 'fp32', label: 'FP32' },
              ]}
            />
            <NumberInput
              label="Save Every"
              value={jobConfig.config.process[0].save.save_every}
              onChange={value => setJobConfig(value, 'config.process[0].save.save_every')}
              placeholder="eg. 250"
              min={1}
              required
            />
            <NumberInput
              label="Max Step Saves to Keep"
              value={jobConfig.config.process[0].save.max_step_saves_to_keep}
              onChange={value => setJobConfig(value, 'config.process[0].save.max_step_saves_to_keep')}
              placeholder="eg. 4"
              min={1}
              required
            />
            <FormGroup label="Options">
              <Checkbox
                label="Archive Optimizer"
                checked={jobConfig.config.process[0].save.archive_optimizer || false}
                onChange={value => setJobConfig(value, 'config.process[0].save.archive_optimizer')}
              />
            </FormGroup>
            <FormGroup label="Options">
              <Checkbox
                  label="Keep step number in final safetensors file name"
                  checked={jobConfig.config.process[0].save.save_with_step_num || false}
                  onChange={value => setJobConfig(value, 'config.process[0].save.save_with_step_num')}
              />
            </FormGroup>
          </Card>claude
        </div>
        <div className={sampleOnly ? 'hidden' : ''}>
          <Card title="Training">
            <div className={trainingBarClass}>
              <div>
                <NumberInput
                  label="Batch Size"
                  value={jobConfig.config.process[0].train.batch_size}
                  onChange={value => setJobConfig(value, 'config.process[0].train.batch_size')}
                  placeholder="eg. 4"
                  min={1}
                  required
                />
                <NumberInput
                  label="Gradient Accumulation"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.gradient_accumulation}
                  onChange={value => setJobConfig(value, 'config.process[0].train.gradient_accumulation')}
                  placeholder="eg. 1"
                  min={1}
                  required
                />
                <NumberInput
                  label="Steps"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.steps}
                  onChange={value => setJobConfig(value, 'config.process[0].train.steps')}
                  placeholder="eg. 2000"
                  min={1}
                  required
                />
              </div>
              <div>
                <SelectInput
                  label="Optimizer"
                  value={jobConfig.config.process[0].train.optimizer}
                  onChange={value => setJobConfig(value, 'config.process[0].train.optimizer')}
                  options={[
                    { value: 'adafactor', label: 'Adafactor' },
                    { value: 'adam', label: 'Adam' },
                    { value: 'adamw', label: 'AdamW' },
                    { value: 'adamw8bit', label: 'AdamW8Bit' },
                    { value: 'automagic', label: 'Automagic' },
                    { value: 'automagic2', label: 'Automagic v2' },
                    { value: 'automagic3', label: 'Automagic v3' },
                    { value: 'prodigyopt', label: 'Prodigy' },
                    { value: 'prodigy8bit', label: 'Prodigy8Bit' },
                  ]}
                />
                {isAudioModel && jobConfig.config.process[0].train.optimizer === 'automagic3' && (
                  <p className="text-xs text-yellow-500 mt-1">
                    This has never worked well for you on audio models — loss has stalled flat in
                    4 of 5 past ACE-Step runs with automagic3 (automagic v3 has no effective max_lr
                    clamp, so per-tensor LR can decay to near-zero before anything is learned).
                    AdamW8Bit or Automagic v2 have consistently trained instead.
                  </p>
                )}
                <NumberInput
                  label="Learning Rate"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.lr}
                  onChange={value => setJobConfig(value, 'config.process[0].train.lr')}
                  placeholder="eg. 0.0001"
                  min={0}
                  required
                />
                <NumberInput
                  label="Weight Decay"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.optimizer_params.weight_decay}
                  onChange={value => setJobConfig(value, 'config.process[0].train.optimizer_params.weight_decay')}
                  placeholder="eg. 0.0001"
                  min={0}
                  required
                />
              </div>
              <div>
                {disableSections.includes('train.timestep_type') ? null : (
                  <SelectInput
                    label="Timestep Type"
                    value={jobConfig.config.process[0].train.timestep_type}
                    disabled={disableSections.includes('train.timestep_type') || false}
                    onChange={value => setJobConfig(value, 'config.process[0].train.timestep_type')}
                    options={[
                      { value: 'sigmoid', label: 'Sigmoid' },
                      { value: 'linear', label: 'Linear' },
                      { value: 'shift', label: 'Shift' },
                      { value: 'weighted', label: 'Weighted' },
                    ]}
                  />
                )}
                <SelectInput
                  label="Timestep Bias"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.content_or_style}
                  onChange={value => setJobConfig(value, 'config.process[0].train.content_or_style')}
                  options={[
                    { value: 'balanced', label: 'Balanced' },
                    { value: 'content', label: 'High Noise' },
                    { value: 'style', label: 'Low Noise' },
                  ]}
                />
                <SelectInput
                  label="Loss Type"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.loss_type}
                  onChange={value => setJobConfig(value, 'config.process[0].train.loss_type')}
                  options={[
                    { value: 'mse', label: 'Mean Squared Error' },
                    { value: 'mae', label: 'Mean Absolute Error' },
                    { value: 'wavelet', label: 'Wavelet' },
                    { value: 'stepped', label: 'Stepped Recovery' },
                  ]}
                />
                {modelArch?.additionalSections?.includes('train.audio_loss_multiplier') && (
                  <NumberInput
                    label="Audio Loss Multiplier"
                    className="pt-2"
                    value={jobConfig.config.process[0].train.audio_loss_multiplier ?? 1.0}
                    onChange={value => setJobConfig(value, 'config.process[0].train.audio_loss_multiplier')}
                    placeholder="eg. 1.0"
                    docKey={'train.audio_loss_multiplier'}
                    min={0}
                  />
                )}
              </div>
              <div>
                <FormGroup label="EMA (Exponential Moving Average)">
                  <Checkbox
                    label="Use EMA"
                    className="pt-1"
                    checked={jobConfig.config.process[0].train.ema_config?.use_ema || false}
                    onChange={value => setJobConfig(value, 'config.process[0].train.ema_config.use_ema')}
                  />
                </FormGroup>
                {jobConfig.config.process[0].train.ema_config?.use_ema && (
                  <NumberInput
                    label="EMA Decay"
                    className="pt-2"
                    value={jobConfig.config.process[0].train.ema_config?.ema_decay as number}
                    onChange={value => setJobConfig(value, 'config.process[0].train.ema_config.ema_decay')}
                    placeholder="eg. 0.99"
                    min={0}
                  />
                )}

                <FormGroup label="Text Encoder Optimizations" className="pt-2">
                  {!disableSections.includes('train.unload_text_encoder') && (
                    <Checkbox
                      label="Unload TE"
                      checked={jobConfig.config.process[0].train.unload_text_encoder || false}
                      docKey={'train.unload_text_encoder'}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.unload_text_encoder');
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.cache_text_embeddings');
                        }
                      }}
                    />
                  )}
                  <Checkbox
                    label="Cache Text Embeddings"
                    checked={jobConfig.config.process[0].train.cache_text_embeddings || false}
                    docKey={'train.cache_text_embeddings'}
                    onChange={value => {
                      setJobConfig(value, 'config.process[0].train.cache_text_embeddings');
                      if (value) {
                        setJobConfig(false, 'config.process[0].train.unload_text_encoder');
                      }
                    }}
                  />
                </FormGroup>
                {jobConfig.config.process[0].datasets.length > 1 && (
                  <FormGroup label="Multi-Dataset" className="pt-2">
                    <Checkbox
                      label="Combine Datasets for Bucketing"
                      checked={jobConfig.config.process[0].train.combine_datasets || false}
                      docKey={'train.combine_datasets'}
                      onChange={value => setJobConfig(value, 'config.process[0].train.combine_datasets')}
                    />
                  </FormGroup>
                )}
              </div>
              <div>
                {disableSections.includes('train.diff_output_preservation') ||
                disableSections.includes('train.blank_prompt_preservation') ? null : (
                  <FormGroup label="Regularization">
                    <></>
                  </FormGroup>
                )}
                {disableSections.includes('train.diff_output_preservation') ? null : (
                  <>
                    <Checkbox
                      label="Differential Output Preservation"
                      docKey={'train.diff_output_preservation'}
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.diff_output_preservation || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.diff_output_preservation');
                        if (value && jobConfig.config.process[0].train.blank_prompt_preservation) {
                          // only one can be enabled at a time
                          setJobConfig(false, 'config.process[0].train.blank_prompt_preservation');
                        }
                      }}
                    />
                    {jobConfig.config.process[0].train.diff_output_preservation && (
                      <>
                        <NumberInput
                          label="DOP Loss Multiplier"
                          className="pt-2"
                          value={jobConfig.config.process[0].train.diff_output_preservation_multiplier as number}
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.diff_output_preservation_multiplier')
                          }
                          placeholder="eg. 1.0"
                          min={0}
                        />
                        <TextInput
                          label="DOP Preservation Class"
                          className="pt-2 pb-4"
                          value={jobConfig.config.process[0].train.diff_output_preservation_class as string}
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.diff_output_preservation_class')
                          }
                          placeholder="eg. woman"
                        />
                      </>
                    )}
                  </>
                )}
                {disableSections.includes('train.blank_prompt_preservation') ? null : (
                  <>
                    <Checkbox
                      label="Blank Prompt Preservation"
                      docKey={'train.blank_prompt_preservation'}
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.blank_prompt_preservation || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.blank_prompt_preservation');
                        if (value && jobConfig.config.process[0].train.diff_output_preservation) {
                          // only one can be enabled at a time
                          setJobConfig(false, 'config.process[0].train.diff_output_preservation');
                        }
                      }}
                    />
                    {jobConfig.config.process[0].train.blank_prompt_preservation && (
                      <>
                        <NumberInput
                          label="BPP Loss Multiplier"
                          className="pt-2"
                          value={
                            (jobConfig.config.process[0].train.blank_prompt_preservation_multiplier as number) || 1.0
                          }
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.blank_prompt_preservation_multiplier')
                          }
                          placeholder="eg. 1.0"
                          min={0}
                        />
                      </>
                    )}
                  </>
                )}
                <FormGroup label="Other" className="pt-2">
                  <>
                    <Checkbox
                      label="Contrastive Guidance Loss"
                      docKey={'train.do_guidance_loss'}
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.do_guidance_loss || false}
                      onChange={value => {
                        if (value) {
                          setJobConfig(true, 'config.process[0].train.do_guidance_loss');
                          if (!jobConfig.config.process[0].train.guidance_loss_target) {
                            setJobConfig(4.0, 'config.process[0].train.guidance_loss_target');
                          }
                        } else {
                          setJobConfig(undefined, 'config.process[0].train.do_guidance_loss');
                          setJobConfig(undefined, 'config.process[0].train.guidance_loss_target');
                        }
                      }}
                    />
                    {jobConfig.config.process[0].train.do_guidance_loss && (
                      <>
                        <NumberInput
                          label="Guidance Loss Target"
                          docKey={'train.guidance_loss_target'}
                          value={(jobConfig.config.process[0].train.guidance_loss_target as number) || 4.0}
                          onChange={value => setJobConfig(value, 'config.process[0].train.guidance_loss_target')}
                          placeholder="eg. 3.0"
                          min={0}
                        />
                      </>
                    )}
                  </>
                </FormGroup>
              </div>
            </div>
          </Card>
        </div>
        <div className={sampleOnly ? 'hidden' : ''}>
          <Card
            title="Validation"
            toggled={!!validationConfig}
            onToggle={value => {
              if (value) {
                setJobConfig(
                  {
                    validation_items: isAudioModel
                      ? [{ audio_path: '', caption_path: '', prompt: '' }]
                      : [{ image_path: '', prompt: '' }],
                    resolution: 1024,
                    validate_every_n_steps: 1,
                    validation_sigmas: [0.5],
                  },
                  'config.process[0].train.validation_config',
                );
              } else {
                setJobConfig(undefined, 'config.process[0].train.validation_config');
              }
            }}
          >
            {validationConfig && (
              <>
                <p className="text-sm text-gray-400 mb-4">
                  {isAudioModel ? (
                    <>
                      Validation runs a stable loss check on a fixed set of held-out audio files. Each file is
                      encoded once at startup and predicted at the selected sigmas with fixed seeds, so the result is
                      always deterministic and comparable across the run. The average loss is logged as val/loss
                      every time validation runs. Point each item at an audio file and either a caption file (using
                      the same {'<CAPTION>/<LYRICS>/<BPM>/...'} tagged format as training captions) or a prompt typed
                      directly, but{' '}
                      <span className="font-bold text-gray-300">
                        do not include the validation audio in the training dataset
                      </span>
                      .
                    </>
                  ) : (
                    <>
                      Validation runs a stable loss check on a fixed set of images. Each image is encoded once at
                      startup and predicted at the selected sigmas with fixed seeds, so the result is always
                      deterministic and comparable across the run. The average loss is logged as val/loss every time
                      validation runs. The images need to match the concept of your dataset, but{' '}
                      <span className="font-bold text-gray-300">
                        do not include the validation images in the dataset
                      </span>
                      . They must be images containing the concept you want to train, but not an image trained on.
                    </>
                  )}
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <NumberInput
                    label="Validate Every"
                    value={validationConfig.validate_every_n_steps}
                    onChange={value =>
                      setJobConfig(value, 'config.process[0].train.validation_config.validate_every_n_steps')
                    }
                    placeholder="eg. 10"
                    min={1}
                    required
                  />
                  {!isAudioModel && (
                    <NumberInput
                      label="Validation Resolution"
                      value={validationConfig.resolution}
                      onChange={value => setJobConfig(value, 'config.process[0].train.validation_config.resolution')}
                      placeholder="eg. 512"
                      min={64}
                      required
                    />
                  )}
                  <SelectInput
                    label="Validation Sigmas"
                    value={(validationConfig.validation_sigmas ?? [1.0, 0.75, 0.5, 0.25]).join(', ')}
                    onChange={value =>
                      setJobConfig(
                        value.split(',').map((v: string) => parseFloat(v)),
                        'config.process[0].train.validation_config.validation_sigmas',
                      )
                    }
                    options={[
                      { value: '0.5', label: '0.5' },
                      { value: '1, 0.5', label: '1.0, 0.5' },
                      { value: '1, 0.66, 0.33', label: '1.0, 0.66, 0.33' },
                      { value: '1, 0.75, 0.5, 0.25', label: '1.0, 0.75, 0.5, 0.25' },
                    ]}
                  />
                </div>
                <div className="mt-4">
                  <label className="block text-xs text-gray-300 mb-2">
                    {isAudioModel ? 'Validation Audio' : 'Validation Images'} (
                    {validationConfig.validation_items.length})
                  </label>
                  {validationConfig.validation_items.map((item, i) => (
                    <div key={i} className="rounded-lg pl-4 pr-1 py-3 mb-4 bg-gray-950">
                      {isAudioModel ? (
                        <div className="flex items-center space-x-4">
                          <div className="flex-1 grid grid-cols-1 md:grid-cols-3 gap-4">
                            <TextInput
                              label="Audio Path"
                              value={item.audio_path ?? ''}
                              onChange={value =>
                                setJobConfig(
                                  value,
                                  `config.process[0].train.validation_config.validation_items[${i}].audio_path`,
                                )
                              }
                              placeholder="C:\path\to\Girl.mp3"
                            />
                            <TextInput
                              label="Caption Path"
                              value={item.caption_path ?? ''}
                              onChange={value =>
                                setJobConfig(
                                  value,
                                  `config.process[0].train.validation_config.validation_items[${i}].caption_path`,
                                )
                              }
                              placeholder="C:\path\to\Girl.caption"
                            />
                            <TextInput
                              label="Prompt (used if no caption path)"
                              value={item.prompt}
                              onChange={value =>
                                setJobConfig(
                                  value,
                                  `config.process[0].train.validation_config.validation_items[${i}].prompt`,
                                )
                              }
                              placeholder="<CAPTION>...</CAPTION><LYRICS>...</LYRICS>"
                            />
                          </div>
                          <div>
                            <button
                              type="button"
                              onClick={() =>
                                setJobConfig(
                                  validationConfig.validation_items.filter((_, index) => index !== i),
                                  'config.process[0].train.validation_config.validation_items',
                                )
                              }
                              className="rounded-full p-1 text-sm"
                            >
                              <X />
                            </button>
                          </div>
                        </div>
                      ) : (
                        <div className="flex items-center space-x-4">
                          <SampleControlImage
                            instruction="Add Image"
                            src={item.image_path === '' ? null : item.image_path}
                            onNewImageSelected={imagePath => {
                              setJobConfig(
                                imagePath ?? '',
                                `config.process[0].train.validation_config.validation_items[${i}].image_path`,
                              );
                            }}
                          />
                          <div className="flex-1">
                            <TextInput
                              label="Prompt"
                              value={item.prompt}
                              onChange={value =>
                                setJobConfig(
                                  value,
                                  `config.process[0].train.validation_config.validation_items[${i}].prompt`,
                                )
                              }
                              placeholder="Enter prompt"
                            />
                          </div>
                          <div>
                            <button
                              type="button"
                              onClick={() =>
                                setJobConfig(
                                  validationConfig.validation_items.filter((_, index) => index !== i),
                                  'config.process[0].train.validation_config.validation_items',
                                )
                              }
                              className="rounded-full p-1 text-sm"
                            >
                              <X />
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                  <button
                    type="button"
                    onClick={() =>
                      setJobConfig(
                        [
                          ...validationConfig.validation_items,
                          isAudioModel
                            ? { audio_path: '', caption_path: '', prompt: '' }
                            : { image_path: '', prompt: '' },
                        ],
                        'config.process[0].train.validation_config.validation_items',
                      )
                    }
                    className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                  >
                    Add Validation Image
                  </button>
                </div>
              </>
            )}
          </Card>
        </div>
        <div className={sampleOnly ? 'hidden' : ''}>
          <Card title="Advanced" collapsible>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div>
                <Checkbox
                  label="Do Differential Guidance"
                  docKey={'train.do_differential_guidance'}
                  className="pt-1"
                  checked={jobConfig.config.process[0].train.do_differential_guidance || false}
                  onChange={value => {
                    let newValue = value == false ? undefined : value;
                    setJobConfig(newValue, 'config.process[0].train.do_differential_guidance');
                    if (!newValue) {
                      setJobConfig(undefined, 'config.process[0].train.differential_guidance_scale');
                    } else if (
                      jobConfig.config.process[0].train.differential_guidance_scale === undefined ||
                      jobConfig.config.process[0].train.differential_guidance_scale === null
                    ) {
                      // set default differential guidance scale to 3.0
                      setJobConfig(3.0, 'config.process[0].train.differential_guidance_scale');
                    }
                  }}
                />
                {jobConfig.config.process[0].train.differential_guidance_scale && (
                  <>
                    <NumberInput
                      label="Differential Guidance Scale"
                      className="pt-2"
                      value={(jobConfig.config.process[0].train.differential_guidance_scale as number) || 3.0}
                      onChange={value => setJobConfig(value, 'config.process[0].train.differential_guidance_scale')}
                      placeholder="eg. 3.0"
                      min={0}
                    />
                  </>
                )}
              </div>
            </div>
          </Card>
        </div>
        <div className={sampleOnly ? 'hidden' : ''}>
          <Card title="Datasets">
            <>
              {jobConfig.config.process[0].datasets.map((dataset, i) => (
                <div key={i} className="p-4 rounded-lg bg-gray-800 relative">
                  <div className="absolute top-2 right-2 flex gap-1">
                    <button
                      type="button"
                      onClick={() => {
                        const duplicated = objectCopy(dataset);
                        const datasets = [...jobConfig.config.process[0].datasets];
                        datasets.splice(i + 1, 0, duplicated);
                        setJobConfig(datasets, 'config.process[0].datasets');
                      }}
                      className="bg-gray-700 hover:bg-gray-600 rounded-full p-2 text-sm transition-colors"
                      title="Duplicate Dataset"
                    >
                      <Copy className="w-4 h-4" />
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        setJobConfig(
                          jobConfig.config.process[0].datasets.filter((_, index) => index !== i),
                          'config.process[0].datasets',
                        )
                      }
                      className="bg-red-600 hover:bg-red-700 text-white rounded-full p-2 text-sm transition-colors"
                      title="Remove Dataset"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                  <h2 className="text-lg font-bold mb-4">Dataset {i + 1}</h2>
                  <div className={datasetStyleClass}>
                    <div>
                      <SelectInput
                        label="Target Dataset"
                        value={dataset.folder_path}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].folder_path`)}
                        options={[{ value: defaultDatasetConfig.folder_path, label: 'Please select...' }, ...datasetOptions]}
                      />
                      {modelArch?.additionalSections?.includes('datasets.control_path') && (
                        <SelectInput
                          label="Control Dataset"
                          docKey="datasets.control_path"
                          value={dataset.control_path ?? ''}
                          className="pt-2"
                          onChange={value =>
                            setJobConfig(value == '' ? null : value, `config.process[0].datasets[${i}].control_path`)
                          }
                          options={[{ value: '', label: 'None' }, ...datasetOptions]}
                        />
                      )}
                      {modelArch?.additionalSections?.includes('datasets.multi_control_paths') && (
                        <>
                          <SelectInput
                            label="Control Dataset 1"
                            docKey="datasets.multi_control_paths"
                            value={dataset.control_path_1 ?? ''}
                            className="pt-2"
                            onChange={value =>
                              setJobConfig(
                                value == '' ? null : value,
                                `config.process[0].datasets[${i}].control_path_1`,
                              )
                            }
                            options={[{ value: '', label: 'None' }, ...datasetOptions]}
                          />
                          <SelectInput
                            label="Control Dataset 2"
                            docKey="datasets.multi_control_paths"
                            value={dataset.control_path_2 ?? ''}
                            className="pt-2"
                            onChange={value =>
                              setJobConfig(
                                value == '' ? null : value,
                                `config.process[0].datasets[${i}].control_path_2`,
                              )
                            }
                            options={[{ value: '', label: 'None' }, ...datasetOptions]}
                          />
                          <SelectInput
                            label="Control Dataset 3"
                            docKey="datasets.multi_control_paths"
                            value={dataset.control_path_3 ?? ''}
                            className="pt-2"
                            onChange={value =>
                              setJobConfig(
                                value == '' ? null : value,
                                `config.process[0].datasets[${i}].control_path_3`,
                              )
                            }
                            options={[{ value: '', label: 'None' }, ...datasetOptions]}
                          />
                        </>
                      )}
                      <NumberInput
                        label="LoRA Weight"
                        value={dataset.network_weight}
                        className="pt-2"
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].network_weight`)}
                        placeholder="eg. 1.0"
                      />
                      <NumberInput
                        label="Num Repeats"
                        value={dataset.num_repeats || 1}
                        className="pt-2"
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].num_repeats`)}
                        placeholder="eg. 1"
                        docKey={'dataset.num_repeats'}
                      />
                    </div>
                    <div>
                      <TextInput
                        label="Default Caption"
                        value={dataset.default_caption}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].default_caption`)}
                        placeholder="eg. A photo of a cat"
                      />
                      <TextInput
                        label="Trigger Word"
                        className="pt-2"
                        docKey="datasets.trigger_word"
                        value={dataset.trigger_word || ''}
                        onChange={(value: string | null) => {
                          if (value?.trim() === '') {
                            value = null;
                          }
                          setJobConfig(value, `config.process[0].datasets[${i}].trigger_word`);
                        }}
                        placeholder="Uses the global Trigger Word if left blank"
                      />
                      <NumberInput
                        label="Caption Dropout Rate"
                        className="pt-2"
                        docKey="datasets.caption_dropout_rate"
                        value={dataset.caption_dropout_rate}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].caption_dropout_rate`)}
                        placeholder="eg. 0.05"
                        min={0}
                        required
                      />
                      <CreatableSelectInput
                        label="Caption Extension"
                        className="pt-2"
                        value={dataset.caption_ext || 'txt'}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].caption_ext`)}
                        options={[
                          { value: 'txt', label: 'txt' },
                          { value: 'json', label: 'json' },
                          { value: 'caption', label: 'caption' },
                        ]}
                      />

                      {modelArch?.additionalSections?.includes('datasets.num_frames') && !dataset.auto_frame_count && (
                        <NumberInput
                          label="Num Frames"
                          className="pt-2"
                          docKey="datasets.num_frames"
                          value={dataset.num_frames}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].num_frames`)}
                          placeholder="eg. 41"
                          min={1}
                          required
                        />
                      )}
                    </div>
                    <div>
                      <FormGroup label="Settings" className="">
                        <Checkbox
                          label="Cache Latents"
                          checked={dataset.cache_latents_to_disk || false}
                          onChange={value =>
                            setJobConfig(value, `config.process[0].datasets[${i}].cache_latents_to_disk`)
                          }
                        />
                        <Checkbox
                          label="Is Regularization"
                          checked={dataset.is_reg || false}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].is_reg`)}
                        />
                        {modelArch?.additionalSections?.includes('datasets.auto_frame_count') && (
                          <Checkbox
                            label="Auto Frame Count"
                            checked={dataset.auto_frame_count || false}
                            onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].auto_frame_count`)}
                            docKey="datasets.auto_frame_count"
                          />
                        )}
                        {modelArch?.additionalSections?.includes('datasets.do_i2v') && (
                          <Checkbox
                            label="Do I2V"
                            checked={dataset.do_i2v || false}
                            onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].do_i2v`)}
                            docKey="datasets.do_i2v"
                          />
                        )}
                        {modelArch?.additionalSections?.includes('datasets.do_audio') && (
                          <Checkbox
                            label="Do Audio"
                            checked={dataset.do_audio || false}
                            onChange={value => {
                              if (!value) {
                                setJobConfig(undefined, `config.process[0].datasets[${i}].do_audio`);
                              } else {
                                setJobConfig(value, `config.process[0].datasets[${i}].do_audio`);
                              }
                            }}
                            docKey="datasets.do_audio"
                          />
                        )}
                        {modelArch?.additionalSections?.includes('datasets.audio_normalize') && (
                          <Checkbox
                            label="Audio Normalize"
                            checked={dataset.audio_normalize || false}
                            onChange={value => {
                              if (!value) {
                                setJobConfig(undefined, `config.process[0].datasets[${i}].audio_normalize`);
                              } else {
                                setJobConfig(value, `config.process[0].datasets[${i}].audio_normalize`);
                              }
                            }}
                            docKey="datasets.audio_normalize"
                          />
                        )}
                        {modelArch?.additionalSections?.includes('datasets.audio_preserve_pitch') && (
                          <Checkbox
                            label="Audio Preserve Pitch"
                            checked={dataset.audio_preserve_pitch || false}
                            onChange={value => {
                              if (!value) {
                                setJobConfig(undefined, `config.process[0].datasets[${i}].audio_preserve_pitch`);
                              } else {
                                setJobConfig(value, `config.process[0].datasets[${i}].audio_preserve_pitch`);
                              }
                            }}
                            docKey="datasets.audio_preserve_pitch"
                          />
                        )}
                      </FormGroup>
                      {!isAudioModel && (
                        <FormGroup label="Flipping" docKey={'datasets.flip'} className="mt-2">
                          <Checkbox
                            label={
                              <>
                                Flip X <FlipHorizontal2 className="inline-block w-4 h-4 ml-1" />
                              </>
                            }
                            checked={dataset.flip_x || false}
                            onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].flip_x`)}
                          />
                          <Checkbox
                            label={
                              <>
                                Flip Y <FlipVertical2 className="inline-block w-4 h-4 ml-1" />
                              </>
                            }
                            checked={dataset.flip_y || false}
                            onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].flip_y`)}
                          />
                        </FormGroup>
                      )}
                    </div>
                    {!isAudioModel && (
                      <div>
                        <FormGroup label="Resolutions" className="pt-2">
                          <div className="grid grid-cols-2 gap-2">
                            {[
                              [256, 512, 768, 1024],
                              [1280, 1328, 1536, 2048],
                            ].map(resGroup => (
                              <div key={resGroup[0]} className="space-y-2">
                                {resGroup.map(res => (
                                  <Checkbox
                                    key={res}
                                    label={res.toString()}
                                    checked={dataset.resolution.includes(res)}
                                    onChange={value => {
                                      const resolutions = dataset.resolution.includes(res)
                                        ? dataset.resolution.filter(r => r !== res)
                                        : [...dataset.resolution, res];
                                      setJobConfig(resolutions, `config.process[0].datasets[${i}].resolution`);
                                    }}
                                  />
                                ))}
                              </div>
                            ))}
                          </div>
                        </FormGroup>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <button
                type="button"
                onClick={() => {
                  const newDataset = objectCopy(defaultDatasetConfig);
                  // automatically add the controls for a new dataset
                  const controls = modelArch?.controls ?? [];
                  newDataset.controls = controls;
                  setJobConfig([...jobConfig.config.process[0].datasets, newDataset], 'config.process[0].datasets');
                }}
                className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              >
                Add Dataset
              </button>
            </>
          </Card>
        </div>
        <div>
          <Card title="Sample">
            <div className={sampleTopStyleClass}>
              <div>
                <NumberInput
                  label="Sample Every"
                  value={jobConfig.config.process[0].sample.sample_every}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sample_every')}
                  placeholder="eg. 250"
                  min={1}
                  required
                />
                <NumberInput
                  label="Sample Start Step"
                  value={jobConfig.config.process[0].sample.sample_start_step ?? 0}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sample_start_step')}
                  placeholder="eg. 0"
                  className="pt-2"
                  min={0}
                  required
                />
                <SelectInput
                  label="Sampler"
                  className="pt-2"
                  value={jobConfig.config.process[0].sample.sampler}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sampler')}
                  options={[
                    { value: 'flowmatch', label: 'FlowMatch' },
                    { value: 'ddpm', label: 'DDPM' },
                  ]}
                />
                <NumberInput
                  label="Guidance Scale"
                  value={jobConfig.config.process[0].sample.guidance_scale}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.guidance_scale')}
                  placeholder="eg. 1.0"
                  className="pt-2"
                  min={0}
                  required
                />
                <NumberInput
                  label="Sample Steps"
                  value={jobConfig.config.process[0].sample.sample_steps}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sample_steps')}
                  placeholder="eg. 1"
                  className="pt-2"
                  min={1}
                  required
                />
              </div>

              {!isAudioModel && (
                <div>
                  <NumberInput
                    label="Width"
                    value={jobConfig.config.process[0].sample.width}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.width')}
                    placeholder="eg. 1024"
                    min={0}
                    required
                  />
                  <NumberInput
                    label="Height"
                    value={jobConfig.config.process[0].sample.height}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.height')}
                    placeholder="eg. 1024"
                    className="pt-2"
                    min={0}
                    required
                  />
                  {isVideoModel && (
                    <div>
                      <div className="pt-2 flex items-end gap-3">
                        <NumberInput
                          label="Duration (seconds)"
                          value={sampleDuration}
                          onChange={value => setSampleDuration(value)}
                          placeholder="eg. 3"
                          className="flex-1"
                          min={0}
                          required
                        />
                        <div className="text-xs text-gray-400 pb-2 whitespace-nowrap">
                          {sampleFrameCount} frames
                          {Math.abs(frameCountToDuration(sampleFrameCount, sampleFps) - sampleDuration) > 0.005 && (
                            <span className="text-gray-500">
                              {' '}
                              · {formatDuration(frameCountToDuration(sampleFrameCount, sampleFps))}s
                            </span>
                          )}
                        </div>
                      </div>
                      <NumberInput
                        label="FPS"
                        value={sampleFps}
                        onChange={value => {
                          setJobConfig(value, 'config.process[0].sample.fps');
                          // keep the derived frame count in step with the new fps
                          setSampleDuration(sampleDuration, value ?? 0);
                        }}
                        placeholder="eg. 24"
                        className="pt-2"
                        min={0}
                        required
                      />
                      {fpsUntested && (
                        <div className="pt-1 text-xs text-yellow-500">
                          Untested framerate for this model
                          {modelArch?.supportedFps ? ` (tested: ${modelArch.supportedFps.join(', ')} fps)` : ''}.
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              <div>
                <NumberInput
                  label="Seed"
                  value={jobConfig.config.process[0].sample.seed}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.seed')}
                  placeholder="eg. 0"
                  min={0}
                  required
                />
                <Checkbox
                  label="Walk Seed"
                  className="pt-4 pl-2"
                  checked={jobConfig.config.process[0].sample.walk_seed}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.walk_seed')}
                />
              </div>
              <div>
                <FormGroup label="Advanced Sampling" className="pt-2">
                  <div>
                    <Checkbox
                      label="Skip First Sample"
                      className="pt-4"
                      checked={jobConfig.config.process[0].train.skip_first_sample || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.skip_first_sample');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.force_first_sample');
                        }
                      }}
                    />
                  </div>
                  <div>
                    <Checkbox
                      label="Force First Sample"
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.force_first_sample || false}
                      docKey={'train.force_first_sample'}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.force_first_sample');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.skip_first_sample');
                        }
                      }}
                    />
                  </div>
                  <div>
                    <Checkbox
                      label="Disable Sampling"
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.disable_sampling || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.disable_sampling');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.force_first_sample');
                        }
                      }}
                    />
                  </div>
                </FormGroup>
              </div>
            </div>
            <div className="pt-2 mb-2 flex items-center justify-between">
              <label className="block text-xs text-gray-300">
                Sample Prompts ({jobConfig.config.process[0].sample.samples.length})
              </label>
              {modelArch?.additionalSections?.includes('ideogram_4_prompt') && (
                <button
                  type="button"
                  disabled={jobConfig.config.process[0].sample.samples.length === 0}
                  onClick={() => {
                    const sampleCfg = jobConfig.config.process[0].sample;
                    const items = sampleCfg.samples
                      .map((s, i) => ({
                        index: i,
                        prompt: s.prompt || '',
                        aspectRatio: toAspectRatio(s.width || sampleCfg.width, s.height || sampleCfg.height),
                      }))
                      .filter(it => it.prompt.trim() !== '');
                    if (items.length === 0) return;
                    openUpsamplePromptsModal(items, (index, newPrompt) => {
                      setJobConfig(newPrompt, `config.process[0].sample.samples[${index}].prompt`);
                    });
                  }}
                  className="px-3 py-1.5 text-sm bg-purple-600 hover:bg-purple-700 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-md inline-flex items-center gap-2"
                >
                  <Wand2 className="w-4 h-4" />
                  Upsample Prompts
                </button>
              )}
            </div>
            {jobConfig.config.process[0].sample.samples.map((sample, i) => (
              <div key={i} className="rounded-lg pl-4 pr-1 mb-4 bg-gray-950">
                <div className="flex items-center space-x-2">
                  <div className="flex-1">
                    <div className="flex">
                      <div className="flex-1">
                        {modelArch?.sampleTags && taggedSampleArr && modelArchTagSections ? (
                          <>
                            {modelArchTagSections.map((sampleTagSection, sti) => (
                              <div key={sti} className="grid w-full lg:grid-flow-col lg:auto-cols-fr gap-4 mt-2">
                                {Object.entries(sampleTagSection).map(([tagKey, tag]) => (
                                  <div key={tagKey} className="mb-2">
                                    {tag.type === 'text' && (
                                      <TextInput
                                        label={tag.title}
                                        value={taggedSampleArr[i][tagKey] ?? ''}
                                        onChange={value => {
                                          let taggedSample = { ...taggedSampleArr[i] };
                                          taggedSample[tagKey] = value;
                                          setJobConfig(
                                            objToTags(taggedSample),
                                            `config.process[0].sample.samples[${i}].prompt`,
                                          );
                                        }}
                                        placeholder={`Enter ${tag.title.toLowerCase()}`}
                                      />
                                    )}
                                    {tag.type === 'multiline' && (
                                      <TextAreaInput
                                        label={tag.title}
                                        value={taggedSampleArr[i][tagKey] ?? ''}
                                        onChange={value => {
                                          let taggedSample = { ...taggedSampleArr[i] };
                                          taggedSample[tagKey] = value;
                                          setJobConfig(
                                            objToTags(taggedSample),
                                            `config.process[0].sample.samples[${i}].prompt`,
                                          );
                                        }}
                                        placeholder={`Enter ${tag.title.toLowerCase()}`}
                                      />
                                    )}
                                    {tag.type === 'number' && (
                                      <NumberInput
                                        label={tag.title}
                                        value={taggedSampleArr[i][tagKey] ?? ''}
                                        onChange={value => {
                                          let taggedSample = { ...taggedSampleArr[i] };
                                          taggedSample[tagKey] = value;
                                          setJobConfig(
                                            objToTags(taggedSample),
                                            `config.process[0].sample.samples[${i}].prompt`,
                                          );
                                        }}
                                        placeholder={`Enter ${tag.title.toLowerCase()}`}
                                      />
                                    )}
                                  </div>
                                ))}
                              </div>
                            ))}
                          </>
                        ) : (
                          <TextAreaInput
                            label={`Prompt`}
                            value={sample.prompt}
                            onChange={value => setJobConfig(value, `config.process[0].sample.samples[${i}].prompt`)}
                            placeholder="Enter prompt"
                            required
                          />
                        )}

                        {modelArch?.additionalSections?.includes('ideogram_4_prompt') && (
                          <div className="mt-2">
                            <button
                              type="button"
                              onClick={() => {
                                const sampleCfg = jobConfig.config.process[0].sample;
                                openPromptBoxEditor({
                                  prompt: sample.prompt || '',
                                  aspectRatio: toAspectRatio(
                                    sample.width || sampleCfg.width,
                                    sample.height || sampleCfg.height,
                                  ),
                                  title: `Prompt #${i + 1}`,
                                  onApply: newPrompt =>
                                    setJobConfig(newPrompt, `config.process[0].sample.samples[${i}].prompt`),
                                });
                              }}
                              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md border border-gray-600 text-gray-300 hover:bg-gray-800 transition-colors"
                            >
                              <SquareDashed className="w-3.5 h-3.5" />
                              Edit caption &amp; boxes
                            </button>
                          </div>
                        )}

                        <div className="grid w-full lg:grid-flow-col lg:auto-cols-fr gap-4 mt-2">
                          {!isAudioModel && (
                            <TextInput
                              label={`Width`}
                              value={sample.width ? `${sample.width}` : ''}
                              onChange={value => {
                                // remove any non-numeric characters
                                value = value.replace(/\D/g, '');
                                if (value === '') {
                                  // remove the key from the config if empty
                                  let newConfig = objectCopy(jobConfig);
                                  if (newConfig.config.process[0].sample.samples[i]) {
                                    delete newConfig.config.process[0].sample.samples[i].width;
                                    setJobConfig(
                                      newConfig.config.process[0].sample.samples,
                                      'config.process[0].sample.samples',
                                    );
                                  }
                                } else {
                                  const intValue = parseInt(value);
                                  if (!isNaN(intValue)) {
                                    setJobConfig(intValue, `config.process[0].sample.samples[${i}].width`);
                                  } else {
                                    console.warn('Invalid width value:', value);
                                  }
                                }
                              }}
                              placeholder={`${jobConfig.config.process[0].sample.width} (default)`}
                            />
                          )}
                          {!isAudioModel && (
                            <TextInput
                              label={`Height`}
                              value={sample.height ? `${sample.height}` : ''}
                              onChange={value => {
                                // remove any non-numeric characters
                                value = value.replace(/\D/g, '');
                                if (value === '') {
                                  // remove the key from the config if empty
                                  let newConfig = objectCopy(jobConfig);
                                  if (newConfig.config.process[0].sample.samples[i]) {
                                    delete newConfig.config.process[0].sample.samples[i].height;
                                    setJobConfig(
                                      newConfig.config.process[0].sample.samples,
                                      'config.process[0].sample.samples',
                                    );
                                  }
                                } else {
                                  const intValue = parseInt(value);
                                  if (!isNaN(intValue)) {
                                    setJobConfig(intValue, `config.process[0].sample.samples[${i}].height`);
                                  } else {
                                    console.warn('Invalid height value:', value);
                                  }
                                }
                              }}
                              placeholder={`${jobConfig.config.process[0].sample.height} (default)`}
                            />
                          )}
                          <TextInput
                            label={`Seed`}
                            value={sample.seed ? `${sample.seed}` : ''}
                            onChange={value => {
                              // remove any non-numeric characters
                              value = value.replace(/\D/g, '');
                              if (value === '') {
                                // remove the key from the config if empty
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].seed;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                const intValue = parseInt(value);
                                if (!isNaN(intValue)) {
                                  setJobConfig(intValue, `config.process[0].sample.samples[${i}].seed`);
                                } else {
                                  console.warn('Invalid seed value:', value);
                                }
                              }
                            }}
                            placeholder={`${jobConfig.config.process[0].sample.walk_seed ? jobConfig.config.process[0].sample.seed + i : jobConfig.config.process[0].sample.seed} (default)`}
                          />
                          <TextInput
                            label={`LoRA Scale`}
                            value={sample.network_multiplier ? `${sample.network_multiplier}` : ''}
                            onChange={value => {
                              // remove any non-numeric, - or . characters
                              value = value.replace(/[^0-9.-]/g, '');
                              if (value === '') {
                                // remove the key from the config if empty
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].network_multiplier;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                // set it as a string
                                setJobConfig(value, `config.process[0].sample.samples[${i}].network_multiplier`);
                                return;
                              }
                            }}
                            placeholder={`1.0 (default)`}
                          />
                        </div>
                      </div>
                      {modelArch?.additionalSections?.includes('datasets.multi_control_paths') && (
                        <FormGroup label="Control Images" className="pt-2 ml-4">
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mt-2 mt-2">
                            {['ctrl_img_1', 'ctrl_img_2', 'ctrl_img_3'].map((ctrlKey, ctrl_idx) => (
                              <SampleControlImage
                                key={ctrlKey}
                                instruction={`Add Control Image ${ctrl_idx + 1}`}
                                className=""
                                src={sample[ctrlKey as keyof typeof sample] as string}
                                onNewImageSelected={imagePath => {
                                  if (!imagePath) {
                                    let newSamples = objectCopy(jobConfig.config.process[0].sample.samples);
                                    delete newSamples[i][ctrlKey as keyof typeof sample];
                                    setJobConfig(newSamples, 'config.process[0].sample.samples');
                                  } else {
                                    setJobConfig(imagePath, `config.process[0].sample.samples[${i}].${ctrlKey}`);
                                  }
                                }}
                              />
                            ))}
                          </div>
                        </FormGroup>
                      )}
                      {modelArch?.additionalSections?.includes('sample.ctrl_img') && (
                        <SampleControlImage
                          className="mt-6 ml-4"
                          src={sample.ctrl_img}
                          onNewImageSelected={imagePath => {
                            if (!imagePath) {
                              let newSamples = objectCopy(jobConfig.config.process[0].sample.samples);
                              delete newSamples[i].ctrl_img;
                              setJobConfig(newSamples, 'config.process[0].sample.samples');
                            } else {
                              setJobConfig(imagePath, `config.process[0].sample.samples[${i}].ctrl_img`);
                            }
                          }}
                        />
                      )}
                    </div>
                    <div className="pb-4"></div>
                  </div>
                  <div>
                    <button
                      type="button"
                      onClick={() =>
                        setJobConfig(
                          jobConfig.config.process[0].sample.samples.filter((_, index) => index !== i),
                          'config.process[0].sample.samples',
                        )
                      }
                      className="rounded-full p-1 text-sm"
                    >
                      <X />
                    </button>
                  </div>
                </div>
              </div>
            ))}
            <button
              type="button"
              onClick={() =>
                setJobConfig(
                  [...jobConfig.config.process[0].sample.samples, { prompt: '' }],
                  'config.process[0].sample.samples',
                )
              }
              className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
            >
              Add Prompt
            </button>
            {!modelArch?.disableSections?.includes('sample.neg') && (
              <>
                <TextAreaInput
                  label="Negative Prompt"
                  value={jobConfig.config.process[0].sample.neg || ''}
                  onChange={value => setJobConfig(value || false, 'config.process[0].sample.neg')}
                  placeholder="eg. blurry, low quality"
                  className="pt-2"
                />
                {jobConfig.config.process[0].sample.guidance_scale <= 1 && !!jobConfig.config.process[0].sample.neg && (
                  <p className="text-xs text-yellow-500 mt-1">
                    CFG = 1 — negative prompt will be ignored.
                  </p>
                )}
              </>
            )}

            {/* Sampling LoRA — WAN 2.2 (two-stage LightX2V: high-noise + low-noise) */}
            {modelArch?.additionalSections?.includes('sample.lightx2v_loras') && (
              <div className="mt-4 border-t border-gray-700 pt-4">
                <Checkbox
                  label="Enable LoRA during sampling"
                  docKey="sample.lightx2v_loras"
                  checked={
                    jobConfig.config.process[0].sample.sample_lora_path !== null &&
                    jobConfig.config.process[0].sample.sample_lora_path !== undefined
                  }
                  onChange={value => {
                    if (value) {
                      setJobConfig('', 'config.process[0].sample.sample_lora_path');
                      setJobConfig('', 'config.process[0].sample.sample_lora_path_2');
                      setJobConfig(1.0, 'config.process[0].sample.sample_lora_strength');
                      setJobConfig(1.0, 'config.process[0].sample.sample_lora_strength_2');
                    } else {
                      setJobConfig(null, 'config.process[0].sample.sample_lora_path');
                      setJobConfig(null, 'config.process[0].sample.sample_lora_path_2');
                    }
                  }}
                />
                {jobConfig.config.process[0].sample.sample_lora_path !== null &&
                  jobConfig.config.process[0].sample.sample_lora_path !== undefined && (
                  <div className="mt-2 pl-2 space-y-2">
                    <LoraPathInput
                      label="Stage 1 LoRA (High Noise)"
                      value={jobConfig.config.process[0].sample.sample_lora_path ?? ''}
                      onChange={v => setJobConfig(v, 'config.process[0].sample.sample_lora_path')}
                    />
                    <SliderInput
                      label="Stage 1 Strength"
                      value={jobConfig.config.process[0].sample.sample_lora_strength ?? 1.0}
                      onChange={value => setJobConfig(value, 'config.process[0].sample.sample_lora_strength')}
                      min={0}
                      max={2}
                      step={0.05}
                    />
                    <LoraPathInput
                      label="Stage 2 LoRA (Low Noise)"
                      value={jobConfig.config.process[0].sample.sample_lora_path_2 ?? ''}
                      onChange={v => setJobConfig(v, 'config.process[0].sample.sample_lora_path_2')}
                    />
                    <SliderInput
                      label="Stage 2 Strength"
                      value={jobConfig.config.process[0].sample.sample_lora_strength_2 ?? 1.0}
                      onChange={value => setJobConfig(value, 'config.process[0].sample.sample_lora_strength_2')}
                      min={0}
                      max={2}
                      step={0.05}
                    />
                  </div>
                )}
              </div>
            )}

            {/* Sampling LoRA — LTX-2 / LTX-2.3 */}
            {modelArch?.additionalSections?.includes('sample.distill_lora') && (
              <div className="mt-4 border-t border-gray-700 pt-4">
                <Checkbox
                  label="Enable LoRA during sampling"
                  docKey="sample.distill_lora"
                  checked={
                    jobConfig.config.process[0].sample.sample_lora_path !== null &&
                    jobConfig.config.process[0].sample.sample_lora_path !== undefined
                  }
                  onChange={value => {
                    if (value) {
                      setJobConfig('', 'config.process[0].sample.sample_lora_path');
                      setJobConfig(1.0, 'config.process[0].sample.sample_lora_strength');
                    } else {
                      setJobConfig(null, 'config.process[0].sample.sample_lora_path');
                    }
                  }}
                />
                {jobConfig.config.process[0].sample.sample_lora_path !== null &&
                  jobConfig.config.process[0].sample.sample_lora_path !== undefined && (
                  <div className="mt-2 pl-2 space-y-2">
                    <LoraPathInput
                      label="LoRA Path"
                      value={jobConfig.config.process[0].sample.sample_lora_path ?? ''}
                      onChange={v => setJobConfig(v, 'config.process[0].sample.sample_lora_path')}
                    />
                    {!jobConfig.config.process[0].model.spatial_upscaler_path && (
                      <SliderInput
                        label="Strength"
                        value={jobConfig.config.process[0].sample.sample_lora_strength ?? 1.0}
                        onChange={value => setJobConfig(value, 'config.process[0].sample.sample_lora_strength')}
                        min={0}
                        max={2}
                        step={0.05}
                      />
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Sampling LoRA — Qwen Image */}
            {modelArch?.additionalSections?.includes('sample.sampling_lora') && (
              <div className="mt-4 border-t border-gray-700 pt-4">
                <Checkbox
                  label="Enable LoRA during sampling"
                  docKey="sample.sampling_lora"
                  checked={
                    jobConfig.config.process[0].sample.sample_lora_path !== null &&
                    jobConfig.config.process[0].sample.sample_lora_path !== undefined
                  }
                  onChange={value => {
                    if (value) {
                      setJobConfig('', 'config.process[0].sample.sample_lora_path');
                      setJobConfig(1.0, 'config.process[0].sample.sample_lora_strength');
                    } else {
                      setJobConfig(null, 'config.process[0].sample.sample_lora_path');
                    }
                  }}
                />
                {jobConfig.config.process[0].sample.sample_lora_path !== null &&
                  jobConfig.config.process[0].sample.sample_lora_path !== undefined && (
                  <div className="mt-2 pl-2 space-y-2">
                    <LoraPathInput
                      label="LoRA Path"
                      value={jobConfig.config.process[0].sample.sample_lora_path ?? ''}
                      onChange={v => setJobConfig(v, 'config.process[0].sample.sample_lora_path')}
                    />
                    <SliderInput
                      label="Strength"
                      value={jobConfig.config.process[0].sample.sample_lora_strength ?? 1.0}
                      onChange={value => setJobConfig(value, 'config.process[0].sample.sample_lora_strength')}
                      min={0}
                      max={2}
                      step={0.05}
                    />
                  </div>
                )}
              </div>
            )}
            {/* Sampling LoRA — MiniMax-H3 (4-step turbo / step distill) */}
            {modelArch?.additionalSections?.includes('sample.minimax_h3_turbo_lora') && (
              <div className="mt-4 border-t border-gray-700 pt-4">
                <Checkbox
                  label="Use turbo LoRA during sampling"
                  docKey="sample.minimax_h3_turbo_lora"
                  checked={
                    jobConfig.config.process[0].sample.sample_lora_path !== null &&
                    jobConfig.config.process[0].sample.sample_lora_path !== undefined
                  }
                  onChange={value => {
                    if (value) {
                      setJobConfig('', 'config.process[0].sample.sample_lora_path');
                      setJobConfig(1.0, 'config.process[0].sample.sample_lora_strength');
                    } else {
                      setJobConfig(null, 'config.process[0].sample.sample_lora_path');
                    }
                  }}
                />
                {jobConfig.config.process[0].sample.sample_lora_path !== null &&
                  jobConfig.config.process[0].sample.sample_lora_path !== undefined && (
                  <div className="mt-2 pl-2 space-y-2">
                    <LoraPathInput
                      label="Turbo LoRA Path"
                      value={jobConfig.config.process[0].sample.sample_lora_path ?? ''}
                      onChange={v => setJobConfig(v, 'config.process[0].sample.sample_lora_path')}
                    />
                    <SliderInput
                      label="Strength"
                      value={jobConfig.config.process[0].sample.sample_lora_strength ?? 1.0}
                      onChange={value => setJobConfig(value, 'config.process[0].sample.sample_lora_strength')}
                      min={0}
                      max={2}
                      step={0.05}
                    />
                    <p className="text-xs text-gray-500">
                      Applied only while generating samples and removed again before training resumes. Drop Sample Steps
                      to 4–8 to get the speedup.
                    </p>
                  </div>
                )}
              </div>
            )}
            {modelArch?.additionalSections?.includes('sample.krea2_sampling_lora') && (
              <div className="mt-4 border-t border-gray-700 pt-4">
                <Checkbox
                  label="Apply LoRA during sampling"
                  docKey="sample.sampling_lora"
                  checked={
                    jobConfig.config.process[0].sample.sample_lora_path !== null &&
                    jobConfig.config.process[0].sample.sample_lora_path !== undefined
                  }
                  onChange={value => {
                    if (value) {
                      setJobConfig('', 'config.process[0].sample.sample_lora_path');
                      setJobConfig(1, 'config.process[0].sample.sample_lora_strength');
                    } else {
                      setJobConfig(null, 'config.process[0].sample.sample_lora_path');
                      setJobConfig(null, 'config.process[0].sample.sample_lora_path_2');
                    }
                  }}
                />
                {jobConfig.config.process[0].sample.sample_lora_path !== null &&
                  jobConfig.config.process[0].sample.sample_lora_path !== undefined && (
                  <div className="mt-2 pl-2 space-y-3">
                    {jobConfig.config.process[0].model.layer_offloading && (
                      <p className="text-xs text-yellow-500">
                        Layer offloading is on — sample LoRA merging currently fails on offloaded (quantized)
                        layers and falls back to dequantizing them on the fly. That undoes the memory savings
                        layer offloading gives you on those layers (risking OOM on high-res buckets) and,
                        since a turbo/distill LoRA's whole point is speed, quietly gives up that benefit too.
                        Turn off layer offloading, or drop the sample LoRA, to avoid this.
                      </p>
                    )}
                    <div className="space-y-2">
                      <LoraPathInput
                        label="LoRA 1 Path"
                        value={jobConfig.config.process[0].sample.sample_lora_path ?? ''}
                        onChange={v => setJobConfig(v, 'config.process[0].sample.sample_lora_path')}
                      />
                      <NumberInput
                        label="Strength"
                        value={jobConfig.config.process[0].sample.sample_lora_strength ?? 1}
                        onChange={value => setJobConfig(value, 'config.process[0].sample.sample_lora_strength')}
                        min={0.01}
                        max={1000}
                      />
                    </div>
                    <div className="space-y-2">
                      <LoraPathInput
                        label="LoRA 2 Path (optional)"
                        value={jobConfig.config.process[0].sample.sample_lora_path_2 ?? ''}
                        onChange={v => setJobConfig(v || null, 'config.process[0].sample.sample_lora_path_2')}
                      />
                      {jobConfig.config.process[0].sample.sample_lora_path_2 && (
                        <NumberInput
                          label="Strength"
                          value={jobConfig.config.process[0].sample.sample_lora_strength_2 ?? 4}
                          onChange={value => setJobConfig(value, 'config.process[0].sample.sample_lora_strength_2')}
                          min={0.01}
                          max={1000}
                        />
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </Card>
        </div>

        {status === 'success' && <p className="text-green-500 text-center">Training saved successfully!</p>}
        {status === 'error' && <p className="text-red-500 text-center">Error saving training. Please try again.</p>}
      </form>
      <AddSingleImageModal />
    </>
  );
}
