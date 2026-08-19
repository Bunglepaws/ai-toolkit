import path from 'path';
import prisma from './prisma';

export const TOOLKIT_ROOT = path.resolve('@', '..', '..');
export const defaultTrainFolder = path.join(TOOLKIT_ROOT, 'output');
export const defaultDatasetsFolder = path.join(TOOLKIT_ROOT, 'datasets');
export const defaultDataRoot = path.join(TOOLKIT_ROOT, 'data');
export const defaultModelsFolder = path.join(TOOLKIT_ROOT, 'models');

// Forked file-server workers set AI_TOOLKIT_QUIET_PATHS so this line prints
// once per launched process group, not once per worker.
if (!process.env.AI_TOOLKIT_QUIET_PATHS) {
  console.log('TOOLKIT_ROOT:', TOOLKIT_ROOT);
}

export const getTrainingFolder = async () => {
  const key = 'TRAINING_FOLDER';
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  let trainingRoot = defaultTrainFolder;
  if (row?.value && row.value !== '') {
    trainingRoot = row.value;
  }
  return trainingRoot as string;
};

export const getHFToken = async () => {
  const key = 'HF_TOKEN';
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  let token = '';
  if (row?.value && row.value !== '') {
    token = row.value;
  }
  return token;
};

export const getGemmaApiKey = async () => {
  const key = 'GEMMA_API_KEY';
  let row = await prisma.settings.findFirst({
    where: { key: key },
  });
  let apiKey = '';
  if (row?.value && row.value !== '') {
    apiKey = row.value;
  }
  return apiKey;
};

// See ui/src/server/settings.ts getGemmaApiModelIdSource -- same setting, this
// copy is what startJob.ts reads to inject GEMMA_API_MODEL_ID_SOURCE into the
// trainer's environment (cron runs from compiled dist/cron, separate from the
// Next server, hence the duplicated getter).
export const getGemmaApiModelIdSource = async () => {
  const key = 'GEMMA_API_MODEL_ID_SOURCE';
  let row = await prisma.settings.findFirst({
    where: { key: key },
  });
  let path = '';
  if (row?.value && row.value !== '') {
    path = row.value;
  }
  return path;
};

export const getQuantizationCacheDir = async () => {
  const key = 'QUANTIZATION_CACHE_DIR';
  let row = await prisma.settings.findFirst({
    where: { key: key },
  });
  if (row?.value && row.value !== '') {
    return row.value;
  }
  // Default: {trainingFolder}/quantized
  const trainingFolder = await getTrainingFolder();
  return path.join(trainingFolder, 'quantized');
};

export const getEnableHotModelReload = async () => {
  const key = 'ENABLE_HOT_MODEL_RELOAD';
  let row = await prisma.settings.findFirst({ where: { key } });
  // default true — matches prior (always-on) behavior when the setting has never been saved
  return row?.value ? row.value === 'true' : true;
};

// See ui/src/server/settings.ts getSamplePreviewEnabled -- same setting, this
// copy is what startJob.ts reads to inject AITK_SAMPLE_PREVIEW into the
// trainer's environment (cron runs from compiled dist/cron, separate from the
// Next server, hence the duplicated getter).
export const getSamplePreviewEnabled = async () => {
  const key = 'AITK_SAMPLE_PREVIEW';
  let row = await prisma.settings.findFirst({ where: { key } });
  // default true — matches toolkit/sample_preview.py's own env-var default
  return row?.value ? row.value === 'true' : true;
};

export const getOmniVoiceModelPath = async () => {
  // TTS weights for voice-clip generation. Machine-level infrastructure, so it lives in
  // Settings rather than per job. Empty means the feature errors with a clear message
  // rather than silently starting a multi-GB HuggingFace download.
  const key = 'OMNIVOICE_MODEL_PATH';
  const row = await prisma.settings.findFirst({ where: { key: key } });
  return row?.value && row.value !== '' ? row.value : '';
};

export const getModelsPath = async () => {
  const key = 'MODELS_PATH';
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  let modelsPath = '';
  if (row?.value && row.value !== '' && row.value !== defaultModelsFolder) {
    modelsPath = row.value;
  }
  return modelsPath;
};
