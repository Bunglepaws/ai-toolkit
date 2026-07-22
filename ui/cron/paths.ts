import path from 'path';
import prisma from './prisma';

export const TOOLKIT_ROOT = path.resolve('@', '..', '..');
export const defaultTrainFolder = path.join(TOOLKIT_ROOT, 'output');
export const defaultDatasetsFolder = path.join(TOOLKIT_ROOT, 'datasets');
export const defaultDataRoot = path.join(TOOLKIT_ROOT, 'data');

console.log('TOOLKIT_ROOT:', TOOLKIT_ROOT);

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
