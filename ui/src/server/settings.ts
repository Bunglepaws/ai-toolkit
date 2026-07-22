import prisma from '@/server/prisma';
import { defaultDatasetsFolder, defaultDataRoot } from '@/paths';
import { defaultTrainFolder } from '@/paths';
import NodeCache from 'node-cache';

const myCache = new NodeCache();

export const flushCache = () => {
  myCache.flushAll();
};

export const getDatasetsRoot = async () => {
  const key = 'DATASETS_FOLDER';
  let datasetsPath = myCache.get(key) as string;
  if (datasetsPath) {
    return datasetsPath;
  }
  let row = await prisma.settings.findFirst({
    where: {
      key: 'DATASETS_FOLDER',
    },
  });
  datasetsPath = defaultDatasetsFolder;
  if (row?.value && row.value !== '') {
    datasetsPath = row.value;
  }
  myCache.set(key, datasetsPath);
  return datasetsPath as string;
};

export const getTrainingFolder = async () => {
  const key = 'TRAINING_FOLDER';
  let trainingRoot = myCache.get(key) as string;
  if (trainingRoot) {
    return trainingRoot;
  }
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  trainingRoot = defaultTrainFolder;
  if (row?.value && row.value !== '') {
    trainingRoot = row.value;
  }
  myCache.set(key, trainingRoot);
  return trainingRoot as string;
};

export const getGemmaApiKey = async () => {
  const key = 'GEMMA_API_KEY';
  let apiKey = myCache.get(key) as string;
  if (apiKey) {
    return apiKey;
  }
  let row = await prisma.settings.findFirst({
    where: { key: key },
  });
  apiKey = '';
  if (row?.value && row.value !== '') {
    apiKey = row.value;
  }
  myCache.set(key, apiKey);
  return apiKey;
};

export const getHFToken = async () => {
  const key = 'HF_TOKEN';
  let token = myCache.get(key) as string;
  if (token) {
    return token;
  }
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  token = '';
  if (row?.value && row.value !== '') {
    token = row.value;
  }
  myCache.set(key, token);
  return token;
};

export const getCheckConfigEnableWebSearch = async () => {
  const key = 'CHECK_CONFIG_ENABLE_WEB_SEARCH';
  let val = myCache.get(key) as string | undefined;
  if (val !== undefined) return val === 'true';
  const row = await prisma.settings.findFirst({ where: { key } });
  val = row?.value || 'false';
  myCache.set(key, val);
  return val === 'true';
};

export const getCheckConfigApiBaseUrl = async () => {
  const key = 'CHECK_CONFIG_API_BASE_URL';
  let val = myCache.get(key) as string;
  if (val !== undefined) return val;
  const row = await prisma.settings.findFirst({ where: { key } });
  val = row?.value || '';
  myCache.set(key, val);
  return val;
};

export const getCheckConfigApiKey = async () => {
  const key = 'CHECK_CONFIG_API_KEY';
  let val = myCache.get(key) as string;
  if (val !== undefined) return val;
  const row = await prisma.settings.findFirst({ where: { key } });
  val = row?.value || '';
  myCache.set(key, val);
  return val;
};

export const getCheckConfigModel = async () => {
  const key = 'CHECK_CONFIG_MODEL';
  let val = myCache.get(key) as string;
  if (val !== undefined) return val;
  const row = await prisma.settings.findFirst({ where: { key } });
  val = row?.value || 'claude-sonnet-5';
  myCache.set(key, val);
  return val;
};

export const getQuantizationCacheDir = async () => {
  const key = 'QUANTIZATION_CACHE_DIR';
  let cacheDir = myCache.get(key) as string;
  if (cacheDir) {
    return cacheDir;
  }
  let row = await prisma.settings.findFirst({
    where: { key: key },
  });
  cacheDir = '';
  if (row?.value && row.value !== '') {
    cacheDir = row.value;
  }
  myCache.set(key, cacheDir);
  return cacheDir;
};

export const getDataRoot = async () => {
  const key = 'DATA_ROOT';
  let dataRoot = myCache.get(key) as string;
  if (dataRoot) {
    return dataRoot;
  }
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  dataRoot = defaultDataRoot;
  if (row?.value && row.value !== '') {
    dataRoot = row.value;
  }
  myCache.set(key, dataRoot);
  return dataRoot;
};
