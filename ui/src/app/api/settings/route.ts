import { NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { defaultTrainFolder, defaultDatasetsFolder } from '@/paths';
import { flushCache } from '@/server/settings';
import fs from 'fs';
import path from 'path';

export async function GET() {
  try {
    const settings = await prisma.settings.findMany();
    const settingsObject = settings.reduce((acc: any, setting) => {
      acc[setting.key] = setting.value;
      return acc;
    }, {});
    // if TRAINING_FOLDER is not set, use default
    if (!settingsObject.TRAINING_FOLDER || settingsObject.TRAINING_FOLDER === '') {
      settingsObject.TRAINING_FOLDER = defaultTrainFolder;
    }
    // if DATASETS_FOLDER is not set, use default
    if (!settingsObject.DATASETS_FOLDER || settingsObject.DATASETS_FOLDER === '') {
      settingsObject.DATASETS_FOLDER = defaultDatasetsFolder;
    }
    // QUANTIZATION_CACHE_DIR defaults to empty (cron/paths.ts computes the fallback at job start)
    if (!settingsObject.QUANTIZATION_CACHE_DIR) {
      settingsObject.QUANTIZATION_CACHE_DIR = '';
    }

    // Check Config AI settings
    if (!settingsObject.CHECK_CONFIG_API_BASE_URL) settingsObject.CHECK_CONFIG_API_BASE_URL = '';
    if (!settingsObject.CHECK_CONFIG_API_KEY) settingsObject.CHECK_CONFIG_API_KEY = '';
    if (!settingsObject.CHECK_CONFIG_MODEL) settingsObject.CHECK_CONFIG_MODEL = 'claude-sonnet-5';
    if (!settingsObject.CHECK_CONFIG_ENABLE_WEB_SEARCH) settingsObject.CHECK_CONFIG_ENABLE_WEB_SEARCH = 'false';

    // Read version from version.py in root
    let version = 'unknown';
    try {
      const versionPath = path.join(process.cwd(), '..', 'version.py');
      const versionContent = fs.readFileSync(versionPath, 'utf8');
      const match = versionContent.match(/VERSION = ["']([^"']+)["']/);
      if (match) {
        version = match[1];
      }
    } catch (e) {
      console.error('Error reading version.py:', e);
    }
    settingsObject.VERSION = version;

    return NextResponse.json(settingsObject);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch settings' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const {
      HF_TOKEN, GEMMA_API_KEY, TRAINING_FOLDER, DATASETS_FOLDER, QUANTIZATION_CACHE_DIR,
      CHECK_CONFIG_API_BASE_URL, CHECK_CONFIG_API_KEY, CHECK_CONFIG_MODEL,
      CHECK_CONFIG_ENABLE_WEB_SEARCH,
    } = body;

    const upsert = (key: string, value: string) =>
      prisma.settings.upsert({ where: { key }, update: { value }, create: { key, value } });

    // Upsert all settings
    await Promise.all([
      upsert('HF_TOKEN', HF_TOKEN ?? ''),
      upsert('GEMMA_API_KEY', GEMMA_API_KEY ?? ''),
      upsert('TRAINING_FOLDER', TRAINING_FOLDER ?? ''),
      upsert('DATASETS_FOLDER', DATASETS_FOLDER ?? ''),
      upsert('QUANTIZATION_CACHE_DIR', QUANTIZATION_CACHE_DIR ?? ''),
      upsert('CHECK_CONFIG_API_BASE_URL', CHECK_CONFIG_API_BASE_URL ?? ''),
      upsert('CHECK_CONFIG_API_KEY', CHECK_CONFIG_API_KEY ?? ''),
      upsert('CHECK_CONFIG_MODEL', CHECK_CONFIG_MODEL ?? ''),
      upsert('CHECK_CONFIG_ENABLE_WEB_SEARCH', CHECK_CONFIG_ENABLE_WEB_SEARCH ?? 'false'),
    ]);

    flushCache();

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to update settings' }, { status: 500 });
  }
}
