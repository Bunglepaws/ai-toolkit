import fs from 'fs';
import os from 'os';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { modelArchs } from '@/app/jobs/new/options';
import { audioExtensions, imgExtensions, videoExtensions } from '@/utils/basic';

const execFileAsync = promisify(execFile);

export type ModelFamily = 'image' | 'video' | 'audio';

export interface MediaItem {
  dataUrl: string;
  label: string;
}

export interface SampleMediaResult {
  items: MediaItem[];
  ffmpegUnavailable?: boolean;
}

const MAX_SAMPLE_IMAGES = 4;
const MAX_DATASET_IMAGES = 6;
const MAX_VIDEO_FRAMES = 4;
/** We keep images small so multimodal prompts stay within context budgets. */
const MAX_IMAGE_BYTES = 1_500_000;

export function getModelFamily(archOrProcessType: string): ModelFamily {
  const arch =
    modelArchs.find(a => a.name === archOrProcessType) ??
    modelArchs.find(a => archOrProcessType.includes(a.name));

  if (arch?.group === 'audio') return 'audio';
  if (arch?.group === 'video' || arch?.isVideoModel) return 'video';

  // Heuristic fallback when arch is missing / mistyped
  const lower = archOrProcessType.toLowerCase();
  if (lower.includes('ace_step') || lower.includes('audio')) return 'audio';
  if (
    lower.includes('wan') ||
    lower.includes('ltx') ||
    lower.includes('video') ||
    lower.includes('hunyuan')
  ) {
    return 'video';
  }
  return 'image';
}

function mimeForExt(ext: string): string {
  switch (ext.toLowerCase()) {
    case '.png':
      return 'image/png';
    case '.webp':
      return 'image/webp';
    case '.gif':
      return 'image/gif';
    case '.jpg':
    case '.jpeg':
    default:
      return 'image/jpeg';
  }
}

async function fileToDataUrl(filePath: string): Promise<string | null> {
  try {
    const data = await fs.promises.readFile(filePath);
    if (data.length === 0 || data.length > MAX_IMAGE_BYTES) return null;
    const ext = path.extname(filePath).toLowerCase();
    return `data:${mimeForExt(ext)};base64,${data.toString('base64')}`;
  } catch {
    return null;
  }
}

async function listFiles(dir: string): Promise<string[]> {
  try {
    const entries = await fs.promises.readdir(dir, { withFileTypes: true });
    return entries.filter(e => e.isFile() && !e.name.startsWith('.')).map(e => path.join(dir, e.name));
  } catch {
    return [];
  }
}

async function mtimeMs(filePath: string): Promise<number> {
  try {
    return (await fs.promises.stat(filePath)).mtimeMs;
  } catch {
    return 0;
  }
}

async function sortByMtimeDesc(files: string[]): Promise<string[]> {
  const withMtime = await Promise.all(files.map(async f => ({ f, m: await mtimeMs(f) })));
  withMtime.sort((a, b) => b.m - a.m);
  return withMtime.map(x => x.f);
}

async function ffmpegAvailable(): Promise<boolean> {
  try {
    await execFileAsync('ffmpeg', ['-version'], { timeout: 5000 });
    return true;
  } catch {
    return false;
  }
}

/** Extract up to `count` JPEG frames from a video; returns data URLs (temp files cleaned up). */
async function extractVideoFrameDataUrls(videoPath: string, count: number): Promise<string[]> {
  const tmpDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'aitk-check-config-'));
  const outPattern = path.join(tmpDir, 'frame_%02d.jpg');

  try {
    // fps filter picks ~count frames across a short clip without needing duration first
    await execFileAsync(
      'ffmpeg',
      [
        '-y',
        '-i',
        videoPath,
        '-vf',
        `fps=1/${Math.max(1, Math.round(8 / count))},scale=512:-2`,
        '-frames:v',
        String(count),
        '-q:v',
        '5',
        outPattern,
      ],
      { timeout: 30_000 },
    );

    const frames = (await listFiles(tmpDir))
      .filter(f => f.endsWith('.jpg'))
      .sort()
      .slice(0, count);

    const urls: string[] = [];
    for (const framePath of frames) {
      const dataUrl = await fileToDataUrl(framePath);
      if (dataUrl) urls.push(dataUrl);
    }
    return urls;
  } catch {
    return [];
  } finally {
    try {
      await fs.promises.rm(tmpDir, { recursive: true, force: true });
    } catch {
      /* ignore */
    }
  }
}

export async function collectSampleMedia(
  trainingFolder: string,
  jobName: string,
  modelFamily: ModelFamily,
): Promise<SampleMediaResult> {
  if (modelFamily === 'audio') {
    return { items: [] };
  }

  const samplesFolder = path.join(trainingFolder, jobName, 'samples');
  const files = await listFiles(samplesFolder);
  if (files.length === 0) return { items: [] };

  const images = files.filter(f => imgExtensions.includes(path.extname(f).toLowerCase()));
  const videos = files.filter(f => videoExtensions.includes(path.extname(f).toLowerCase()));

  // Prefer still samples when present; otherwise pull frames from the newest MP4
  if (images.length > 0) {
    const recent = (await sortByMtimeDesc(images)).slice(0, MAX_SAMPLE_IMAGES);
    const items: MediaItem[] = [];
    for (const filePath of recent) {
      const dataUrl = await fileToDataUrl(filePath);
      if (dataUrl) {
        items.push({ dataUrl, label: `sample:${path.basename(filePath)}` });
      }
    }
    return { items };
  }

  if (modelFamily === 'video' && videos.length > 0) {
    if (!(await ffmpegAvailable())) {
      return { items: [], ffmpegUnavailable: true };
    }
    const newest = (await sortByMtimeDesc(videos))[0];
    const frameUrls = await extractVideoFrameDataUrls(newest, MAX_VIDEO_FRAMES);
    if (frameUrls.length === 0) {
      return { items: [], ffmpegUnavailable: true };
    }

    return {
      items: frameUrls.map((dataUrl, i) => ({
        dataUrl,
        label: `sample-frame:${path.basename(newest)}#${i + 1}`,
      })),
    };
  }

  return { items: [] };
}

async function collectImagesFromDir(dir: string, limit: number): Promise<string[]> {
  const results: string[] = [];

  async function walk(current: string) {
    if (results.length >= limit) return;
    let entries;
    try {
      entries = await fs.promises.readdir(current, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (results.length >= limit) return;
      if (entry.name.startsWith('.') || entry.name === '_controls') continue;
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) {
        await walk(full);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).toLowerCase();
        if (imgExtensions.includes(ext) && !audioExtensions.includes(ext)) {
          results.push(full);
        }
      }
    }
  }

  await walk(dir);
  return results;
}

export async function collectDatasetImages(
  datasetPaths: string[],
  modelFamily: ModelFamily,
): Promise<MediaItem[]> {
  if (modelFamily === 'audio' || datasetPaths.length === 0) return [];

  // Spread the budget across datasets so multi-dataset jobs aren't biased to the first folder
  const perDataset = Math.max(1, Math.ceil(MAX_DATASET_IMAGES / datasetPaths.length));
  const candidates: string[] = [];

  for (const folder of datasetPaths) {
    if (!folder || folder.startsWith('/path/to/')) continue;
    const found = await collectImagesFromDir(folder, perDataset * 3);
    const recent = (await sortByMtimeDesc(found)).slice(0, perDataset);
    candidates.push(...recent);
    if (candidates.length >= MAX_DATASET_IMAGES) break;
  }

  const items: MediaItem[] = [];
  for (const filePath of candidates.slice(0, MAX_DATASET_IMAGES)) {
    const dataUrl = await fileToDataUrl(filePath);
    if (dataUrl) {
      items.push({ dataUrl, label: `dataset:${path.basename(filePath)}` });
    }
  }
  return items;
}
