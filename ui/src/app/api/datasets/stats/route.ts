import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';
import { imgExtensions, videoExtensions, audioExtensions } from '@/utils/basic';

// Media files count towards the item count. Caption files only contribute to the
// modified date — editing a caption is a real change to the dataset, but captions
// are not items.
const mediaExtensions = new Set([...imgExtensions, ...videoExtensions, ...audioExtensions]);
const captionExtensions = new Set(['.txt', '.json', '.caption']);

export interface DatasetStats {
  count: number;
  modified: number | null; // epoch ms of the newest media/caption file, null if empty
}

// Scanning a folder is cheap but not free (one stat per file), and this endpoint is
// called for every dataset on every visit to the datasets page. A short TTL cache keeps
// repeat visits instant while still picking up changes made elsewhere (caption edits from
// the dataset detail page, files dropped in by hand) within a minute. The client can force
// a rescan with `refresh: true`, which is what the table's refresh button does.
const CACHE_TTL_MS = 60_000;
const cache = new Map<string, { at: number; stats: DatasetStats }>();

async function scanDataset(datasetsPath: string, name: string): Promise<DatasetStats> {
  // Only the top level of the dataset folder. Subfolders hold cached/derived files
  // (latent caches, _controls, etc.) which are not part of the dataset's content.
  const folder = path.join(datasetsPath, name);
  let entries: fs.Dirent[];
  try {
    entries = await fs.promises.readdir(folder, { withFileTypes: true });
  } catch {
    return { count: 0, modified: null };
  }

  let count = 0;
  let modified: number | null = null;

  const files = entries.filter(entry => entry.isFile() && !entry.name.startsWith('.'));
  await Promise.all(
    files.map(async entry => {
      const ext = path.extname(entry.name).toLowerCase();
      const isMedia = mediaExtensions.has(ext);
      if (!isMedia && !captionExtensions.has(ext)) return;
      if (isMedia) count++;
      try {
        const stat = await fs.promises.stat(path.join(folder, entry.name));
        const mtime = stat.mtimeMs;
        if (modified === null || mtime > modified) modified = mtime;
      } catch {
        // file vanished between readdir and stat; ignore
      }
    }),
  );

  return { count, modified };
}

export async function POST(request: Request) {
  try {
    const datasetsPath = await getDatasetsRoot();
    const body = await request.json();
    const names: string[] = Array.isArray(body?.names) ? body.names : [];
    const refresh: boolean = body?.refresh === true;

    const now = Date.now();
    const stats: Record<string, DatasetStats> = {};

    await Promise.all(
      names.map(async name => {
        // Reject anything that isn't a plain folder name so a crafted name can't
        // walk out of the datasets root.
        if (typeof name !== 'string' || name === '' || name !== path.basename(name)) return;

        const cached = cache.get(name);
        if (!refresh && cached && now - cached.at < CACHE_TTL_MS) {
          stats[name] = cached.stats;
          return;
        }
        const result = await scanDataset(datasetsPath, name);
        cache.set(name, { at: Date.now(), stats: result });
        stats[name] = result;
      }),
    );

    return NextResponse.json({ stats });
  } catch (error) {
    console.error('Error fetching dataset stats:', error);
    return NextResponse.json({ error: 'Failed to fetch dataset stats' }, { status: 500 });
  }
}
