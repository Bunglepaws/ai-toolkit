import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import { TOOLKIT_ROOT } from '@/paths';
import { getDatasetsRoot } from '@/server/settings';
import { resolvePythonPath } from '../../../../../../cron/pythonPath';

export const runtime = 'nodejs';

const SCAN_SCRIPT = path.join(TOOLKIT_ROOT, 'ui_scripts', 'scan_dataset_buckets.py');

function runScan(args: string[]): Promise<{ ok: boolean; result: unknown; error?: string }> {
  return new Promise(resolve => {
    const child = spawn(resolvePythonPath(), ['-u', SCAN_SCRIPT, ...args], {
      cwd: TOOLKIT_ROOT,
      env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
      windowsHide: true,
    });

    let stdout = '';
    let stderr = '';

    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      resolve({ ok: false, result: null, error: 'Scan timed out after 60 s' });
    }, 60_000);

    child.stdout.on('data', (c: Buffer) => { stdout += c.toString('utf-8'); });
    child.stderr.on('data', (c: Buffer) => { stderr += c.toString('utf-8'); });

    child.on('error', err => {
      clearTimeout(timer);
      resolve({ ok: false, result: null, error: err.message });
    });

    child.on('close', code => {
      clearTimeout(timer);
      const lines = stdout.trimEnd().split(/\r?\n/);
      for (let i = lines.length - 1; i >= 0; i--) {
        const line = lines[i].trim();
        if (line.startsWith('{')) {
          try {
            resolve({ ok: code === 0, result: JSON.parse(line) });
            return;
          } catch {}
        }
      }
      resolve({ ok: false, result: null, error: stderr.trim() || 'No JSON output from scan script' });
    });
  });
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ datasetName: string }> },
) {
  const { datasetName } = await params;

  let body: { resolution?: number; divisibility?: number } = {};
  try {
    body = await request.json();
  } catch {
    // body is optional — defaults apply
  }

  const datasetsRoot = await getDatasetsRoot();
  const datasetDir = path.join(datasetsRoot, datasetName);

  const resolution = typeof body.resolution === 'number' ? body.resolution : 1024;
  const divisibility = typeof body.divisibility === 'number' ? body.divisibility : 8;

  const { ok, result, error } = await runScan([
    '--dataset-dir', datasetDir,
    '--resolution', String(resolution),
    '--divisibility', String(divisibility),
  ]);

  if (!ok || !result) {
    return NextResponse.json({ error: error ?? 'Scan failed' }, { status: 500 });
  }

  return NextResponse.json(result);
}
