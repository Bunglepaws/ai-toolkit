import { NextRequest } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import os from 'os';
import crypto from 'crypto';
import { TOOLKIT_ROOT } from '@/paths';
import { getDatasetsRoot } from '@/server/settings';
import { resolvePythonPath } from '../../../../../../cron/pythonPath';

export const runtime = 'nodejs';
export const maxDuration = 1200;

const CROP_SCRIPT = path.join(TOOLKIT_ROOT, 'ui_scripts', 'crop_resize_dataset.py');

function uniqueOutputDir(base: string): string {
  if (!fs.existsSync(base)) return base;
  for (let n = 2; n < 100; n++) {
    const candidate = `${base}_${n}`;
    if (!fs.existsSync(candidate)) return candidate;
  }
  return `${base}_${Date.now()}`;
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ datasetName: string }> },
) {
  const { datasetName } = await params;

  let body: {
    resolution?: number;
    divisibility?: number;
    mode?: 'resize' | 'crop';
    decisions?: Record<string, { anchor_x: number; anchor_y: number }>;
  };
  try {
    body = await request.json();
  } catch {
    return new Response('Invalid JSON', { status: 400 });
  }

  const resolution = typeof body.resolution === 'number' ? body.resolution : 1024;
  const divisibility = typeof body.divisibility === 'number' ? body.divisibility : 8;
  const mode = body.mode === 'crop' ? 'crop' : 'resize';
  const decisions = body.decisions ?? {};

  const datasetsRoot = await getDatasetsRoot();
  const inputDir = path.join(datasetsRoot, datasetName);
  const outputBase = path.join(datasetsRoot, `${datasetName}_bucket_${resolution}x${resolution}`);
  const outputDir = uniqueOutputDir(outputBase);
  const outputName = path.basename(outputDir);

  // Write the decisions map to a temp file so the CLI arg stays short
  const decisionsFile = path.join(os.tmpdir(), `decisions-${crypto.randomUUID()}.json`);
  await fs.promises.writeFile(decisionsFile, JSON.stringify(decisions), 'utf-8');

  const args = [
    '--input-dir', inputDir,
    '--output-dir', outputDir,
    '--resolution', String(resolution),
    '--divisibility', String(divisibility),
    '--mode', mode,
    '--decisions', decisionsFile,
  ];

  const child = spawn(resolvePythonPath(), ['-u', CROP_SCRIPT, ...args], {
    cwd: TOOLKIT_ROOT,
    env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
    windowsHide: true,
  });

  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    start(controller) {
      let stdoutBuf = '';

      child.stdout.on('data', (chunk: Buffer) => {
        const text = chunk.toString('utf-8');
        stdoutBuf += text;
        // Forward each NDJSON line to the client, enriched with the output dataset name
        for (const line of text.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            const obj = JSON.parse(trimmed);
            if (obj.type === 'done') {
              obj.outputName = outputName;
            }
            controller.enqueue(encoder.encode(JSON.stringify(obj) + '\n'));
          } catch {
            // non-JSON line — pass through as a log event
            controller.enqueue(encoder.encode(JSON.stringify({ type: 'log', message: trimmed }) + '\n'));
          }
        }
      });

      child.stderr.on('data', (chunk: Buffer) => {
        const text = chunk.toString('utf-8').trim();
        if (text) {
          controller.enqueue(encoder.encode(JSON.stringify({ type: 'log', message: text }) + '\n'));
        }
      });

      child.on('error', err => {
        controller.enqueue(encoder.encode(JSON.stringify({ type: 'error', message: err.message }) + '\n'));
        controller.close();
        fs.promises.unlink(decisionsFile).catch(() => {});
      });

      child.on('close', () => {
        controller.close();
        fs.promises.unlink(decisionsFile).catch(() => {});
      });
    },
    cancel() {
      if (!child.killed) child.kill('SIGKILL');
      fs.promises.unlink(decisionsFile).catch(() => {});
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'application/x-ndjson; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      'X-Accel-Buffering': 'no',
    },
  });
}
