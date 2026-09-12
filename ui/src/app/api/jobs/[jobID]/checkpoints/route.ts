import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

export interface CheckpointEntry {
  step: number;
  filename: string;
  hasOptimizer: boolean;
  sizeBytes: number;
  mtimeMs: number;
}

// Matches plain (_00001.safetensors) and WAN 2.2 split variants
// (_00010016_high_noise.safetensors, _00010016_low_noise.safetensors).
// \d{5,} covers both the current 5-digit padding and legacy 9-digit files
// that haven't been renamed by a resumed training run yet.
const STEP_RE = /_(\d{5,})(?:_(?:high|low)_noise)?\.safetensors$/;

export async function GET(_req: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  if (job.status === 'running') return NextResponse.json({ error: 'Job is running' }, { status: 400 });

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);

  if (!fs.existsSync(jobFolder)) {
    return NextResponse.json({ checkpoints: [], currentStep: job.step });
  }

  let files: string[];
  try {
    files = fs.readdirSync(jobFolder);
  } catch {
    return NextResponse.json({ checkpoints: [], currentStep: job.step });
  }

  // Collect one entry per step (prefer the primary file — named exactly {job.name}_{step}.safetensors)
  const stepMap = new Map<number, CheckpointEntry>();
  for (const f of files) {
    const m = f.match(STEP_RE);
    if (!m) continue;
    const step = parseInt(m[1], 10);
    const filePath = path.join(jobFolder, f);
    let stat: fs.Stats;
    try { stat = fs.statSync(filePath); } catch { continue; }

    const existing = stepMap.get(step);
    // Prefer: plain > _high_noise > _low_noise (first seen wins for equal priority)
    const isPlain = f === `${job.name}_${m[1]}.safetensors`;
    const isHighNoise = f.endsWith('_high_noise.safetensors');
    const existingIsPlain = existing && existing.filename === `${job.name}_${m[1]}.safetensors`;
    const existingIsHighNoise = existing && existing.filename.endsWith('_high_noise.safetensors');
    const shouldReplace = !existing || isPlain || (isHighNoise && !existingIsPlain && !existingIsHighNoise);
    if (shouldReplace) {
      stepMap.set(step, {
        step,
        filename: f,
        hasOptimizer: false, // filled below
        sizeBytes: stat.size,
        mtimeMs: stat.mtimeMs,
      });
    }
  }

  // Check for optimizer archives. Try the current 5-digit padding first, then
  // fall back to legacy 9-digit padding for files not yet renamed.
  for (const [step, entry] of stepMap) {
    const archive5 = `optimizer_${String(step).padStart(5, '0')}.pt`;
    const archive9 = `optimizer_${String(step).padStart(9, '0')}.pt`;
    entry.hasOptimizer =
      fs.existsSync(path.join(jobFolder, archive5)) || fs.existsSync(path.join(jobFolder, archive9));
  }

  const checkpoints = Array.from(stepMap.values()).sort((a, b) => b.step - a.step);
  return NextResponse.json({ checkpoints, currentStep: job.step });
}
