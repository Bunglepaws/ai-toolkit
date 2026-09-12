import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

/**
 * Live sampling preview for a job.
 *
 * The trainer writes preview.mp4 + preview.json into the job folder while a
 * sample is being denoised (see toolkit/sample_preview.py). This reports the
 * sidecar so the UI can decide whether a preview is worth showing, and hands
 * back a path the existing /api/files route can stream.
 *
 * Returns { available: false } rather than an error when there is nothing to
 * show — the normal case for most of a run — so the poller stays quiet.
 */
export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const jsonPath = path.join(jobFolder, 'preview.json');

  let meta: any;
  try {
    meta = JSON.parse(await fs.promises.readFile(jsonPath, 'utf8'));
  } catch {
    return NextResponse.json({ available: false });
  }

  // The writer replaces both files atomically, but a reader can still land
  // between the two replacements. Treating a missing clip as "nothing yet"
  // keeps that race invisible instead of flashing a broken player.
  const clipPath = path.join(jobFolder, meta?.file || 'preview.mp4');
  let mtimeMs = 0;
  try {
    mtimeMs = (await fs.promises.stat(clipPath)).mtimeMs;
  } catch {
    return NextResponse.json({ available: false });
  }

  // A preview only means anything while the sample that produced it is still
  // running. Left visible afterwards it reads as the finished output, which is
  // exactly the wrong impression — it is a low-quality approximation.
  const stale = Date.now() - mtimeMs > 120_000;

  return NextResponse.json({
    available: !stale,
    path: clipPath,
    // cache-buster: the file is replaced in place, so the URL never changes
    version: Math.round(mtimeMs),
    sample: meta?.sample ?? null,
    of: meta?.of ?? null,
    step: meta?.step ?? null,
    total: meta?.total ?? null,
  });
}
