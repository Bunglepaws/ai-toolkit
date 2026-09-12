import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

const prisma = new PrismaClient();

/**
 * A checkpoint already exists for the job's current step — either nothing has
 * trained yet (step 0) or a periodic save already landed on this exact step,
 * so an on-demand save would just write an identical duplicate.
 */
async function checkpointAlreadyExists(jobName: string, step: number): Promise<boolean> {
  if (!step || step <= 0) return true;
  const trainingFolder = await getTrainingFolder();
  const filename = `${jobName}_${String(step).padStart(5, '0')}.safetensors`;
  return fs.existsSync(path.join(trainingFolder, jobName, filename));
}

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const alreadySaved = await checkpointAlreadyExists(job.name, job.step);

  if (!alreadySaved) {
    // NOTE: deliberately does NOT set `stop`. The trainer's stop-watcher thread
    // treats `stop` as "raise SIGINT now" and would kill the training step before
    // the checkpoint was ever written. stop_after_save is only read by
    // maybe_stop(), and only once no save is still pending, so the save always
    // lands first and the stop happens cleanly straight after it.
    await prisma.job.update({
      where: { id: jobID },
      data: {
        save_now: true,
        stop_after_save: true,
        info: 'Saving snapshot and pausing...',
      },
    });
  } else {
    // Already have a checkpoint for this exact step — nothing new to save, so
    // this is a plain stop and the watcher may interrupt immediately.
    await prisma.job.update({
      where: { id: jobID },
      data: {
        stop: true,
        info: 'Stopping job...',
      },
    });
  }

  return NextResponse.json(job);
}
