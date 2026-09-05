import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';
import { invalidateCache } from '@/server/apiCache';

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

/**
 * "Save and Stop Queue": save on the next step, put this job back at the HEAD of
 * its queue, and stop the queue so nothing else starts.
 *
 * Pinning the position is the whole point, and it is not free. A running job
 * keeps whatever queue_position it held when it was picked, but every renumber
 * in the system — the reorder API and processQueue's renumberQueue — only
 * touches rows with status 'queued'. So reordering the queue while this job runs
 * renumbers the queued rows to 0..n-1 and leaves the running job's stale
 * position colliding with one of them. When Python flips the row to 'queued' the
 * tie falls to created_at, and the job that was running can come back BELOW a
 * job it should be above. That is the "it didn't stay at the top" bug.
 *
 * So renumber explicitly here: this job to 0, everything already queued to
 * 1..n, in the same transaction as the flags and the queue stop.
 */
export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const alreadySaved = await checkpointAlreadyExists(job.name, job.step);

  // Order preserved from what the UI and the reorder API both show
  // ([queue_position asc, created_at asc]); they are pushed down one slot, not
  // rearranged.
  const queuedJobs = await prisma.job.findMany({
    where: { gpu_ids: job.gpu_ids, status: 'queued', id: { not: jobID } },
    orderBy: [{ queue_position: 'asc' }, { created_at: 'asc' }],
  });

  const queue = job.gpu_ids ? await prisma.queue.findUnique({ where: { gpu_ids: job.gpu_ids } }) : null;

  const writes: any[] = [];

  // Stop the queue so it won't advance to the next job after Python exits.
  if (queue) {
    writes.push(prisma.queue.update({ where: { id: queue.id }, data: { is_running: false } }));
  }

  if (!alreadySaved) {
    // As in save_and_pause: no `stop`, or the watcher SIGINTs the step before the
    // checkpoint is written. return_to_queue is itself cooperative (only maybe_stop()
    // reads it, the watcher does not), so it is safe to set here — but it must not
    // fire before the save, which the stop_after_save gate in maybe_stop() ensures.
    writes.push(
      prisma.job.update({
        where: { id: jobID },
        data: {
          save_now: true,
          stop_after_save: true,
          return_to_queue: true,
          queue_position: 0,
          info: 'Saving snapshot and returning to queue...',
        },
      }),
    );
  } else {
    // Already have a checkpoint for this exact step — nothing new to save.
    writes.push(
      prisma.job.update({
        where: { id: jobID },
        data: {
          return_to_queue: true,
          queue_position: 0,
          info: 'Returning to queue...',
        },
      }),
    );
  }

  queuedJobs.forEach((j, idx) => {
    if (j.queue_position !== idx + 1) {
      writes.push(prisma.job.update({ where: { id: j.id }, data: { queue_position: idx + 1 } }));
    }
  });

  // One transaction: there must be no tick in which the queue is stopped but the
  // save flags are not yet set. processQueue's stopped-queue branch sets
  // return_to_queue on any still-running job, and a trainer that polled in that
  // window would requeue without ever writing the checkpoint.
  await prisma.$transaction(writes);

  // The active-jobs list is cached for 5s; without this the client's refresh
  // right after the click is served the pre-change order.
  invalidateCache('jobs-active');

  return NextResponse.json(job);
}
