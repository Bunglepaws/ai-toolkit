import prisma from '../prisma';

import { Job, Queue } from '@prisma/client';
import fs from 'fs';
import startJob from './startJob';

/**
 * Is this PID a live trainer process?
 *
 * A job's `status` alone is not proof its GPU is free. A stop request flips the
 * status immediately, but the trainer exits cooperatively and can stay alive for
 * minutes finishing a step, saving, or sitting in an uninterruptible section
 * (model load, quantization, sampling). Trusting status alone is how two jobs end
 * up sharing one GPU.
 *
 * PIDs are recycled, so a bare liveness check can produce false positives that
 * would deadlock the queue forever. Where /proc is available (Linux/WSL, where
 * trainers actually run) we confirm the process is really one of ours.
 */
function isTrainerAlive(pid: number | null): boolean {
  if (pid == null) return false;

  try {
    // Signal 0 performs the permission/existence check without delivering anything.
    process.kill(pid, 0);
  } catch (e: any) {
    // EPERM: the process exists but we may not signal it — still alive.
    // ESRCH (or anything else): treat as dead.
    if (e?.code !== 'EPERM') return false;
  }

  try {
    // /proc/<pid>/cmdline is NUL-separated
    return fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').includes('run_ui.py');
  } catch {
    // No /proc (e.g. native Windows) — fall back to the liveness result above.
    return true;
  }
}

/** Returns a job whose trainer process is still alive on these GPUs, if any. */
async function findLiveTrainerOnGpu(gpuIds: string): Promise<Job | null> {
  const candidates: Job[] = await prisma.job.findMany({
    where: { gpu_ids: gpuIds, pid: { not: null } },
  });
  return candidates.find(job => isTrainerAlive(job.pid)) ?? null;
}

export default async function processQueue() {
  const queues: Queue[] = await prisma.queue.findMany({
    orderBy: {
      id: 'asc',
    },
  });

  for (const queue of queues) {
    if (!queue.is_running) {
      // stop any running jobs first
      const runningJobs: Job[] = await prisma.job.findMany({
        where: {
          status: 'running',
          gpu_ids: queue.gpu_ids,
        },
      });

      for (const job of runningJobs) {
        console.log(`Stopping job ${job.id} on GPU(s) ${job.gpu_ids}`);
        await prisma.job.update({
          where: { id: job.id },
          data: {
            return_to_queue: true,
            info: 'Stopping job...',
          },
        });
      }

      // Re-queue any stopped jobs that were flagged for requeue (e.g. Save and Stop Queue)
      const stoppedRequeueJobs: Job[] = await prisma.job.findMany({
        where: {
          status: 'stopped',
          return_to_queue: true,
          gpu_ids: queue.gpu_ids,
        },
      });

      for (const job of stoppedRequeueJobs) {
        console.log(`Re-queuing job ${job.id} on GPU(s) ${job.gpu_ids}`);
        await prisma.job.update({
          where: { id: job.id },
          data: {
            status: 'queued',
            return_to_queue: false,
            stop: false,
            info: 'Job queued',
          },
        });
      }
    }
    if (queue.is_running) {
      // first see if one is already running, status of running or stopping
      const runningJob: Job | null = await prisma.job.findFirst({
        where: {
          status: { in: ['running', 'stopping'] },
          gpu_ids: queue.gpu_ids,
        },
      });

      if (runningJob) {
        // already running, nothing to do
        continue; // skip to next queue
      }

      // Status said the GPU is free. Confirm no trainer is actually still alive on
      // it before handing the slot to another job — see isTrainerAlive above.
      const liveTrainer = await findLiveTrainerOnGpu(queue.gpu_ids);
      if (liveTrainer) {
        console.log(
          `GPU(s) ${queue.gpu_ids} still busy: job ${liveTrainer.name} (pid ${liveTrainer.pid}) is alive ` +
            `despite status '${liveTrainer.status}'. Not starting another job.`,
        );
        continue; // skip to next queue
      }

      // find the next job in the queue
      const nextJob: Job | null = await prisma.job.findFirst({
        where: {
          status: 'queued',
          gpu_ids: queue.gpu_ids,
        },
        orderBy: {
          queue_position: 'asc',
        },
      });
      if (nextJob) {
        console.log(`Starting job ${nextJob.id} on GPU(s) ${nextJob.gpu_ids}`);
        await startJob(nextJob.id);
      } else {
        // find any job that needs sampling
        const sampleJobToRun: Job | null = await prisma.job.findFirst({
          where: {
            sample: true,
            gpu_ids: queue.gpu_ids,
            status: { notIn: ['running', 'stopping'] },
          },
        });
        if (sampleJobToRun) {
          console.log(`Starting sample-only job ${sampleJobToRun.id} on GPU(s) ${sampleJobToRun.gpu_ids}`);
          await startJob(sampleJobToRun.id, true);
        } else {
          // no more jobs, stop the queue
          console.log(`No more jobs in queue for GPU(s) ${queue.gpu_ids}, stopping queue`);
          await prisma.queue.update({
            where: { id: queue.id },
            data: { is_running: false },
          });
        }
      }
    }
  }
}
