import prisma from '../prisma';

import { Job, Queue } from '@prisma/client';
import startJob from './startJob';

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
      } else {
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
}
