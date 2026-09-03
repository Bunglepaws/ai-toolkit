import prisma from '../prisma';

import { Job, Queue } from '@prisma/client';
import fs from 'fs';
import { execFileSync } from 'child_process';
import startJob from './startJob';

/**
 * The process's command line, or null when it cannot be read.
 *
 * null means "could not determine" and is deliberately distinct from an empty
 * result, because the caller treats the two very differently: unknown is
 * resolved conservatively (assume the trainer lives), while a definite "no such
 * process" is what lets a recycled PID be recognised as dead.
 */
function processCommandLine(pid: number): string | null {
  if (process.platform !== 'win32') {
    try {
      // /proc/<pid>/cmdline is NUL-separated
      return fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8');
    } catch {
      return null;
    }
  }

  // Windows has no /proc. Without this the check degrades to "does any process
  // hold this PID", so a recycled PID reads as a live trainer and stalls the
  // queue permanently — the stack runs natively on Windows now, so this is the
  // normal path, not an edge case.
  try {
    const out = execFileSync(
      'powershell.exe',
      [
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        `$p = Get-CimInstance Win32_Process -Filter "ProcessId=${pid}" -ErrorAction SilentlyContinue; ` +
          `if (-not $p) { 'NOPROC' } elseif (-not $p.CommandLine) { 'UNKNOWN' } else { $p.CommandLine }`,
      ],
      { encoding: 'utf8', timeout: 10000, windowsHide: true },
    ).trim();

    if (out === 'NOPROC') return '';
    if (out === 'UNKNOWN' || out === '') return null;
    return out;
  } catch {
    return null;
  }
}

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

  const cmdline = processCommandLine(pid);

  // Could not determine it. Resolve conservatively: a false "dead" hands the
  // GPU to a second job and wrecks both runs, while a false "alive" only stalls
  // the queue, which is visible and recoverable.
  if (cmdline === null) return true;

  // Trainers are launched as `python run_ui.py` (pythonw.exe on Windows, so
  // match on the script rather than the executable name).
  return cmdline.includes('run_ui.py');
}

/**
 * How long a row may sit at 'running' with no pid before we call it dead.
 *
 * A launch is not atomic: startJob flips the row to 'running' first and can only
 * write the pid once the Windows relay has reported it back, which is well over
 * one worker tick. In that window the row looks exactly like a crashed job --
 * 'running', no live pid -- and reconciling it there marks a perfectly healthy
 * trainer "Stopped (trainer process gone)" and frees its GPU for the next job.
 * Wait out the launch instead; a genuinely dead job is still reconciled, just
 * a minute later, and nothing depends on that being instant.
 */
const LAUNCH_GRACE_MS = 90000;

function isLaunching(job: Job): boolean {
  if (job.pid != null) return false;
  return Date.now() - new Date(job.updated_at).getTime() < LAUNCH_GRACE_MS;
}

/** Returns a job whose trainer process is still alive on these GPUs, if any. */
async function findLiveTrainerOnGpu(gpuIds: string): Promise<Job | null> {
  // Only consider jobs in active states. Stopped/completed jobs retain their
  // last pid in the DB but that process is long gone; checking them causes false
  // positives when an unrelated system process recycles the old pid — the UNKNOWN
  // command-line case returns true conservatively and deadlocks the queue forever.
  const candidates: Job[] = await prisma.job.findMany({
    where: { gpu_ids: gpuIds, pid: { not: null }, status: { in: ['running', 'stopping'] } },
  });
  return candidates.find(job => isTrainerAlive(job.pid)) ?? null;
}

/**
 * Rewrite a GPU's queued jobs to positions 0..n-1, preserving their current
 * order. Re-queuing below puts a job back with the position it held before it
 * ran (deliberately — pausing and resuming the queue should resume that job in
 * its old spot), but that position can collide with whatever a reorder has since
 * renumbered into the slot. Duplicate positions make the queue unreorderable:
 * the UI and the reorder API sort the same [queue_position, created_at] pair, so
 * they agree on the order, but a duplicate still lets a plain up/down swap write
 * each row its own value back and do nothing. Renumbering keeps relative order
 * and removes the duplicate.
 */
async function renumberQueue(gpuIds: string): Promise<void> {
  const queued: Job[] = await prisma.job.findMany({
    where: { status: 'queued', gpu_ids: gpuIds },
    orderBy: [{ queue_position: 'asc' }, { created_at: 'asc' }],
  });
  const writes = queued
    .map((job, idx) => ({ job, idx }))
    .filter(({ job, idx }) => job.queue_position !== idx);
  if (writes.length === 0) return;
  await prisma.$transaction(
    writes.map(({ job, idx }) => prisma.job.update({ where: { id: job.id }, data: { queue_position: idx } })),
  );
  console.log(`Renumbered ${writes.length} queued job(s) on GPU(s) ${gpuIds}`);
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

      // The row is 'queued' but still carries its pre-run position, which may
      // now duplicate another job's. Heal it in the same tick. Safe here: this
      // whole block only runs while the queue is stopped, so nothing is being
      // picked, and the renumber preserves order so the resumed job keeps its
      // spot. Worst case a UI poll lands between the two writes and shows the
      // stale position for one second.
      if (stoppedRequeueJobs.length > 0) {
        await renumberQueue(queue.gpu_ids);
      }
    }
    if (queue.is_running) {
      // See if one is already running, status of running or stopping.
      //
      // A 'running' row is not proof a trainer exists. If the process is killed,
      // OOMs, or the box loses power, nothing ever clears the row and this check
      // would skip the queue forever — a permanent deadlock that looks like "the
      // queue just stopped working". So verify the process before trusting it,
      // and reconcile any row whose trainer is gone.
      const claimedJobs: Job[] = await prisma.job.findMany({
        where: {
          status: { in: ['running', 'stopping'] },
          gpu_ids: queue.gpu_ids,
        },
      });

      const liveClaimedJob = claimedJobs.find(job => isTrainerAlive(job.pid) || isLaunching(job));
      if (liveClaimedJob) {
        // genuinely running (or still starting up), nothing to do
        continue; // skip to next queue
      }

      for (const stale of claimedJobs) {
        console.log(
          `Reconciling stale '${stale.status}' job ${stale.name} (pid ${stale.pid}): no trainer process alive.`,
        );
        await prisma.job.update({
          where: { id: stale.id },
          data: { status: 'stopped', pid: null, info: 'Stopped (trainer process gone)' },
        });
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
