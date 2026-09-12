import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';

const isWindows = process.platform === 'win32';

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  // Try to kill the process if we have a PID, in case it is still running
  if (job.pid != null) {
    try {
      if (isWindows) {
        const { execSync } = require('child_process');
        execSync(`taskkill /PID ${job.pid} /T /F`, { stdio: 'ignore' });
      } else {
        process.kill(job.pid, 'SIGINT');
      }
      console.log(`Sent kill signal to PID ${job.pid} for job ${jobID}`);
    } catch (e: any) {
      // ESRCH means the process already exited — expected, not an error
      if (e?.code !== 'ESRCH') {
        console.warn(`Could not kill PID ${job.pid} for job ${jobID}:`, e);
      }
    }
  }

  // Keep the PID. We signalled the process above, but SIGINT is handled
  // cooperatively — the trainer finishes its step and saves before exiting, which
  // can take minutes. Until it actually exits it still holds the GPU, so the PID
  // has to stay so the queue's liveness check can see it.
  // Clear the cooperative flags along with the status. They mean "do this at the
  // next end_step_hook", and this run will never reach another one: we have just
  // force-killed the process (taskkill /F on Windows) and set stop, which the
  // watcher turns into an immediate interrupt. Left set, they are not inert --
  // they are read by the NEXT launch, where a stale stop_after_save stops the job
  // the moment it starts and a stale return_to_queue bounces it straight back to
  // the queue, with nothing in the UI to explain either. Self-healing per the
  // repo rule: the operation that owns the flag clears it rather than leaving it
  // to break a future run.
  await prisma.job.update({
    where: { id: jobID },
    data: {
      stop: true,
      status: 'stopped',
      info: 'Job stopped',
      save_now: false,
      stop_after_save: false,
      return_to_queue: false,
      sample: false,
    },
  });

  console.log(`Job ${jobID} marked as stopped`);

  return NextResponse.json(job);
}
