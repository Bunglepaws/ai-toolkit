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
  await prisma.job.update({
    where: { id: jobID },
    data: {
      stop: true,
      status: 'stopped',
      info: 'Job stopped',
    },
  });

  console.log(`Job ${jobID} marked as stopped`);

  return NextResponse.json(job);
}
