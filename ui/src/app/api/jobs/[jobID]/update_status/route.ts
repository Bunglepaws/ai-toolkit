import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function POST(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const { status, info } = await request.json();

  if (!status) {
    return NextResponse.json({ error: 'Status is required' }, { status: 400 });
  }

  try {
    const updatedJob = await prisma.job.update({
      where: { id: jobID },
      data: {
        status: status,
        info: info || null,
        // Deliberately keep the PID here. "About to exit" is not "exited" — the
        // trainer can stay alive for minutes finishing a step, saving, or sitting
        // in an uninterruptible section. Clearing the PID loses the only handle on
        // that process, which both hides it from the queue's liveness check and
        // makes the orphan unkillable. The PID is cleared on confirmed exit
        // (startJob's exit handler) and stale PIDs are filtered by liveness.
      },
    });
    console.log(`Job ${jobID} status updated to ${status}`);
    return NextResponse.json(updatedJob);
  } catch (error) {
    console.error(`Error updating job ${jobID} status to ${status}:`, error);
    return NextResponse.json({ error: 'Failed to update job status' }, { status: 500 });
  }
}
