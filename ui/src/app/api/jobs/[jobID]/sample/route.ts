import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  // update job status to request sample
  await prisma.job.update({
    where: { id: jobID },
    data: {
      sample: true,
    },
  });

  // If the job is not currently running, ensure the queue is started so
  // processQueue picks up the sample request and runs the job in sample-only mode.
  if (job.status !== 'running') {
    await prisma.queue.upsert({
      where: { gpu_ids: job.gpu_ids },
      create: { gpu_ids: job.gpu_ids, is_running: true },
      update: { is_running: true },
    });
  }

  return NextResponse.json(job);
}
