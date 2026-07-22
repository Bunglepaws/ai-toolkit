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

  // Set stop flag only — no SIGINT. Python's stop-watcher thread polls this
  // every ~5s and exits cleanly without attempting to save a checkpoint.
  await prisma.job.update({
    where: { id: jobID },
    data: {
      stop: true,
      info: 'Stopping job...',
    },
  });

  return NextResponse.json(job);
}
