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

  // request an on-demand save on the next step (ostris' canonical save_now flag)
  await prisma.job.update({
    where: { id: jobID },
    data: {
      save_now: true,
    },
  });

  return NextResponse.json(job);
}
