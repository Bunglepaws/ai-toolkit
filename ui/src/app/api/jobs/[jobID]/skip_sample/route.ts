import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

// Abandons only the clip currently rendering and moves to the next prompt in
// the batch, unlike /stop_sample which abandons the whole batch and resumes
// training. See DiffusionTrainer._maybe_skip_sample_hook, checked every
// denoise step so it takes effect within one step of being requested.
export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  await prisma.job.update({
    where: { id: jobID },
    data: { skip_sample: true },
  });

  return NextResponse.json(job);
}
