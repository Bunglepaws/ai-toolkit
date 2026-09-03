import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { invalidateCache } from '@/server/apiCache';
import { getTrainingFolder } from '@/server/settings';
import path from 'path';
import fs from 'fs';

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingRoot = await getTrainingFolder();
  const trainingFolder = path.join(trainingRoot, job.name);

  // force:true makes this a no-op if the folder is already gone
  await fs.promises.rm(trainingFolder, { recursive: true, force: true });

  await prisma.job.delete({
    where: { id: jobID },
  });

  // Serve the change immediately: the active-jobs list is cached for 5s, so
  // without this the client's next poll is still shown the pre-change list.
  invalidateCache('jobs-active');

  return NextResponse.json(job);
}
