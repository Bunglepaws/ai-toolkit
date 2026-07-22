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
        // If stopping, clear PID as the process is about to exit
        ...(status === 'stopped' || status === 'paused' ? { pid: null } : {}),
      },
    });
    console.log(`Job ${jobID} status updated to ${status}`);
    return NextResponse.json(updatedJob);
  } catch (error) {
    console.error(`Error updating job ${jobID} status to ${status}:`, error);
    return NextResponse.json({ error: 'Failed to update job status' }, { status: 500 });
  }
}
