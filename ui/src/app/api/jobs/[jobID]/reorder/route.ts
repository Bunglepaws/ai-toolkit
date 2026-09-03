import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { invalidateCache } from '@/server/apiCache';

export async function POST(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const { direction, targetIndex } = await request.json();

  if (targetIndex === undefined && !['up', 'down'].includes(direction)) {
    return NextResponse.json({ error: 'Invalid direction or targetIndex' }, { status: 400 });
  }

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job || job.queue_position === null) {
    return NextResponse.json({ error: 'Job not in queue' }, { status: 404 });
  }

  // Find all jobs in the same GPU queue, sorted by position. The secondary
  // created_at key must match the UI's comparator exactly (compareQueueOrder in
  // JobsTable.tsx). When two rows share a position and the two sides break the
  // tie in opposite directions, the row the user sees at #2 is index 0 here, so
  // every attempt to move it up is dropped as a no-op that still reports success.
  const queueJobs = await prisma.job.findMany({
    where: {
      gpu_ids: job.gpu_ids,
      status: 'queued',
    },
    orderBy: [{ queue_position: 'asc' }, { created_at: 'asc' }],
  });

  const currentIndex = queueJobs.findIndex(j => j.id === job.id);
  if (currentIndex === -1) {
    return NextResponse.json({ error: 'Job not found in active queue' }, { status: 404 });
  }

  let destIndex = -1;
  if (targetIndex !== undefined) {
    destIndex = Math.max(0, Math.min(targetIndex, queueJobs.length - 1));
  } else if (direction === 'up' && currentIndex > 0) {
    destIndex = currentIndex - 1;
  } else if (direction === 'down' && currentIndex < queueJobs.length - 1) {
    destIndex = currentIndex + 1;
  }

  const reordered = [...queueJobs];
  if (destIndex !== -1 && destIndex !== currentIndex) {
    const [removed] = reordered.splice(currentIndex, 1);
    reordered.splice(destIndex, 0, removed);
  }

  // Renumber to 0..n-1 rather than swapping the two rows' positions. A swap
  // between rows that happen to share a position writes each one its own value
  // back, so the move silently does nothing. Renumbering also heals the
  // duplicate, so a queue that got into that state fixes itself on the next
  // reorder instead of staying stuck.
  const writes = reordered
    .map((j, idx) => ({ j, idx }))
    .filter(({ j, idx }) => j.queue_position !== idx);

  if (writes.length > 0) {
    await prisma.$transaction(
      writes.map(({ j, idx }) => prisma.job.update({ where: { id: j.id }, data: { queue_position: idx } })),
    );
    if (destIndex !== -1 && destIndex !== currentIndex) {
      console.log(`Job ${job.id} moved to index ${destIndex}`);
    } else {
      console.log(`Renumbered queue on GPU(s) ${job.gpu_ids} (duplicate positions healed)`);
    }
    // The active-jobs list is cached for 5s. Without this the client's refresh
    // right after a reorder is served the pre-move order, the optimistic row
    // snaps back, and the new order only appears when the entry expires — the
    // 5-10s "lag" that looks like a slow reorder but is a stale read.
    invalidateCache('jobs-active');
  }

  return NextResponse.json({ success: true });
}
