import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';
import sqlite3 from 'sqlite3';

const prisma = new PrismaClient();

function openDb(filename: string) {
  const db = new sqlite3.Database(filename);
  db.configure('busyTimeout', 5_000);
  return db;
}

function getOne<T = any>(db: sqlite3.Database, sql: string, params: any[] = []) {
  return new Promise<T | null>((resolve, reject) => {
    db.get(sql, params, (err, row) => {
      if (err) reject(err);
      else resolve((row as T) ?? null);
    });
  });
}

function closeDb(db: sqlite3.Database) {
  return new Promise<void>((resolve, reject) => {
    db.close((err) => (err ? reject(err) : resolve()));
  });
}

async function hasStepsThisSession(logPath: string): Promise<boolean> {
  if (!fs.existsSync(logPath)) return false;
  const db = openDb(logPath);
  try {
    const tbl = await getOne<{ name: string }>(
      db,
      `SELECT name FROM sqlite_master WHERE type='table' AND name='training_sessions'`,
    );
    if (!tbl) return false;
    const session = await getOne<{ start_time: number }>(
      db,
      `SELECT start_time FROM training_sessions ORDER BY start_time DESC LIMIT 1`,
    );
    if (!session) return false;
    const row = await getOne<{ n: number }>(
      db,
      `SELECT COUNT(*) AS n FROM steps WHERE wall_time >= ?`,
      [session.start_time],
    );
    return (row?.n ?? 0) > 0;
  } catch {
    return false;
  } finally {
    await closeDb(db);
  }
}

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  // Stop the queue so it won't advance to the next job after Python exits
  if (job.gpu_ids) {
    const queue = await prisma.queue.findUnique({ where: { gpu_ids: job.gpu_ids } });
    if (queue) {
      await prisma.queue.update({ where: { id: queue.id }, data: { is_running: false } });
    }
  }

  const trainingFolder = await getTrainingFolder();
  const logPath = path.join(trainingFolder, job.name, 'loss_log.db');
  const stepsMade = await hasStepsThisSession(logPath);

  if (stepsMade) {
    // Save then stop — Python uses the normal save+stop path (status becomes 'stopped').
    // return_to_queue: true tells processQueue.ts to flip the job back to 'queued'
    // once it sees the job has stopped, without needing Python to coordinate.
    await prisma.job.update({
      where: { id: jobID },
      data: {
        save_now: true,
        stop: true,
        return_to_queue: true,
        info: 'Saving snapshot and returning to queue...',
      },
    });
  } else {
    // No progress this session — stop immediately and requeue without saving.
    await prisma.job.update({
      where: { id: jobID },
      data: {
        stop: true,
        return_to_queue: true,
        info: 'Returning to queue...',
      },
    });
  }

  return NextResponse.json(job);
}
