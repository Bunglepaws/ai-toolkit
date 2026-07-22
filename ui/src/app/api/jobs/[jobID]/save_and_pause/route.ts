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

/**
 * Returns true if at least one training step has been recorded in the current
 * session (i.e. since the most-recent training_sessions.start_time).
 * Falls back to false if the DB or table does not exist yet.
 */
async function hasStepsThisSession(logPath: string): Promise<boolean> {
  if (!fs.existsSync(logPath)) return false;

  const db = openDb(logPath);
  try {
    // Check that the training_sessions table exists (forward-only feature).
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

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingFolder = await getTrainingFolder();
  const logPath = path.join(trainingFolder, job.name, 'loss_log.db');
  const stepsMade = await hasStepsThisSession(logPath);

  if (stepsMade) {
    // Set both flags — end_step_hook runs maybe_save() (save_now) before
    // maybe_stop() (stop), so the Python trainer saves first then stops cleanly.
    await prisma.job.update({
      where: { id: jobID },
      data: {
        save_now: true,
        stop: true,
        info: 'Saving snapshot and pausing...',
      },
    });
  } else {
    // No progress this session — nothing new to save.  Just stop.
    await prisma.job.update({
      where: { id: jobID },
      data: {
        stop: true,
        info: 'Stopping job...',
      },
    });
  }

  return NextResponse.json(job);
}
