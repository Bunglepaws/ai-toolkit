import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';
import sqlite3 from 'sqlite3';

export const runtime = 'nodejs';

const prisma = new PrismaClient();

function openDb(filename: string) {
  const db = new sqlite3.Database(filename);
  db.configure('busyTimeout', 30_000);
  return db;
}

function run(db: sqlite3.Database, sql: string, params: any[] = []) {
  return new Promise<void>((resolve, reject) => {
    db.run(sql, params, (err) => (err ? reject(err) : resolve()));
  });
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

export async function PUT(
  request: NextRequest,
  { params }: { params: { jobID: string; sessionId: string } },
) {
  const { jobID, sessionId } = await params;
  const sessionIdNum = parseInt(sessionId, 10);
  if (isNaN(sessionIdNum)) {
    return NextResponse.json({ error: 'Invalid session ID' }, { status: 400 });
  }

  let body: { training_seconds_override: number | null };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 });
  }

  const override = body.training_seconds_override;
  if (override !== null && (typeof override !== 'number' || override < 0)) {
    return NextResponse.json({ error: 'training_seconds_override must be a non-negative number or null' }, { status: 400 });
  }

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const trainingFolder = await getTrainingFolder();
  const logPath = path.join(trainingFolder, job.name, 'loss_log.db');

  if (!fs.existsSync(logPath)) {
    return NextResponse.json({ error: 'loss_log.db not found' }, { status: 404 });
  }

  const db = openDb(logPath);
  try {
    // Ensure column exists (may be an older DB)
    const cols = await new Promise<{ name: string }[]>((resolve, reject) => {
      db.all(`PRAGMA table_info(training_sessions);`, [], (err, rows) => {
        if (err) reject(err);
        else resolve(rows as { name: string }[]);
      });
    });
    if (!cols.some(c => c.name === 'training_seconds_override')) {
      await run(db, `ALTER TABLE training_sessions ADD COLUMN training_seconds_override REAL;`);
    }

    const session = await getOne(db, `SELECT id FROM training_sessions WHERE id = ?`, [sessionIdNum]);
    if (!session) {
      return NextResponse.json({ error: 'Session not found' }, { status: 404 });
    }

    await run(
      db,
      `UPDATE training_sessions SET training_seconds_override = ? WHERE id = ?`,
      [override, sessionIdNum],
    );

    return NextResponse.json({ ok: true });
  } finally {
    await closeDb(db);
  }
}
