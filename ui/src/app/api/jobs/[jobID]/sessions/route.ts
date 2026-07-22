import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

import sqlite3 from 'sqlite3';

export const runtime = 'nodejs';

// Gaps between consecutive steps larger than this are treated as session
// boundaries when estimating sessions for older DBs (no training_sessions table).
// Sampling/rendering can take 30+ minutes, so 2 hours is safely above any
// in-session pause while still well below a typical inter-session gap.
const INACTIVE_GAP_SECONDS = 2 * 60 * 60;

const prisma = new PrismaClient();

function openDb(filename: string) {
  const db = new sqlite3.Database(filename);
  db.configure('busyTimeout', 30_000);
  return db;
}

function all<T = any>(db: sqlite3.Database, sql: string, params: any[] = []) {
  return new Promise<T[]>((resolve, reject) => {
    db.all(sql, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows as T[]);
    });
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

interface SessionResult {
  start_time: number;
  end_time: number | null;
  duration_seconds: number | null;
  estimated?: true;
}

async function tableExists(db: sqlite3.Database, name: string): Promise<boolean> {
  const row = await getOne<{ n: number }>(
    db,
    `SELECT COUNT(*) AS n FROM sqlite_master WHERE type='table' AND name=?`,
    [name],
  );
  return (row?.n ?? 0) > 0;
}

/**
 * Estimate sessions from step wall_times for older DBs that lack the
 * training_sessions table.  Groups consecutive steps into sessions wherever
 * the gap exceeds INACTIVE_GAP_SECONDS and sums only active inter-step time.
 */
function estimateSessionsFromSteps(steps: { wall_time: number }[]): SessionResult[] {
  if (steps.length === 0) return [];

  const sessions: SessionResult[] = [];
  let group: number[] = [steps[0].wall_time];

  for (let i = 1; i < steps.length; i++) {
    const gap = steps[i].wall_time - steps[i - 1].wall_time;
    if (gap > INACTIVE_GAP_SECONDS) {
      sessions.push(groupToSession(group));
      group = [];
    }
    group.push(steps[i].wall_time);
  }
  sessions.push(groupToSession(group));
  return sessions;
}

function groupToSession(wallTimes: number[]): SessionResult {
  let active = 0;
  for (let i = 1; i < wallTimes.length; i++) {
    const gap = wallTimes[i] - wallTimes[i - 1];
    // Skip negative gaps (out-of-order wall_times from data anomalies) and
    // large gaps (sampling/renders between sessions).
    if (gap > 0 && gap < INACTIVE_GAP_SECONDS) active += gap;
  }
  return {
    start_time: wallTimes[0],
    end_time: wallTimes[wallTimes.length - 1],
    duration_seconds: active,
    estimated: true,
  };
}

export async function GET(_request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const trainingFolder = await getTrainingFolder();
  const logPath = path.join(trainingFolder, job.name, 'loss_log.db');

  if (!fs.existsSync(logPath)) {
    return NextResponse.json({ sessions: [], total_seconds: 0 });
  }

  const db = openDb(logPath);

  try {
    const hasSessionsTable = await tableExists(db, 'training_sessions');

    let sessions: SessionResult[];

    if (!hasSessionsTable) {
      // ── Pre-feature: estimate from step wall_time gaps ───────────────────────
      const stepRows = await all<{ wall_time: number }>(
        db,
        `SELECT wall_time FROM steps ORDER BY step ASC`,
      );
      sessions = estimateSessionsFromSteps(stepRows);
    } else {
      // ── With feature: use exact training_sessions + sampling_periods ─────────
      const sessionRows = await all<{ start_time: number }>(
        db,
        `SELECT start_time FROM training_sessions ORDER BY start_time ASC`,
      );

      if (sessionRows.length === 0) {
        return NextResponse.json({ sessions: [], total_seconds: 0 });
      }

      // Estimate sessions for any steps that predate the first recorded session
      // (i.e. training runs before this feature was added to the DB).
      const firstSessionStart = sessionRows[0].start_time;
      const preFeatureStepRows = await all<{ wall_time: number }>(
        db,
        `SELECT wall_time FROM steps WHERE wall_time < ? ORDER BY step ASC`,
        [firstSessionStart],
      );
      const preFeatureSessions = estimateSessionsFromSteps(preFeatureStepRows);

      const hasSamplingTable = await tableExists(db, 'sampling_periods');

      const exactSessions = await Promise.all(
        sessionRows.map(async (session, i) => {
          const nextStart = i + 1 < sessionRows.length ? sessionRows[i + 1].start_time : null;

          const range = await getOne<{ min_wt: number | null; max_wt: number | null }>(
            db,
            `SELECT MIN(wall_time) AS min_wt, MAX(wall_time) AS max_wt
             FROM steps
             WHERE wall_time >= ?
               AND (? IS NULL OR wall_time < ?)`,
            [session.start_time, nextStart, nextStart],
          );

          const min_wt = range?.min_wt ?? null;
          const max_wt = range?.max_wt ?? null;

          if (min_wt === null || max_wt === null) {
            return { start_time: session.start_time, end_time: null, duration_seconds: null };
          }

          let sampling_seconds = 0;
          if (hasSamplingTable) {
            const samplingRow = await getOne<{ total: number }>(
              db,
              `SELECT COALESCE(SUM(end_time - start_time), 0) AS total
               FROM sampling_periods
               WHERE start_time >= ? AND end_time IS NOT NULL
                 AND (? IS NULL OR start_time < ?)`,
              [session.start_time, nextStart, nextStart],
            );
            sampling_seconds = samplingRow?.total ?? 0;
          }

          return {
            start_time: session.start_time,
            end_time: max_wt,
            duration_seconds: Math.max(0, (max_wt - min_wt) - sampling_seconds),
          };
        }),
      );

      sessions = [...preFeatureSessions, ...exactSessions];
    }

    const total_seconds = sessions.reduce(
      (acc, s) => acc + (s.duration_seconds ?? 0),
      0,
    );

    return NextResponse.json({ sessions, total_seconds });
  } finally {
    await closeDb(db);
  }
}
