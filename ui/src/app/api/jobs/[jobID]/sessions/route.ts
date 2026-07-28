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
  id: number | null;
  start_time: number;
  end_time: number | null;
  start_step: number | null;
  startup_seconds: number | null;
  sampling_seconds: number | null;
  training_seconds: number | null;
  total_seconds: number | null;
  in_progress: boolean;
  overridden?: true;
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
function estimateSessionsFromSteps(steps: { step: number; wall_time: number }[]): SessionResult[] {
  if (steps.length === 0) return [];

  const sessions: SessionResult[] = [];
  let group: { step: number; wall_time: number }[] = [steps[0]];

  for (let i = 1; i < steps.length; i++) {
    const gap = steps[i].wall_time - steps[i - 1].wall_time;
    if (gap > INACTIVE_GAP_SECONDS) {
      sessions.push(groupToSession(group));
      group = [];
    }
    group.push(steps[i]);
  }
  sessions.push(groupToSession(group));
  return sessions;
}

function groupToSession(rows: { step: number; wall_time: number }[]): SessionResult {
  let active = 0;
  for (let i = 1; i < rows.length; i++) {
    const gap = rows[i].wall_time - rows[i - 1].wall_time;
    // Skip negative gaps (out-of-order wall_times from data anomalies) and
    // large gaps (sampling/renders between sessions).
    if (gap > 0 && gap < INACTIVE_GAP_SECONDS) active += gap;
  }
  return {
    id: null,
    start_time: rows[0].wall_time,
    end_time: rows[rows.length - 1].wall_time,
    start_step: rows[0].step || null,
    startup_seconds: null,
    sampling_seconds: null,
    training_seconds: active,
    total_seconds: active,
    in_progress: false,
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
    return NextResponse.json({
      sessions: [],
      total_seconds: 0,
      startup_total: 0,
      sampling_total: 0,
      training_total: 0,
      grand_total: 0,
    });
  }

  const db = openDb(logPath);

  try {
    const hasSessionsTable = await tableExists(db, 'training_sessions');

    let sessions: SessionResult[];

    if (!hasSessionsTable) {
      // ── Pre-feature: estimate from step wall_time gaps ───────────────────────
      const stepRows = await all<{ step: number; wall_time: number }>(
        db,
        `SELECT step, wall_time FROM steps ORDER BY step ASC`,
      );
      sessions = estimateSessionsFromSteps(stepRows);
    } else {
      // ── With feature: use exact training_sessions + sampling_periods ─────────
      const hasStartupColumn = (
        await all<{ name: string }>(db, `PRAGMA table_info(training_sessions);`)
      ).some(col => col.name === 'startup_seconds');

      const hasOverrideColumn = (
        await all<{ name: string }>(db, `PRAGMA table_info(training_sessions);`)
      ).some(col => col.name === 'training_seconds_override');

      const sessionRows = await all<{
        id: number;
        start_time: number;
        startup_seconds: number | null;
        training_seconds_override: number | null;
      }>(
        db,
        `SELECT id, start_time,
                ${hasStartupColumn ? 'startup_seconds' : 'NULL AS startup_seconds'},
                ${hasOverrideColumn ? 'training_seconds_override' : 'NULL AS training_seconds_override'}
         FROM training_sessions ORDER BY start_time ASC`,
      );

      if (sessionRows.length === 0) {
        return NextResponse.json({
          sessions: [],
          total_seconds: 0,
          startup_total: 0,
          sampling_total: 0,
          training_total: 0,
          grand_total: 0,
        });
      }

      // Estimate sessions for any steps that predate the first recorded session
      // (i.e. training runs before this feature was added to the DB).
      const firstSessionStart = sessionRows[0].start_time;
      const preFeatureStepRows = await all<{ step: number; wall_time: number }>(
        db,
        `SELECT step, wall_time FROM steps WHERE wall_time < ? ORDER BY step ASC`,
        [firstSessionStart],
      );
      const preFeatureSessions = estimateSessionsFromSteps(preFeatureStepRows);

      const hasSamplingTable = await tableExists(db, 'sampling_periods');

      const exactSessions = await Promise.all(
        sessionRows.map(async (session, i): Promise<SessionResult> => {
          const isLast = i === sessionRows.length - 1;
          const nextStart = i + 1 < sessionRows.length ? sessionRows[i + 1].start_time : null;

          const range = await getOne<{ min_wt: number | null; max_wt: number | null; start_step: number | null }>(
            db,
            `SELECT MIN(wall_time) AS min_wt, MAX(wall_time) AS max_wt,
                    (SELECT step FROM steps
                     WHERE wall_time >= ? AND (? IS NULL OR wall_time < ?)
                     ORDER BY step ASC LIMIT 1) AS start_step
             FROM steps
             WHERE wall_time >= ?
               AND (? IS NULL OR wall_time < ?)`,
            [session.start_time, nextStart, nextStart, session.start_time, nextStart, nextStart],
          );

          const min_wt = range?.min_wt ?? null;
          const max_wt = range?.max_wt ?? null;
          const startup_seconds = session.startup_seconds ?? null;

          if (min_wt === null || max_wt === null) {
            return {
              id: session.id,
              start_time: session.start_time,
              end_time: null,
              start_step: null,
              startup_seconds,
              sampling_seconds: null,
              training_seconds: null,
              total_seconds: null,
              in_progress: isLast,
            };
          }

          let sampling_seconds: number | null = null;
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

          const override = session.training_seconds_override ?? null;
          // training_seconds = pure gradient-step time only (sampling subtracted so it
          // doesn't get counted twice when we sum training + sampling + startup for the total)
          const training_seconds = override !== null
            ? override
            : Math.max(0, max_wt - min_wt - (sampling_seconds ?? 0));
          const total_seconds =
            (startup_seconds ?? 0) + (sampling_seconds ?? 0) + training_seconds;

          return {
            id: session.id,
            start_time: session.start_time,
            end_time: max_wt,
            start_step: range?.start_step || null,
            startup_seconds,
            sampling_seconds,
            training_seconds,
            total_seconds,
            in_progress: false,
            ...(override !== null ? { overridden: true as const } : {}),
          };
        }),
      );

      sessions = [...preFeatureSessions, ...exactSessions];
    }

    const sum = (key: 'startup_seconds' | 'sampling_seconds' | 'training_seconds') =>
      sessions.reduce((acc, s) => acc + (s[key] ?? 0), 0);

    const startup_total = sum('startup_seconds');
    const sampling_total = sum('sampling_seconds');
    const training_total = sum('training_seconds');
    const grand_total = startup_total + sampling_total + training_total;

    return NextResponse.json({
      sessions,
      total_seconds: training_total,
      startup_total,
      sampling_total,
      training_total,
      grand_total,
    });
  } finally {
    await closeDb(db);
  }
}
