import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';
import sqlite3 from 'sqlite3';

// Matches plain and WAN 2.2 split variants (_high_noise / _low_noise).
// \d{5,} covers both the current 5-digit padding and legacy 9-digit files
// that haven't been renamed by a resumed training run yet.
const STEP_RE = /_(\d{5,})(?:_(?:high|low)_noise)?\.safetensors$/;
const OPT_ARCHIVE_RE = /^optimizer_(\d{5,})\.pt$/;
const SAMPLE_RE = /[_-](\d{9})[_-]\d+\.(jpg|jpeg|png|webp|mp4)$/;

function openDb(filename: string) {
  const db = new sqlite3.Database(filename);
  db.configure('busyTimeout', 5_000);
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

async function safeRename(src: string, dst: string) {
  try {
    await fs.promises.rename(src, dst);
  } catch (e: any) {
    if (e.code === 'EXDEV') {
      await fs.promises.copyFile(src, dst);
      await fs.promises.unlink(src);
    } else {
      throw e;
    }
  }
}

async function pruneStepsInDb(logPath: string, targetStep: number): Promise<number> {
  if (!fs.existsSync(logPath)) return 0;

  const db = openDb(logPath);
  let pruned = 0;
  try {
    // Check tables exist before deleting
    const metricsTable = await getOne<{ name: string }>(
      db,
      `SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'`,
    );
    if (metricsTable) {
      await run(db, `DELETE FROM metrics WHERE step > ?`, [targetStep]);
    }

    const stepsTable = await getOne<{ name: string }>(
      db,
      `SELECT name FROM sqlite_master WHERE type='table' AND name='steps'`,
    );
    if (stepsTable) {
      const result = await new Promise<{ changes: number }>((resolve, reject) => {
        db.run(`DELETE FROM steps WHERE step > ?`, [targetStep], function (err) {
          if (err) reject(err);
          else resolve({ changes: this.changes });
        });
      });
      pruned = result.changes;
    }
  } finally {
    await closeDb(db);
  }
  return pruned;
}

export async function POST(req: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const body = await req.json().catch(() => ({}));
  const targetStep: number = body.step;
  const deleteSamples: boolean = body.deleteSamples ?? false;

  if (typeof targetStep !== 'number' || !Number.isInteger(targetStep) || targetStep < 0) {
    return NextResponse.json({ error: 'Invalid step value' }, { status: 400 });
  }

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  if (job.status !== 'stopped' && job.status !== 'error') {
    return NextResponse.json({ error: 'Job must be stopped before rolling back' }, { status: 409 });
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);

  // Guard: at least one safetensors file must exist for the target step
  let files: string[];
  try {
    files = fs.readdirSync(jobFolder);
  } catch {
    return NextResponse.json({ error: 'Job output folder not found' }, { status: 404 });
  }

  const targetExists = files.some(f => {
    const m = f.match(STEP_RE);
    return m && parseInt(m[1], 10) === targetStep;
  });
  if (!targetExists) {
    return NextResponse.json({ error: `No checkpoint found for step ${targetStep}` }, { status: 400 });
  }

  const deletedFiles: string[] = [];

  // ── 1. Restore optimizer archive for targetStep (if it exists) ────────────
  // Try the current 5-digit padding first, then fall back to legacy 9-digit
  // padding for archives not yet renamed.
  const optimizerPath = path.join(jobFolder, 'optimizer.pt');
  const archivePath5 = path.join(jobFolder, `optimizer_${String(targetStep).padStart(5, '0')}.pt`);
  const archivePath9 = path.join(jobFolder, `optimizer_${String(targetStep).padStart(9, '0')}.pt`);
  const archivePath = fs.existsSync(archivePath5) ? archivePath5 : archivePath9;
  if (fs.existsSync(archivePath)) {
    // Atomic: rename current optimizer.pt to a temp backup, then restore archive
    const backupPath = path.join(jobFolder, 'optimizer_rollback_backup.pt');
    if (fs.existsSync(optimizerPath)) {
      await safeRename(optimizerPath, backupPath);
    }
    try {
      await safeRename(archivePath, optimizerPath);
      if (fs.existsSync(backupPath)) {
        await fs.promises.unlink(backupPath);
      }
    } catch (e) {
      // Restore on failure
      if (fs.existsSync(backupPath) && !fs.existsSync(optimizerPath)) {
        await safeRename(backupPath, optimizerPath);
      }
      throw e;
    }
    deletedFiles.push(`${path.basename(archivePath)} → optimizer.pt`);
  }

  // ── 2. Delete all safetensors with step > targetStep ─────────────────────
  for (const f of files) {
    const m = f.match(STEP_RE);
    if (!m) continue;
    const step = parseInt(m[1], 10);
    if (step > targetStep) {
      await fs.promises.unlink(path.join(jobFolder, f));
      deletedFiles.push(f);
    }
  }

  // ── 3. Delete optimizer archives with step > targetStep ───────────────────
  // Re-read dir since we just renamed/deleted some files
  const filesAfter = fs.readdirSync(jobFolder);
  for (const f of filesAfter) {
    const m = f.match(OPT_ARCHIVE_RE);
    if (!m) continue;
    const step = parseInt(m[1], 10);
    if (step > targetStep) {
      await fs.promises.unlink(path.join(jobFolder, f));
      deletedFiles.push(f);
    }
  }

  // ── 4. Prune loss_log.db ──────────────────────────────────────────────────
  const logPath = path.join(jobFolder, 'loss_log.db');
  const prunedSteps = await pruneStepsInDb(logPath, targetStep);

  // ── 5. Optionally delete samples ─────────────────────────────────────────
  if (deleteSamples) {
    const samplesDir = path.join(jobFolder, 'samples');
    if (fs.existsSync(samplesDir)) {
      const sampleFiles = fs.readdirSync(samplesDir);
      for (const f of sampleFiles) {
        const m = f.match(SAMPLE_RE);
        if (!m) continue;
        const step = parseInt(m[1], 10);
        if (step > targetStep) {
          await fs.promises.unlink(path.join(samplesDir, f));
          deletedFiles.push(`samples/${f}`);
        }
      }
    }
  }

  // ── 6. Update Prisma job.step ─────────────────────────────────────────────
  await prisma.job.update({
    where: { id: jobID },
    data: { step: targetStep },
  });

  return NextResponse.json({ success: true, deletedFiles, prunedSteps });
}
