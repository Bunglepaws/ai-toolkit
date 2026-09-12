/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { filePath } = body;

    if (!filePath || typeof filePath !== 'string') {
      return new NextResponse('filePath is required', { status: 400 });
    }

    // Decode the path
    const decodedFilePath = decodeURIComponent(filePath);

    // Get allowed directories
    const datasetRoot = await getDatasetsRoot();
    const trainingRoot = await getTrainingFolder();
    const allowedDirs = [datasetRoot, trainingRoot];

    // Security check: resolve so `..` segments collapse, then verify still under
    // an allowed root. Substring `.includes('..')` false-positives on filenames
    // containing `..` as text (e.g. an ellipsis in a filename).
    const resolvedFilePath = path.resolve(decodedFilePath);
    const isAllowed = allowedDirs.some(
      allowedDir => resolvedFilePath === allowedDir || resolvedFilePath.startsWith(allowedDir + path.sep),
    );

    if (!isAllowed) {
      console.warn(`Access denied: ${resolvedFilePath} not in ${allowedDirs.join(', ')}`);
      return new NextResponse('Access denied', { status: 403 });
    }

    // Check if file exists and grab file info in one stat
    let stat;
    try {
      stat = await fs.promises.stat(resolvedFilePath);
    } catch {
      console.warn(`File not found: ${resolvedFilePath}`);
      return new NextResponse('File not found', { status: 404 });
    }
    if (!stat.isFile()) {
      return new NextResponse('Not a file', { status: 400 });
    }

    await fs.promises.unlink(resolvedFilePath);

    // If deleting a step's .safetensors checkpoint, also remove the matching
    // optimizer archive: optimizer_{step:05d}.pt (not same-basename — a separate
    // naming pattern). Handles WAN 2.2's _high_noise/_low_noise suffix too.
    // \d{5,} covers both the current 5-digit padding and legacy 9-digit files.
    const stepMatch = path
      .basename(resolvedFilePath)
      .match(/_(\d{5,})(?:_(?:high|low)_noise)?\.safetensors$/);
    if (stepMatch) {
      const optimizerArchive = path.join(path.dirname(resolvedFilePath), `optimizer_${stepMatch[1]}.pt`);
      try {
        await fs.promises.unlink(optimizerArchive);
      } catch {
        // Archive doesn't exist — that's fine, nothing to do.
      }
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error deleting file:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
