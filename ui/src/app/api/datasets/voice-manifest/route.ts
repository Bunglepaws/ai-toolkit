import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

// Reads the .voice_clone.json a voice-clip generation run leaves in its target dataset.
// The Clone Voice card has no other way to know whether clips exist: generation happens
// inside the training job, so nothing in the browser observes it. Without this the card
// cannot tell "not generated yet" from "generated and up to date", which is exactly the
// confusion that made the rebuild button look like it did nothing.
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const datasetPath = searchParams.get('path');

  if (!datasetPath) {
    return NextResponse.json({ error: 'path is required' }, { status: 400 });
  }

  try {
    const manifestPath = path.join(datasetPath, '.voice_clone.json');
    if (!fs.existsSync(manifestPath)) {
      return NextResponse.json({ exists: false, count: 0 });
    }

    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
    const files: string[] = Array.isArray(manifest?.files) ? manifest.files : [];

    // Report what is actually on disk, not just what the manifest claims -- a clip
    // deleted by hand should show up as missing rather than being counted.
    const present = files.filter(f => fs.existsSync(path.join(datasetPath, f)));

    const fp = manifest?.fingerprint ?? {};
    return NextResponse.json({
      exists: true,
      count: present.length,
      claimed: files.length,
      missing: files.length - present.length,
      target_seconds: fp.target_seconds ?? null,
      voice_description: fp.voice_description ?? null,
      mode: fp.mode ?? null,
    });
  } catch (error) {
    // A corrupt or unreadable manifest is not worth failing the page over -- the card
    // just falls back to showing nothing.
    return NextResponse.json({ exists: false, count: 0 });
  }
}
