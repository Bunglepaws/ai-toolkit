/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { getDatasetsRoot, getTrainingFolder, getDataRoot } from '@/server/settings';

const execFileAsync = promisify(execFile);

// ---------------------------------------------------------------------------
// Shared: extract prompt and seed from a raw A1111 parameters string
// ---------------------------------------------------------------------------
function parseParameters(parameters: string): { prompt: string | null; seed: number | null } {
  const lines = parameters.split('\n');
  const prompt = lines[0]?.trim() || null;
  let seed: number | null = null;
  // Seed appears in the last line as "..., Seed: <N>, ..."
  for (let i = lines.length - 1; i >= 0; i--) {
    const m = lines[i].match(/\bSeed:\s*(\d+)/);
    if (m) {
      seed = parseInt(m[1], 10);
      break;
    }
  }
  return { prompt, seed };
}

// ---------------------------------------------------------------------------
// JPEG: read EXIF UserComment (tag 0x9286) written with UNICODE\x00 + UTF-16-LE
// ---------------------------------------------------------------------------
function readJpegParameters(buf: Buffer): string | null {
  // JPEG starts with FFD8
  if (buf.length < 2 || buf[0] !== 0xff || buf[1] !== 0xd8) return null;

  let offset = 2;
  while (offset + 4 <= buf.length) {
    if (buf[offset] !== 0xff) break;
    const marker = buf[offset + 1];
    // SOS (0xda) and EOI (0xd9) end the header
    if (marker === 0xda || marker === 0xd9) break;
    const segLen = buf.readUInt16BE(offset + 2);
    if (segLen < 2 || offset + 2 + segLen > buf.length) break;

    // APP1 (0xe1) may contain Exif
    if (marker === 0xe1 && segLen > 6) {
      const seg = buf.slice(offset + 4, offset + 2 + segLen);
      if (seg.slice(0, 6).toString('ascii').startsWith('Exif')) {
        // Exif header: "Exif\x00\x00" then TIFF data
        const tiff = seg.slice(6);
        const littleEndian = tiff[0] === 0x49; // 'II'
        const readU16 = (o: number) => littleEndian ? tiff.readUInt16LE(o) : tiff.readUInt16BE(o);
        const readU32 = (o: number) => littleEndian ? tiff.readUInt32LE(o) : tiff.readUInt32BE(o);

        // Helper: extract UserComment bytes from an IFD at the given offset
        const readUserComment = (ifdOffset: number): string | null => {
          if (ifdOffset + 2 > tiff.length) return null;
          const count = readU16(ifdOffset);
          for (let i = 0; i < count; i++) {
            const e = ifdOffset + 2 + i * 12;
            if (e + 12 > tiff.length) break;
            if (readU16(e) !== 0x9286) continue;
            const byteCount = readU32(e + 4);
            const valOffset = readU32(e + 8);
            if (valOffset + byteCount > tiff.length) break;
            const raw = tiff.slice(valOffset, valOffset + byteCount);
            if (raw.length > 8 && raw.slice(0, 7).toString('ascii') === 'UNICODE') {
              return raw.slice(8).toString('utf16le').replace(/\0+$/, '');
            }
            break;
          }
          return null;
        };

        // Walk IFD0: look for ExifIFD pointer (0x8769) AND fallback 0x9286 in IFD0
        // (older files written by PIL incorrectly placed UserComment in IFD0)
        const ifd0Offset = readU32(4);
        const ifd0Count = readU16(ifd0Offset);
        let exifIfdOffset: number | null = null;
        let ifd0UserComment: string | null = null;
        for (let i = 0; i < ifd0Count; i++) {
          const entryOffset = ifd0Offset + 2 + i * 12;
          if (entryOffset + 12 > tiff.length) break;
          const tag = readU16(entryOffset);
          if (tag === 0x8769) {
            exifIfdOffset = readU32(entryOffset + 8);
          } else if (tag === 0x9286) {
            // Mis-placed UserComment (PIL bug) — capture as fallback
            const byteCount = readU32(entryOffset + 4);
            const valOffset = readU32(entryOffset + 8);
            if (valOffset + byteCount <= tiff.length) {
              const raw = tiff.slice(valOffset, valOffset + byteCount);
              if (raw.length > 8 && raw.slice(0, 7).toString('ascii') === 'UNICODE') {
                ifd0UserComment = raw.slice(8).toString('utf16le').replace(/\0+$/, '');
              }
            }
          }
        }

        // Prefer ExifIFD (correct location), fall back to IFD0 (PIL bug)
        const result = (exifIfdOffset !== null ? readUserComment(exifIfdOffset) : null)
                    ?? ifd0UserComment;
        if (result) return result;
      }
    }
    offset += 2 + segLen;
  }
  return null;
}

// ---------------------------------------------------------------------------
// PNG: read the 'parameters' tEXt chunk (A1111 / CivitAI format)
// ---------------------------------------------------------------------------
function readPngParameters(buf: Buffer): string | null {
  if (buf.length < 8) return null;
  const pngSig = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  if (!buf.slice(0, 8).equals(pngSig)) return null;

  let offset = 8;
  while (offset + 12 <= buf.length) {
    const length = buf.readUInt32BE(offset);
    if (offset + 12 + length > buf.length) break;
    const type = buf.slice(offset + 4, offset + 8).toString('ascii');
    if (type === 'tEXt' && length > 0) {
      const data = buf.slice(offset + 8, offset + 8 + length);
      const nullIdx = data.indexOf(0);
      if (nullIdx !== -1 && data.slice(0, nullIdx).toString('latin1') === 'parameters') {
        return data.slice(nullIdx + 1).toString('latin1');
      }
    }
    if (type === 'IDAT' || type === 'IEND') break;
    offset += 12 + length;
  }
  return null;
}

// ---------------------------------------------------------------------------
// MP4 primary: ffprobe — reads the 'parameters' (or 'comment') tag written
// by ffmpeg with -movflags use_metadata_tags (same as ComfyUI VHS approach)
// ---------------------------------------------------------------------------
async function readMp4ViaFfprobe(filepath: string): Promise<string | null> {
  try {
    const { stdout } = await execFileAsync('ffprobe', [
      '-v', 'quiet',
      '-print_format', 'json',
      '-show_format',
      filepath,
    ], { timeout: 10000 });
    const info = JSON.parse(stdout);
    const tags: Record<string, string> | undefined = info?.format?.tags;
    if (!tags) return null;
    // Prefer our custom 'parameters' tag; fall back to 'comment'
    return tags['parameters'] ?? tags['comment'] ?? null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// MP4 fallback: binary parse for the mutagen ©cmt iTunes tag
// (used by older files written before the ffmpeg approach was adopted)
// ---------------------------------------------------------------------------
const CBOX = Buffer.from([0xa9, 0x63, 0x6d, 0x74]); // ©cmt

function findBox(
  buf: Buffer,
  start: number,
  end: number,
  name: Buffer | string,
): { start: number; end: number } | null {
  const target = typeof name === 'string' ? Buffer.from(name, 'ascii') : name;
  let offset = start;
  while (offset + 8 <= end) {
    const size = buf.readUInt32BE(offset);
    if (size < 8 || offset + size > end) break;
    if (buf.slice(offset + 4, offset + 8).equals(target)) {
      return { start: offset + 8, end: offset + size };
    }
    offset += size;
  }
  return null;
}

function readMp4CommentBinary(buf: Buffer): string | null {
  const moov = findBox(buf, 0, buf.length, 'moov');
  if (!moov) return null;
  const udta = findBox(buf, moov.start, moov.end, 'udta');
  if (!udta) return null;
  const meta = findBox(buf, udta.start, udta.end, 'meta');
  if (!meta) return null;
  // meta is a FullBox: skip 4-byte version+flags before children
  const ilst = findBox(buf, meta.start + 4, meta.end, 'ilst');
  if (!ilst) return null;
  const cmt = findBox(buf, ilst.start, ilst.end, CBOX);
  if (!cmt) return null;
  const data = findBox(buf, cmt.start, cmt.end, 'data');
  if (!data || data.end - data.start <= 8) return null;
  // data box: 4-byte type indicator + 4-byte locale = 8 bytes, then UTF-8 text
  return buf.slice(data.start + 8, data.end).toString('utf-8');
}

// ---------------------------------------------------------------------------
// Route
// ---------------------------------------------------------------------------
export async function POST(request: NextRequest) {
  let body;
  try {
    body = await request.json();
  } catch {
    return new NextResponse(null, { status: 499 });
  }

  const { imgPath } = body;
  if (!imgPath) return new NextResponse('Missing imgPath', { status: 400 });

  const [datasetRoot, trainingRoot, dataRoot] = await Promise.all([
    getDatasetsRoot(),
    getTrainingFolder(),
    getDataRoot(),
  ]);

  const resolved = path.resolve(imgPath as string);
  const isAllowed = [datasetRoot, trainingRoot, dataRoot].some(
    root => resolved === root || resolved.startsWith(root + path.sep),
  );
  if (!isAllowed) return new NextResponse('Access denied', { status: 403 });

  const stat = await fs.promises.stat(resolved).catch(() => null);
  if (!stat || !stat.isFile()) return new NextResponse('Not found', { status: 404 });

  const ext = path.extname(resolved).toLowerCase();
  let parameters: string | null = null;

  if (ext === '.png') {
    // PNG: fast binary parse — no subprocess needed
    const MAX_BYTES = 2 * 1024 * 1024; // metadata is always in the first few KB
    const readSize = Math.min(stat.size, MAX_BYTES);
    const buf = Buffer.alloc(readSize);
    const fd = await fs.promises.open(resolved, 'r');
    try {
      await fd.read(buf, 0, readSize, 0);
    } finally {
      await fd.close();
    }
    parameters = readPngParameters(buf);
  } else if (ext === '.jpg' || ext === '.jpeg') {
    // JPEG: read EXIF UserComment (tag 0x9286) written by save_image()
    const MAX_BYTES = 256 * 1024; // EXIF is always in the first few KB
    const readSize = Math.min(stat.size, MAX_BYTES);
    const buf = Buffer.alloc(readSize);
    const fd = await fs.promises.open(resolved, 'r');
    try {
      await fd.read(buf, 0, readSize, 0);
    } finally {
      await fd.close();
    }
    parameters = readJpegParameters(buf);
  } else if (ext === '.mp4' || ext === '.m4v' || ext === '.webm') {
    // MP4: try ffprobe first (reads ffmpeg FFMETADATA1 tags), then binary fallback
    parameters = await readMp4ViaFfprobe(resolved);
    if (!parameters) {
      // Binary fallback for older files that used mutagen ©cmt
      const MAX_BYTES = 50 * 1024 * 1024;
      const readSize = Math.min(stat.size, MAX_BYTES);
      const buf = Buffer.alloc(readSize);
      const fd = await fs.promises.open(resolved, 'r');
      try {
        await fd.read(buf, 0, readSize, 0);
      } finally {
        await fd.close();
      }
      parameters = readMp4CommentBinary(buf);
    }
  }

  if (!parameters) {
    return new NextResponse(JSON.stringify({ prompt: null, seed: null }), {
      headers: { 'Content-Type': 'application/json' },
    });
  }

  const { prompt, seed } = parseParameters(parameters);
  return new NextResponse(JSON.stringify({ prompt, seed }), {
    headers: { 'Content-Type': 'application/json' },
  });
}
