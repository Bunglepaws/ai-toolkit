'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Modal } from '@/components/Modal';
import { apiClient } from '@/utils/api';

// ─── Types ───────────────────────────────────────────────────────────────────

interface ImageInfo {
  path: string;
  width: number;
  height: number;
  bucket_w: number;
  bucket_h: number;
  needs_crop: boolean;
}

interface ScanResult {
  images: ImageInfo[];
  buckets: Record<string, number>;
}

interface AnchorDecision {
  anchor_x: number;
  anchor_y: number;
  source?: 'ai' | 'fallback' | 'user';
}

type Decisions = Record<string, AnchorDecision>;

// Maps 3×3 grid position name to (anchor_x, anchor_y)
const ANCHOR_MAP: Record<string, { anchor_x: number; anchor_y: number }> = {
  'top-left':      { anchor_x: 0.0, anchor_y: 0.0 },
  'top-center':    { anchor_x: 0.5, anchor_y: 0.0 },
  'top-right':     { anchor_x: 1.0, anchor_y: 0.0 },
  'center-left':   { anchor_x: 0.0, anchor_y: 0.5 },
  'center':        { anchor_x: 0.5, anchor_y: 0.5 },
  'center-right':  { anchor_x: 1.0, anchor_y: 0.5 },
  'bottom-left':   { anchor_x: 0.0, anchor_y: 1.0 },
  'bottom-center': { anchor_x: 0.5, anchor_y: 1.0 },
  'bottom-right':  { anchor_x: 1.0, anchor_y: 1.0 },
};

const ANCHOR_LABELS: Record<string, string> = {
  'top-left': '↖', 'top-center': '↑', 'top-right': '↗',
  'center-left': '←', 'center': '·', 'center-right': '→',
  'bottom-left': '↙', 'bottom-center': '↓', 'bottom-right': '↘',
};

const ANCHOR_GRID_ORDER = [
  ['top-left', 'top-center', 'top-right'],
  ['center-left', 'center', 'center-right'],
  ['bottom-left', 'bottom-center', 'bottom-right'],
];

function anchorNameFromDecision(d: AnchorDecision): string {
  for (const [name, { anchor_x, anchor_y }] of Object.entries(ANCHOR_MAP)) {
    if (Math.abs(d.anchor_x - anchor_x) < 0.01 && Math.abs(d.anchor_y - anchor_y) < 0.01) {
      return name;
    }
  }
  return 'center';
}

// ─── Crop overlay ─────────────────────────────────────────────────────────────
// Computes the kept-region rectangle as percentages of the thumbnail dimensions.
// Returns CSS clip-path polygon string that darkens outside the kept region.

function cropOverlayStyle(
  imgW: number, imgH: number,
  bucketW: number, bucketH: number,
  anchorX: number, anchorY: number,
): React.CSSProperties {
  const srcAr = imgW / imgH;
  const tgtAr = bucketW / bucketH;

  let keepL = 0, keepT = 0, keepR = 100, keepB = 100;

  if (srcAr > tgtAr) {
    // Image wider than target — the kept width fraction is tgtAr / srcAr
    const keepFrac = tgtAr / srcAr;
    const excessFrac = 1 - keepFrac;
    keepL = excessFrac * anchorX * 100;
    keepR = keepL + keepFrac * 100;
  } else {
    // Image taller than target — kept height fraction is srcAr / tgtAr
    const keepFrac = srcAr / tgtAr;
    const excessFrac = 1 - keepFrac;
    keepT = excessFrac * anchorY * 100;
    keepB = keepT + keepFrac * 100;
  }

  // clip-path "polygon with hole" — outer CCW then inner CW creates the overlay
  const outer = `0% 0%, 100% 0%, 100% 100%, 0% 100%`;
  const inner = `${keepL}% ${keepT}%, ${keepL}% ${keepB}%, ${keepR}% ${keepB}%, ${keepR}% ${keepT}%`;
  return {
    clipPath: `polygon(evenodd, ${outer}, ${inner})`,
    position: 'absolute' as const,
    inset: 0,
    backgroundColor: 'rgba(0,0,0,0.55)',
    pointerEvents: 'none' as const,
  };
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function BucketBar({ buckets }: { buckets: Record<string, number> }) {
  const total = Object.values(buckets).reduce((a, b) => a + b, 0);
  const sorted = Object.entries(buckets).sort((a, b) => b[1] - a[1]);
  const max = sorted[0]?.[1] ?? 1;

  return (
    <div className="space-y-1.5 mt-2">
      {sorted.map(([key, count]) => (
        <div key={key} className="flex items-center gap-2 text-xs">
          <span className="w-24 text-right text-gray-400 font-mono">{key}</span>
          <div className="flex-1 bg-gray-700 rounded h-4 overflow-hidden">
            <div
              className={`h-full rounded transition-all ${count < 5 ? 'bg-amber-500' : 'bg-blue-500'}`}
              style={{ width: `${(count / max) * 100}%` }}
            />
          </div>
          <span className={`w-8 text-right ${count < 5 ? 'text-amber-400' : 'text-gray-300'}`}>
            {count}
          </span>
        </div>
      ))}
      {sorted.length === 0 && <p className="text-gray-500 text-sm">No images found.</p>}
    </div>
  );
}

function AnchorPicker({
  value,
  onChange,
}: {
  value: string;
  onChange: (name: string) => void;
}) {
  return (
    <div className="grid grid-cols-3 gap-0.5 mt-1" style={{ width: 66 }}>
      {ANCHOR_GRID_ORDER.flat().map(name => (
        <button
          key={name}
          title={name}
          onClick={() => onChange(name)}
          className={`w-5 h-5 text-xs rounded flex items-center justify-center transition-colors ${
            value === name
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
          }`}
        >
          {ANCHOR_LABELS[name]}
        </button>
      ))}
    </div>
  );
}

interface CropCardProps {
  img: ImageInfo;
  decision: AnchorDecision;
  onAnchorChange: (name: string) => void;
}

function CropCard({ img, decision, onAnchorChange }: CropCardProps) {
  const anchorName = anchorNameFromDecision(decision);
  const overlayStyle = cropOverlayStyle(
    img.width, img.height,
    img.bucket_w, img.bucket_h,
    decision.anchor_x, decision.anchor_y,
  );

  return (
    <div className="bg-gray-800 rounded-lg p-2 flex flex-col gap-1.5">
      <div className="relative rounded overflow-hidden" style={{ aspectRatio: `${img.width}/${img.height}` }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`/api/img/${encodeURIComponent(img.path)}`}
          alt=""
          className="w-full h-full object-cover"
          loading="lazy"
        />
        <div style={overlayStyle} />
      </div>
      <p className="text-xs text-gray-400 truncate" title={img.path}>
        {img.path.split(/[\\/]/).pop()}
      </p>
      <p className="text-xs text-gray-500">
        {decision.source === 'ai' ? `AI: ${anchorName}` : `Fallback: ${anchorName}`}
      </p>
      <AnchorPicker value={anchorName} onChange={onAnchorChange} />
    </div>
  );
}

// ─── Main modal ───────────────────────────────────────────────────────────────

interface Props {
  isOpen: boolean;
  onClose: () => void;
  datasetName: string;
}

const RESOLUTION_PRESETS = [512, 768, 1024, 1280];

export function BucketToolsModal({ isOpen, onClose, datasetName }: Props) {
  const [step, setStep] = useState<1 | 2 | 3 | 4 | 5>(1);
  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState<ScanResult | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  const [resolution, setResolution] = useState(1024);
  const [customRes, setCustomRes] = useState('');
  const [mode, setMode] = useState<'resize' | 'crop'>('crop');
  const [decisions, setDecisions] = useState<Decisions>({});
  const [detectProgress, setDetectProgress] = useState({ current: 0, total: 0 });
  const [processProgress, setProcessProgress] = useState({ current: 0, total: 0, warnings: [] as string[] });
  const [outputName, setOutputName] = useState<string | null>(null);
  const detectAbortRef = useRef<(() => void) | null>(null);

  // Reset when opened
  useEffect(() => {
    if (isOpen) {
      setStep(1);
      setScanResult(null);
      setScanError(null);
      setDecisions({});
      setDetectProgress({ current: 0, total: 0 });
      setProcessProgress({ current: 0, total: 0, warnings: [] });
      setOutputName(null);
      runScan(resolution);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  const runScan = useCallback(async (res: number) => {
    setScanning(true);
    setScanError(null);
    setScanResult(null);
    try {
      const { data } = await apiClient.post<ScanResult>(
        `/api/datasets/${encodeURIComponent(datasetName)}/scan-buckets`,
        { resolution: res },
      );
      setScanResult(data);
    } catch (e: any) {
      setScanError(e?.message ?? 'Scan failed');
    } finally {
      setScanning(false);
    }
  }, [datasetName]);

  const handleResolutionChange = (res: number) => {
    setResolution(res);
    runScan(res);
  };

  const startDetect = useCallback(async () => {
    if (!scanResult) return;
    const needsCrop = scanResult.images.filter(i => i.needs_crop);
    if (needsCrop.length === 0) {
      // Nothing to detect — go straight to step 3 with empty decisions
      setStep(3);
      return;
    }

    setStep(2);
    setDetectProgress({ current: 0, total: needsCrop.length });

    // Initialise all as center fallback
    const initial: Decisions = {};
    for (const img of needsCrop) {
      initial[img.path] = { anchor_x: 0.5, anchor_y: 0.5, source: 'fallback' };
    }
    setDecisions(initial);

    let aborted = false;
    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
    detectAbortRef.current = () => { aborted = true; reader?.cancel(); };

    try {
      const response = await fetch('/api/datasets/focal-points', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imagePaths: needsCrop.map(i => i.path) }),
      });

      if (!response.body) { setStep(3); return; }
      reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      let count = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done || aborted) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          const trimmed = line.replace(/^data:\s*/, '').trim();
          if (!trimmed) continue;
          try {
            const evt = JSON.parse(trimmed);
            if (evt.done) { break; }
            if (evt.path && evt.anchor && ANCHOR_MAP[evt.anchor]) {
              count++;
              setDetectProgress(p => ({ ...p, current: count }));
              setDecisions(prev => ({
                ...prev,
                [evt.path]: { ...ANCHOR_MAP[evt.anchor], source: evt.source ?? 'fallback' },
              }));
            }
          } catch {}
        }
      }
    } catch {}

    if (!aborted) setStep(3);
  }, [scanResult]);

  const skipDetect = useCallback(() => {
    detectAbortRef.current?.();
    // Fill remaining images with center
    if (scanResult) {
      const init: Decisions = {};
      for (const img of scanResult.images.filter(i => i.needs_crop)) {
        if (!decisions[img.path]) {
          init[img.path] = { anchor_x: 0.5, anchor_y: 0.5, source: 'fallback' };
        }
      }
      setDecisions(prev => ({ ...prev, ...init }));
    }
    setStep(3);
  }, [decisions, scanResult]);

  const startProcess = useCallback(async () => {
    setStep(4);
    setProcessProgress({ current: 0, total: 0, warnings: [] });

    try {
      const response = await fetch(
        `/api/datasets/${encodeURIComponent(datasetName)}/process-buckets`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ resolution, mode, decisions }),
        },
      );

      if (!response.body) { setStep(5); return; }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            const evt = JSON.parse(trimmed);
            if (evt.type === 'progress') {
              setProcessProgress(p => ({ ...p, current: evt.current, total: evt.total }));
            } else if (evt.type === 'warning') {
              setProcessProgress(p => ({ ...p, warnings: [...p.warnings, `${evt.name}: ${evt.error}`] }));
            } else if (evt.type === 'done') {
              setOutputName(evt.outputName ?? null);
              setProcessProgress(p => ({ ...p, total: evt.total, current: evt.total }));
              setStep(5);
            }
          } catch {}
        }
      }
    } catch (e: any) {
      setProcessProgress(p => ({ ...p, warnings: [...p.warnings, `Processing error: ${e?.message}`] }));
    }
  }, [datasetName, decisions, mode, resolution]);

  const needsCropImages = scanResult?.images.filter(i => i.needs_crop) ?? [];
  const resizeOnlyCount = (scanResult?.images.length ?? 0) - needsCropImages.length;

  const handleClose = () => {
    if (step === 4) {
      if (!window.confirm('Processing is running. Close anyway?')) return;
    }
    onClose();
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleClose}
      title="Bucket Tools"
      size="xl"
      closeOnOverlayClick={step !== 4}
    >
      {/* Step indicator */}
      <div className="flex items-center gap-1 mb-4 text-xs text-gray-500">
        {(['Scan', 'Detect', 'Review', 'Process', 'Done'] as const).map((label, i) => (
          <span key={label} className="flex items-center gap-1">
            {i > 0 && <span>›</span>}
            <span className={step === i + 1 ? 'text-blue-400 font-semibold' : ''}>{label}</span>
          </span>
        ))}
      </div>

      {/* ── Step 1: Scan + Configure ──────────────────────────────────────── */}
      {step === 1 && (
        <div className="space-y-4">
          {/* Resolution picker */}
          <div>
            <label className="block text-sm font-medium mb-1">Target resolution</label>
            <div className="flex gap-2 flex-wrap">
              {RESOLUTION_PRESETS.map(r => (
                <button
                  key={r}
                  onClick={() => handleResolutionChange(r)}
                  className={`px-3 py-1 rounded text-sm border transition-colors ${
                    resolution === r
                      ? 'bg-blue-600 border-blue-500 text-white'
                      : 'bg-gray-700 border-gray-600 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {r}
                </button>
              ))}
              <input
                type="number"
                placeholder="Custom…"
                value={customRes}
                onChange={e => setCustomRes(e.target.value)}
                onBlur={() => {
                  const v = parseInt(customRes, 10);
                  if (v > 0) handleResolutionChange(v);
                }}
                className="w-24 px-2 py-1 rounded text-sm bg-gray-700 border border-gray-600 text-gray-300"
              />
            </div>
          </div>

          {/* Mode */}
          <div>
            <label className="block text-sm font-medium mb-1">Operation mode</label>
            <div className="flex gap-3">
              {(['resize', 'crop'] as const).map(m => (
                <label key={m} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="mode"
                    value={m}
                    checked={mode === m}
                    onChange={() => setMode(m)}
                    className="accent-blue-500"
                  />
                  <span className="text-sm capitalize">{m === 'resize' ? 'Resize only (keep aspect ratio)' : 'Crop to exact ratio'}</span>
                </label>
              ))}
            </div>
            {mode === 'resize' && (
              <p className="text-xs text-gray-500 mt-1">
                Images keep their original aspect ratio — the trainer handles multiple sizes within the pixel budget.
              </p>
            )}
          </div>

          {/* Bucket distribution */}
          <div>
            <p className="text-sm font-medium mb-1">
              Current bucket distribution
              {scanning && <span className="ml-2 text-gray-400 text-xs">scanning…</span>}
            </p>
            {scanError && <p className="text-red-400 text-sm">{scanError}</p>}
            {scanResult && <BucketBar buckets={scanResult.buckets} />}
            {scanResult && (
              <p className="text-xs text-gray-500 mt-2">
                {scanResult.images.length} images total ·{' '}
                {needsCropImages.length} would need cropping ·{' '}
                {resizeOnlyCount} resize-only
              </p>
            )}
          </div>

          <div className="flex justify-end pt-2">
            <button
              onClick={() => mode === 'crop' ? startDetect() : setStep(4)}
              disabled={scanning || !scanResult}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed rounded text-sm font-medium text-white transition-colors"
            >
              {mode === 'crop' ? 'Continue →' : 'Process Now →'}
            </button>
          </div>
        </div>
      )}

      {/* ── Step 2: Focal point detection ─────────────────────────────────── */}
      {step === 2 && (
        <div className="space-y-4">
          <p className="text-sm text-gray-300">
            Detecting focal points in {detectProgress.total} images that need cropping…
          </p>
          <div className="bg-gray-700 rounded-full h-3 overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all"
              style={{
                width: detectProgress.total > 0
                  ? `${(detectProgress.current / detectProgress.total) * 100}%`
                  : '0%',
              }}
            />
          </div>
          <p className="text-sm text-gray-400">
            {detectProgress.current} / {detectProgress.total}
          </p>
          <button
            onClick={skipDetect}
            className="text-sm text-gray-400 hover:text-gray-200 underline"
          >
            Skip detection (use center crop for remaining)
          </button>
        </div>
      )}

      {/* ── Step 3: Per-image review grid ─────────────────────────────────── */}
      {step === 3 && (
        <div className="space-y-4">
          {needsCropImages.length === 0 ? (
            <p className="text-gray-400 text-sm">No images need cropping — all will be resized only.</p>
          ) : (
            <>
              <p className="text-sm text-gray-300">
                Review and adjust crop anchors. The darkened region will be removed.
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 max-h-[60vh] overflow-y-auto pr-1">
                {needsCropImages.map(img => (
                  <CropCard
                    key={img.path}
                    img={img}
                    decision={decisions[img.path] ?? { anchor_x: 0.5, anchor_y: 0.5, source: 'fallback' }}
                    onAnchorChange={name => setDecisions(prev => ({
                      ...prev,
                      [img.path]: { ...ANCHOR_MAP[name], source: 'user' },
                    }))}
                  />
                ))}
              </div>
              {resizeOnlyCount > 0 && (
                <p className="text-xs text-gray-500">
                  + {resizeOnlyCount} image{resizeOnlyCount !== 1 ? 's' : ''} will be resized without cropping.
                </p>
              )}
            </>
          )}
          <div className="flex justify-between pt-2">
            <button
              onClick={() => setStep(1)}
              className="text-sm text-gray-400 hover:text-gray-200"
            >
              ← Back
            </button>
            <button
              onClick={startProcess}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-sm font-medium text-white transition-colors"
            >
              Process {(scanResult?.images.length ?? 0)} images →
            </button>
          </div>
        </div>
      )}

      {/* ── Step 4: Processing ────────────────────────────────────────────── */}
      {step === 4 && (
        <div className="space-y-4">
          <p className="text-sm text-gray-300">Processing images…</p>
          <div className="bg-gray-700 rounded-full h-3 overflow-hidden">
            <div
              className="h-full bg-green-500 transition-all"
              style={{
                width: processProgress.total > 0
                  ? `${(processProgress.current / processProgress.total) * 100}%`
                  : '5%',
              }}
            />
          </div>
          <p className="text-sm text-gray-400">
            {processProgress.current} / {processProgress.total || '…'}
          </p>
          {processProgress.warnings.length > 0 && (
            <div className="space-y-1">
              {processProgress.warnings.map((w, i) => (
                <p key={i} className="text-xs text-amber-400">{w}</p>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Step 5: Done ──────────────────────────────────────────────────── */}
      {step === 5 && (
        <div className="space-y-4">
          <p className="text-green-400 font-medium">
            ✓ Processed {processProgress.total} images
          </p>
          {outputName && (
            <p className="text-sm text-gray-300">
              Output dataset: <span className="font-mono text-blue-300">{outputName}</span>
            </p>
          )}
          {processProgress.warnings.length > 0 && (
            <div className="space-y-1">
              <p className="text-xs text-amber-400 font-medium">
                {processProgress.warnings.length} image{processProgress.warnings.length !== 1 ? 's' : ''} had errors:
              </p>
              {processProgress.warnings.map((w, i) => (
                <p key={i} className="text-xs text-amber-400">{w}</p>
              ))}
            </div>
          )}
          <div className="flex gap-3 pt-2">
            {outputName && (
              <a
                href={`/datasets/${encodeURIComponent(outputName)}`}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-sm font-medium text-white transition-colors"
              >
                Open new dataset
              </a>
            )}
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm text-gray-300 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </Modal>
  );
}
