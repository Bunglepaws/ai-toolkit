'use client';

import { useState, useEffect, useRef } from 'react';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { JobConfig } from '@/types';
import { setNestedValue } from '@/utils/hooks';

export interface CheckFinding {
  field: string | null;
  current_value: string | null;
  suggested_value: string | null;
  reason: string;
  confidence: 'high' | 'medium' | 'low';
  references: string[];
  applyable: boolean;
  severity: 'info' | 'warning' | 'error';
}

interface CheckConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  jobId: string | null;
  jobConfig: JobConfig;
  onApply: (field: string, value: string) => void;
  /** If true, skips the options step and starts analysis immediately on open. */
  autoRun?: boolean;
  /** Optional context message shown at the top of results (e.g. "OOM detected"). */
  contextMessage?: string;
}

const CONFIDENCE_LABELS: Record<string, string> = {
  high: 'High Confidence',
  medium: 'Medium Confidence',
  low: 'Low Confidence (Speculative)',
};

const CONFIDENCE_COLORS: Record<string, string> = {
  high: 'text-green-400 border-green-500/40 bg-green-500/5',
  medium: 'text-amber-400 border-amber-500/40 bg-amber-500/5',
  low: 'text-gray-400 border-gray-500/40 bg-gray-500/5',
};

const SEVERITY_BADGES: Record<string, string> = {
  info: 'bg-blue-900/40 text-blue-300',
  warning: 'bg-amber-900/40 text-amber-300',
  error: 'bg-red-900/40 text-red-300',
};

type Phase = 'options' | 'loading' | 'results';

export default function CheckConfigModal({ isOpen, onClose, jobId, jobConfig, onApply, autoRun = false, contextMessage }: CheckConfigModalProps) {
  const [phase, setPhase] = useState<Phase>('options');
  const [includeImages, setIncludeImages] = useState(false);
  const autoRanRef = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const [findings, setFindings] = useState<CheckFinding[]>([]);
  const [speculativeOk, setSpeculativeOk] = useState(false);
  const [applied, setApplied] = useState<Set<string>>(new Set());
  const [applyingField, setApplyingField] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>('');

  // Auto-fire analysis when modal opens with autoRun=true (e.g. triggered by OOM)
  useEffect(() => {
    if (isOpen && autoRun && phase === 'options' && jobId && !autoRanRef.current) {
      autoRanRef.current = true;
      runAnalysis();
    }
    if (!isOpen) {
      autoRanRef.current = false;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, autoRun, jobId]);

  const handleClose = () => {
    onClose();
    setTimeout(() => {
      setPhase('options');
      setFindings([]);
      setError(null);
      setSpeculativeOk(false);
      setApplied(new Set());
    }, 200);
  };

  const runAnalysis = async () => {
    if (!jobId) return;
    setPhase('loading');
    setFindings([]);
    setError(null);
    setSpeculativeOk(false);
    setApplied(new Set());
    setStatusMessage('Starting…');

    try {
      const res = await fetch('/api/jobs/check-config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId, includeImages }),
      });

      if (!res.ok || !res.body) {
        const text = await res.text().catch(() => '');
        let msg = 'Failed to reach Check Config API';
        try { msg = JSON.parse(text)?.error ?? msg; } catch { /* ignore */ }
        setError(msg);
        setPhase('results');
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE lines are separated by \n\n; each starts with "data: "
        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith('data:')) continue;
          const json = line.slice('data:'.length).trim();
          let evt: any;
          try { evt = JSON.parse(json); } catch { continue; }

          if (evt.type === 'progress') {
            setStatusMessage(evt.message ?? '');
          } else if (evt.type === 'done') {
            setFindings(evt.findings ?? []);
            setPhase('results');
          } else if (evt.type === 'error') {
            setError(evt.message ?? 'Check Config failed');
            setPhase('results');
          }
        }
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to reach Check Config API');
      setPhase('results');
    }
  };

  const handleApply = async (finding: CheckFinding) => {
    if (!finding.field || finding.suggested_value == null) return;
    const key = `${finding.field}:${finding.suggested_value}`;
    setApplyingField(key);
    try {
      onApply(finding.field, finding.suggested_value);
      setApplied(prev => new Set(prev).add(key));
    } finally {
      setApplyingField(null);
    }
  };

  const grouped: Record<string, CheckFinding[]> = { high: [], medium: [], low: [] };
  for (const f of findings) {
    (grouped[f.confidence] || grouped.low).push(f);
  }

  const renderFinding = (finding: CheckFinding, idx: number) => {
    const key = `${finding.field}:${finding.suggested_value}`;
    const isApplied = applied.has(key);
    const isApplying = applyingField === key;
    const canApply =
      finding.applyable &&
      finding.field &&
      finding.suggested_value != null &&
      !isApplied &&
      (finding.confidence !== 'low' || speculativeOk);

    return (
      <div key={idx} className="border border-gray-700 rounded-lg p-3 space-y-2">
        <div className="flex items-start gap-2 flex-wrap">
          {finding.severity && (
            <span className={`text-xs px-2 py-0.5 rounded font-semibold uppercase tracking-wide ${SEVERITY_BADGES[finding.severity] || SEVERITY_BADGES.info}`}>
              {finding.severity}
            </span>
          )}
          {finding.field && (
            <span className="text-xs font-mono bg-gray-800 px-2 py-0.5 rounded text-gray-300">
              {finding.field}
            </span>
          )}
        </div>

        {finding.field && finding.current_value != null && finding.suggested_value != null && (
          <div className="text-xs font-mono flex items-center gap-2">
            <span className="text-red-400 line-through">{finding.current_value}</span>
            <span className="text-gray-500">→</span>
            <span className="text-green-400">{finding.suggested_value}</span>
          </div>
        )}

        <p className="text-sm text-gray-300">{finding.reason}</p>

        {finding.references.length > 0 && (
          <div className="space-y-0.5">
            <p className="text-xs text-gray-500 uppercase tracking-wide font-semibold">Sources</p>
            {finding.references.map((ref, ri) => (
              <p key={ri} className="text-xs text-gray-400 italic">
                {ref.startsWith('http') ? (
                  <a href={ref} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
                    {ref}
                  </a>
                ) : ref}
              </p>
            ))}
          </div>
        )}

        {finding.applyable && (
          <div className="pt-1">
            {isApplied ? (
              <span className="text-green-400 text-xs font-semibold">Applied ✓</span>
            ) : (
              <button
                onClick={() => handleApply(finding)}
                disabled={!canApply || isApplying}
                className="text-xs px-3 py-1 rounded bg-blue-700 hover:bg-blue-600 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                title={finding.confidence === 'low' && !speculativeOk ? 'Check the speculative box below to enable' : ''}
              >
                {isApplying ? 'Applying…' : 'Apply'}
              </button>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <Dialog open={isOpen} onClose={handleClose} className="relative z-50">
      <DialogBackdrop className="fixed inset-0 bg-black/60 backdrop-blur-sm" />
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <DialogPanel className="w-full max-w-2xl max-h-[85vh] flex flex-col bg-gray-900 rounded-xl shadow-2xl border border-gray-700">
          <div className="flex items-center justify-between px-5 py-4 border-b border-gray-700 flex-shrink-0">
            <DialogTitle className="text-lg font-semibold text-white flex items-center gap-2">
              <span className="text-purple-400">✦</span> Config Review
            </DialogTitle>
            <button onClick={handleClose} className="text-gray-500 hover:text-gray-300 text-xl leading-none">×</button>
          </div>

          <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">

            {/* ── Options phase ── */}
            {phase === 'options' && (
              <div className="space-y-5">
                <div className="text-sm text-gray-400 space-y-1">
                  <p>The AI will review your <span className="text-gray-200">job config</span>, <span className="text-gray-200">dataset stats</span>, and <span className="text-gray-200">recent loss curve</span> for issues.</p>
                  {!jobId && (
                    <p className="text-amber-400 text-xs mt-2">Save the job first to run a full analysis.</p>
                  )}
                </div>

                <div className="border border-gray-700 rounded-lg p-4 space-y-2">
                  <label className="flex items-start gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={includeImages}
                      onChange={e => setIncludeImages(e.target.checked)}
                      className="mt-0.5 accent-purple-500 flex-shrink-0"
                    />
                    <div>
                      <p className="text-sm text-gray-200 font-medium">Include visual analysis</p>
                      <p className="text-xs text-gray-500 mt-0.5">
                        Sends recent sample outputs and dataset images to the vision model.
                        Slower and uses more context — but catches white noise, mode collapse,
                        and dataset quality issues a text-only review misses.
                      </p>
                    </div>
                  </label>
                </div>
              </div>
            )}

            {/* ── Loading phase ── */}
            {phase === 'loading' && (
              <div className="flex flex-col items-center justify-center h-40 gap-3 text-gray-400">
                <div className="w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
                <p className="text-sm">{statusMessage || 'Starting…'}</p>
              </div>
            )}

            {/* ── Results phase ── */}
            {phase === 'results' && (
              <>
                {contextMessage && (
                  <div className="rounded-lg bg-red-900/30 border border-red-700 p-3 text-red-300 text-sm flex items-start gap-2">
                    <span className="text-lg leading-none mt-0.5">⚠</span>
                    <span>{contextMessage}</span>
                  </div>
                )}
                {error && (
                  <div className="rounded-lg bg-red-900/30 border border-red-700 p-4 text-red-300 text-sm">
                    {error}
                  </div>
                )}

                {!error && findings.length === 0 && (
                  <div className="flex flex-col items-center justify-center h-40 text-gray-500 text-sm">
                    <p className="text-2xl mb-2">✓</p>
                    <p>No issues found — config looks good!</p>
                  </div>
                )}

                {(['high', 'medium', 'low'] as const).map(level => {
                  const group = grouped[level];
                  if (group.length === 0) return null;
                  return (
                    <div key={level}>
                      <div className={`flex items-center gap-2 mb-3 pb-1 border-b ${CONFIDENCE_COLORS[level]} border-opacity-40`}>
                        <span className={`text-xs font-bold uppercase tracking-widest ${CONFIDENCE_COLORS[level].split(' ')[0]}`}>
                          {CONFIDENCE_LABELS[level]}
                        </span>
                        <span className="text-xs text-gray-600">({group.length})</span>
                      </div>
                      <div className="space-y-3">
                        {group.map((f, i) => renderFinding(f, i))}
                      </div>
                    </div>
                  );
                })}
              </>
            )}
          </div>

          {/* Speculative footer — results only */}
          {phase === 'results' && grouped.low.length > 0 && (
            <div className="px-5 py-3 border-t border-gray-700 flex-shrink-0">
              <label className="flex items-start gap-2 cursor-pointer text-sm text-gray-400">
                <input
                  type="checkbox"
                  checked={speculativeOk}
                  onChange={e => setSpeculativeOk(e.target.checked)}
                  className="mt-0.5 accent-purple-500"
                />
                <span>I understand the low-confidence suggestions are speculative and not based on direct documentation for this model</span>
              </label>
            </div>
          )}

          <div className="px-5 py-3 border-t border-gray-700 flex justify-between items-center flex-shrink-0">
            {phase === 'results' && (
              <button
                onClick={() => setPhase('options')}
                className="px-3 py-1.5 rounded text-sm text-gray-400 hover:text-gray-200 transition-colors"
              >
                ← Options
              </button>
            )}
            {phase !== 'results' && <div />}

            <div className="flex gap-2">
              <button
                onClick={handleClose}
                className="px-4 py-1.5 rounded bg-gray-700 hover:bg-gray-600 text-sm text-white transition-colors"
              >
                Close
              </button>
              {phase === 'options' && (
                <button
                  onClick={runAnalysis}
                  disabled={!jobId}
                  className="px-4 py-1.5 rounded bg-purple-700 hover:bg-purple-600 text-sm text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  title={!jobId ? 'Save the job first' : ''}
                >
                  Run Analysis
                </button>
              )}
            </div>
          </div>
        </DialogPanel>
      </div>
    </Dialog>
  );
}
