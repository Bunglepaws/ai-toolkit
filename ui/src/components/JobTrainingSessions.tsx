'use client';

import { Job } from '@prisma/client';
import { useState, useEffect, useCallback, useRef } from 'react';
import { MdExpandMore, MdExpandLess, MdFileDownload } from 'react-icons/md';

interface Session {
  id: number | null;
  start_time: number;
  end_time: number | null;
  start_step: number | null;
  startup_seconds: number | null;
  sampling_seconds: number | null;
  training_seconds: number | null;
  total_seconds: number | null;
  in_progress: boolean;
  overridden?: boolean;
  estimated?: boolean;
}

interface SessionsData {
  sessions: Session[];
  total_seconds: number;
  startup_total: number;
  sampling_total: number;
  training_total: number;
  grand_total: number;
}

const DASH = '–';

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.round(seconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  return `${m}m ${s}s`;
}

function formatDateTime(unixSeconds: number): string {
  return new Date(unixSeconds * 1000).toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatEndDateTime(startUnix: number, endUnix: number): string {
  const start = new Date(startUnix * 1000);
  const end = new Date(endUnix * 1000);
  const sameDay =
    start.getFullYear() === end.getFullYear() &&
    start.getMonth() === end.getMonth() &&
    start.getDate() === end.getDate();
  if (sameDay) {
    return end.toLocaleString(undefined, { hour: '2-digit', minute: '2-digit' });
  }
  return formatDateTime(endUnix);
}

function durationCell(seconds: number | null): string {
  return seconds !== null ? formatDuration(seconds) : DASH;
}

/** Parse "2h 15m", "135m", "8100s", or plain seconds number → seconds. Returns null on failure. */
function parseDuration(raw: string): number | null {
  const s = raw.trim();
  if (!s) return null;

  let total = 0;
  const hMatch = s.match(/(\d+(?:\.\d+)?)\s*h/i);
  const mMatch = s.match(/(\d+(?:\.\d+)?)\s*m(?!s)/i);
  const sMatch = s.match(/(\d+(?:\.\d+)?)\s*s/i);

  if (hMatch || mMatch || sMatch) {
    if (hMatch) total += parseFloat(hMatch[1]) * 3600;
    if (mMatch) total += parseFloat(mMatch[1]) * 60;
    if (sMatch) total += parseFloat(sMatch[1]);
    return total > 0 ? total : null;
  }

  // plain number interpreted as minutes
  const n = parseFloat(s);
  if (!isNaN(n) && n >= 0) return n * 60;
  return null;
}

const gridCols = '1.6fr 1fr 0.7fr 0.9fr 0.9fr 1fr 1fr';

function csvEscape(value: string): string {
  if (/[",\n]/.test(value)) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

function downloadSessionsCsv(jobName: string, sessions: Session[], totals: {
  startup: number;
  sampling: number;
  training: number;
  grand: number;
}) {
  const header = ['Start', 'End', 'Start Step', 'Startup (s)', 'Sampling (s)', 'Training (s)', 'Total (s)'];
  const rows = sessions.map(s => [
    formatDateTime(s.start_time),
    s.in_progress ? 'in progress' : s.end_time !== null ? formatDateTime(s.end_time) : '',
    s.start_step ? String(s.start_step) : '',
    s.startup_seconds !== null ? String(Math.round(s.startup_seconds)) : '',
    s.sampling_seconds !== null ? String(Math.round(s.sampling_seconds)) : '',
    s.training_seconds !== null ? String(Math.round(s.training_seconds)) : '',
    s.total_seconds !== null ? String(Math.round(s.total_seconds)) : '',
  ]);
  const subtotalRow = [
    'Subtotals',
    '',
    '',
    String(Math.round(totals.startup)),
    String(Math.round(totals.sampling)),
    String(Math.round(totals.training)),
    String(Math.round(totals.grand)),
  ];

  const lines = [header, subtotalRow, ...rows].map(row => row.map(csvEscape).join(','));
  const csv = lines.join('\r\n');

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${jobName}_training_sessions.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

interface Props {
  job: Job;
}

export default function JobTrainingSessions({ job }: Props) {
  const [data, setData] = useState<SessionsData | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editValue, setEditValue] = useState('');
  const [editOriginalSeconds, setEditOriginalSeconds] = useState<number | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch(`/api/jobs/${job.id}/sessions`);
      if (!res.ok) return;
      const json: SessionsData = await res.json();
      setData(json);
    } catch {
      // silently ignore — sessions are informational
    }
  }, [job.id]);

  useEffect(() => {
    fetchSessions();
    const interval = setInterval(fetchSessions, 10_000);
    return () => clearInterval(interval);
  }, [fetchSessions]);

  useEffect(() => {
    if (editingId !== null) inputRef.current?.focus();
  }, [editingId]);

  const startEdit = (session: Session) => {
    if (session.id === null) return;
    const formatted = session.training_seconds !== null ? formatDuration(session.training_seconds) : '';
    setEditingId(session.id);
    setEditValue(formatted);
    setEditOriginalSeconds(session.training_seconds);
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditValue('');
  };

  const commitEdit = async (sessionId: number) => {
    const seconds = editValue.trim() === '' ? null : parseDuration(editValue);
    if (editValue.trim() !== '' && seconds === null) {
      // bad input — keep editing
      return;
    }
    // No real change — parsed value matches what was there before
    if (seconds === editOriginalSeconds) {
      cancelEdit();
      return;
    }
    setEditingId(null);
    setEditValue('');
    try {
      await fetch(`/api/jobs/${job.id}/sessions/${sessionId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ training_seconds_override: seconds }),
      });
    } catch {
      // ignore — next poll will reflect the actual state
    }
    fetchSessions();
  };

  if (!data || data.sessions.length === 0) return null;

  // Hide sessions with no steps unless it's the one currently in progress.
  const visibleSessions = data.sessions
    .filter(s => s.training_seconds !== null || s.in_progress)
    .sort((a, b) => a.start_time - b.start_time);

  if (visibleSessions.length === 0) return null;

  const anyEstimated = visibleSessions.some(s => s.estimated);

  const sumVisible = (key: 'startup_seconds' | 'sampling_seconds' | 'training_seconds') =>
    visibleSessions.reduce((acc, s) => acc + (s[key] ?? 0), 0);

  const visibleStartupTotal = sumVisible('startup_seconds');
  const visibleSamplingTotal = sumVisible('sampling_seconds');
  const visibleTrainingTotal = sumVisible('training_seconds');
  const visibleGrandTotal = visibleStartupTotal + visibleSamplingTotal + visibleTrainingTotal;

  return (
    <div className="text-sm">
      <button
        type="button"
        onClick={() => setExpanded(v => !v)}
        className="w-full flex items-center justify-between px-4 py-2 text-gray-400 hover:text-gray-300 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="h-1.5 w-1.5 rounded-full bg-purple-500/70 shrink-0" />
          <span className="text-gray-400">Training time</span>
          <span className="text-gray-600">
            {formatDuration(visibleGrandTotal)} total
            <span className="ml-2">
              ({visibleSessions.length} {visibleSessions.length === 1 ? 'session' : 'sessions'})
            </span>
          </span>
        </div>
        {expanded ? (
          <MdExpandLess className="text-gray-600 shrink-0" />
        ) : (
          <MdExpandMore className="text-gray-600 shrink-0" />
        )}
      </button>

      {expanded && (
        <div className="px-4 pt-1 pb-2">
          {anyEstimated && (
            <p className="text-xs text-yellow-700/60 pb-1">
              ~ estimated from step timestamps — model load time not included
            </p>
          )}
          <div className="flex items-center justify-between pb-1">
            <p className="text-xs text-gray-700">Click a training time to override it.</p>
            <button
              type="button"
              onClick={() =>
                downloadSessionsCsv(job.name, visibleSessions, {
                  startup: visibleStartupTotal,
                  sampling: visibleSamplingTotal,
                  training: visibleTrainingTotal,
                  grand: visibleGrandTotal,
                })
              }
              className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 transition-colors"
              title="Export to CSV"
            >
              <MdFileDownload />
              Export CSV
            </button>
          </div>
          <div className="max-h-80 overflow-y-auto overflow-x-auto">
            <div className="min-w-[560px]">
              <div
                className="grid gap-x-2 text-xs text-gray-500 pb-1 border-b border-gray-800"
                style={{ gridTemplateColumns: gridCols }}
              >
                <div>Start</div>
                <div>End</div>
                <div className="text-right">Step</div>
                <div className="text-right">Startup</div>
                <div className="text-right">Sampling</div>
                <div className="text-right">Training</div>
                <div className="text-right">Total</div>
              </div>

              <div
                className="grid gap-x-2 py-1 text-xs font-medium text-gray-300 border-b border-gray-700"
                style={{ gridTemplateColumns: gridCols }}
              >
                <div>Subtotals</div>
                <div />
                <div />
                <div className="text-right font-mono">{formatDuration(visibleStartupTotal)}</div>
                <div className="text-right font-mono">{formatDuration(visibleSamplingTotal)}</div>
                <div className="text-right font-mono">{formatDuration(visibleTrainingTotal)}</div>
                <div className="text-right font-mono text-purple-400">
                  {formatDuration(visibleGrandTotal)}
                </div>
              </div>

              <div className="space-y-0.5 pt-1">
                {visibleSessions.map((session, i) => (
                  <div
                    key={i}
                    className="grid gap-x-2 py-1 text-xs text-gray-500"
                    style={{ gridTemplateColumns: gridCols }}
                  >
                    <div>{formatDateTime(session.start_time)}</div>
                    <div className={session.in_progress ? 'text-gray-700' : ''}>
                      {session.in_progress
                        ? 'in progress'
                        : session.end_time !== null
                          ? formatEndDateTime(session.start_time, session.end_time)
                          : DASH}
                    </div>
                    <div className="text-right">{session.start_step || DASH}</div>
                    <div className="text-right font-mono">{durationCell(session.startup_seconds)}</div>
                    <div className="text-right font-mono">{durationCell(session.sampling_seconds)}</div>
                    <div
                      className={`text-right font-mono ${session.estimated ? 'text-yellow-700/70' : ''}`}
                    >
                      {editingId === session.id ? (
                        <input
                          ref={inputRef}
                          value={editValue}
                          onChange={e => setEditValue(e.target.value)}
                          onKeyDown={e => {
                            if (e.key === 'Enter') commitEdit(session.id!);
                            if (e.key === 'Escape') cancelEdit();
                          }}
                          onBlur={() => commitEdit(session.id!)}
                          className="w-full bg-gray-900 border border-gray-600 rounded px-1 text-right font-mono text-xs text-gray-200 focus:outline-none focus:border-purple-500"
                          placeholder="e.g. 2h 15m"
                        />
                      ) : (
                        <span
                          onClick={() => session.id !== null && startEdit(session)}
                          className={`cursor-pointer hover:text-gray-300 ${session.overridden ? 'text-blue-400/70' : ''}`}
                          title={session.id !== null ? 'Click to override' : undefined}
                        >
                          {session.training_seconds !== null
                            ? `${session.estimated ? '~' : ''}${formatDuration(session.training_seconds)}${session.overridden ? ' *' : ''}`
                            : DASH}
                        </span>
                      )}
                    </div>
                    <div className="text-right font-mono text-gray-400">
                      {durationCell(session.total_seconds)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
