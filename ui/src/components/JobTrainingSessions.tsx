'use client';

import { Job } from '@prisma/client';
import { useState, useEffect, useCallback } from 'react';
import { MdExpandMore, MdExpandLess } from 'react-icons/md';

interface Session {
  start_time: number;
  end_time: number | null;
  duration_seconds: number | null;
  estimated?: boolean;
}

interface SessionsData {
  sessions: Session[];
  total_seconds: number;
}

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

interface Props {
  job: Job;
}

export default function JobTrainingSessions({ job }: Props) {
  const [data, setData] = useState<SessionsData | null>(null);
  const [expanded, setExpanded] = useState(false);

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

  if (!data || data.sessions.length === 0) return null;

  // Hide sessions with no steps unless it's the very last one (currently running).
  const visibleSessions = data.sessions.filter(
    (s, i) => s.duration_seconds !== null || i === data.sessions.length - 1,
  );

  if (visibleSessions.length === 0) return null;

  const anyEstimated = visibleSessions.some(s => s.estimated);
  const lastSession = visibleSessions[visibleSessions.length - 1];
  const isLastInProgress = lastSession.duration_seconds === null;

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
            {formatDuration(data.total_seconds)} total
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
          <div className="max-h-64 overflow-y-auto space-y-0.5">
            {visibleSessions.map((session, i) => (
              <div key={i} className="flex items-center justify-between py-1 text-xs">
                <div className="flex items-center gap-2 text-gray-500">
                  <span className="w-5 text-right">{i + 1}.</span>
                  <span>{formatDateTime(session.start_time)}</span>
                </div>
                <span className={`font-mono ${session.estimated ? 'text-yellow-700/70' : 'text-gray-600'}`}>
                  {session.duration_seconds !== null
                    ? `${session.estimated ? '~' : ''}${formatDuration(session.duration_seconds)}`
                    : <span className="text-gray-700">in progress</span>}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
