'use client';

import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { CgSpinner } from 'react-icons/cg';
import useJobsList from '@/hooks/useJobsList';
import { getTotalSteps } from '@/utils/jobs';

// one job card is 59px tall, plus the 8px bottom padding on the list wrapper.
// below this the panel can't show a single card without clipping it, so we
// render the spacer empty instead of a scrolling sliver.
const MIN_PANEL_HEIGHT = 67;

export default function ActiveJobWidget() {
  const { jobs } = useJobsList({ onlyActive: true, reloadInterval: 5000 });
  // the panel is a flex-1 spacer, so its height comes from the leftover sidebar
  // space rather than its content — measuring it can't feed back into itself
  const panelRef = useRef<HTMLDivElement>(null);
  const [hasRoom, setHasRoom] = useState(true);

  useEffect(() => {
    const el = panelRef.current;
    if (!el) return;
    const observer = new ResizeObserver(entries => {
      setHasRoom(entries[0].contentRect.height >= MIN_PANEL_HEIGHT);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const showJobs = hasRoom && jobs && jobs.length > 0;

  return (
    <div ref={panelRef} className="flex-1 min-h-0 overflow-y-auto">
      {showJobs && (
        <div className="px-3 pb-2">
          <ul className="space-y-2">
            {jobs.map(job => {
            let totalSteps: number | undefined;
            try {
              totalSteps = getTotalSteps(job);
            } catch {
              totalSteps = undefined;
            }
            const pct = totalSteps ? Math.min(100, (job.step / totalSteps) * 100) : 0;
            const isRunning = job.status === 'running' || job.status === 'stopping';

            let label = job.name;
            let href = `/jobs/${job.id}`;
            if (job.job_type === 'caption') {
              const splits = (job.job_ref ?? '').split(/[/\\]/);
              label = splits[splits.length - 1] || job.name;
              href = `/datasets/${label}`;
            }

            let statusColor = 'text-gray-400';
            if (job.status === 'running') statusColor = 'text-blue-400';
            if (job.status === 'queued') statusColor = 'text-yellow-400';
            if (job.status === 'stopping') statusColor = 'text-orange-400';

            return (
              <li key={job.id} className="min-w-0">
                <Link
                  href={href}
                  className="block px-3 py-2 bg-gray-800 hover:bg-gray-950 rounded-lg transition-colors min-w-0"
                >
                  <div className="flex items-center gap-1.5 min-w-0">
                    {isRunning && <CgSpinner className="animate-spin text-blue-400 flex-shrink-0" />}
                    <span className="text-xs text-gray-100 truncate min-w-0 flex-1">{label}</span>
                  </div>
                  {totalSteps ? (
                    <div className="mt-1.5">
                      <div className="bg-gray-700 rounded-full h-1">
                        <div className="bg-blue-500 h-1 rounded-full transition-all" style={{ width: `${pct}%` }} />
                      </div>
                      <div className="flex justify-between gap-2 mt-0.5 min-w-0">
                        <span className={`text-[10px] truncate min-w-0 ${statusColor}`}>{job.info || job.status}</span>
                        <span className="text-[10px] text-gray-400 flex-shrink-0">
                          {job.step} / {totalSteps}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className={`text-[10px] mt-0.5 truncate ${statusColor}`}>{job.info || job.status}</div>
                  )}
                </Link>
              </li>
            );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
