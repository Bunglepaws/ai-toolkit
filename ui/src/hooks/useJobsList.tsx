'use client';

import { useEffect, useRef, useState } from 'react';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';

// how soon the next poll fires after a manual refresh (e.g. after reordering
// the queue), so position/status changes show up quickly instead of waiting
// out the full reloadInterval
const FAST_FOLLOWUP_INTERVAL = 1000;

type UseJobsListProps = {
  onlyActive?: boolean;
  reloadInterval?: number | null;
  job_type?: string | null;
};

export default function useJobsList({
  onlyActive = false,
  reloadInterval = null,
  job_type = null,
}: UseJobsListProps = {}) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const isFetchingRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Bumped by refreshJobs(). Anything started under an older generation is
  // stale: its payload predates the local mutation that triggered the refresh
  // (e.g. a queue reorder), so applying it would revert the optimistic order.
  const generationRef = useRef(0);

  const fetchJobs = (force = false) => {
    // `force` is for manual refreshes, which must not be swallowed just because
    // a background poll happens to be mid-flight
    if (isFetchingRef.current && !force) return Promise.resolve();
    const gen = generationRef.current;
    isFetchingRef.current = true;
    setStatus('loading');
    const params: Record<string, string> = {};
    if (job_type) {
      params.job_type = job_type;
    }
    if (onlyActive) {
      params.only_active = 'true';
    }
    return apiClient
      .get('/api/jobs', { params })
      .then(res => res.data)
      .then(data => {
        // a manual refresh superseded this request while it was in flight
        if (gen !== generationRef.current) return;
        console.log('Jobs:', data);
        if (data.error) {
          console.log('Error fetching jobs:', data.error);
          setStatus('error');
        } else {
          setJobs(data.jobs);
          setStatus('success');
        }
      })
      .catch(error => {
        console.error('Error fetching jobs:', error);
        setStatus('error');
      })
      .finally(() => {
        isFetchingRef.current = false;
      });
  };

  // schedules the next poll only after the current one settles, so a slow
  // server can't stack overlapping requests the way setInterval does
  const scheduleNext = (delay: number) => {
    if (timerRef.current) clearTimeout(timerRef.current);
    if (!reloadInterval) return;
    const gen = generationRef.current;
    timerRef.current = setTimeout(async () => {
      await fetchJobs();
      // A manual refresh has taken over the schedule since this poll was
      // queued. Re-arming here would clear its fast follow-up and put the
      // full reloadInterval back, which is what made reorders take 5s.
      if (gen !== generationRef.current) return;
      scheduleNext(reloadInterval);
    }, delay);
  };

  useEffect(() => {
    fetchJobs();
    scheduleNext(reloadInterval || 0);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Manual refresh (e.g. right after reordering the queue). Invalidates any
  // in-flight poll so a pre-mutation payload can't land, fetches immediately
  // rather than waiting out the cadence, then fast-tracks one follow-up in case
  // the write hadn't committed server-side yet.
  const refreshJobs = () => {
    generationRef.current += 1;
    fetchJobs(true);
    scheduleNext(FAST_FOLLOWUP_INTERVAL);
  };

  return { jobs, setJobs, status, refreshJobs };
}
