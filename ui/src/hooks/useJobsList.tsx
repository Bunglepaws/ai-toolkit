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

  const fetchJobs = () => {
    if (isFetchingRef.current) return Promise.resolve();
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
    timerRef.current = setTimeout(async () => {
      await fetchJobs();
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

  // manual refresh (e.g. after reordering the queue) also fast-tracks the next poll
  const refreshJobs = () => {
    fetchJobs();
    scheduleNext(FAST_FOLLOWUP_INTERVAL);
  };

  return { jobs, setJobs, status, refreshJobs };
}
