'use client';

import { useEffect, useRef, useState } from 'react';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';

// how soon the next poll fires after a manual refresh (e.g. after clicking Stop),
// so status changes show up quickly instead of waiting out the full reloadInterval
const FAST_FOLLOWUP_INTERVAL = 1000;

export default function useJob(jobID: string, reloadInterval: null | number = null) {
  const [job, setJob] = useState<Job | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchJob = () => {
    setStatus('loading');
    return apiClient
      .get(`/api/jobs?id=${jobID}`)
      .then(res => res.data)
      .then(data => {
        console.log('Job:', data);
        setJob(data);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching datasets:', error);
        setStatus('error');
      });
  };

  // schedules the next poll only after the current one settles, so a slow
  // server can't stack overlapping requests the way setInterval does
  const scheduleNext = (delay: number) => {
    if (timerRef.current) clearTimeout(timerRef.current);
    if (!reloadInterval) return;
    timerRef.current = setTimeout(async () => {
      await fetchJob();
      scheduleNext(reloadInterval);
    }, delay);
  };

  useEffect(() => {
    fetchJob();
    scheduleNext(reloadInterval || 0);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [jobID]);

  // manual refresh (e.g. after stopping a job) also fast-tracks the next poll
  const refreshJob = () => {
    fetchJob();
    scheduleNext(FAST_FOLLOWUP_INTERVAL);
  };
  return { job, setJob, status, refreshJob };
}
