'use client';

import { useEffect, useRef, useState } from 'react';
import { Queue } from '@prisma/client';
import { apiClient } from '@/utils/api';

type UseQueueListProps = {
  reloadInterval?: number | null;
};

export default function useQueueList({ reloadInterval = null }: UseQueueListProps = {}) {
  const [queues, setQueues] = useState<Queue[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchQueues = () => {
    setStatus('loading');
    return apiClient
      .get('/api/queue')
      .then(res => res.data)
      .then(data => {
        console.log('Queues:', data);
        if (data.error) {
          console.log('Error fetching queues:', data.error);
          setStatus('error');
        } else {
          setQueues(data.queues);
          setStatus('success');
        }
      })
      .catch(error => {
        console.error('Error fetching queues:', error);
        setStatus('error');
      });
  };

  const scheduleNext = (delay: number) => {
    if (timerRef.current) clearTimeout(timerRef.current);
    if (!reloadInterval) return;
    timerRef.current = setTimeout(async () => {
      await fetchQueues();
      scheduleNext(reloadInterval);
    }, delay);
  };

  useEffect(() => {
    fetchQueues();
    scheduleNext(reloadInterval || 0);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refreshQueues = () => {
    fetchQueues();
    scheduleNext(reloadInterval || 0);
  };

  return { queues, setQueues, status, refreshQueues };
}
