'use client';

import useQueueList from '@/hooks/useQueueList';
import { startQueue, stopQueue } from '@/utils/queue';

export default function QueueStatusWidget() {
  const { queues, refreshQueues } = useQueueList({ reloadInterval: 5000 });

  if (!queues || queues.length === 0) return null;

  return (
    <div className="px-3 pb-2">
      <ul className="w-[196px] space-y-1.5">
        {queues.map(queue => {
          const gpuIds = queue.gpu_ids as string;
          return (
            <li
              key={queue.id}
              className={`flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-lg ${
                queue.is_running ? 'bg-green-900/60' : 'bg-red-900/60'
              }`}
            >
              <div className="min-w-0 flex-1">
                <div className="text-[10px] text-gray-400 truncate">GPU {gpuIds}</div>
                <div className={`text-xs truncate ${queue.is_running ? 'text-green-300' : 'text-red-300'}`}>
                  {queue.is_running ? 'Running' : 'Stopped'}
                </div>
              </div>
              {queue.is_running ? (
                <button
                  onClick={async () => {
                    await stopQueue(gpuIds);
                    refreshQueues();
                  }}
                  className="flex-shrink-0 text-[10px] text-white bg-red-600 hover:bg-red-700 px-2 py-1 rounded"
                >
                  STOP
                </button>
              ) : (
                <button
                  onClick={async () => {
                    await startQueue(gpuIds);
                    refreshQueues();
                  }}
                  className="flex-shrink-0 text-[10px] text-white bg-green-600 hover:bg-green-700 px-2 py-1 rounded"
                >
                  START
                </button>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
