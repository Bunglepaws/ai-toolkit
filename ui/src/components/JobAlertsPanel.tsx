'use client';

import React from 'react';
import { clearJobAlerts } from '@/utils/jobs';

export interface JobAlert {
  timestamp: string;
  step: number;
  type: 'loss_spike' | 'white_noise_samples' | string;
  message: string;
  data?: Record<string, unknown>;
}

interface JobAlertsPanelProps {
  jobID: string;
  alerts: JobAlert[];
  onClose: () => void;
  onCleared: () => void;
}

const ALERT_ICONS: Record<string, string> = {
  loss_spike: '📈',
  white_noise_samples: '🌫️',
};

const ALERT_LABELS: Record<string, string> = {
  loss_spike: 'Loss Spike',
  white_noise_samples: 'White Noise Samples',
};

export default function JobAlertsPanel({ jobID, alerts, onClose, onCleared }: JobAlertsPanelProps) {
  const [clearing, setClearing] = React.useState(false);

  const handleClear = async () => {
    setClearing(true);
    try {
      await clearJobAlerts(jobID);
      onCleared();
      onClose();
    } catch (e) {
      console.error('Failed to clear alerts:', e);
    } finally {
      setClearing(false);
    }
  };

  // Close on Escape key
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const sorted = [...alerts].reverse(); // newest first

  return (
    <div className="mt-1 rounded-lg border border-amber-500/40 bg-gray-900 shadow-xl z-10 w-full">
      <div className="flex items-center justify-between px-3 py-2 border-b border-amber-500/30">
        <span className="text-amber-400 font-semibold text-sm flex items-center gap-1.5">
          ⚠ Training Alerts ({alerts.length})
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={handleClear}
            disabled={clearing}
            className="text-xs text-gray-400 hover:text-red-400 transition-colors disabled:opacity-50"
          >
            {clearing ? 'Clearing…' : 'Clear all'}
          </button>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-300 transition-colors text-lg leading-none"
            aria-label="Close alerts panel"
          >
            ×
          </button>
        </div>
      </div>
      <div className="max-h-64 overflow-y-auto divide-y divide-gray-800">
        {sorted.map((alert, i) => (
          <div key={i} className="px-3 py-2">
            <div className="flex items-start gap-2">
              <span className="text-base mt-0.5 flex-shrink-0">
                {ALERT_ICONS[alert.type] ?? '⚠'}
              </span>
              <div className="min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-amber-400 text-xs font-semibold uppercase tracking-wide">
                    {ALERT_LABELS[alert.type] ?? alert.type}
                  </span>
                  <span className="text-gray-500 text-xs">
                    step {alert.step}
                  </span>
                  <span className="text-gray-600 text-xs">
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <p className="text-gray-300 text-xs mt-0.5 break-words">{alert.message}</p>
                {alert.data && Object.keys(alert.data).length > 0 && (
                  <div className="mt-0.5 flex gap-3 flex-wrap">
                    {Object.entries(alert.data).map(([k, v]) => (
                      <span key={k} className="text-gray-500 text-xs">
                        {k}: <span className="text-gray-400">{String(v)}</span>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
