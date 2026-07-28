'use client';

import { useState, useEffect } from 'react';
import { createGlobalState } from 'react-global-hooks';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { History } from 'lucide-react';
import { Job } from '@prisma/client';
import { listCheckpoints, resumeFromCheckpoint } from '@/utils/jobs';
import type { CheckpointEntry } from '@/app/api/jobs/[jobID]/checkpoints/route';

export interface ResumeFromCheckpointState {
  job: Job;
  onRefresh?: () => void;
}

export const resumeFromCheckpointState = createGlobalState<ResumeFromCheckpointState | null>(null);

export const openResumeFromCheckpointModal = (props: ResumeFromCheckpointState) => {
  resumeFromCheckpointState.set(props);
};

type Phase = 'loading' | 'list' | 'confirm' | 'applying';

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(ms: number): string {
  return new Date(ms).toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

export default function ResumeFromCheckpointModal() {
  const [state, setState] = resumeFromCheckpointState.use();
  const [isOpen, setIsOpen] = useState(false);
  const [phase, setPhase] = useState<Phase>('loading');
  const [checkpoints, setCheckpoints] = useState<CheckpointEntry[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [selected, setSelected] = useState<CheckpointEntry | null>(null);
  const [deleteSamples, setDeleteSamples] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (state?.job) setIsOpen(true);
  }, [state]);

  useEffect(() => {
    if (!isOpen || !state?.job) return;
    setPhase('loading');
    setCheckpoints([]);
    setSelected(null);
    setError(null);
    setDeleteSamples(false);

    listCheckpoints(state.job.id)
      .then((res: any) => {
        const data = res.data ?? res;
        setCheckpoints(data.checkpoints ?? []);
        setCurrentStep(data.currentStep ?? 0);
        setPhase('list');
      })
      .catch((e: any) => {
        setError(e?.response?.data?.error ?? e?.message ?? 'Failed to load checkpoints');
        setPhase('list');
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen && state) {
      const timer = setTimeout(() => setState(null), 300);
      return () => clearTimeout(timer);
    }
  }, [isOpen, state, setState]);

  const handleClose = () => {
    setIsOpen(false);
    setTimeout(() => {
      setPhase('loading');
      setCheckpoints([]);
      setSelected(null);
      setError(null);
    }, 300);
  };

  const handleSelect = (entry: CheckpointEntry) => {
    setSelected(entry);
    setPhase('confirm');
  };

  const handleBack = () => {
    setSelected(null);
    setPhase('list');
  };

  const handleConfirm = async () => {
    if (!selected || !state?.job) return;
    setPhase('applying');
    setError(null);
    try {
      await resumeFromCheckpoint(state.job.id, selected.step, deleteSamples);
      if (state.onRefresh) state.onRefresh();
      handleClose();
    } catch (e: any) {
      setError(e?.response?.data?.error ?? e?.message ?? 'Rollback failed');
      setPhase('confirm');
    }
  };

  const stepsToDelete = checkpoints.filter(c => c.step > (selected?.step ?? 0));

  return (
    <Dialog open={isOpen} onClose={handleClose} className="relative z-50">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />
      <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
        <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
          <DialogPanel
            transition
            className="relative transform overflow-hidden rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in sm:my-8 sm:w-full sm:max-w-xl data-closed:sm:translate-y-0 data-closed:sm:scale-95"
          >
            {/* Header */}
            <div className="bg-gray-800 px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
              <div className="sm:flex sm:items-start">
                <div className="mx-auto flex size-12 shrink-0 items-center justify-center rounded-full bg-indigo-600 sm:mx-0 sm:size-10">
                  <History aria-hidden="true" className="size-5 text-white" />
                </div>
                <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left flex-1 min-w-0">
                  <DialogTitle as="h3" className="text-base font-semibold text-indigo-400">
                    Resume From Checkpoint
                  </DialogTitle>
                  <p className="text-sm text-gray-400 mt-1 truncate">
                    {state?.job?.name}
                  </p>
                </div>
              </div>
            </div>

            {/* Body */}
            <div className="px-4 sm:px-6 pb-4 sm:pb-5 space-y-4">

              {/* Loading */}
              {phase === 'loading' && (
                <div className="flex items-center justify-center h-32 gap-3 text-gray-400">
                  <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                  <span className="text-sm">Loading checkpoints…</span>
                </div>
              )}

              {/* Applying */}
              {phase === 'applying' && (
                <div className="flex items-center justify-center h-32 gap-3 text-gray-400">
                  <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                  <span className="text-sm">Rolling back…</span>
                </div>
              )}

              {/* List */}
              {phase === 'list' && (
                <>
                  {error && (
                    <div className="rounded-lg bg-red-900/30 border border-red-700 p-3 text-red-300 text-sm">
                      {error}
                    </div>
                  )}
                  {!error && checkpoints.length === 0 && (
                    <p className="text-sm text-gray-400 text-center py-8">No checkpoints found in the output folder.</p>
                  )}
                  {checkpoints.length > 0 && (
                    <>
                      <p className="text-sm text-gray-400">
                        Select a checkpoint to roll back to. Checkpoints newer than your selection will be deleted.
                      </p>
                      <div className="max-h-72 overflow-y-auto rounded-lg border border-gray-700 divide-y divide-gray-700">
                        {checkpoints.map(c => (
                          <button
                            key={c.step}
                            onClick={() => handleSelect(c)}
                            className="w-full text-left px-4 py-3 hover:bg-gray-700 transition-colors flex items-center justify-between gap-3 group"
                          >
                            <div className="flex items-center gap-3 min-w-0">
                              <span className="font-mono text-sm text-gray-200 shrink-0 tabular-nums">
                                {c.step.toLocaleString()}
                              </span>
                              <span className="text-xs text-gray-500 truncate">{c.filename}</span>
                            </div>
                            <div className="flex items-center gap-3 shrink-0 text-xs text-gray-500">
                              {c.hasOptimizer ? (
                                <span className="text-emerald-400 font-mono" title="Optimizer archive available">opt ✓</span>
                              ) : (
                                <span className="text-gray-600 font-mono" title="No optimizer archive">opt –</span>
                              )}
                              <span>{formatBytes(c.sizeBytes)}</span>
                              <span>{formatDate(c.mtimeMs)}</span>
                              {c.step === currentStep && (
                                <span className="text-indigo-400 font-semibold">current</span>
                              )}
                            </div>
                          </button>
                        ))}
                      </div>
                    </>
                  )}
                </>
              )}

              {/* Confirm */}
              {phase === 'confirm' && selected && (
                <>
                  <div className="rounded-lg bg-gray-750 border border-gray-700 p-4 space-y-3">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-gray-300">Rolling back to step</span>
                      <span className="font-mono text-indigo-400 font-semibold tabular-nums">
                        {selected.step.toLocaleString()}
                      </span>
                    </div>

                    {stepsToDelete.length > 0 && (
                      <div>
                        <p className="text-xs text-gray-500 uppercase tracking-wide font-semibold mb-1">
                          Checkpoints to delete ({stepsToDelete.length})
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {stepsToDelete.map(c => (
                            <span key={c.step} className="font-mono text-xs bg-red-900/30 text-red-400 px-2 py-0.5 rounded tabular-nums">
                              {c.step.toLocaleString()}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    <div>
                      <p className="text-xs text-gray-500 uppercase tracking-wide font-semibold mb-1">Optimizer state</p>
                      {selected.hasOptimizer ? (
                        <p className="text-sm text-emerald-400">
                          Archive found — optimizer will be restored to step {selected.step.toLocaleString()}.
                        </p>
                      ) : (
                        <p className="text-sm text-amber-400">
                          No optimizer archive at this step. Weights will roll back but optimizer momentum/statistics
                          will remain from the latest step. Training will still converge.
                        </p>
                      )}
                    </div>

                    <div>
                      <p className="text-xs text-gray-500 uppercase tracking-wide font-semibold mb-1">Loss log</p>
                      <p className="text-sm text-gray-400">
                        Steps beyond {selected.step.toLocaleString()} will be pruned from the loss graph immediately.
                      </p>
                    </div>
                  </div>

                  <label className="flex items-start gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={deleteSamples}
                      onChange={e => setDeleteSamples(e.target.checked)}
                      className="mt-0.5 accent-indigo-500 flex-shrink-0"
                    />
                    <div>
                      <p className="text-sm text-gray-300 font-medium">Also delete sample images after step {selected.step.toLocaleString()}</p>
                      <p className="text-xs text-gray-500 mt-0.5">Removes generated sample outputs from those steps.</p>
                    </div>
                  </label>

                  {error && (
                    <div className="rounded-lg bg-red-900/30 border border-red-700 p-3 text-red-300 text-sm">
                      {error}
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Footer */}
            <div className="bg-gray-700 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6 gap-2">
              {phase === 'confirm' && (
                <button
                  type="button"
                  onClick={handleConfirm}
                  className="inline-flex w-full justify-center rounded-md bg-red-700 hover:bg-red-600 px-3 py-2 text-sm font-semibold text-white shadow-xs sm:ml-3 sm:w-auto cursor-pointer transition-colors"
                >
                  Roll Back to Step {selected?.step.toLocaleString()}
                </button>
              )}
              {phase === 'confirm' && (
                <button
                  type="button"
                  onClick={handleBack}
                  className="mt-2 sm:mt-0 inline-flex w-full justify-center rounded-md bg-gray-800 px-3 py-2 text-sm font-semibold text-gray-200 hover:bg-gray-600 sm:w-auto cursor-pointer transition-colors"
                >
                  ← Back
                </button>
              )}
              <button
                type="button"
                onClick={handleClose}
                className="mt-2 sm:mt-0 inline-flex w-full justify-center rounded-md bg-gray-800 px-3 py-2 text-sm font-semibold text-gray-200 hover:bg-gray-600 sm:w-auto cursor-pointer transition-colors"
              >
                Cancel
              </button>
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>
  );
}
