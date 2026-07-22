'use client';
import { useState, useEffect } from 'react';
import { createGlobalState } from 'react-global-hooks';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { OctagonX } from 'lucide-react';
import React from 'react';
import { Job } from '@prisma/client';
import { gracefulStopJob, saveAndPauseJob, saveAndRequeueJob } from '@/utils/jobs';
import classNames from 'classnames';

export interface StopJobState {
  job: Job | null;
  onRefresh?: () => void;
}

export const stopJobState = createGlobalState<StopJobState | null>(null);

export const openStopJobModal = (props: StopJobState) => {
  stopJobState.set(props);
};

export default function StopJobModal() {
  const [state, setState] = stopJobState.use();
  const [isOpen, setIsOpen] = useState(false);
  const [isStopping, setIsStopping] = useState(false);

  useEffect(() => {
    if (state?.job) {
      setIsOpen(true);
    }
  }, [state]);

  useEffect(() => {
    if (!isOpen && state) {
      const timer = setTimeout(() => {
        setState(null);
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [isOpen, state, setState]);

  const handleStop = async (mode: 'graceful' | 'saveJob' | 'saveQueue') => {
    if (!state?.job || isStopping) return;

    setIsStopping(true);
    setIsOpen(false);
    try {
      if (mode === 'saveQueue') {
        // Route handles stopping the queue + save + return_to_queue atomically
        await saveAndRequeueJob(state.job.id);
      } else if (mode === 'saveJob') {
        await saveAndPauseJob(state.job.id);
      } else {
        await gracefulStopJob(state.job.id);
      }
      if (state.onRefresh) {
        state.onRefresh();
      }
    } catch (e) {
      console.error('Error stopping job:', e);
      alert('Failed to stop job. Check console for details.');
    } finally {
      setIsStopping(false);
    }
  };

  const onCancel = () => {
    setIsOpen(false);
  };

  const btnBase = 'inline-flex w-full justify-center rounded-md px-3 py-2 text-sm font-semibold text-white shadow-xs sm:ml-3 sm:w-auto cursor-pointer';

  return (
    <Dialog open={isOpen} onClose={onCancel} className="relative z-50">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />

      <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
        <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
          <DialogPanel
            transition
            className="relative transform overflow-hidden rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in sm:my-8 sm:w-full sm:max-w-lg data-closed:sm:translate-y-0 data-closed:sm:scale-95"
          >
            <div className="bg-gray-800 px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
              <div className="sm:flex sm:items-start">
                <div className="mx-auto flex size-12 shrink-0 items-center justify-center rounded-full bg-blue-500 sm:mx-0 sm:size-10">
                  <OctagonX aria-hidden="true" className="size-6 text-blue-950" />
                </div>
                <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left flex-1">
                  <DialogTitle as="h3" className="text-base font-semibold text-blue-500">
                    Stop "{state?.job?.name}"
                  </DialogTitle>
                  <div className="mt-2">
                    <p className="text-sm text-gray-200">
                      Do you want to save a snapshot before stopping, or stop immediately without saving?
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div className="bg-gray-700 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6 gap-y-2 flex-wrap">
              <button
                type="button"
                onClick={() => handleStop('saveQueue')}
                disabled={isStopping}
                className={classNames(btnBase, 'bg-blue-700 hover:bg-blue-600', { 'opacity-50 cursor-not-allowed': isStopping })}
              >
                {isStopping ? 'Processing...' : 'Save and Stop Queue'}
              </button>
              <button
                type="button"
                onClick={() => handleStop('saveJob')}
                disabled={isStopping}
                className={classNames(btnBase, 'bg-blue-600 hover:bg-blue-500', { 'opacity-50 cursor-not-allowed': isStopping })}
              >
                {isStopping ? 'Processing...' : 'Save and Stop Job'}
              </button>
              <button
                type="button"
                onClick={() => handleStop('graceful')}
                disabled={isStopping}
                className={classNames(btnBase, 'bg-gray-600 hover:bg-gray-500', { 'opacity-50 cursor-not-allowed': isStopping })}
              >
                {isStopping ? 'Processing...' : 'Stop'}
              </button>
              <button
                type="button"
                data-autofocus
                onClick={onCancel}
                disabled={isStopping}
                className="mt-3 inline-flex w-full justify-center rounded-md bg-gray-800 px-3 py-2 text-sm font-semibold text-gray-200 hover:bg-gray-700 sm:mt-0 sm:w-auto ring-0 cursor-pointer"
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
