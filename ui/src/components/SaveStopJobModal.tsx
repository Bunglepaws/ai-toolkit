'use client';
import { useState, useEffect } from 'react';
import { createGlobalState } from 'react-global-hooks';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { Save } from 'lucide-react';
import React from 'react';
import { Job } from '@prisma/client';
import { saveJobNow, saveAndPauseJob, saveAndRequeueJob, gracefulStopJob } from '@/utils/jobs';
import classNames from 'classnames';

/**
 * The single save/stop dialog for a running job, opened by both the save icon
 * and the pause icon in JobActionBar. It replaces the old pair of dialogs
 * (SaveSnapshotModal + StopJobModal), which offered overlapping options under
 * different names for the same underlying routes.
 *
 * Every save option below asks the trainer to write its checkpoint on the NEXT
 * step; they differ only in what happens to the job and the queue afterwards.
 * None of them sets `stop` alongside a pending save -- the stop-watcher thread
 * turns `stop` into an immediate SIGINT and would kill the step before the
 * checkpoint landed. See the route handlers and DiffusionTrainer.maybe_stop().
 *
 *   continue   /save_now          save_now
 *                                   -> keeps training this job
 *   stopQueue  /save_and_requeue  save_now + stop_after_save + return_to_queue,
 *                                 queue.is_running = false, queue_position = 0
 *                                   -> job returns to the head of the queue, queue stops
 *   trainNext  /save_and_pause    save_now + stop_after_save
 *                                   -> job leaves the queue, queue starts the next job
 *   stop       /stop_graceful     stop
 *                                   -> no checkpoint, job stops now
 */
export type SaveStopMode = 'continue' | 'stopQueue' | 'trainNext' | 'stop';

export interface SaveStopJobState {
  job: Job | null;
  onRefresh?: () => void;
}

export const saveStopJobState = createGlobalState<SaveStopJobState | null>(null);

export const openSaveStopJobModal = (props: SaveStopJobState) => {
  saveStopJobState.set(props);
};

export default function SaveStopJobModal() {
  const [state, setState] = saveStopJobState.use();
  const [isOpen, setIsOpen] = useState(false);
  const [isWorking, setIsWorking] = useState(false);

  useEffect(() => {
    if (state?.job) {
      setIsOpen(true);
    }
  }, [state]);

  useEffect(() => {
    if (!isOpen && state) {
      // use timeout to allow the dialog to close before resetting the state
      const timer = setTimeout(() => {
        setState(null);
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [isOpen, state, setState]);

  const handleAction = async (mode: SaveStopMode) => {
    if (!state?.job || isWorking) return;

    const jobID = state.job.id;
    setIsWorking(true);
    setIsOpen(false);
    try {
      if (mode === 'continue') {
        await saveJobNow(jobID);
      } else if (mode === 'stopQueue') {
        await saveAndRequeueJob(jobID);
      } else if (mode === 'trainNext') {
        await saveAndPauseJob(jobID);
      } else {
        await gracefulStopJob(jobID);
      }
      if (state.onRefresh) {
        state.onRefresh();
      }
    } catch (e) {
      console.error(`Error running "${mode}" on job ${jobID}:`, e);
      alert('Failed to process request. Check console for details.');
    } finally {
      setIsWorking(false);
    }
  };

  const onCancel = () => {
    setIsOpen(false);
  };

  const btnBase =
    'inline-flex w-full justify-center rounded-md px-3 py-2 text-sm font-semibold text-white shadow-xs sm:ml-3 sm:w-auto cursor-pointer';

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
            className="relative transform overflow-hidden rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in sm:my-8 sm:w-full sm:max-w-2xl data-closed:sm:translate-y-0 data-closed:sm:scale-95"
          >
            <div className="bg-gray-800 px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
              <div className="sm:flex sm:items-start">
                <div className="mx-auto flex size-12 shrink-0 items-center justify-center rounded-full bg-blue-500 sm:mx-0 sm:size-10">
                  <Save aria-hidden="true" className="size-6 text-blue-950" />
                </div>
                <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left flex-1">
                  <DialogTitle as="h3" className="text-base font-semibold text-blue-500">
                    Save Snapshot for &quot;{state?.job?.name}&quot;
                  </DialogTitle>
                  <div className="mt-2">
                    <p className="text-sm text-gray-200">
                      A snapshot is written on the next training step. Choose what happens to this job and the queue
                      afterwards.
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div className="bg-gray-700 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6 gap-y-2 flex-wrap">
              <button
                type="button"
                onClick={() => handleAction('continue')}
                disabled={isWorking}
                title="Saves on the next step and then continues training the same job."
                className={classNames(btnBase, 'bg-blue-700 hover:bg-blue-600', {
                  'opacity-50 cursor-not-allowed': isWorking,
                })}
              >
                {isWorking ? 'Processing...' : 'Save and Continue'}
              </button>
              <button
                type="button"
                onClick={() => handleAction('stopQueue')}
                disabled={isWorking}
                title="Saves on the next step and then stops the queue, leaving the job order the same so this job remains first in the queue."
                className={classNames(btnBase, 'bg-blue-600 hover:bg-blue-500', {
                  'opacity-50 cursor-not-allowed': isWorking,
                })}
              >
                {isWorking ? 'Processing...' : 'Save and Stop Queue'}
              </button>
              <button
                type="button"
                onClick={() => handleAction('trainNext')}
                disabled={isWorking}
                title="Saves on the next step and removes this job from the queue, then proceeds with the next job in the queue."
                className={classNames(btnBase, 'bg-blue-600 hover:bg-blue-500', {
                  'opacity-50 cursor-not-allowed': isWorking,
                })}
              >
                {isWorking ? 'Processing...' : 'Save and Train Next'}
              </button>
              <button
                type="button"
                onClick={() => handleAction('stop')}
                disabled={isWorking}
                title="Stops the job now without saving anything. Everything since the last checkpoint is lost."
                className={classNames(btnBase, 'bg-gray-600 hover:bg-gray-500', {
                  'opacity-50 cursor-not-allowed': isWorking,
                })}
              >
                {isWorking ? 'Processing...' : 'Stop Without Saving'}
              </button>
              <button
                type="button"
                data-autofocus
                onClick={onCancel}
                disabled={isWorking}
                title="Close this dialog and leave the job exactly as it is."
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
