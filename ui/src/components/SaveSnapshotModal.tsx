'use client';
import { useRef } from 'react';
import { useState, useEffect } from 'react';
import { createGlobalState } from 'react-global-hooks';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { Save } from 'lucide-react';
import React from 'react';
import { Job } from '@prisma/client';
import { saveJob, saveAndPauseJob } from '@/utils/jobs';
import classNames from 'classnames';

export interface SaveSnapshotState {
  job: Job | null;
  onRefresh?: () => void;
}

export const saveSnapshotState = createGlobalState<SaveSnapshotState | null>(null);

export const openSaveSnapshotModal = (props: SaveSnapshotState) => {
  console.log('Opening save snapshot modal for job:', props.job?.id);
  saveSnapshotState.set(props);
};

export default function SaveSnapshotModal() {
  const [state, setState] = saveSnapshotState.use();
  const [isOpen, setIsOpen] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    if (state?.job) {
      console.log('SaveSnapshotModal: Job detected, opening...');
      setIsOpen(true);
    }
  }, [state]);

  useEffect(() => {
    if (!isOpen && state) {
      // use timeout to allow the dialog to close before resetting the state
      const timer = setTimeout(() => {
        console.log('SaveSnapshotModal: Resetting state');
        setState(null);
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [isOpen, state, setState]);

  const handleSave = async (andStop: boolean) => {
    if (!state?.job || isSaving) return;

    console.log(`HandleSave: andStop=${andStop}, jobID=${state.job.id}`);
    setIsSaving(true);
    setIsOpen(false);
    try {
      if (andStop) {
        // Save and pause: set both flags atomically without sending a kill signal.
        // Python checks should_save() before should_stop() at the end of each step,
        // so it will write the .safetensors file before stopping cleanly.
        await saveAndPauseJob(state.job.id);
        console.log('Save-and-pause requested successfully');
      } else {
        // Save only: just set the save flag, training continues.
        await saveJob(state.job.id);
        console.log('Save requested successfully');
      }
      if (state.onRefresh) {
        state.onRefresh();
      }
    } catch (e) {
      console.error('Error in handleSave:', e);
      alert('Failed to process request. Check console for details.');
    } finally {
      setIsSaving(false);
    }
  };

  const onCancel = () => {
    console.log('SaveSnapshotModal: Cancelled');
    setIsOpen(false);
  };

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
                <div
                  className={`mx-auto flex size-12 shrink-0 items-center justify-center rounded-full bg-blue-500 sm:mx-0 sm:size-10`}
                >
                  <Save aria-hidden="true" className={`size-6 text-blue-950`} />
                </div>
                <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left flex-1">
                  <DialogTitle as="h3" className={`text-base font-semibold text-blue-500`}>
                    Save Snapshot for "{state?.job?.name}"
                  </DialogTitle>
                  <div className="mt-2">
                    <p className="text-sm text-gray-200">
                      Do you want to save a snapshot and continue training, or save and pause the job?
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div className="bg-gray-700 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6">
              <button
                type="button"
                onClick={() => handleSave(false)}
                disabled={isSaving}
                className={classNames(
                  `inline-flex w-full justify-center rounded-md bg-blue-700 px-3 py-2 text-sm font-semibold text-white shadow-xs sm:ml-3 sm:w-auto cursor-pointer`,
                  { 'opacity-50 cursor-not-allowed': isSaving },
                )}
              >
                {isSaving ? 'Processing...' : 'Save and Continue'}
              </button>
              <button
                type="button"
                onClick={() => handleSave(true)}
                disabled={isSaving}
                className={classNames(
                  `inline-flex w-full justify-center rounded-md bg-blue-600 px-3 py-2 text-sm font-semibold text-white shadow-xs sm:ml-3 sm:w-auto cursor-pointer`,
                  { 'opacity-50 cursor-not-allowed': isSaving },
                )}
              >
                {isSaving ? 'Processing...' : 'Save and Pause'}
              </button>
              <button
                type="button"
                data-autofocus
                onClick={onCancel}
                disabled={isSaving}
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
