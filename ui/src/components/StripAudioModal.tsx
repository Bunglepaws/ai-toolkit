'use client';
import React, { useEffect, useRef, useState } from 'react';
import { createGlobalState } from 'react-global-hooks';
import { Modal } from './Modal';
import { TextInput, SelectInput } from '@/components/formInputs';
import { getFilename } from '@/utils/basic';
import { callScriptStream } from '@/utils/callScript';
import { SelectOption } from '@/types';

export interface StripAudioModalState {
  folderPath: string;
  files: { path: string }[];
  onClose?: () => void;
}

export const stripAudioModalState = createGlobalState<StripAudioModalState | null>(null);

export const openStripAudioModal = (
  folderPath: string,
  files: { path: string }[],
  onClose?: () => void,
) => {
  stripAudioModalState.set({ folderPath, files, onClose });
};

const loraLabel = (path: string) => getFilename(path).replace('.safetensors', '');

const defaultOutputPath = (inputPath: string) => {
  const base = inputPath.replace(/\.safetensors$/, '');
  return `${base}.no_audio.safetensors`;
};

const StripAudioModal: React.FC = () => {
  const [modalInfo, setModalInfo] = stripAudioModalState.use();
  const isOpen = modalInfo !== null;

  const [selectedPath, setSelectedPath] = useState('');
  const [outputPath, setOutputPath] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [logOutput, setLogOutput] = useState('');
  const logRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!modalInfo) {
      setSelectedPath('');
      setOutputPath('');
      setIsRunning(false);
      setIsDone(false);
      setHasError(false);
      setLogOutput('');
    }
  }, [modalInfo]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logOutput]);

  const handleSelectFile = (path: string) => {
    setSelectedPath(path);
    setOutputPath(path ? defaultOutputPath(path) : '');
  };

  const onClose = () => {
    if (isRunning) return;
    setModalInfo(null);
    modalInfo?.onClose?.();
  };

  const onSubmit = async () => {
    if (isRunning || !modalInfo || !selectedPath || !outputPath) return;
    setIsRunning(true);
    setIsDone(false);
    setHasError(false);
    setLogOutput('');

    const append = (chunk: string) => setLogOutput(prev => prev + chunk);

    try {
      const finalEvent = await callScriptStream('strip_audio_from_ltx_lora.py', {
        args: { input_path: selectedPath, output_path: outputPath },
        onStdout: append,
        onStderr: append,
      });

      const ok = finalEvent?.type === 'exit' && finalEvent.ok === true;
      if (!ok) {
        setHasError(true);
        if (finalEvent?.type === 'error' && finalEvent.message) {
          append(`\n${finalEvent.message}\n`);
        } else if (finalEvent?.type === 'exit' && finalEvent.timedOut) {
          append('\nScript timed out.\n');
        } else if (finalEvent?.type === 'exit') {
          append(`\nScript exited with code ${finalEvent.exitCode}.\n`);
        }
      }
    } catch (err: any) {
      setHasError(true);
      append(`\n${err?.message || 'Unknown error'}\n`);
    } finally {
      setIsRunning(false);
      setIsDone(true);
    }
  };

  const files = modalInfo?.files ?? [];
  const options: SelectOption[] = files.map(f => ({ value: f.path, label: loraLabel(f.path) }));

  const showLog = isRunning || isDone;

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Strip Audio from LoRA"
      size="lg"
      showCloseButton={!isRunning}
      closeOnOverlayClick={!isRunning}
    >
      {showLog ? (
        <div>
          <div className="mb-2 text-sm">
            {isRunning && <span className="text-amber-400">Stripping audio tensors… please do not close this window.</span>}
            {isDone && hasError && <span className="text-rose-400">Script failed. See log below.</span>}
            {isDone && !hasError && <span className="text-emerald-400">Done — audio tensors removed.</span>}
          </div>
          <div
            ref={logRef}
            className="font-mono text-xs whitespace-pre-wrap break-all overflow-y-auto rounded-md p-3 min-h-[400px] max-h-[60vh] bg-white text-gray-900 dark:bg-black dark:text-gray-100"
          >
            {logOutput || (isRunning ? 'Starting…\n' : '')}
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              disabled={isRunning}
              className="px-4 py-2 text-sm bg-gray-700 hover:bg-gray-600 disabled:opacity-40 disabled:cursor-not-allowed text-gray-100 rounded-md"
            >
              Close
            </button>
          </div>
        </div>
      ) : (
        <form onSubmit={e => { e.preventDefault(); onSubmit(); }}>
          <SelectInput
            label="Checkpoint to strip"
            multiple={false}
            value={selectedPath}
            onChange={handleSelectFile}
            options={options}
          />

          {selectedPath && (
            <div className="mt-4">
              <TextInput
                label="Output filename"
                value={getFilename(outputPath).replace('.safetensors', '')}
                suffix=".safetensors"
                onChange={value => {
                  const folder = outputPath.substring(0, outputPath.lastIndexOf(outputPath.includes('\\') ? '\\' : '/') + 1);
                  setOutputPath(`${folder}${value}.safetensors`);
                }}
                placeholder="Output filename"
              />
              <p className="mt-1 text-xs text-gray-500 break-all">{outputPath}</p>
            </div>
          )}

          <div className="mt-6 flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm text-gray-300 hover:text-gray-100 rounded-md"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!selectedPath || !outputPath}
              className="px-4 py-2 text-sm bg-cyan-700 hover:bg-cyan-600 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-md"
            >
              Strip Audio
            </button>
          </div>
        </form>
      )}
    </Modal>
  );
};

export default StripAudioModal;
