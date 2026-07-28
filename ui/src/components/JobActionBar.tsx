import Link from 'next/link';
import { Eye, Trash2, Pen, Play, Pause, Cog, X, Copy, Save, OctagonX, Camera, ArrowLeft, History, Image } from 'lucide-react';
import { Button } from '@headlessui/react';
import { openConfirm } from '@/components/ConfirmModal';
import { openSaveSnapshotModal } from '@/components/SaveSnapshotModal';
import { openResumeFromCheckpointModal } from '@/components/ResumeFromCheckpointModal';
import { openStopJobModal } from '@/components/StopJobModal';
import { Job } from '@prisma/client';
import {
  startJob,
  stopJob,
  deleteJob,
  getAvaliableJobActions,
  markJobAsStopped,
  saveJob,
  sampleJob,
  stopSampleJob,
  sampleJobNow,
} from '@/utils/jobs';
import { startQueue } from '@/utils/queue';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import { openCaptionDatasetModal } from '@/components/CaptionDatasetModal';
import { useState } from 'react';

interface JobActionBarProps {
  job: Job;
  onRefresh?: () => void;
  afterDelete?: () => void;
  hideView?: boolean;
  hideSample?: boolean;
  className?: string;
  autoStartQueue?: boolean;
  isAnyJobRunning?: boolean;
  hasSamples?: boolean;
  // Where the gear menu opens; defaults to below. Use 'top end' when the bar is
  // pinned to the bottom of the screen so the menu opens upward into view.
  menuAnchor?: 'bottom' | 'top end';
}

export default function JobActionBar({
  job,
  onRefresh,
  afterDelete,
  className,
  hideView,
  hideSample = false,
  autoStartQueue = false,
  isAnyJobRunning = false,
  hasSamples = false,
  menuAnchor = 'bottom',
}: JobActionBarProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const { canStart, canStop, canDelete, canEdit, canEditSample, canRemoveFromQueue, canSave, canSample, isBusy } = getAvaliableJobActions(
    job,
    isAnyJobRunning,
    hasSamples,
  );

  if (!afterDelete) afterDelete = onRefresh;

  const handleAction = async (action: () => Promise<void>) => {
    if (isProcessing) return;
    setIsProcessing(true);
    try {
      await action();
    } catch (e) {
      console.error('Error performing job action:', e);
    } finally {
      setIsProcessing(false);
      if (onRefresh) onRefresh();
    }
  };

  const disabled = isProcessing || isBusy;
  const iconSizeClass = 'w-5 h-5 sm:w-6 sm:h-6';

  return (
    <div className={`flex items-center flex-shrink-0 ${className ?? ''}`}>
      {canStart && (
        <Button
          onClick={() => handleAction(async () => {
            await startJob(job.id);
            if (autoStartQueue) {
              await startQueue(job.gpu_ids);
            }
          })}
          disabled={disabled}
          className={`ml-1 sm:ml-2 opacity-100 disabled:opacity-30 disabled:cursor-not-allowed`}
          title="Start Job"
        >
          <Play className={iconSizeClass} />
        </Button>
      )}
      {canRemoveFromQueue && (
        <Button
          onClick={() => handleAction(() => markJobAsStopped(job.id))}
          disabled={disabled}
          className={`ml-1 sm:ml-2 opacity-100 disabled:opacity-30 disabled:cursor-not-allowed`}
          title="Remove from Queue"
        >
          <X className={iconSizeClass} />
        </Button>
      )}
      {canSave && (
        <Button
          onClick={() => {
            if (disabled) return;
            openSaveSnapshotModal({
              job,
              onRefresh,
            });
          }}
          disabled={disabled}
          className={`ml-1 sm:ml-2 opacity-100 disabled:opacity-30 disabled:cursor-not-allowed`}
          title="Save Snapshot"
        >
          <Save className={iconSizeClass} />
        </Button>
      )}
      {canSample && !hideSample && (
        <Button
          onClick={() => handleAction(() => sampleJob(job.id))}
          disabled={disabled}
          className={`ml-1 sm:ml-2 opacity-100 disabled:opacity-30 disabled:cursor-not-allowed`}
          title="Generate Samples Now"
        >
          <Camera className={iconSizeClass} />
        </Button>
      )}
      {job.sample && job.status === 'running' && (
        <Button
          onClick={() => handleAction(() => stopSampleJob(job.id))}
          disabled={isProcessing}
          className={`ml-1 sm:ml-2 opacity-100 disabled:opacity-30 disabled:cursor-not-allowed text-yellow-400`}
          title="Stop sampling and return to training"
        >
          <ArrowLeft className={iconSizeClass} />
        </Button>
      )}
      {canStop && (
        <Button
          onClick={() => {
            if (!canStop) return;
            openStopJobModal({ job, onRefresh: onRefresh });
          }}
          disabled={isProcessing}
          className={`ml-1 sm:ml-2 opacity-100 disabled:opacity-30 disabled:cursor-not-allowed`}
          title="Stop Job"
        >
          <Pause className={iconSizeClass} />
        </Button>
      )}
      {!hideView && (
        <Link href={`/jobs/${job.id}`} className="ml-1 sm:ml-2 text-gray-200 hover:text-gray-100 inline-block" title="View Job Details">
          <Eye className={iconSizeClass} />
        </Link>
      )}
      {job.job_type === 'caption' && canEdit && (
        <div
          className="ml-1 sm:ml-2 hover:text-gray-100 inline-block cursor-pointer"
          onClick={() =>
            openCaptionDatasetModal(
              job.job_ref || '',
              () => {
                if (onRefresh) onRefresh();
              },
              { jobId: job.id },
            )
          }
        >
          <Pen className={iconSizeClass} />
        </div>
      )}
      {job.job_type === 'train' && canEdit && (
        <Link href={`/jobs/new?id=${job.id}`} className="ml-1 sm:ml-2 hover:text-gray-100 inline-block" title="Edit Job Config">
          <Pen className={iconSizeClass} />
        </Link>
      )}
      {job.job_type === 'train' && canEditSample && !canEdit && (
        <Link href={`/jobs/new?id=${job.id}&sampleOnly=true`} className="ml-1 sm:ml-2 hover:text-gray-100 inline-block" title="Edit Sample Prompts">
          <Pen className={iconSizeClass} />
        </Link>
      )}
      <Button
        onClick={() => {
          let message = `Are you sure you want to delete the job "${job.name}"? This will also permanently remove it from your disk.`;
          if (job.status === 'running') {
            message += ' WARNING: The job is currently running. You should stop it first if you can.';
          }
          openConfirm({
            title: 'Delete Job',
            message: message,
            type: 'warning',
            confirmText: 'Delete',
            onConfirm: async () => {
              await handleAction(async () => {
                if (job.status === 'running') {
                  await stopJob(job.id);
                }
                await deleteJob(job.id);
                if (afterDelete) afterDelete();
              });
            },
          });
        }}
        disabled={disabled}
        className={`ml-1 sm:ml-2 opacity-100 disabled:opacity-30 disabled:cursor-not-allowed`}
        title="Delete Job"
      >
        <Trash2 className={iconSizeClass} />
      </Button>
      <div className="border-r border-1 border-gray-700 ml-1 sm:ml-2 inline"></div>
      <Menu>
        <MenuButton className={'ml-1 sm:ml-2'} title="More Actions">
          <Cog className={iconSizeClass} />
        </MenuButton>
        <MenuItems
          anchor={{ to: menuAnchor, gap: 16 }}
          className="bg-gray-900 border border-gray-700 rounded shadow-lg w-60 px-2 py-2 z-50"
        >
          {job.job_type === 'train' && (
            <MenuItem>
              <Link
                href={`/jobs/new?cloneId=${job.id}`}
                className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded flex items-center gap-2"
              >
                <Copy className="w-4 h-4" />
                Clone Job
              </Link>
            </MenuItem>
          )}
          {job.job_type === 'train' && canStop && (
            <MenuItem>
              <div
                className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded flex items-center gap-2"
                onClick={async () => {
                  await saveJob(job.id);
                  if (onRefresh) onRefresh();
                }}
              >
                <Save className="w-4 h-4" />
                Save Next Step
              </div>
            </MenuItem>
          )}
          {job.job_type === 'train' && canStop && (
            <MenuItem>
              <div
                className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded flex items-center gap-2"
                onClick={async () => {
                  await sampleJobNow(job.id);
                  if (onRefresh) onRefresh();
                }}
              >
                <Image className="w-4 h-4" />
                Sample Next Step
              </div>
            </MenuItem>
          )}
          {job.job_type === 'train' && (canStart || job.status === 'queued') && (
            <MenuItem>
              <div
                className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded flex items-center gap-2"
                onClick={() => openResumeFromCheckpointModal({ job, onRefresh })}
              >
                <History className="w-4 h-4" />
                Resume From Checkpoint…
              </div>
            </MenuItem>
          )}
          <MenuItem>
            <div
              className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded flex items-center gap-2"
              onClick={() => {
                let message = `Are you sure you want to mark this job as stopped? This will set the job status to 'stopped' if the status is hung. Only do this if you are 100% sure the job is stopped. This will NOT stop the job.`;
                openConfirm({
                  title: 'Mark Job as Stopped',
                  message: message,
                  type: 'warning',
                  confirmText: 'Mark as Stopped',
                  onConfirm: async () => {
                    await handleAction(() => markJobAsStopped(job.id));
                  },
                });
              }}
            >
              <OctagonX className="w-4 h-4" />
              Mark as Stopped
            </div>
          </MenuItem>
        </MenuItems>
      </Menu>
    </div>
  );
}
