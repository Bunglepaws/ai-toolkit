import { useMemo, useState } from 'react';
import useJobsList from '@/hooks/useJobsList';
import { JobConfig } from '@/types';
import Link from 'next/link';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { GpuInfo } from '@/types';
import JobActionBar from './JobActionBar';
import { Job, Queue } from '@prisma/client';
import useQueueList from '@/hooks/useQueueList';
import classNames from 'classnames';
import { startQueue, stopQueue } from '@/utils/queue';
import { CgSpinner } from 'react-icons/cg';
import useGPUInfo from '@/hooks/useGPUInfo';
import { ChevronUp, ChevronDown, ChevronsUp, GripVertical, Trash2 } from 'lucide-react';
import { openConfirm } from '@/components/ConfirmModal';
import { deleteJob, getTotalSteps, reorderJob, reorderJobToIndex, stopJob } from '@/utils/jobs';
import JobAlertsPanel, { JobAlert } from '@/components/JobAlertsPanel';

interface JobsTableProps {
  autoStartQueue?: boolean;
  onlyActive?: boolean;
  filter?: string;
  job_type?: string | null;
}

export default function JobsTable({ onlyActive = false, filter = '', job_type = null }: JobsTableProps) {
  const { jobs, setJobs, status, refreshJobs } = useJobsList({ onlyActive, reloadInterval: 5000, job_type });
  const { queues, status: queueStatus, refreshQueues } = useQueueList();
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();

  const isAnyJobRunning = jobs.some(j => j.status === 'running');

  const refresh = () => {
    refreshJobs();
    refreshQueues();
  };

  const [draggedJobId, setDraggedJobId] = useState<string | null>(null);
  const [dragOverJobId, setDragOverJobId] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [deleteProgress, setDeleteProgress] = useState<{ done: number; total: number } | null>(null);
  const [openAlertsPanelId, setOpenAlertsPanelId] = useState<string | null>(null);

  const isDeleting = deleteProgress !== null;
  const allSelected = jobs.length > 0 && jobs.every(job => selectedIds.has(job.id));

  const toggleRow = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleAll = () => {
    setSelectedIds(allSelected ? new Set() : new Set(jobs.map(job => job.id)));
  };

  const onMassDelete = () => {
    const jobsToDelete = jobs.filter(job => selectedIds.has(job.id));
    if (jobsToDelete.length === 0) return;
    const runningCount = jobsToDelete.filter(job => job.status === 'running').length;
    let message = `Are you sure you want to delete ${jobsToDelete.length} job${
      jobsToDelete.length === 1 ? '' : 's'
    }? This will also permanently remove them from your disk.`;
    if (runningCount > 0) {
      message += ` WARNING: ${runningCount} of them ${
        runningCount === 1 ? 'is' : 'are'
      } currently running and will be stopped first.`;
    }
    openConfirm({
      title: 'Delete Jobs',
      message: message,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: async () => {
        setDeleteProgress({ done: 0, total: jobsToDelete.length });
        for (let i = 0; i < jobsToDelete.length; i++) {
          const job = jobsToDelete[i];
          try {
            if (job.status === 'running') {
              try { await stopJob(job.id); } catch (e) { console.error('Error stopping job before deleting:', e); }
            }
            await deleteJob(job.id);
            setSelectedIds(prev => { const next = new Set(prev); next.delete(job.id); return next; });
          } catch (e) {
            console.error('Error deleting job:', job.name, e);
          }
          setDeleteProgress({ done: i + 1, total: jobsToDelete.length });
          refreshJobs();
        }
        setDeleteProgress(null);
        refresh();
      },
    });
  };

  const handleDragStart = (e: React.DragEvent, jobId: string) => {
    e.dataTransfer.effectAllowed = 'move';
    setDraggedJobId(jobId);
  };

  const handleDragOver = (e: React.DragEvent, jobId: string) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    if (dragOverJobId !== jobId) setDragOverJobId(jobId);
  };

  const handleDrop = async (e: React.DragEvent, targetJobId: string, queuedJobs: Job[]) => {
    e.preventDefault();
    if (!draggedJobId || draggedJobId === targetJobId) {
      setDraggedJobId(null);
      setDragOverJobId(null);
      return;
    }
    const targetIndex = queuedJobs.findIndex(j => j.id === targetJobId);
    if (targetIndex === -1) return;
    try {
      await reorderJobToIndex(draggedJobId, targetIndex);
      refresh();
    } catch (err) {
      console.error('Failed to reorder job:', err);
    } finally {
      setDraggedJobId(null);
      setDragOverJobId(null);
    }
  };

  const handleDragEnd = () => {
    setDraggedJobId(null);
    setDragOverJobId(null);
  };

  const filteredJobs = useMemo(() => {
    if (!filter) return jobs;

    const escapeRegExp = (string: string) => string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

    const matchesTerm = (job: Job, term: string) => {
      term = term.trim();
      if (!term) return true;
      let modelName = '';
      try {
        const jobConfig: JobConfig = JSON.parse(job.job_config);
        modelName = jobConfig?.config?.process?.[0]?.model?.name_or_path || '';
      } catch { /* malformed config */ }
      const jobRef = job.job_ref || '';
      const searchableText = `${job.name} ${modelName} ${jobRef}`.toLowerCase();
      if (term.startsWith('"') && term.endsWith('"')) {
        const exactTerm = term.slice(1, -1);
        if (!exactTerm) return true;
        return new RegExp(`(^|[^a-zA-Z0-9_])${escapeRegExp(exactTerm)}([^a-zA-Z0-9_]|$)`, 'i').test(searchableText);
      }
      return searchableText.includes(term.toLowerCase());
    };

    const splitByOperator = (input: string, operator: 'and' | 'or') => {
      const regex = new RegExp(`\\s+${operator}\\s+`, 'gi');
      const parts: string[] = [];
      let lastIndex = 0;
      let match;
      while ((match = regex.exec(input)) !== null) {
        const part = input.slice(lastIndex, match.index).trim();
        if ((part.match(/"/g) || []).length % 2 === 0) { parts.push(part); lastIndex = regex.lastIndex; }
      }
      parts.push(input.slice(lastIndex).trim());
      return parts.filter(p => p !== '');
    };

    const orParts = splitByOperator(filter, 'or');
    if (orParts.length > 1) {
      return jobs.filter(job => orParts.some(part => {
        const andParts = splitByOperator(part, 'and');
        return andParts.length > 1 ? andParts.every(sub => matchesTerm(job, sub)) : matchesTerm(job, part);
      }));
    }
    const andParts = splitByOperator(filter, 'and');
    if (andParts.length > 1) return jobs.filter(job => andParts.every(part => matchesTerm(job, part)));
    return jobs.filter(job => matchesTerm(job, filter));
  }, [jobs, filter]);

  const handleReorder = async (jobID: string, direction: 'up' | 'down') => {
    setJobs(prev => {
      const job = prev.find(j => j.id === jobID);
      if (!job) return prev;
      const queueJobs = prev.filter(j => j.status === 'queued' && j.gpu_ids === job.gpu_ids)
        .sort((a, b) => (a.queue_position ?? 0) - (b.queue_position ?? 0));
      const idx = queueJobs.findIndex(j => j.id === jobID);
      const swapIdx = direction === 'up' ? idx - 1 : idx + 1;
      if (swapIdx < 0 || swapIdx >= queueJobs.length) return prev;
      const neighbour = queueJobs[swapIdx];
      return prev.map(j => {
        if (j.id === jobID) return { ...j, queue_position: neighbour.queue_position };
        if (j.id === neighbour.id) return { ...j, queue_position: job.queue_position };
        return j;
      });
    });
    try { await reorderJob(jobID, direction); } catch (e) { console.error('Failed to reorder job:', e); }
    refresh();
  };

  const handleMoveToTop = async (jobID: string) => {
    setJobs(prev => {
      const job = prev.find(j => j.id === jobID);
      if (!job) return prev;
      const queueJobs = prev.filter(j => j.status === 'queued' && j.gpu_ids === job.gpu_ids)
        .sort((a, b) => (a.queue_position ?? 0) - (b.queue_position ?? 0));
      if (queueJobs[0]?.id === jobID) return prev;
      const reordered = [job, ...queueJobs.filter(j => j.id !== jobID)];
      const basePos = Math.min(...queueJobs.map(j => j.queue_position ?? 0));
      const updated = new Map(reordered.map((j, i) => [j.id, { ...j, queue_position: basePos + i }]));
      return prev.map(j => updated.get(j.id) ?? j);
    });
    try { await reorderJobToIndex(jobID, 0); } catch (e) { console.error('Failed to move job to top:', e); }
    refresh();
  };

  const columns: TableColumn[] = [
    {
      title: (
        <input
          type="checkbox"
          checked={allSelected}
          onChange={toggleAll}
          disabled={isDeleting}
          className="cursor-pointer accent-blue-500"
        />
      ),
      key: 'select',
      className: 'w-8',
      render: row => (
        <input
          type="checkbox"
          checked={selectedIds.has(row.id)}
          onChange={() => toggleRow(row.id)}
          disabled={isDeleting}
          className="cursor-pointer accent-blue-500"
        />
      ),
    },
    {
      title: 'Name',
      key: 'name',
      render: row => {
        let title: React.ReactNode = row.name;
        let href = `/jobs/${row.id}`;
        // if (row.job_type === 'train') title = `Train: ${title}`;
        if (row.job_type === 'caption') {
          let paths: string[] = [];
          try {
            const jobConfig: JobConfig = JSON.parse(row.job_config);
            const pathToCaption = (jobConfig as any)?.config?.process?.[0]?.caption?.path_to_caption;
            if (Array.isArray(pathToCaption)) {
              paths = pathToCaption;
            } else if (typeof pathToCaption === 'string') {
              paths = pathToCaption.split('|');
            }
          } catch { /* malformed config */ }
          paths = paths.map(p => p.trim()).filter(Boolean);
          if (paths.length === 0) paths = [row.job_ref || ''];
          const names = paths.map(p => {
            const splits = p.split(/[/\\]/);
            return splits[splits.length - 1];
          });
          href = `/datasets/${names[0]}`;
          title = (
            <>
              <small className="opacity-50">CAPTION: </small> {names.join(', ')}
            </>
          );
        }
        return (
          <div className="flex items-center">
            {row.status === 'queued' && (
              <>
                <div className="mr-1 text-gray-600 cursor-grab" title="Drag to reorder">
                  <GripVertical size={16} />
                </div>
                <div className="flex flex-col mr-3 text-gray-500">
                  <button
                    onClick={() => handleMoveToTop(row.id)}
                    className="hover:text-white transition-colors"
                    title="Move to Top"
                  >
                    <ChevronsUp size={16} />
                  </button>
                  <button
                    onClick={() => handleReorder(row.id, 'up')}
                    className="hover:text-white transition-colors"
                    title="Move Up"
                  >
                    <ChevronUp size={16} />
                  </button>
                  <button
                    onClick={() => handleReorder(row.id, 'down')}
                    className="hover:text-white transition-colors"
                    title="Move Down"
                  >
                    <ChevronDown size={16} />
                  </button>
                </div>
              </>
            )}
            <Link href={href} className="font-medium whitespace-nowrap">
              {['running', 'stopping'].includes(row.status) ? (
                <CgSpinner className="inline animate-spin mr-2 text-blue-400" />
              ) : null}
              {title}
            </Link>
          </div>
        );
      },
    },
    {
      title: 'Steps',
      key: 'steps',
      render: row => {
        const totalSteps = getTotalSteps(row);
        if (!totalSteps) {
          return <></>;
        }

        return (
          <div>
            <div className="text-xs text-gray-400">
              {row.step} / {totalSteps}
            </div>
            <div className="bg-gray-700 rounded-full h-1.5">
              <div
                className="bg-blue-500 h-1.5 rounded-full"
                style={{ width: `${(row.step / totalSteps) * 100}%` }}
              ></div>
            </div>
          </div>
        );
      },
    },
    {
      title: 'GPU',
      key: 'gpu_ids',
    },
    {
      title: 'Status',
      key: 'status',
      render: row => {
        let statusClass = 'text-gray-400';
        if (row.status === 'completed') statusClass = 'text-green-400';
        if (row.status === 'failed') statusClass = 'text-red-400';
        if (row.status === 'running') statusClass = 'text-blue-400';

        let alerts: JobAlert[] = [];
        try {
          alerts = JSON.parse((row as any).alerts || '[]');
        } catch { /* ignore */ }

        return (
          <div className="flex items-center gap-2 flex-wrap">
            <span className={statusClass}>{row.status}</span>
            {alerts.length > 0 && (
              <button
                onClick={e => {
                  e.stopPropagation();
                  setOpenAlertsPanelId(prev => prev === row.id ? null : row.id);
                }}
                className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-semibold bg-amber-900/40 text-amber-400 border border-amber-500/40 hover:bg-amber-900/60 transition-colors"
                title="View training alerts"
              >
                ⚠ {alerts.length}
              </button>
            )}
          </div>
        );
      },
    },
    {
      title: 'Info',
      key: 'info',
      className: 'truncate max-w-xs',
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'text-right',
      render: row => {
        return (
          <JobActionBar
            job={row}
            onRefresh={refreshJobs}
            autoStartQueue={false}
            isAnyJobRunning={isAnyJobRunning}
          />
        );
      },
    },
  ];

  const jobsDict = useMemo(() => {
    if (!isGPUInfoLoaded) return {};
    if (filteredJobs.length === 0) return {};
    let jd: { [key: string]: { name: string; jobs: Job[] } } = {};
    gpuList.forEach(gpu => {
      jd[`${gpu.index}`] = { name: `${gpu.name}`, jobs: [] };
    });
    jd['Idle'] = { name: 'Idle', jobs: [] };
    filteredJobs.forEach(job => {
      const gpu = gpuList.find(gpu => job.gpu_ids?.split(',').includes(gpu.index.toString())) as GpuInfo;
      const key = `${gpu?.index || '0'}`;
      if (['queued', 'running', 'stopping'].includes(job.status) && key in jd) {
        jd[key].jobs.push(job);
      } else {
        jd['Idle'].jobs.push(job);
      }
    });
    // sort the queued/running jobs by queue position
    Object.keys(jd).forEach(key => {
      if (key === 'Idle') {
        jd[key].jobs.sort((a, b) => {
          // sort by updated_at, newest first
          return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
        });
      } else {
        jd[key].jobs.sort((a, b) => {
          const aIsActive = ['running', 'stopping'].includes(a.status);
          const bIsActive = ['running', 'stopping'].includes(b.status);
          if (aIsActive && !bIsActive) return -1;
          if (!aIsActive && bIsActive) return 1;
          if (a.queue_position === null) return 1;
          if (b.queue_position === null) return -1;
          return a.queue_position - b.queue_position;
        });
      }
    });
    return jd;
  }, [filteredJobs, queues, isGPUInfoLoaded]);

  let isLoading = status === 'loading' || queueStatus === 'loading' || !isGPUInfoLoaded;

  // if job dict is populated, we are always loaded
  if (Object.keys(jobsDict).length > 0) isLoading = false;

  return (
    <div>
      {(selectedIds.size > 0 || isDeleting) && (
        <div className="fixed top-16 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 px-4 py-2 bg-gray-800 rounded-lg border border-gray-700 shadow-lg">
          {isDeleting ? (
            <>
              <CgSpinner className="inline animate-spin text-red-400" />
              <span className="text-sm text-gray-300">
                Deleting {deleteProgress.done} / {deleteProgress.total}...
              </span>
            </>
          ) : (
            <>
              <span className="text-sm text-gray-300 flex-1">
                {selectedIds.size} job{selectedIds.size === 1 ? '' : 's'} selected
              </span>
              <button
                onClick={() => setSelectedIds(new Set())}
                className="text-xs text-gray-300 bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded"
              >
                Clear
              </button>
              <button
                onClick={onMassDelete}
                className="text-xs text-white bg-red-600 hover:bg-red-700 px-2 py-1 rounded flex items-center gap-1"
              >
                <Trash2 className="w-3.5 h-3.5" />
                Delete Selected
              </button>
            </>
          )}
        </div>
      )}
      {Object.keys(jobsDict)
        .sort()
        .filter(key => key !== 'Idle')
        .map(gpuKey => {
          const queue = queues.find(q => `${q.gpu_ids}` === gpuKey) as Queue;
          return (
            <div key={gpuKey} className="mb-6">
              <div
                className={classNames(
                  'text-md flex flex-wrap gap-y-1 px-2 sm:px-4 py-1 rounded-t-lg',
                  { 'bg-green-600 dark:bg-green-900': queue?.is_running },
                  { 'bg-red-600 dark:bg-red-900': !queue?.is_running },
                )}
              >
                <div className="flex items-center space-x-2 flex-1 min-w-0 py-2">
                  <h2 className="font-semibold text-white truncate">{jobsDict[gpuKey].name}</h2>
                  <span className="px-2 py-0.5 bg-gray-700 rounded-full text-xs text-gray-300 flex-shrink-0">
                    # {queue?.gpu_ids}
                  </span>
                </div>
                <div className="text-sm text-gray-300 italic flex items-center flex-shrink-0">
                  {queue?.is_running ? (
                    <>
                      <span className="text-green-100 dark:text-green-400 mr-2">Queue Running</span>
                      <button
                        onClick={async () => {
                          await stopQueue(queue.gpu_ids as string);
                          refresh();
                        }}
                        className="ml-2 sm:ml-4 text-xs text-white bg-red-600 hover:bg-red-700 px-2 py-1 rounded"
                      >
                        STOP
                      </button>
                    </>
                  ) : (
                    <>
                      <span className="text-red-100 dark:text-red-400 mr-2">Queue Stopped</span>
                      <button
                        onClick={async () => {
                          await startQueue(gpuKey);
                          refresh();
                        }}
                        className="ml-2 sm:ml-4 text-xs text-white bg-green-600 hover:bg-green-700 px-2 py-1 rounded"
                      >
                        START
                      </button>
                    </>
                  )}
                </div>
              </div>
              <UniversalTable
                columns={columns}
                rows={jobsDict[gpuKey].jobs}
                isLoading={isLoading}
                onRefresh={refresh}
                theadClassName={
                  queue?.is_running
                    ? 'bg-green-700 dark:bg-green-950 text-white dark:text-gray-400'
                    : 'bg-red-700 dark:bg-red-950 text-white dark:text-gray-400'
                }
                rowProps={(row) => {
                  if (row.status !== 'queued') return {};
                  const queuedJobs = jobsDict[gpuKey].jobs.filter((j: Job) => j.status === 'queued');
                  const isDragging = row.id === draggedJobId;
                  const isDragOver = row.id === dragOverJobId && row.id !== draggedJobId;
                  return {
                    draggable: true,
                    onDragStart: (e: React.DragEvent<HTMLTableRowElement>) => handleDragStart(e, row.id),
                    onDragOver: (e: React.DragEvent<HTMLTableRowElement>) => handleDragOver(e, row.id),
                    onDrop: (e: React.DragEvent<HTMLTableRowElement>) => handleDrop(e, row.id, queuedJobs),
                    onDragEnd: handleDragEnd,
                    className: classNames(
                      isDragging && 'opacity-40',
                      isDragOver && 'border-t-2 border-blue-400',
                    ),
                  };
                }}
                afterRow={(row) => {
                  if (openAlertsPanelId !== row.id) return null;
                  let alerts: JobAlert[] = [];
                  try { alerts = JSON.parse((row as any).alerts || '[]'); } catch { /* ignore */ }
                  return (
                    <JobAlertsPanel
                      jobID={row.id}
                      alerts={alerts}
                      onClose={() => setOpenAlertsPanelId(null)}
                      onCleared={() => { setOpenAlertsPanelId(null); refresh(); }}
                    />
                  );
                }}
              />
            </div>
          );
        })}
      {!onlyActive && Object.keys(jobsDict).includes('Idle') && (
        <div className="mb-6 opacity-50">
          <div className="text-md flex px-4 py-1 rounded-t-lg bg-slate-600">
            <div className="flex items-center space-x-2 flex-1 py-2">
              <h2 className="font-semibold text-gray-100">Idle</h2>
            </div>
          </div>
          <UniversalTable
            columns={columns}
            rows={jobsDict['Idle'].jobs}
            isLoading={isLoading}
            onRefresh={refresh}
            afterRow={(row) => {
              if (openAlertsPanelId !== row.id) return null;
              let alerts: JobAlert[] = [];
              try { alerts = JSON.parse((row as any).alerts || '[]'); } catch { /* ignore */ }
              return (
                <JobAlertsPanel
                  jobID={row.id}
                  alerts={alerts}
                  onClose={() => setOpenAlertsPanelId(null)}
                  onCleared={() => { setOpenAlertsPanelId(null); refresh(); }}
                />
              );
            }}
          />
        </div>
      )}
    </div>
  );
}
