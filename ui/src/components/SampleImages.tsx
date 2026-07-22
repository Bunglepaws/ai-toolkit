import { useMemo, useState, useRef, useCallback } from 'react';
import { Virtuoso, VirtuosoHandle } from 'react-virtuoso';
import useSampleImages from '@/hooks/useSampleImages';
import SampleImageCard from './SampleImageCard';
import { Job } from '@prisma/client';
import { JobConfig } from '@/types';
import { LuImageOff, LuLoader, LuBan, LuArrowLeft } from 'react-icons/lu';
import { Camera } from 'lucide-react';
import { Button } from '@headlessui/react';
import { FaDownload } from 'react-icons/fa';
import { apiClient } from '@/utils/api';
import classNames from 'classnames';
import { FaCaretDown, FaCaretUp } from 'react-icons/fa';
import SampleImageViewer from './SampleImageViewer';
import { getAvaliableJobActions, getTotalSteps, sampleJob, stopSampleJob } from '@/utils/jobs';

interface SampleImagesMenuProps {
  job: Job;
  onRefresh?: () => void;
  hasSamples?: boolean;
  isAnyJobRunning?: boolean;
}

export const SampleImagesMenu = ({ job, onRefresh, hasSamples, isAnyJobRunning }: SampleImagesMenuProps) => {
  const [isZipping, setIsZipping] = useState(false);
  const { canSample, isActivelySampling } = getAvaliableJobActions(job, isAnyJobRunning, hasSamples);
  const totalSteps = getTotalSteps(job);

  const downloadZip = async () => {
    if (isZipping) return;
    setIsZipping(true);

    try {
      const res = await apiClient.post('/api/zip', {
        zipTarget: 'samples',
        jobName: job?.name,
      });

      const zipPath = res.data.zipPath; // e.g. /mnt/Train2/out/ui/.../samples.zip
      if (!zipPath) throw new Error('No zipPath in response');

      const downloadPath = `/api/files/${encodeURIComponent(zipPath)}`;
      const a = document.createElement('a');
      a.href = downloadPath;
      // optional: suggest filename (browser may ignore if server sets Content-Disposition)
      a.download = 'samples.zip';
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      console.error('Error downloading zip:', err);
    } finally {
      setIsZipping(false);
    }
  };
  return (
    <div className="flex items-center">
      {totalSteps > 0 && (
        <span className="hidden sm:inline text-xs text-gray-400 dark:text-gray-500 mr-2 sm:mr-3 whitespace-nowrap">
          Step {job.step} of {totalSteps}
        </span>
      )}
      {canSample && !isActivelySampling && (
        <Button
          onClick={async () => {
            if (!canSample) return;
            await sampleJob(job.id);
            if (onRefresh) onRefresh();
          }}
          className={classNames(`px-2 sm:px-4 py-1 h-8 hover:bg-gray-200 dark:hover:bg-gray-700 mr-1 sm:mr-2 flex items-center`)}
        >
          <Camera className="inline-block sm:mr-2 w-4 h-4" />
          <span className="hidden sm:inline">Generate Samples Now</span>
        </Button>
      )}
      {isActivelySampling && (
        <Button
          onClick={async () => {
            await stopSampleJob(job.id);
            if (onRefresh) onRefresh();
          }}
          className={classNames(`px-2 sm:px-4 py-1 h-8 hover:bg-gray-200 dark:hover:bg-gray-700 mr-1 sm:mr-2 flex items-center text-yellow-500 dark:text-yellow-400`)}
          title="Cancels current sample generations for this snapshot and resumes training"
        >
          <LuArrowLeft className="inline-block sm:mr-2" />
          <span className="hidden sm:inline">Return to Training</span>
        </Button>
      )}
      <Button
        onClick={downloadZip}
        className={classNames(
          `flex-1 sm:flex-initial justify-center px-2 sm:px-4 py-1 h-8 hover:bg-gray-200 dark:hover:bg-gray-700 flex items-center`,
          { 'opacity-50 cursor-not-allowed': isZipping },
        )}
      >
        {isZipping ? (
          <LuLoader className="animate-spin inline-block sm:mr-2" />
        ) : (
          <FaDownload className="inline-block sm:mr-2" />
        )}
        <span className="hidden sm:inline">{isZipping ? 'Preparing' : 'Download'}</span>
      </Button>
    </div>
  );
};

interface SampleImagesProps {
  job: Job;
}

export default function SampleImages({ job }: SampleImagesProps) {
  const { sampleImages, status, refreshSampleImages } = useSampleImages(job.id, 5000);
  const [selectedSamplePath, setSelectedSamplePath] = useState<string | null>(null);
  const [scrollParent, setScrollParent] = useState<HTMLDivElement | null>(null);
  const scrollParentCallback = useCallback((el: HTMLDivElement | null) => setScrollParent(el), []);
  const virtuosoRef = useRef<VirtuosoHandle>(null);
  const configNumSamples = useMemo(() => {
    if (job?.job_config) {
      const jobConfig = JSON.parse(job.job_config) as JobConfig;
      const sampleConfig = jobConfig.config.process[0].sample;
      const numPrompts = sampleConfig.prompts ? sampleConfig.prompts.length : 0;
      const numSamples = sampleConfig.samples.length;
      return Math.max(numPrompts, numSamples, 1);
    }
    return 10;
  }, [job]);

  // Build sampleSlots: a sparse (string | null)[] with null for steps that haven't been sampled yet,
  // plus the sorted steps array so rows can show a step-count label.
  const { sampleSlots, numSamples, steps } = useMemo(() => {
    const defaultRes = { sampleSlots: sampleImages as (string | null)[], numSamples: configNumSamples, steps: [] as number[] };
    if (sampleImages.length === 0) return defaultRes;

    // 1. Parse filenames to extract timestamp, step, promptIdx.
    // Filenames have two formats:
    //   new: {timestamp}__{step}_{promptIdx}.ext  (each file gets its own timestamp)
    //   old: {step}_{promptIdx}.ext
    const parsedImages = sampleImages.map(path => {
      const filename = (path.includes('\\') ? path.split('\\') : path.split('/')).pop() || null;
      if (!filename) return { path, timestamp: '', step: -1, promptIdx: -1 };
      const basename = filename.split('.')[0];
      const parts = basename.split('_').filter(p => p !== '');
      if (parts.length >= 2) {
        const promptIdx = parseInt(parts[parts.length - 1]);
        const step = parseInt(parts[parts.length - 2]);
        const timestamp = basename.includes('__') ? parts[0] : '';
        return { path, timestamp, step, promptIdx };
      }
      return { path, timestamp: '', step: -1, promptIdx: -1 };
    });

    const validParsed = parsedImages.filter(
      img => !isNaN(img.step) && img.step !== -1 && !isNaN(img.promptIdx) && img.promptIdx !== -1,
    );
    if (validParsed.length === 0) return defaultRes;

    const maxIdxInFiles = Math.max(...validParsed.map(img => img.promptIdx));
    const eNumSamples = Math.max(configNumSamples, maxIdxInFiles + 1);

    // 2. For each step, detect multiple sampling runs by counting duplicate promptIdx values.
    //    Each file gets its own timestamp, so we can't group by timestamp. Instead we sort
    //    each (step, promptIdx) group by timestamp and assign to runs by ordinal:
    //    first occurrence → run 0, second occurrence → run 1, etc.
    const sortedSteps = Array.from(new Set(validParsed.map(img => img.step))).sort((a, b) => a - b);

    const slots: (string | null)[] = [];
    const steps: number[] = [];

    sortedSteps.forEach(step => {
      const stepImages = validParsed.filter(img => img.step === step);

      // Group by promptIdx, sort each group oldest-first by timestamp
      const byPromptIdx = new Map<number, typeof validParsed>();
      stepImages.forEach(img => {
        if (!byPromptIdx.has(img.promptIdx)) byPromptIdx.set(img.promptIdx, []);
        byPromptIdx.get(img.promptIdx)!.push(img);
      });
      byPromptIdx.forEach(imgs => imgs.sort((a, b) => a.timestamp.localeCompare(b.timestamp)));

      // Number of rows = max images at any single promptIdx (e.g. 2 samplings → 2 rows)
      const numRuns = Math.max(...Array.from(byPromptIdx.values()).map(imgs => imgs.length));

      for (let runIdx = 0; runIdx < numRuns; runIdx++) {
        steps.push(step);
        for (let pIdx = 0; pIdx < eNumSamples; pIdx++) {
          const imgs = byPromptIdx.get(pIdx);
          const found = imgs?.[runIdx];
          slots.push(found ? found.path : null);
        }
      }
    });

    return { sampleSlots: slots, numSamples: eNumSamples, steps };
  }, [sampleImages, configNumSamples]);

  // Group sampleSlots into rows of `numSamples` for the virtualized list — one row per sample iteration.
  const rows = useMemo(() => {
    const out: (string | null)[][] = [];
    for (let i = 0; i < sampleSlots.length; i += numSamples) {
      out.push(sampleSlots.slice(i, i + numSamples));
    }
    return out;
  }, [sampleSlots, numSamples]);

  const scrollToBottom = () => {
    virtuosoRef.current?.scrollToIndex({ index: 'LAST', align: 'end' });
  };

  const scrollToTop = () => {
    virtuosoRef.current?.scrollToIndex({ index: 0, align: 'start' });
  };

  const PageInfoContent = useMemo(() => {
    let icon = null;
    let text = '';
    let subtitle = '';
    let showIt = false;
    let bgColor = '';
    let textColor = '';
    let iconColor = '';

    if (sampleImages.length > 0) return null;

    if (status == 'loading') {
      icon = <LuLoader className="animate-spin w-8 h-8" />;
      text = 'Loading Samples';
      subtitle = 'Please wait while we fetch your samples...';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }
    if (status == 'error') {
      icon = <LuBan className="w-8 h-8" />;
      text = 'Error Loading Samples';
      subtitle = 'There was a problem fetching the samples.';
      showIt = true;
      bgColor = 'bg-red-50 dark:bg-red-950/20';
      textColor = 'text-red-900 dark:text-red-100';
      iconColor = 'text-red-600 dark:text-red-400';
    }
    if (status == 'success' && sampleImages.length === 0) {
      icon = <LuImageOff className="w-8 h-8" />;
      text = 'No Samples Found';
      subtitle = 'No samples have been generated yet';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }

    if (!showIt) return null;

    return (
      <div
        className={`mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed ${bgColor} ${textColor} mx-auto max-w-md text-center`}
      >
        <div className={`${iconColor} mb-4`}>{icon}</div>
        <h3 className="text-lg font-semibold mb-2">{text}</h3>
        <p className="text-sm opacity-75 leading-relaxed">{subtitle}</p>
      </div>
    );
  }, [status, sampleImages.length]);

  // Inline style instead of Tailwind grid-cols-N classes — Tailwind only ships grid-cols-1..12,
  // so class-based columns silently break for larger sample counts.
  const gridCols = Math.max(numSamples, 3);

  const sampleConfig = useMemo(() => {
    if (job?.job_config) {
      const jobConfig = JSON.parse(job.job_config) as JobConfig;
      return jobConfig.config.process[0].sample;
    }
    return null;
  }, [job]);

  return (
    <div ref={scrollParentCallback} className="absolute top-[80px] left-0 right-0 bottom-0 overflow-y-auto">
      <div className="pb-4">
        {PageInfoContent}
        {sampleImages && rows.length > 0 && scrollParent && (
          <Virtuoso
            ref={virtuosoRef}
            customScrollParent={scrollParent}
            totalCount={rows.length}
            initialTopMostItemIndex={rows.length - 1}
            followOutput="auto"
            increaseViewportBy={400}
            computeItemKey={index => rows[index]?.find(s => s !== null) ?? index}
            itemContent={index => {
              const row = rows[index];
              if (!row) return null;

              // Only pad the final row when numSamples < MIN_COLS and the row is short.
              const MIN_COLS = 3;
              const shouldPad = numSamples < MIN_COLS && row.length < MIN_COLS;
              const padsNeeded = shouldPad ? MIN_COLS - row.length : 0;

              // Step label for this row (shown on the first slot, real or placeholder)
              const rowStepLabel = steps.length > index ? steps[index] : undefined;

              return (
                // pb-1 recreates the vertical gap between rows that the original single CSS grid provided via `gap-1`.
                <div className="grid gap-1 pb-1" style={{ gridTemplateColumns: `repeat(${gridCols}, minmax(0, 1fr))` }}>
                  {row.map((sample, slotIdx) =>
                    sample ? (
                      <SampleImageCard
                        key={sample}
                        imageUrl={sample}
                        numSamples={numSamples}
                        sampleImages={sampleImages}
                        alt="Sample Image"
                        onClick={() => setSelectedSamplePath(sample)}
                        observerRoot={scrollParent}
                        stepLabel={slotIdx === 0 ? rowStepLabel : undefined}
                      />
                    ) : (
                      <div key={`empty-${index}-${slotIdx}`} className="flex flex-col">
                        <div className="relative w-full" style={{ paddingBottom: '100%' }}>
                          <div
                            className="absolute inset-0 rounded-t-lg shadow-md bg-gray-950 flex items-center justify-center border border-gray-800"
                            style={{ containerType: 'size' }}
                          >
                            <span className="text-[10px] text-gray-500 font-mono">NOT SAMPLED</span>
                            {slotIdx === 0 && rowStepLabel !== undefined && (
                              <div
                                className="absolute top-0 left-0 z-10 text-white font-bold leading-none select-none pointer-events-none"
                                style={{
                                  fontSize: '10cqmin',
                                  padding: '0.1em 0.15em',
                                  textShadow: '0 0 6px rgba(0,0,0,1), 0 1px 4px rgba(0,0,0,0.9)',
                                }}
                              >
                                {rowStepLabel}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ),
                  )}
                  {Array.from({ length: padsNeeded }).map((_, i) => (
                    <div key={`pad-${index}-${i}`} className="invisible" />
                  ))}
                </div>
              );
            }}
          />
        )}
      </div>
      <SampleImageViewer
        imgPath={selectedSamplePath}
        numSamples={numSamples}
        sampleImages={sampleSlots}
        onChange={setPath => setSelectedSamplePath(setPath)}
        sampleConfig={sampleConfig}
        refreshSampleImages={refreshSampleImages}
      />
      <div
        className="hidden md:flex fixed top-20 mt-4 right-6 w-10 h-10 rounded-full bg-gray-900 shadow-lg items-center justify-center text-white opacity-80 hover:opacity-100 cursor-pointer"
        onClick={scrollToTop}
        title="Scroll to Top"
      >
        <FaCaretUp className="text-gray-500 dark:text-gray-400" />
      </div>
      <div
        className="hidden md:flex fixed bottom-5 right-6 w-10 h-10 rounded-full bg-gray-900 shadow-lg items-center justify-center text-white opacity-80 hover:opacity-100 cursor-pointer"
        onClick={scrollToBottom}
        title="Scroll to Bottom"
      >
        <FaCaretDown className="text-gray-500 dark:text-gray-400" />
      </div>
    </div>
  );
}
