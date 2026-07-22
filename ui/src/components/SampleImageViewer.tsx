'use client';
import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Dialog, DialogBackdrop, DialogPanel } from '@headlessui/react';
import { SampleConfig, SampleItem } from '@/types';
import { Cog, SquareDashed } from 'lucide-react';
import classNames from 'classnames';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import { openConfirm } from './ConfirmModal';
import { apiClient } from '@/utils/api';
import { isVideo, isAudio } from '@/utils/basic';
import AudioPlayer from './AudioPlayer';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';
import BoundingBoxOverlay, { parseBoundingBoxes } from './BoundingBoxOverlay';

interface Props {
  imgPath: string | null; // current image path
  numSamples: number; // number of samples per row
  sampleImages: (string | null)[]; // all sample images (can contain null for gaps)
  sampleConfig: SampleConfig | null;
  onChange: (nextPath: string | null) => void; // parent setter
  refreshSampleImages?: () => void;
}

export default function SampleImageViewer({
  imgPath,
  numSamples,
  sampleImages,
  sampleConfig,
  onChange,
  refreshSampleImages,
}: Props) {
  const [mounted, setMounted] = useState(false);
  const [isOpen, setIsOpen] = useState(Boolean(imgPath));
  const [showingControlIdx, setShowingControlIdx] = useState<number | null>(null);
  const [promptExpanded, setPromptExpanded] = useState(false);
  const [showBoxes, setShowBoxes] = useState<boolean>(false);
  const [imgBlobUrl, setImgBlobUrl] = useState<string | null>(null);
  const [imgLoadError, setImgLoadError] = useState<string | null>(null);

  useEffect(() => setMounted(true), []);

  // open/close based on external value
  useEffect(() => {
    setIsOpen(Boolean(imgPath));
  }, [imgPath]);

  // after close, clear parent state post-transition
  useEffect(() => {
    if (!isOpen && imgPath) {
      const t = setTimeout(() => onChange(null), 300);
      return () => clearTimeout(t);
    }
  }, [isOpen, imgPath, onChange]);

  const onCancel = useCallback(() => {
    setShowingControlIdx(null);
    setIsOpen(false);
  }, []);

  const imgInfo = useMemo(() => {
    // handle windows C:\\Apps\\AI-Toolkit\\AI-Toolkit\\output\\LoRA-Name\\samples\\1763563000704__000004000_0.jpg
    const ii = { filename: '', step: 0, promptIdx: 0 };
    if (imgPath) {
      // handle windows
      let filename: string | null = null;
      if (imgPath.includes('\\')) {
        const parts = imgPath.split('\\');
        filename = parts[parts.length - 1];
      } else {
        filename = imgPath.split('/').pop() || null;
      }
      if (!filename) {
        console.error('Filename could not be determined from imgPath:', imgPath);
        return ii;
      }
      ii.filename = filename;
      const parts = filename
        .split('.')[0]
        .split('_')
        .filter(p => p !== '');
      if (parts.length >= 2) {
        ii.promptIdx = parseInt(parts[parts.length - 1]);
        ii.step = parseInt(parts[parts.length - 2]);
      } else {
        console.error('Unexpected filename format for sample image:', filename);
      }
    }
    return ii;
  }, [imgPath]);

  const setImageAtIndex = useCallback(
    (idx: number) => {
      if (idx < 0 || idx >= sampleImages.length) return;
      const nextPath = sampleImages[idx];
      if (!nextPath) return; // skip if null
      setShowingControlIdx(null);
      onChange(nextPath);
    },
    [sampleImages, onChange],
  );

  const currentIndex = useMemo(() => {
    if (!imgPath) return -1;
    return sampleImages.findIndex(img => img === imgPath);
  }, [imgPath, sampleImages]);

  const handleArrowUp = useCallback(() => {
    if (currentIndex === -1) return;
    setImageAtIndex(currentIndex - numSamples);
  }, [numSamples, currentIndex, setImageAtIndex]);

  const handleArrowDown = useCallback(() => {
    if (currentIndex === -1) return;
    setImageAtIndex(currentIndex + numSamples);
  }, [numSamples, currentIndex, setImageAtIndex]);

  const handleArrowLeft = useCallback(() => {
    if (currentIndex === -1) return;
    if (imgInfo.promptIdx === 0) return;
    const minIdx = currentIndex - imgInfo.promptIdx;
    const nextIdx = currentIndex - 1;
    if (nextIdx < minIdx) return;
    setImageAtIndex(nextIdx);
  }, [currentIndex, imgInfo.promptIdx, setImageAtIndex]);

  const handleArrowRight = useCallback(() => {
    if (currentIndex === -1) return;
    const stepMinIdx = currentIndex - imgInfo.promptIdx;
    const maxIdx = stepMinIdx + numSamples - 1;
    const nextIdx = currentIndex + 1;
    if (nextIdx > maxIdx) return;
    setImageAtIndex(nextIdx);
  }, [numSamples, currentIndex, imgInfo.promptIdx, setImageAtIndex]);

  const handleDelete = useCallback(() => {
    if (!imgPath) return;
    openConfirm({
      title: 'Delete Sample',
      message: `Are you sure you want to delete this sample? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: () => {
        apiClient
          .post('/api/img/delete', { imgPath: imgPath })
          .then(() => {
            console.log('Image deleted:', imgPath);
            onChange(null);
            if (refreshSampleImages) {
              refreshSampleImages();
            }
          })
          .catch(error => {
            console.error('Error deleting image:', error);
          });
      },
    });
  }, [imgPath, onChange, refreshSampleImages]);

  const sampleItem = useMemo<SampleItem | null>(() => {
    if (!sampleConfig) return null;
    if (imgInfo.promptIdx < 0) return null;
    if (imgInfo.promptIdx >= sampleConfig.samples.length) return null;
    return sampleConfig.samples[imgInfo.promptIdx];
  }, [sampleConfig, imgInfo.promptIdx]);

  const [metadataPrompt, setMetadataPrompt] = useState<string | null>(null);
  const [metadataSeed, setMetadataSeed] = useState<number | null>(null);

  useEffect(() => {
    if (!imgPath) {
      setMetadataPrompt(null);
      setMetadataSeed(null);
      return;
    }
    setMetadataPrompt(null);
    setMetadataSeed(null);
    let cancelled = false;
    apiClient
      .post('/api/img/metadata', { imgPath })
      .then(res => {
        if (cancelled) return;
        if (res.data?.prompt) setMetadataPrompt(res.data.prompt);
        if (res.data?.seed != null) setMetadataSeed(res.data.seed);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [imgPath]);

  const displayedPrompt = useMemo(() => {
    // Prefer metadata (embedded in image file), fall back to sample config prompt.
    // For Ideogram bbox-JSON, extract the human-readable high_level_description
    // rather than dumping the full JSON blob into the caption area.
    const candidates = [metadataPrompt, sampleItem?.prompt ?? null];
    for (const raw of candidates) {
      if (!raw) continue;
      const trimmed = raw.trim();
      if (trimmed.startsWith('{')) {
        try {
          const parsed = JSON.parse(trimmed);
          if (typeof parsed?.high_level_description === 'string') return parsed.high_level_description;
        } catch {}
        // Incomplete / unparseable JSON (e.g. metadata contains only the first line) — skip
        continue;
      }
      return raw; // plain-text prompt, use as-is
    }
    return null;
  }, [metadataPrompt, sampleItem?.prompt]);

  const controlImages = useMemo<string[]>(() => {
    if (!imgPath) return [];
    let controlImageArr: string[] = [];
    if (sampleItem?.ctrl_img) {
      // can be a an array of paths, or a single path
      if (Array.isArray(sampleItem.ctrl_img)) {
        controlImageArr = sampleItem.ctrl_img;
      } else {
        controlImageArr = [sampleItem.ctrl_img];
      }
    } else if (sampleItem?.ctrl_img_1) {
      controlImageArr.push(sampleItem.ctrl_img_1);
    }
    if (sampleItem?.ctrl_img_2) {
      controlImageArr.push(sampleItem.ctrl_img_2);
    }
    if (sampleItem?.ctrl_img_3) {
      controlImageArr.push(sampleItem.ctrl_img_3);
    }
    // filter out nulls
    controlImageArr = controlImageArr.filter(ci => ci !== null && ci !== undefined && ci !== '');
    return controlImageArr;
  }, [sampleItem, imgPath]);

  const seed = useMemo(() => {
    if (metadataSeed != null) return metadataSeed;
    if (!sampleItem) return '?';
    if (sampleItem.seed !== undefined) return sampleItem.seed;
    if (sampleConfig?.walk_seed) {
      return (sampleConfig.seed || 0) + imgInfo.promptIdx;
    }
    return sampleConfig?.seed ?? '?';
  }, [metadataSeed, sampleItem, sampleConfig, imgInfo.promptIdx]);

  const displayedImgPath = useMemo(() => {
    if (showingControlIdx !== null && controlImages[showingControlIdx]) {
      return controlImages[showingControlIdx];
    }
    return imgPath;
  }, [showingControlIdx, controlImages, imgPath]);

  // Load the displayed image via fetch → blob URL, mirroring SampleImageCard behavior.
  // This avoids the silent-failure mode where <img src> collapses to 0×0 on error,
  // making the image area invisible (user sees "just the caption").
  useEffect(() => {
    if (!displayedImgPath || isAudio(displayedImgPath) || isVideo(displayedImgPath)) {
      setImgBlobUrl(null);
      setImgLoadError(null);
      return;
    }

    setImgBlobUrl(null);
    setImgLoadError(null);

    let cancelled = false;
    let objectUrl: string | null = null;

    const controller = new AbortController();
    fetch(`/api/img/${encodeURIComponent(displayedImgPath)}`, { signal: controller.signal })
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.blob();
      })
      .then(blob => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setImgBlobUrl(objectUrl);
      })
      .catch(err => {
        if (cancelled || err?.name === 'AbortError') return;
        console.error('SampleImageViewer: image load failed:', displayedImgPath, err);
        setImgLoadError(err?.message ?? 'Failed to load');
      });

    return () => {
      cancelled = true;
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      setImgBlobUrl(null);
    };
  }, [displayedImgPath]);

  const boundingBoxes = useMemo(
    () => (sampleItem?.prompt ? parseBoundingBoxes(sampleItem.prompt) : null),
    [sampleItem],
  );
  const canShowBoxes = Boolean(
    boundingBoxes &&
      showingControlIdx === null &&
      displayedImgPath &&
      !isAudio(displayedImgPath) &&
      !isVideo(displayedImgPath),
  );

  // keyboard events while open
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isOpen) return;
      switch (event.key) {
        case 'Escape':
          onCancel();
          break;
        case 'ArrowUp':
          handleArrowUp();
          break;
        case 'ArrowDown':
          handleArrowDown();
          break;
        case 'ArrowLeft':
          handleArrowLeft();
          break;
        case 'ArrowRight':
          handleArrowRight();
          break;
        case 'Delete':
        case 'Backspace':
          handleDelete();
          break;
        default:
          break;
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onCancel, handleArrowUp, handleArrowDown, handleArrowLeft, handleArrowRight, handleDelete]);

  // Touch swipe navigation
  const touchStartRef = useRef<{ x: number; y: number } | null>(null);
  const multiTouchRef = useRef(false);
  const zoomedRef = useRef(false);
  const SWIPE_THRESHOLD = 40; // px before a swipe counts

  const onTouchStart = useCallback((e: React.TouchEvent) => {
    if (e.touches.length > 1) {
      multiTouchRef.current = true;
      touchStartRef.current = null;
      return;
    }
    multiTouchRef.current = false;
    const t = e.touches[0];
    if (!t) return;
    touchStartRef.current = { x: t.clientX, y: t.clientY };
  }, []);

  const onTouchEnd = useCallback(
    (e: React.TouchEvent) => {
      const start = touchStartRef.current;
      touchStartRef.current = null;
      // If a pinch happened or we're zoomed in, don't treat as a swipe.
      if (multiTouchRef.current || zoomedRef.current) {
        if (e.touches.length === 0) multiTouchRef.current = false;
        return;
      }
      if (!start) return;
      const t = e.changedTouches[0];
      if (!t) return;
      const dx = t.clientX - start.x;
      const dy = t.clientY - start.y;
      const absX = Math.abs(dx);
      const absY = Math.abs(dy);
      if (absX < SWIPE_THRESHOLD && absY < SWIPE_THRESHOLD) return;
      if (absX > absY) {
        // Horizontal swipe: left = next (ArrowRight), right = prev (ArrowLeft)
        if (dx < 0) handleArrowRight();
        else handleArrowLeft();
      } else {
        // Vertical swipe: up = next step (ArrowDown), down = prev step (ArrowUp)
        if (dy < 0) handleArrowDown();
        else handleArrowUp();
      }
    },
    [handleArrowLeft, handleArrowRight, handleArrowUp, handleArrowDown],
  );

  if (!mounted) return null;

  return createPortal(
    <Dialog open={isOpen} onClose={onCancel} className="relative z-10">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />
      <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
        <div className="flex min-h-full items-center justify-center p-0 sm:p-4 text-center">
          <DialogPanel
            transition
            onTouchStart={onTouchStart}
            onTouchEnd={onTouchEnd}
            className="relative transform rounded-none sm:rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in w-full sm:w-auto sm:max-w-[95%] max-h-[95vh] data-closed:sm:translate-y-0 data-closed:sm:scale-95 flex flex-col overflow-hidden touch-pan-y"
          >
            <div className="flex-1 min-h-0 overflow-hidden flex items-center justify-center">
              {displayedImgPath &&
                (isAudio(displayedImgPath) ? (
                  <div className="w-[500px] h-[260px] max-w-full sm:max-w-[95vw] max-h-[55vh] self-stretch">
                    <AudioPlayer
                      src={`/api/img/${encodeURIComponent(displayedImgPath)}`}
                      title={displayedImgPath.replace(/^.*[\\/]/, '')}
                      autoPlay
                    />
                  </div>
                ) : isVideo(displayedImgPath) ? (
                  <video
                    src={`/api/img/${encodeURIComponent(displayedImgPath)}`}
                    className="w-auto h-auto max-w-full sm:max-w-[95vw] max-h-[82vh] object-contain"
                    preload="none"
                    playsInline
                    loop
                    autoPlay
                    controls={true}
                  />
                ) : imgLoadError ? (
                  <div className="w-full sm:max-w-[95vw] max-h-[82vh] flex items-center justify-center p-8 text-red-400 text-sm text-center">
                    <div>
                      <div className="font-semibold mb-1">Failed to load image</div>
                      <div className="text-xs text-gray-500">{imgLoadError}</div>
                    </div>
                  </div>
                ) : !imgBlobUrl ? (
                  <div className="w-64 h-64 sm:max-w-[95vw] max-h-[82vh] flex items-center justify-center">
                    <div className="animate-pulse bg-gray-700 w-48 h-48 rounded" />
                  </div>
                ) : (
                  <TransformWrapper
                    key={displayedImgPath}
                    initialScale={1}
                    minScale={1}
                    maxScale={6}
                    doubleClick={{ mode: 'toggle', step: 2 }}
                    wheel={{ step: 0.2 }}
                    panning={{ disabled: false, allowRightClickPan: false }}
                    onTransform={(_ref, state) => {
                      zoomedRef.current = state.scale > 1.01;
                    }}
                  >
                    <TransformComponent>
                      <div className="relative">
                        <img
                          src={imgBlobUrl}
                          alt="Sample Image"
                          draggable={false}
                          className="w-auto h-auto max-w-full sm:max-w-[95vw] max-h-[82vh] object-contain select-none !pointer-events-auto"
                        />
                        {showBoxes && canShowBoxes && boundingBoxes && <BoundingBoxOverlay boxes={boundingBoxes} />}
                      </div>
                    </TransformComponent>
                  </TransformWrapper>
                ))}
            </div>
            {/* # make full width */}
            <div className="bg-gray-950 text-sm flex justify-between items-start px-4 py-2 flex-shrink-0 max-h-[35vh] overflow-y-auto">
              <div className="flex-1 min-w-0 mr-4">
                {displayedPrompt && (
                  <div>
                    <span className="text-gray-400 mr-1">Prompt:</span>
                    <span
                      className={`whitespace-pre-wrap break-words cursor-pointer select-text block overflow-hidden transition-all duration-200 ${
                        promptExpanded ? '' : 'line-clamp-3'
                      }`}
                      onClick={() => setPromptExpanded(true)}
                      onBlur={() => setPromptExpanded(false)}
                      tabIndex={0}
                      title={promptExpanded ? 'Click outside to collapse' : 'Click to expand'}
                    >
                      {displayedPrompt}
                    </span>
                  </div>
                )}
              </div>
              {controlImages.length > 0 && (
                <div key={imgPath} className="flex space-x-2 mr-4">
                  {showingControlIdx !== null && (
                    <img
                      src={`/api/img/${encodeURIComponent(imgPath!)}`}
                      alt="Main"
                      className="max-h-12 max-w-12 object-contain bg-black border-2 border-gray-700 hover:border-gray-500 rounded cursor-pointer"
                      onClick={() => setShowingControlIdx(null)}
                      title="Main image"
                    />
                  )}
                  {controlImages.map((ci, idx) => (
                    <img
                      key={idx}
                      src={`/api/img/${encodeURIComponent(ci)}`}
                      alt={`Control ${idx + 1}`}
                      className={`max-h-12 max-w-12 object-contain bg-black border-2 rounded cursor-pointer ${
                        showingControlIdx === idx ? 'border-blue-500' : 'border-gray-700 hover:border-gray-500'
                      }`}
                      onClick={() => setShowingControlIdx(idx)}
                      title={`Control image ${idx + 1}`}
                    />
                  ))}
                </div>
              )}

              <div className="text-xs">
                <div>
                  <span className="text-gray-400">Step:</span> {imgInfo.step.toLocaleString()}
                </div>
                <div>
                  <span className="text-gray-400">Sample #:</span> {imgInfo.promptIdx + 1}
                </div>
                <div>
                  <span className="text-gray-400">Seed:</span> {seed}
                </div>
              </div>
            </div>
            <div className="absolute top-2 right-2 flex items-center gap-2 z-20">
              {canShowBoxes && (
                <button
                  type="button"
                  onClick={() => setShowBoxes(v => !v)}
                  title={showBoxes ? 'Hide bounding boxes' : 'Show bounding boxes'}
                  className={classNames('bg-gray-900 rounded-full p-1 leading-[0px] hover:opacity-100', {
                    'opacity-100 text-blue-400': showBoxes,
                    'opacity-50': !showBoxes,
                  })}
                >
                  <SquareDashed />
                </button>
              )}
              <div className="bg-gray-900 rounded-full p-1 leading-[0px] opacity-50 hover:opacity-100">
                <Menu>
                  <MenuButton>
                    <Cog />
                  </MenuButton>
                  <MenuItems
                    anchor="bottom end"
                    className="bg-gray-900 border border-gray-700 rounded shadow-lg w-48 px-2 py-2 mt-1 z-50"
                  >
                    {imgPath && isAudio(imgPath) && (
                      <MenuItem>
                        <a
                          className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded block"
                          href={`/api/img/${encodeURIComponent(imgPath)}`}
                          download={imgPath.replace(/^.*[\\/]/, '')}
                        >
                          Download
                        </a>
                      </MenuItem>
                    )}
                    <MenuItem>
                      <div className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded" onClick={handleDelete}>
                        Delete Sample
                      </div>
                    </MenuItem>
                  </MenuItems>
                </Menu>
              </div>
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>,
    document.body,
  );
}
