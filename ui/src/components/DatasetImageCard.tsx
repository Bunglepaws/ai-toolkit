import React, { useEffect, useState, ReactNode, KeyboardEvent, useRef } from 'react';
import { FaTrashAlt, FaExpandAlt } from 'react-icons/fa';
import { openConfirm } from './ConfirmModal';
import classNames from 'classnames';
import { apiClient } from '@/utils/api';
import AudioPlayer from './AudioPlayer';
import { isVideo, isAudio } from '@/utils/basic';
import useCaptionBatch, { setCachedCaption } from '@/hooks/useCaptionBatch';

interface DatasetImageCardProps {
  imageUrl: string;
  alt: string;
  isAutoCaptioning: boolean;
  children?: ReactNode;
  className?: string;
  onDelete?: () => void;
  onCaptionSave?: (newCaption: string, imageUrl: string) => void;
  onImageClick?: () => void;
  /** When provided (even as empty string), the caption is considered pre-loaded and no fetch is issued. */
  initialCaption?: string;
  captionRefreshKey?: number;
  isHighlighted?: boolean;
  highlightText?: string;
  highlightCharIndex?: number;
  /** Increment this to collapse the caption on cards that are not currently highlighted (e.g. on Find navigation). */
  resetEditKey?: number;
  observerRoot?: Element | null;
  rootMargin?: string;
  captionExt?: string;
}

const DatasetImageCard: React.FC<DatasetImageCardProps> = ({
  imageUrl,
  alt,
  isAutoCaptioning,
  children,
  className = '',
  onDelete = () => {},
  onCaptionSave,
  onImageClick,
  initialCaption,
  captionRefreshKey = 0,
  isHighlighted = false,
  highlightText = '',
  highlightCharIndex = -1,
  resetEditKey,
  observerRoot = null,
  rootMargin = '200px 0px',
  captionExt = 'txt',
}) => {
  const textAreaRef = useRef<HTMLTextAreaElement>(null);
  const [loaded, setLoaded] = useState<boolean>(false);
  const [showAudioPlayer, setShowAudioPlayer] = useState(false);
  const [pollTick, setPollTick] = useState(0);
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);
  // Persists the blob URL across visibility changes so the image stays shown while
  // the card remains mounted. Revoked only when imageUrl changes or on unmount.
  const blobObjectUrlRef = useRef<string | null>(null);
  // Tracks whether the card is actively being edited. Used in the IntersectionObserver
  // callback to suppress visibility-loss transitions caused by textarea expansion
  // shifting layout and briefly pushing the card outside the scroll container boundary.
  const effectivelyEditingRef = useRef(false);

  const isItAVideo = isVideo(imageUrl);
  const isItAudio = isAudio(imageUrl);
  const isItImage = !isItAVideo && !isItAudio;

  // Keep the editing ref in sync so the observer callback can read it without deps.
  const isEditingOrHighlighted = isEditing || isHighlighted;
  useEffect(() => {
    effectivelyEditingRef.current = isEditingOrHighlighted;
  });

  // Track actual viewport visibility — Virtuoso keeps a buffer of cards mounted
  // outside the visible region, so we can't rely on mount/unmount alone.
  useEffect(() => {
    const el = cardRef.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      entries => {
        for (const entry of entries) {
          if (entry.target === el) {
            // Suppress visibility-loss while editing: expanding the textarea shifts
            // layout and can push the card outside the scroll container boundary,
            // which would trigger image blob revocation and interrupt the edit.
            if (!entry.isIntersecting && effectivelyEditingRef.current) return;
            setIsVisible(entry.isIntersecting);
          }
        }
      },
      {
        root: observerRoot ?? null,
        threshold: 0.01,
        rootMargin,
      },
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [observerRoot, rootMargin]);

  // Drive image loads through fetch + AbortController so scrolling past actually
  // cancels in-flight requests. Debounced 80ms so fast scroll-throughs never
  // start a request.
  // We do NOT revoke the blob URL when isVisible goes false — Virtuoso keeps cards
  // mounted during layout shifts (e.g. at the bottom of the list), and revoking on
  // every brief visibility-loss causes the flickering the user sees. The blob is
  // kept alive until imageUrl changes or the component unmounts (see effect below).
  useEffect(() => {
    if (!isItImage) return;
    if (!isVisible) return;
    if (blobObjectUrlRef.current) return; // already loaded for this imageUrl

    const controller = new AbortController();
    let cancelled = false;

    const timer = window.setTimeout(() => {
      fetch(`/api/img/${encodeURIComponent(imageUrl)}`, { signal: controller.signal })
        .then(r => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.blob();
        })
        .then(blob => {
          if (cancelled) return;
          const url = URL.createObjectURL(blob);
          blobObjectUrlRef.current = url;
          setBlobUrl(url);
          setLoaded(true);
        })
        .catch(err => {
          if (err?.name !== 'AbortError') console.error('Dataset image fetch failed:', err);
        });
    }, 80);

    return () => {
      cancelled = true;
      clearTimeout(timer);
      controller.abort();
      // Intentionally not revoking blobObjectUrlRef here — see comment above.
    };
  }, [imageUrl, isItImage, isVisible]);

  // Revoke blob URL and reset image state when imageUrl changes or on unmount.
  useEffect(() => {
    return () => {
      if (blobObjectUrlRef.current) {
        URL.revokeObjectURL(blobObjectUrlRef.current);
        blobObjectUrlRef.current = null;
      }
      setBlobUrl(null);
      setLoaded(false);
    };
  }, [imageUrl]);

  // Not gated by isVisible: when a card scrolls outside Virtuoso's overscan window it
  // unmounts, and gating on visibility fed `null` here, which useCaptionBatch treats as
  // "clear the caption" (caption='', isLoaded=false). Scrolling back caused a moment
  // (or, on cache miss, indefinitely) where the caption showed blank. Fetching is cheap
  // here (module-level batched + cached in useCaptionBatch), so just always fetch.
  const combinedRefreshKey = captionRefreshKey + pollTick;
  const { caption: fetchedCaption, isLoaded: isCaptionLoaded } = useCaptionBatch(
    imageUrl,
    combinedRefreshKey,
    captionExt,
  );

  const [caption, setCaption] = useState<string>(initialCaption ?? '');
  const [savedCaption, setSavedCaption] = useState<string>((initialCaption ?? '').trim());
  const dirtyRef = useRef<boolean>(false);

  // Sync from fetched caption, but don't clobber unsaved local edits.
  useEffect(() => {
    if (!isCaptionLoaded) return;
    if (dirtyRef.current) return;
    setCaption(fetchedCaption);
    setSavedCaption(fetchedCaption.trim());
  }, [fetchedCaption, isCaptionLoaded]);

  // If initialCaption changes externally (e.g. after find/replace), sync it.
  useEffect(() => {
    if (initialCaption === undefined) return;
    if (dirtyRef.current) return;
    setCaption(initialCaption);
    setSavedCaption(initialCaption.trim());
  }, [initialCaption]);

  // Poll while auto-captioning so backend-written captions show up.
  useEffect(() => {
    if (!isAutoCaptioning) return;
    const interval = setInterval(() => setPollTick(t => t + 1), 5000);
    return () => clearInterval(interval);
  }, [isAutoCaptioning]);

  const effectivelyLoaded = isCaptionLoaded || initialCaption !== undefined;

  useEffect(() => {
    if (isHighlighted && highlightText && highlightCharIndex !== -1 && effectivelyLoaded) {
      if (textAreaRef.current) {
        textAreaRef.current.focus();
        textAreaRef.current.setSelectionRange(highlightCharIndex, highlightCharIndex + highlightText.length);
      }
    }
    // NOTE: highlightText excluded from deps intentionally — see comment on resetEditKey effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isHighlighted, highlightCharIndex, effectivelyLoaded]);

  // Collapse non-highlighted cards on Find navigation.
  useEffect(() => {
    if (resetEditKey === undefined) return;
    if (!isHighlighted) {
      setIsEditing(false);
      textAreaRef.current?.blur();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resetEditKey]);

  const saveCaption = (valueToSave?: string) => {
    const trimmedCaption = (valueToSave !== undefined ? valueToSave : caption).trim();
    if (trimmedCaption === savedCaption) {
      dirtyRef.current = false;
      return;
    }
    setSavedCaption(trimmedCaption);
    setCachedCaption(imageUrl, trimmedCaption, captionExt);
    dirtyRef.current = false;
    apiClient
      .post('/api/img/caption', { imgPath: imageUrl, caption: trimmedCaption, ext: captionExt })
      .then(() => {
        onCaptionSave?.(trimmedCaption, imageUrl);
      })
      .catch(error => {
        console.error('Error saving caption:', error);
      });
  };

  // Save any pending edit if the card unmounts (e.g. scrolled out of the virtualized window).
  const latestRef = useRef({ caption, savedCaption, imageUrl, captionExt });
  useEffect(() => {
    latestRef.current = { caption, savedCaption, imageUrl, captionExt };
  });
  useEffect(() => {
    return () => {
      if (!dirtyRef.current) return;
      const { caption: c, savedCaption: s, imageUrl: url, captionExt: ext } = latestRef.current;
      const trimmed = c.trim();
      if (trimmed === s) return;
      apiClient
        .post('/api/img/caption', { imgPath: url, caption: trimmed, ext })
        .then(() => setCachedCaption(url, trimmed, ext))
        .catch(err => console.error('Error saving caption on unmount:', err));
    };
  }, []);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      saveCaption(e.currentTarget.value);
      setIsEditing(false);
      (e.target as HTMLTextAreaElement).blur();
    }
  };

  const handleCaptionChange = (value: string) => {
    dirtyRef.current = value.trim() !== savedCaption;
    setCaption(value);
  };

  const isCaptionCurrent = caption.trim() === savedCaption;

  const effectivelyEditing = isEditing || isHighlighted;


  return (
    <div ref={cardRef} className={`flex flex-col ${className}`}>
      <div className="relative w-full" style={{ paddingBottom: '100%' }}>
        <div
          className={classNames('absolute inset-0 rounded-t-lg shadow-md bg-gray-900', {
            'animate-pulse': isItImage && !loaded,
          })}
        >
          {isItAVideo && (
            <video
              src={`/api/img/${encodeURIComponent(imageUrl)}`}
              className="w-full h-full object-contain"
              autoPlay={false}
              loop
              muted
              controls
            />
          )}
          {isItAudio && !showAudioPlayer && (
            <div
              className="w-full h-full cursor-pointer flex flex-col items-center justify-center gap-2 bg-gray-900 text-gray-400 hover:bg-gray-800 hover:text-gray-200 transition-colors"
              onClick={() => setShowAudioPlayer(true)}
              title="Click to play"
            >
              <div className="text-5xl select-none">♪</div>
              <div className="text-xs px-2 w-full text-center truncate">
                {imageUrl.replace(/^.*[\\/]/, '')}
              </div>
              <div className="text-xs text-gray-500">Click to play</div>
            </div>
          )}
          {isItAudio && showAudioPlayer && (
            <AudioPlayer src={`/api/img/${encodeURIComponent(imageUrl)}`} title={imageUrl.replace(/^.*[\\/]/, '')} />
          )}
          {isItImage && blobUrl && (
            <img
              src={blobUrl}
              alt={alt}
              onClick={onImageClick}
              className={classNames('w-full h-full object-contain', {
                'cursor-zoom-in': !!onImageClick,
              })}
            />
          )}
          {children && <div className="absolute inset-0 flex items-center justify-center">{children}</div>}
          <div className="absolute top-1 right-1 flex space-x-2 z-10">
            {isItAudio && onImageClick && (
              <button
                className="bg-gray-800 rounded-full p-2"
                title="Open song editor"
                onClick={e => {
                  e.stopPropagation();
                  onImageClick();
                }}
              >
                <FaExpandAlt />
              </button>
            )}
            <button
              className="bg-gray-800 rounded-full p-2"
              onClick={() => {
                const mediaLabel = isItAVideo ? 'video' : isItAudio ? 'song' : 'image';
                openConfirm({
                  title: `Delete ${mediaLabel}`,
                  message: `Are you sure you want to delete this ${mediaLabel}? This action cannot be undone.`,
                  type: 'warning',
                  confirmText: 'Delete',
                  onConfirm: () => {
                    apiClient
                      .post('/api/img/delete', { imgPath: imageUrl })
                      .then(() => {
                        onDelete();
                      })
                      .catch(error => {
                        console.error('Error deleting image:', error);
                      });
                  },
                });
              }}
            >
              <FaTrashAlt />
            </button>
          </div>
        </div>
      </div>
      <div
        className={classNames('w-full p-2 bg-gray-800 text-white text-sm rounded-b-lg', {
          'border-blue-500 border-2': !isCaptionCurrent,
          'border-transparent border-2': isCaptionCurrent,
          'h-[75px] overflow-hidden': !effectivelyEditing,
          // Capped (not unbounded) growth: long captions (e.g. songs, which embed
          // CAPTION/LYRICS/BPM tags) would otherwise grow the card taller than the
          // grid row Virtuoso allocated, overlapping the row below. Clicking into that
          // overlap hit the next card instead of this textarea, blurring and closing
          // the editor. Scrolling internally keeps the card self-contained.
          'max-h-80 overflow-y-auto z-10': effectivelyEditing,
        })}
      >
        {effectivelyLoaded ? (
          isItAudio && onImageClick ? (
            // Songs embed long lyrics/tag captions that don't fit this card well and
            // caused the in-card textarea to collapse or fail to open reliably. Send
            // editing to the full dialog instead, which has room for it.
            <div
              className="w-full h-full cursor-text whitespace-pre-wrap opacity-90"
              onClick={onImageClick}
              title="Click to edit caption"
            >
              {caption || <span className="text-gray-500">Click to add a caption...</span>}
            </div>
          ) : (
            <form
              onSubmit={e => {
                e.preventDefault();
                saveCaption();
              }}
              onBlur={e => {
                saveCaption(e.currentTarget.querySelector('textarea')?.value);
                setIsEditing(false);
              }}
            >
              <textarea
                ref={textAreaRef}
                className={classNames(
                  'w-full bg-transparent resize-none outline-none focus:ring-0 focus:outline-none',
                  {
                    'opacity-50 cursor-not-allowed': isAutoCaptioning,
                  },
                )}
                style={effectivelyEditing ? ({ fieldSizing: 'content', minHeight: '75px' } as any) : {}}
                value={caption}
                rows={effectivelyEditing ? undefined : 3}
                readOnly={isAutoCaptioning}
                onChange={e => handleCaptionChange(e.target.value)}
                onKeyDown={handleKeyDown}
                onFocus={() => setIsEditing(true)}
              />
            </form>
          )
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-400">Loading caption...</div>
        )}
      </div>
    </div>
  );
};

export default DatasetImageCard;
