'use client';

import { useEffect, useState, use, useMemo, useRef, useCallback } from 'react';
import { LuImageOff, LuLoader, LuBan, LuSearch } from 'react-icons/lu';
import { FaChevronLeft, FaChevronUp, FaChevronDown, FaExclamationTriangle } from 'react-icons/fa';
import { VirtuosoGrid, VirtuosoGridHandle } from 'react-virtuoso';
import DatasetImageCard from '@/components/DatasetImageCard';
import DatasetImageViewer from '@/components/DatasetImageViewer';
import { Button } from '@headlessui/react';
import AddImagesModal, { openImagesModal, useOpenImagesModalOnDrag } from '@/components/AddImagesModal';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import { Modal } from '@/components/Modal';
import { FloatingWindow } from '@/components/FloatingWindow';
import { TextInput, Checkbox } from '@/components/formInputs';
import classNames from 'classnames';
import { CaptionDatasetModal, openCaptionDatasetModal } from '@/components/CaptionDatasetModal';
import useSettings from '@/hooks/useSettings';
import { pathJoin } from '@/utils/basic';
import AutoCaptionButton from '@/components/AutoCaptionButton';
import CaptionMonitor from '@/components/CaptionMonitor';
import { CreatableSelectInput } from '@/components/formInputs';

export default function DatasetPage({ params }: { params: Promise<{ datasetName: string }> }) {
  const { datasetName } = use(params);
  const [imgList, setImgList] = useState<{ img_path: string; caption: string; captions?: Record<string, string>; captionExists?: Record<string, boolean> }[]>([]);
  const [isAutoCaptioning, setIsAutoCaptioning] = useState(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [filter, setFilter] = useState('');
  const [filterHistory, setFilterHistory] = useState<string[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [filterDropdownRect, setFilterDropdownRect] = useState<{ top: number; left: number; width: number } | null>(null);
  const filterInputRef = useRef<HTMLInputElement>(null);
  const [isFindReplaceOpen, setIsFindReplaceOpen] = useState(false);
  const [findText, setFindText] = useState('');
  const [replaceText, setReplaceText] = useState('');
  const [wholeWord, setWholeWord] = useState(false);
  const [matchCase, setMatchCase] = useState(false);
  const [findNextIndex, setFindNextIndex] = useState(-1);
  const [findMatchCharIndex, setFindMatchCharIndex] = useState(-1);
  const [findResultStatus, setFindResultStatus] = useState<'none' | 'found' | 'not-found'>('none');
  // Incremented on every successful Find navigation so non-highlighted cards collapse.
  const [findNavKey, setFindNavKey] = useState(0);
  const findInputRef = useRef<HTMLInputElement>(null);
  const imgListRef = useRef<typeof imgList>([]);
  useEffect(() => { imgListRef.current = imgList; }, [imgList]);
  const loadedExtsRef = useRef<Set<string>>(new Set(['txt']));
  const { settings, isSettingsLoaded } = useSettings();
  const [selectedImgPath, setSelectedImgPath] = useState<string | null>(null);
  const [captionExt, setCaptionExt] = useState<string>(() => {
    if (typeof window === 'undefined') return 'txt';
    return localStorage.getItem(`captionExt:${datasetName}`) ?? 'txt';
  });
  // re-read on datasetName change too: Next.js can reuse this component instance when
  // navigating between datasets, so the lazy initializer above won't rerun on its own
  useEffect(() => {
    setCaptionExt(localStorage.getItem(`captionExt:${datasetName}`) ?? 'txt');
  }, [datasetName]);
  const [captionRefreshKeys, setCaptionRefreshKeys] = useState<Record<string, number>>({});
  const [scrollParent, setScrollParent] = useState<HTMLDivElement | null>(null);
  const [captionBarHeight, setCaptionBarHeight] = useState(0);
  const scrollParentCallback = useCallback((el: HTMLDivElement | null) => setScrollParent(el), []);
  const virtuosoRef = useRef<VirtuosoGridHandle | null>(null);
  const isRefreshingRef = useRef(false);

  useEffect(() => {
    const savedHistory = localStorage.getItem('filterHistory');
    if (savedHistory) {
      try {
        setFilterHistory(JSON.parse(savedHistory));
      } catch (e) {
        console.error('Error parsing filter history:', e);
      }
    }
  }, []);

  const addToHistory = (value: string) => {
    const trimmed = value.trim();
    if (!trimmed) return;
    setFilterHistory(prev => {
      const newHistory = [trimmed, ...prev.filter(item => item !== trimmed)].slice(0, 10);
      localStorage.setItem('filterHistory', JSON.stringify(newHistory));
      return newHistory;
    });
  };

  const refreshImageList = (dbName: string) => {
    if (isRefreshingRef.current) return;
    isRefreshingRef.current = true;
    loadedExtsRef.current = new Set(['txt']);
    setStatus('loading');
    apiClient
      .post('/api/datasets/listImages', { datasetName: dbName })
      .then((res: any) => {
        const data = res.data;
        // Server sends a shared root (with trailing OS separator) + each file's sub-path to
        // keep the payload small. Plain concat rebuilds the native absolute path on any OS.
        // Server already sorts; avoid a client-side sort on large lists.
        const root = data.root;
        setImgList(data.images.map((item: any) => ({
          img_path: root + item.subPath,
          caption: item.caption ?? '',
          captions: item.captions,
          captionExists: item.captionExists,
        })));
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching images:', error);
        setStatus('error');
      })
      .finally(() => {
        isRefreshingRef.current = false;
      });
  };

  const { captionCount, totalCount } = useMemo(() => {
    return {
      captionCount: imgList.filter(img => {
        if (img.captionExists) return img.captionExists[captionExt] ?? false;
        // fallback for legacy data without captionExists (txt assumed)
        return captionExt === 'txt' && !!(img.caption && img.caption.trim().length > 0);
      }).length,
      totalCount: imgList.length,
    };
  }, [imgList, captionExt]);

  const filteredImgList = useMemo(() => {
    if (!filter) return imgList;

    const getCaption = (img: typeof imgList[number]) =>
      img.captions?.[captionExt] ?? (captionExt === 'txt' ? img.caption : '') ?? '';

    const escapeRegExp = (string: string) => {
      return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    };

    const matchesTerm = (caption: string, term: string) => {
      term = term.trim();
      if (!term) return true;

      if (term.startsWith('"') && term.endsWith('"')) {
        const exactTerm = term.slice(1, -1);
        if (!exactTerm) return true;
        const regex = new RegExp(`(^|[^a-zA-Z0-9_])${escapeRegExp(exactTerm)}([^a-zA-Z0-9_]|$)`, 'i');
        return regex.test(caption);
      }

      return caption.toLowerCase().includes(term.toLowerCase());
    };

    const splitByOperator = (input: string, operator: 'and' | 'or') => {
      const regex = new RegExp(`\\s+${operator}\\s+`, 'gi');
      const parts: string[] = [];
      let lastIndex = 0;
      let match;

      while ((match = regex.exec(input)) !== null) {
        const part = input.slice(lastIndex, match.index).trim();
        const quoteCount = (part.match(/"/g) || []).length;
        if (quoteCount % 2 === 0) {
          parts.push(part);
          lastIndex = regex.lastIndex;
        }
      }
      parts.push(input.slice(lastIndex).trim());
      return parts.filter(p => p !== '');
    };

    const orParts = splitByOperator(filter, 'or');
    if (orParts.length > 1) {
      return imgList.filter(img => {
        const caption = getCaption(img);
        return orParts.some(part => {
          const andParts = splitByOperator(part, 'and');
          if (andParts.length > 1) {
            return andParts.every(subPart => matchesTerm(caption, subPart));
          }
          return matchesTerm(caption, part);
        });
      });
    }

    const andParts = splitByOperator(filter, 'and');
    if (andParts.length > 1) {
      return imgList.filter(img => {
        const caption = getCaption(img);
        return andParts.every(part => matchesTerm(caption, part));
      });
    }

    return imgList.filter(img => matchesTerm(getCaption(img), filter));
  }, [imgList, filter, captionExt]);
  useOpenImagesModalOnDrag(datasetName, () => refreshImageList(datasetName));

  const imgPaths = useMemo(() => imgList.map(img => img.img_path), [imgList]);

  useEffect(() => {
    if (datasetName) {
      refreshImageList(datasetName);
    }
  }, [datasetName]);

  const escapeRegExp = (string: string) => {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  };

  const getSearchRegex = (text: string, isWholeWord: boolean, isMatchCase: boolean, global: boolean = false) => {
    if (!text) return null;
    let pattern = escapeRegExp(text);
    if (isWholeWord) {
      pattern = `(^|[^a-zA-Z0-9_])${pattern}([^a-zA-Z0-9_]|$)`;
    }
    return new RegExp(pattern, (isMatchCase ? '' : 'i') + (global ? 'g' : ''));
  };

  const getActiveCaption = (img: typeof imgList[number]) =>
    img.captions?.[captionExt] ?? (captionExt === 'txt' ? img.caption : '') ?? '';

  const prefetchCaptionsForExt = useCallback(async (ext: string): Promise<void> => {
    if (ext === 'txt') return;
    if (loadedExtsRef.current.has(ext)) return;
    const paths = imgListRef.current.map(img => img.img_path);
    if (paths.length === 0) return;
    try {
      const res = await apiClient.post('/api/caption/getBatch', { imgPaths: paths, ext });
      const captions: Record<string, string> = res.data?.captions ?? {};
      loadedExtsRef.current.add(ext);
      const newList = imgListRef.current.map(img => ({
        ...img,
        captions: { ...(img.captions ?? {}), [ext]: captions[img.img_path] ?? '' },
      }));
      imgListRef.current = newList;
      setImgList(newList);
    } catch (err) {
      console.error('Error prefetching captions for find/replace:', err);
    }
  }, []);

  const handleFind = async (startIndex: number = 0, direction: 'next' | 'prev' | 'start' = 'start') => {
    if (!findText) return;

    await prefetchCaptionsForExt(captionExt);

    const currentImgList = imgListRef.current;
    const regex = getSearchRegex(findText, wholeWord, matchCase);
    if (!regex) return;

    let searchIdx = startIndex;
    if (direction === 'next') {
        searchIdx = (findNextIndex + 1) % currentImgList.length;
    } else if (direction === 'prev') {
        searchIdx = (findNextIndex - 1 + currentImgList.length) % currentImgList.length;
    }

    const getActiveCaptionFromItem = (img: typeof imgList[number]) =>
      img.captions?.[captionExt] ?? (captionExt === 'txt' ? img.caption : '') ?? '';

    let found = false;

    for (let i = 0; i < currentImgList.length; i++) {
      const idx = direction === 'prev'
        ? (searchIdx - i + currentImgList.length) % currentImgList.length
        : (searchIdx + i) % currentImgList.length;

      const caption = getActiveCaptionFromItem(currentImgList[idx]);
      const match = caption.match(regex);
      if (match) {
        let charIndex = match.index || 0;
        if (wholeWord && match[1]) {
            charIndex += match[1].length;
        }

        setFindNextIndex(idx);
        setFindMatchCharIndex(charIndex);
        setFindResultStatus('found');
        setFindNavKey(prev => prev + 1);
        found = true;

        // Scroll to the item via VirtuosoGrid (works even with virtualized rendering).
        const filteredIdx = filteredImgList.findIndex(i => i.img_path === currentImgList[idx].img_path);
        if (filteredIdx !== -1 && virtuosoRef.current) {
          virtuosoRef.current.scrollToIndex({ index: filteredIdx, behavior: 'smooth', align: 'center' });
        }
        break;
      }
    }

    if (!found) {
        setFindResultStatus('not-found');
    }
  };

  const handleReplace = (moveNext: boolean = false) => {
    if (findNextIndex === -1 || !findText) return;

    const currentImg = imgList[findNextIndex];
    const regex = getSearchRegex(findText, wholeWord, matchCase, true);
    if (!regex) return;

    const oldCaption = getActiveCaption(currentImg);
    let newCaption = oldCaption;

    if (wholeWord) {
      newCaption = oldCaption.replace(regex, (match, p1, p2) => {
        return (p1 || '') + replaceText + (p2 || '');
      });
    } else {
      newCaption = oldCaption.replace(regex, replaceText);
    }

    if (newCaption !== oldCaption) {
      apiClient
        .post('/api/img/caption', { imgPath: currentImg.img_path, caption: newCaption, ext: captionExt })
        .then(() => {
          setImgList(prev =>
            prev.map((item, idx) =>
              idx === findNextIndex
                ? { ...item, captions: { ...item.captions, [captionExt]: newCaption }, caption: captionExt === 'txt' ? newCaption : item.caption }
                : item,
            ),
          );
        })
        .catch(err => console.error('Error replacing caption:', err));
    }

    if (moveNext) {
      handleFind(findNextIndex, 'next');
    }
  };

  const handleReplaceAll = async () => {
    if (!findText) return;
    await prefetchCaptionsForExt(captionExt);

    const regex = getSearchRegex(findText, wholeWord, matchCase, true);
    if (!regex) return;

    const updates: { img_path: string; caption: string }[] = [];
    const newList = imgList.map(img => {
      const oldCaption = getActiveCaption(img);
      let newCaption = oldCaption;

      if (wholeWord) {
        newCaption = oldCaption.replace(regex, (match, p1, p2) => {
          return (p1 || '') + replaceText + (p2 || '');
        });
      } else {
        newCaption = oldCaption.replace(regex, replaceText);
      }

      if (newCaption !== oldCaption) {
        updates.push({ img_path: img.img_path, caption: newCaption });
        return {
          ...img,
          captions: { ...img.captions, [captionExt]: newCaption },
          caption: captionExt === 'txt' ? newCaption : img.caption,
        };
      }
      return img;
    });

    if (updates.length === 0) return;

    setImgList(newList);

    Promise.all(
      updates.map(update =>
        apiClient.post('/api/img/caption', { imgPath: update.img_path, caption: update.caption, ext: captionExt }),
      ),
    ).catch(err => {
      console.error('Error during replace all:', err);
      refreshImageList(datasetName);
    });
  };

  const openFindReplace = useCallback(() => {
    setIsFindReplaceOpen(true);
    setFindResultStatus('none');
    prefetchCaptionsForExt(captionExt);
    setTimeout(() => {
      findInputRef.current?.focus();
      findInputRef.current?.select();
    }, 100);
  }, [prefetchCaptionsForExt, captionExt]);

  // keep captions for the active extension loaded so the search box (and find/replace)
  // can match against them, not just the .txt caption. Re-runs after a list refresh too,
  // since refreshImageList() clears the loaded-extensions cache.
  useEffect(() => {
    if (status === 'success') {
      prefetchCaptionsForExt(captionExt);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [captionExt, status]);

  useEffect(() => {
    if (!isFindReplaceOpen) return;
    setFindNextIndex(-1);
    setFindMatchCharIndex(-1);
    setFindResultStatus('none');
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [captionExt, isFindReplaceOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        e.preventDefault();
        openFindReplace();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [openFindReplace]);

  const PageInfoContent = useMemo(() => {
    let icon = null;
    let text = '';
    let subtitle = '';
    let showIt = false;
    let bgColor = '';
    let textColor = '';
    let iconColor = '';

    if (status == 'loading') {
      icon = <LuLoader className="animate-spin w-8 h-8" />;
      text = 'Loading Images';
      subtitle = 'Please wait while we fetch your dataset images...';
      showIt = true;
      bgColor = 'bg-gray-800/50';
      textColor = 'text-gray-100';
      iconColor = 'text-gray-400';
    }
    if (status == 'error') {
      icon = <LuBan className="w-8 h-8" />;
      text = 'Error Loading Images';
      subtitle = 'There was a problem fetching the images. Please try refreshing the page.';
      showIt = true;
      bgColor = 'bg-red-600/20';
      textColor = 'text-red-100';
      iconColor = 'text-red-400';
    }
    if (status == 'success' && imgList.length === 0) {
      icon = <LuImageOff className="w-8 h-8" />;
      text = 'No Images Found';
      subtitle = 'This dataset is empty. Click "Add Images" to get started.';
      showIt = true;
      bgColor = 'bg-gray-800/50';
      textColor = 'text-gray-100';
      iconColor = 'text-gray-400';
    }

    if (status == 'success' && imgList.length > 0 && filteredImgList.length === 0) {
      icon = <LuImageOff className="w-8 h-8" />;
      text = 'No Matches';
      subtitle = `No images match your filter: "${filter}"`;
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
  }, [status, imgList.length, filteredImgList.length, filter]);

  return (
    <>
      {/* Fixed top bar */}
      <TopBar>
        <div className="flex-shrink-0">
          <Button className="text-gray-500 dark:text-gray-300 px-2 sm:px-3 mt-1" onClick={() => history.back()}>
            <FaChevronLeft />
          </Button>
        </div>
        <div className="min-w-0 flex-shrink">
          <h1 className="text-base sm:text-lg truncate">
            <span className="hidden sm:inline">Dataset: </span>
            {datasetName}
          </h1>
        </div>
        <div className="flex-1 max-w-xl mx-4">
          <input
            ref={filterInputRef}
            type="text"
            className="w-full bg-slate-800 border border-slate-700 rounded-md px-3 py-1 text-sm text-gray-200 focus:outline-none focus:ring-1 focus:ring-slate-500"
            placeholder="Filter by caption (supports AND, OR)..."
            value={filter}
            onChange={e => setFilter(e.target.value)}
            onFocus={() => {
              if (filterInputRef.current) {
                const r = filterInputRef.current.getBoundingClientRect();
                setFilterDropdownRect({ top: r.bottom + 2, left: r.left, width: r.width });
              }
              setShowHistory(true);
            }}
            onBlur={() => {
              setTimeout(() => setShowHistory(false), 150);
              addToHistory(filter);
            }}
            onKeyDown={e => {
              if (e.key === 'Enter') {
                addToHistory(filter);
                (e.target as HTMLInputElement).blur();
              }
            }}
          />
          {showHistory && filterHistory.length > 0 && filterDropdownRect && (
            <div
              style={{ position: 'fixed', top: filterDropdownRect.top, left: filterDropdownRect.left, width: filterDropdownRect.width, zIndex: 9999 }}
              className="bg-slate-800 border border-slate-700 rounded-md shadow-xl max-h-60 overflow-y-auto"
            >
              {filterHistory.map((item, index) => (
                <div
                  key={index}
                  className="px-3 py-2 text-sm text-gray-200 hover:bg-slate-700 cursor-pointer"
                  onMouseDown={e => {
                    e.preventDefault();
                    setFilter(item);
                    setShowHistory(false);
                  }}
                >
                  {item}
                </div>
              ))}
            </div>
          )}
        </div>
        <div>
          <Button
            className="text-gray-200 bg-slate-600 px-3 py-1 rounded-md mr-2 flex items-center gap-2"
            onClick={openFindReplace}
          >
            <LuSearch size={16} /> Find/Replace
          </Button>
        </div>
        <div className="flex-1"></div>
        {status === 'success' && totalCount > 0 && (
          <div className="text-sm text-gray-400 mr-2 sm:mr-4 flex-shrink-0">
            Caption count: {captionCount}/{totalCount}
          </div>
        )}
        <div className="flex-shrink-0 flex items-center gap-1 sm:gap-2">
          <div className="flex items-center gap-1">
            <label className="text-xs text-gray-400 hidden sm:inline whitespace-nowrap">Caption ext</label>
            <CreatableSelectInput
              className="w-44"
              value={captionExt}
              onChange={value => {
                setCaptionExt(value);
                localStorage.setItem(`captionExt:${datasetName}`, value);
              }}
              options={[
                { value: 'txt', label: 'txt' },
                { value: 'json', label: 'json' },
                { value: 'caption', label: 'caption' },
              ]}
            />
          </div>
          <AutoCaptionButton
            datasetPath={`${pathJoin(settings.DATASETS_FOLDER, datasetName)}`}
            setIsAutoCaptioning={setIsAutoCaptioning}
            captionExt={captionExt}
          />
          <Button
            className="text-white bg-slate-600 px-2 sm:px-3 py-1 rounded-md text-sm sm:text-base whitespace-nowrap"
            onClick={() => openImagesModal(datasetName, () => refreshImageList(datasetName))}
          >
            <span className="sm:hidden">+ Add</span>
            <span className="hidden sm:inline">Add Images</span>
          </Button>
        </div>
      </TopBar>
      <MainContent ref={scrollParentCallback}>
        {PageInfoContent}
        {status === 'success' && filteredImgList.length > 0 && scrollParent && (
          <VirtuosoGrid
            ref={virtuosoRef}
            totalCount={filteredImgList.length}
            customScrollParent={scrollParent}
            overscan={400}
            listClassName="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4"
            itemContent={index => {
              const img = filteredImgList[index];
              if (!img) return null;
              const globalIndex = imgList.findIndex(i => i.img_path === img.img_path);
              const isMatch = globalIndex === findNextIndex;
              return (
                <DatasetImageCard
                  alt="image"
                  imageUrl={img.img_path}
                  className={classNames({
                    'ring-4 ring-blue-500 rounded-lg': isMatch,
                  })}
                  isHighlighted={isMatch}
                  highlightText={isMatch ? findText : undefined}
                  highlightCharIndex={isMatch ? findMatchCharIndex : -1}
                  isAutoCaptioning={isAutoCaptioning}
                  resetEditKey={findNavKey}
                  onDelete={() => refreshImageList(datasetName)}
                  onImageClick={() => setSelectedImgPath(img.img_path)}
                  onCaptionSave={(newCaption, imgPath) => {
                    setImgList(prev =>
                      prev.map(item => (item.img_path === imgPath ? { ...item, caption: newCaption } : item)),
                    );
                  }}
                  captionRefreshKey={captionRefreshKeys[img.img_path] || 0}
                  initialCaption={img.captions?.[captionExt] ?? img.caption}
                  observerRoot={scrollParent}
                  captionExt={captionExt}
                />
              );
            }}
            computeItemKey={index => filteredImgList[index]?.img_path ?? index}
          />
        )}
        {/* Spacer so the last cards stay accessible above the floating caption bar.
            Always keeps a baseline gap, plus the bar height when it is showing. */}
        <div style={{ height: `${captionBarHeight + 24}px` }} className="transition-[height] duration-300" />
      </MainContent>
      <AddImagesModal />
      {isSettingsLoaded && (
        <CaptionMonitor
          datasetPath={`${pathJoin(settings.DATASETS_FOLDER, datasetName)}`}
          onHeightChange={setCaptionBarHeight}
        />
      )}
      <DatasetImageViewer
        imgPath={selectedImgPath}
        imageList={imgPaths}
        onChange={setSelectedImgPath}
        refreshImages={() => refreshImageList(datasetName)}
        onCaptionSaved={path => setCaptionRefreshKeys(prev => ({ ...prev, [path]: (prev[path] || 0) + 1 }))}
        captionExt={captionExt}
      />

      <FloatingWindow
        isOpen={isFindReplaceOpen}
        onClose={() => setIsFindReplaceOpen(false)}
        title="Find and Replace"
      >
        <div className="space-y-4">
          <TextInput
            label="Find"
            value={findText}
            onChange={(val) => {
                setFindText(val);
                setFindResultStatus('none');
            }}
            placeholder="Text to find..."
            ref={findInputRef}
            onKeyDown={e => {
                if (e.key === 'Enter') {
                    handleFind(findNextIndex === -1 ? 0 : findNextIndex, 'next');
                }
            }}
          />
          <TextInput
            label="Replace"
            value={replaceText}
            onChange={setReplaceText}
            placeholder="Replacement text..."
          />

          <div className="flex gap-4">
            <Checkbox label="Whole Word" checked={wholeWord} onChange={setWholeWord} />
            <Checkbox label="Match Case" checked={matchCase} onChange={setMatchCase} />
          </div>

          <div className="flex flex-wrap gap-2 pt-2">
            <Button
              className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-md transition-colors flex items-center gap-2"
              onClick={() => handleFind(findNextIndex === -1 ? 0 : findNextIndex, 'prev')}
              title="Find Previous"
            >
              <FaChevronUp size={12} /> Previous
            </Button>
            <Button
              className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-md transition-colors flex items-center gap-2"
              onClick={() => handleFind(findNextIndex === -1 ? 0 : findNextIndex, 'next')}
              title="Find Next"
            >
              <FaChevronDown size={12} /> Next
            </Button>
            {replaceText !== '' && (
              <>
                <Button
                  className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-md transition-colors"
                  onClick={() => handleReplace(true)}
                >
                  Replace
                </Button>
                <Button
                  className="bg-blue-700 hover:bg-blue-600 text-white px-4 py-2 rounded-md transition-colors"
                  onClick={() => handleReplaceAll()}
                >
                  Replace All
                </Button>
              </>
            )}
          </div>

          {findResultStatus === 'not-found' && (
            <div className="flex items-center gap-2 text-amber-500 text-sm mt-2 animate-in fade-in slide-in-from-top-1">
              <FaExclamationTriangle size={16} />
              <span>No matches found</span>
            </div>
          )}
        </div>
      </FloatingWindow>
      <CaptionDatasetModal />
    </>
  );
}
