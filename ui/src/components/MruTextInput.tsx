'use client';
import { useState, useEffect, useRef, useCallback } from 'react';

const MRU_MAX = 10;

function getMru(key: string): string[] {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveMru(key: string, value: string, current: string[]): string[] {
  const trimmed = value.trim();
  if (!trimmed) return current;
  const next = [trimmed, ...current.filter(v => v !== trimmed)].slice(0, MRU_MAX);
  localStorage.setItem(key, JSON.stringify(next));
  return next;
}

interface Props {
  value: string;
  onChange: (v: string) => void;
  mruKey: string;
  placeholder?: string;
  /** Extra classes applied to the <input> element */
  inputClassName?: string;
  label?: string;
}

export default function MruTextInput({ value, onChange, mruKey, placeholder, inputClassName, label }: Props) {
  const [mru, setMru] = useState<string[]>([]);
  const [showMru, setShowMru] = useState(false);
  const [rect, setRect] = useState<{ top: number; left: number; width: number } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setMru(getMru(mruKey));
  }, [mruKey]);

  const handleFocus = useCallback(() => {
    if (inputRef.current) {
      const r = inputRef.current.getBoundingClientRect();
      setRect({ top: r.bottom + 2, left: r.left, width: r.width });
    }
    setShowMru(true);
  }, []);

  const handleBlur = useCallback(() => {
    // Small delay so onMouseDown on a list item fires before we hide.
    setTimeout(() => {
      setShowMru(false);
      setMru(prev => saveMru(mruKey, value, prev));
    }, 150);
  }, [mruKey, value]);

  const pick = useCallback((item: string) => {
    onChange(item);
    setShowMru(false);
  }, [onChange]);

  return (
    <div className="relative w-full">
      {label && <label className="block text-sm text-gray-400 mb-1">{label}</label>}
      <input
        ref={inputRef}
        type="text"
        className={inputClassName ?? 'w-full bg-slate-800 border border-slate-700 rounded-md px-3 py-1 text-sm text-gray-200 focus:outline-none focus:ring-1 focus:ring-slate-500'}
        placeholder={placeholder}
        value={value}
        onChange={e => onChange(e.target.value)}
        onFocus={handleFocus}
        onBlur={handleBlur}
      />
      {showMru && mru.length > 0 && rect && (
        <div
          style={{ position: 'fixed', top: rect.top, left: rect.left, width: rect.width, zIndex: 9999 }}
          className="bg-slate-800 border border-slate-700 rounded-md shadow-xl max-h-60 overflow-y-auto"
        >
          {mru.map((item, i) => (
            <div
              key={i}
              className="px-3 py-2 text-sm text-gray-200 hover:bg-slate-700 cursor-pointer truncate"
              title={item}
              onMouseDown={e => {
                e.preventDefault(); // prevent blur before click
                pick(item);
              }}
            >
              {item}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
