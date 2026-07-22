import { useState, useEffect } from 'react';

export function useSessionFilter(key: string): [string, (v: string) => void] {
  const [value, setValue] = useState<string>(() => {
    if (typeof window === 'undefined') return '';
    return sessionStorage.getItem(key) ?? '';
  });

  useEffect(() => {
    sessionStorage.setItem(key, value);
  }, [key, value]);

  return [value, setValue];
}
