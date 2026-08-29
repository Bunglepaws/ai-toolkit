'use client';

import { useCallback, useRef } from 'react';

// Global (per-browser) mute preference for audible <video> players, so muting one
// video keeps the next one you open quiet. Grid thumbnails are hard-muted and do
// not participate.
const STORAGE_KEY = 'aitk_video_muted';

export const readVideoMuted = (): boolean => {
  try {
    return localStorage.getItem(STORAGE_KEY) === 'true';
  } catch {
    return false;
  }
};

const writeVideoMuted = (muted: boolean) => {
  try {
    localStorage.setItem(STORAGE_KEY, muted ? 'true' : 'false');
  } catch {
    // private mode / storage disabled — preference just doesn't persist
  }
};

/**
 * Returns props to spread onto an audible <video>. Applies the remembered mute
 * state on mount and writes it back whenever the user toggles the mute control.
 */
export const useVideoMute = () => {
  // What we last applied ourselves. Setting `.muted` in code also fires
  // volumechange, so we compare against this to only persist real user changes.
  const appliedRef = useRef<boolean | null>(null);

  const ref = useCallback((el: HTMLVideoElement | null) => {
    if (!el) return;
    const muted = readVideoMuted();
    appliedRef.current = muted;
    el.muted = muted;
    if (!muted) {
      // Autoplay with sound can be blocked before the browser trusts the page.
      // Fall back to muted playback without clobbering the stored preference.
      const p = el.play();
      if (p && typeof p.catch === 'function') {
        p.catch(() => {
          appliedRef.current = true;
          el.muted = true;
        });
      }
    }
  }, []);

  const onVolumeChange = useCallback((e: React.SyntheticEvent<HTMLVideoElement>) => {
    const muted = e.currentTarget.muted;
    if (muted === appliedRef.current) return;
    appliedRef.current = muted;
    writeVideoMuted(muted);
  }, []);

  return { ref, onVolumeChange };
};

export default useVideoMute;
