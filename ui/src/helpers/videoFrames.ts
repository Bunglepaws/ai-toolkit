// Frame-count math for video sample configs.
//
// Each video architecture only accepts frame counts on its own grid:
//   WAN      fps * seconds + 1   (any count of the form n + 1)
//   LTX-2.x  8n + 1
//   H3       17n + 5
// The UI lets the user think in seconds and snaps the resulting frame count
// onto the model's grid, so the number shown is always one the model will
// actually produce.

export interface FrameGrid {
  // valid counts are `multiple * n + offset` for n >= 0
  multiple: number;
  offset: number;
}

export const defaultFrameGrid: FrameGrid = { multiple: 1, offset: 1 };

/** Snap an arbitrary frame count onto the grid, to the nearest valid value. */
export function snapFrameCount(frames: number, grid: FrameGrid = defaultFrameGrid): number {
  const { multiple, offset } = grid;
  if (!isFinite(frames) || multiple < 1) return offset;
  const n = Math.max(0, Math.round((frames - offset) / multiple));
  return multiple * n + offset;
}

/** Frame count for a duration in seconds at a given fps, snapped to the grid. */
export function durationToFrameCount(
  durationSeconds: number,
  fps: number,
  grid: FrameGrid = defaultFrameGrid,
): number {
  if (!durationSeconds || !fps || durationSeconds <= 0 || fps <= 0) return grid.offset;
  return snapFrameCount(durationSeconds * fps + 1, grid);
}

/** Inverse: the duration a given frame count actually plays for. */
export function frameCountToDuration(frames: number, fps: number): number {
  if (!frames || !fps || fps <= 0) return 0;
  return (frames - 1) / fps;
}

/** Trim float noise for display, e.g. 3.0416666 -> "3.04", 3 -> "3". */
export function formatDuration(seconds: number): string {
  return `${Math.round(seconds * 100) / 100}`;
}

/**
 * fps values we have actually tested per architecture. Anything else still
 * works, we just warn that it is untested.
 */
export function isUntestedFps(fps: number, supported?: number[]): boolean {
  if (!supported || supported.length === 0) return false;
  if (!fps || fps <= 0) return false;
  return !supported.includes(fps);
}
