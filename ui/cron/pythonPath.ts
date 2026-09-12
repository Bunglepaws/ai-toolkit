import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT } from './paths';

const isWindows = process.platform === 'win32';

// Shared resolver used by both the cron worker and Next.js API routes
// so the Python interpreter is configured in exactly one place.
export const resolvePythonPath = (): string => {
  // An explicit override wins. run_ai_toolkit.sh sets AITK_PYTHON because the
  // venv lives on the distro's ext4 rather than under /mnt/c: importing torch +
  // transformers + diffusers costs ~67s across the drvfs bridge versus ~2s
  // native (56k small files, drvfs's worst case).
  //
  // A symlink at <repo>/venv is NOT sufficient — Python keeps the invocation
  // path as sys.prefix, so resolving through /mnt/c still translates every
  // site-packages read across the bridge (measured 30s vs 2s). The interpreter
  // has to be invoked by its real path, which is what this override carries.
  const override = process.env.AITK_PYTHON;
  if (override && fs.existsSync(override)) {
    return override;
  }

  const candidates: string[] = [];

  if (isWindows) {
    candidates.push(path.join(TOOLKIT_ROOT, '.venv', 'Scripts', 'python.exe'));
    candidates.push(path.join(TOOLKIT_ROOT, 'venv', 'Scripts', 'python.exe'));
  } else {
    candidates.push(path.join(TOOLKIT_ROOT, '.venv', 'bin', 'python'));
    candidates.push(path.join(TOOLKIT_ROOT, 'venv', 'bin', 'python'));
  }

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return isWindows ? 'python.exe' : 'python3';
};

// Interpreter for jobs we detach so they outlive the UI. On Windows detaching
// means DETACHED_PROCESS, and Windows gives any *console* app started that way
// a fresh console with a visible window. pythonw.exe is the GUI-subsystem
// build, so it never gets one; stdout/stderr still go to the handles we pass.
export const resolveDetachedPythonPath = (): string => {
  const pythonPath = resolvePythonPath();
  if (!isWindows) return pythonPath;

  // Always take the pythonw next to the interpreter we already resolved, so
  // both come from the same environment. Falling back to python.exe still
  // works, it just shows a console window.
  const pythonwPath = path.join(path.dirname(pythonPath), 'pythonw.exe');
  return fs.existsSync(pythonwPath) ? pythonwPath : pythonPath;
};
