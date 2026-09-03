import sys
import os
from datetime import datetime
from toolkit.accelerator import get_accelerator

# Captured once at import time, before any Logger wrapping happens. Reusing
# sys.stdout/stderr directly in Logger.__init__ would chain through a previous
# Logger on every subsequent setup_log_to_file() call (persistent process
# handing off between jobs), duplicating writes into every prior job's log file.
_REAL_STDOUT = sys.stdout
_REAL_STDERR = sys.stderr

# The trainer's stdout is a pipe, so Python picks the Windows locale encoding
# (cp1252) rather than UTF-8. Any print carrying a non-ASCII character then
# raises UnicodeEncodeError -- and because prints happen inside end_step_hook,
# that exception propagates out of the training loop and kills the job. A
# single U+26A0 in a loss-spike alert did exactly that at step ~1250 of
# sdxl_speedo_tan_line. Switch the real streams to replacement errors so an
# unencodable character degrades to '?' instead of ending a run. Done at import
# time so it also covers output written before setup_log_to_file() wraps them,
# and direct writes to the real streams by library code.
for _stream in (_REAL_STDOUT, _REAL_STDERR):
    try:
        _stream.reconfigure(errors='replace')
    except Exception:
        # Not a TextIOWrapper (already wrapped, or redirected to something
        # exotic). Logger.write's own fallback still covers it.
        pass


def print_acc(*args, **kwargs):
    if get_accelerator().is_local_main_process:
        print(*args, **kwargs)


def timing_enabled():
    """Startup/phase timing output is opt-in via AITK_PROFILE_STARTUP=1.

    Useful when chasing where time-to-first-step goes, but noise in a normal
    run, so it stays off unless asked for. Read at call time rather than import
    time so it can be toggled for a single job without a restart.
    """
    return os.environ.get('AITK_PROFILE_STARTUP', '0').lower() in ('1', 'true', 'yes')


def print_timing(*args, **kwargs):
    """print_acc, but only when AITK_PROFILE_STARTUP is set."""
    if timing_enabled():
        print_acc(*args, **kwargs)


class Logger:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log = log_file
        self._at_line_start = True

    def _stamp(self, message):
        """Prefix each new line written to the log file with a wall clock time.

        Without this, working out where startup time goes means hand-adding
        timers to the code and restarting the job for every question. With it,
        the gap between any two log lines is readable directly.

        Only the log file is stamped, not the terminal, and only real line
        starts are: tqdm redraws its progress bars with '\\r' and no newline, so
        splitting on '\\r' too would stamp every single progress tick.
        """
        if not message:
            return message
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        parts = message.split('\n')
        out = []
        for i, part in enumerate(parts):
            if self._at_line_start and part.strip():
                out.append(f'[{ts}] {part}')
            else:
                out.append(part)
            if i != len(parts) - 1:
                out.append('\n')
                self._at_line_start = True
            else:
                self._at_line_start = (part == '')
        return ''.join(out)

    def write(self, message):
        """Write to both the console and the log file, never raising.

        A diagnostic must not be able to abort training. The streams can still
        reject a character (a terminal whose encoding cannot be reconfigured,
        for instance), so each write falls back to an ASCII-safe form of the
        message rather than letting the error escape into the training loop.
        """
        self._safe_write(self.terminal, message)
        self._safe_write(self.log, self._stamp(message))
        try:
            self.log.flush()  # Make sure it's written immediately
        except Exception:
            pass

    @staticmethod
    def _safe_write(stream, message):
        try:
            stream.write(message)
        except UnicodeEncodeError:
            encoding = getattr(stream, 'encoding', None) or 'ascii'
            try:
                stream.write(
                    message.encode(encoding, errors='replace').decode(encoding, errors='replace')
                )
            except Exception:
                pass
        except Exception:
            pass

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()


def setup_log_to_file(filename):
    if get_accelerator().is_local_main_process:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
    # Close the previous log file handle before replacing it (persistent process
    # handing off between jobs would otherwise leak one fd per job forever).
    # Both wrappers share a single handle, so closing stdout's covers stderr too.
    if isinstance(sys.stdout, Logger):
        try:
            sys.stdout.log.close()
        except Exception:
            pass
    # Wrap the real streams captured at import time — wrapping the
    # already-replaced sys.stdout would chain through the previous Logger and
    # double-write every message into every prior job's log file.
    # Explicit UTF-8: without it the log file inherits the same cp1252 locale
    # encoding as the console and rejects the exact characters the console
    # rejects, so hardening only the terminal write would just move the crash
    # one line down.
    log_file = open(filename, 'a', encoding='utf-8', errors='replace')
    sys.stdout = Logger(_REAL_STDOUT, log_file)
    sys.stderr = Logger(_REAL_STDERR, log_file)
