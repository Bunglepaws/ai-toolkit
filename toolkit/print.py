import sys
import os
from toolkit.accelerator import get_accelerator

# Captured once at import time, before any Logger wrapping happens. Reusing
# sys.stdout/stderr directly in Logger.__init__ would chain through a previous
# Logger on every subsequent setup_log_to_file() call (persistent process
# handing off between jobs), duplicating writes into every prior job's log file.
_REAL_STDOUT = sys.stdout
_REAL_STDERR = sys.stderr


def print_acc(*args, **kwargs):
    if get_accelerator().is_local_main_process:
        print(*args, **kwargs)


class Logger:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Make sure it's written immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()


def setup_log_to_file(filename):
    if get_accelerator().is_local_main_process:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
    # Close previous log file handles before replacing them (persistent process
    # handing off between jobs would otherwise leak one fd per job forever).
    # Both wrappers share a single handle, so closing stdout's covers stderr too.
    if isinstance(sys.stdout, Logger):
        try:
            sys.stdout.log.close()
        except Exception:
            pass
    if isinstance(sys.stderr, Logger):
        try:
            sys.stderr.log.close()
        except Exception:
            pass
    # Capture the real streams before replacing them — wrapping the
    # already-replaced sys.stdout as the stderr Logger's "terminal" would
    # double-write every stderr message to the file. Both wrappers share a
    # single file handle.
    log_file = open(filename, 'a')
    sys.stdout = Logger(_REAL_STDOUT, log_file)
    sys.stderr = Logger(_REAL_STDERR, log_file)
