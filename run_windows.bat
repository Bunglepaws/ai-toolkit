@echo off&&cd /d %~dp0
REM Update-and-run script for Windows - thin bootstrap over the in-repo manager.
REM
REM Everything (venv via uv-managed Python, torch for your GPU, requirements,
REM portable Node.js / FFmpeg / Git, dependency updates) is handled by
REM `python -m manager`; this script only makes sure uv + a Python interpreter
REM exist, then delegates.
setlocal EnableDelayedExpansion
Title AI Toolkit

echo.
echo      _     ___   _____               _  _     _  _
echo     / \   ^|_ _^| ^|_   _^|  ___    ___ ^| ^|^| ^| __(_)^| ^|_
echo    / _ \   ^| ^|    ^| ^|   / _ \  / _ \^| ^|^| ^|/ /^| ^|^| __^|
echo   / ___ \  ^| ^|    ^| ^|  ^| (_) ^|^| (_) ^| ^|^|   ^< ^| ^|^| ^|_
echo  /_/   \_\^|___^|   ^|_^|   \___/  \___/^|_^|^|_^|\_\^|_^| \__^|
echo.
echo   AI Toolkit Manager - Windows
echo.

REM If the UI is already up (someone already ran this bat, or the worker/UI
REM survived a crashed console), don't tear it down and rebuild on top of it.
REM Ask what to do instead of exiting silently - a window that vanishes before
REM it can be read looks identical to the launcher being broken.
for /f %%S in ('curl -s -o nul -w "%%{http_code}" http://localhost:8675 2^>nul') do set "PORT_CHECK=%%S"
if "%PORT_CHECK%"=="200" goto :already_running

:start_fresh

REM Testing expandable_segments to fix early OOM on the uncut H3 job under
REM torch 2.13.0+cu130 (native Windows). Remove if it causes the async
REM "device not ready" failure seen previously on a different job.
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Clear env vars that let a stray conda/pyenv/system Python hijack things
set PYTHONPATH=
set PYTHONHOME=
set PYTHONSTARTUP=
set PYTHONUSERBASE=
set PIP_CONFIG_FILE=
set VIRTUAL_ENV=
set CONDA_PREFIX=
set CONDA_DEFAULT_ENV=
set PYENV_ROOT=
set PYENV_VERSION=

REM ---- 1. Ensure uv (prebuilt static binary, kept inside the repo) ----
set "PATH=%~dp0.uv;%PATH%"
set "UV_PYTHON_INSTALL_DIR=%~dp0.uv\python"
where uv.exe >nul 2>&1
if errorlevel 1 (
    echo Downloading uv ^(package/python manager^) into .uv\ ...
    powershell -NoProfile -ExecutionPolicy ByPass -Command ^
        "$env:UV_INSTALL_DIR = Join-Path '%~dp0' '.uv'; $env:UV_NO_MODIFY_PATH = '1'; irm https://astral.sh/uv/install.ps1 | iex"
    where uv.exe >nul 2>&1
    if errorlevel 1 (
        echo ERROR: uv download failed. See https://docs.astral.sh/uv/
        pause
        exit /b 1
    )
)

REM ---- 2. Find a Python to run the manager (stdlib-only, needs 3.9+) ----
set "PY="
for %%C in (python.exe py.exe) do (
    if not defined PY (
        %%C -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
        if not errorlevel 1 set "PY=%%C"
    )
)
if not defined PY (
    echo No system Python found - provisioning one with uv...
    uv python install 3.12
    for /f "delims=" %%P in ('uv python find 3.12') do set "PY=%%P"
)
if not defined PY (
    echo ERROR: could not find or install a Python interpreter.
    pause
    exit /b 1
)

REM ---- 3. Check for upstream updates (like run_ai_toolkit.sh used to) ----
REM `manager update` only does a fast-forward pull, which can never succeed on
REM this heavily-diverged fork (it has hundreds of commits origin doesn't have)
REM - it would just skip silently or die. This just fetches and reports what's
REM new; it does NOT attempt to merge - batch can't resolve conflicts, and the
REM old Claude-merge-from-bash automation never actually worked. Bring it to a
REM Claude Code session to merge when this shows updates are available.
git rev-parse --is-inside-work-tree >nul 2>&1
if not errorlevel 1 (
    echo.
    echo ---- Checking for updates from origin/main...
    git fetch origin main --quiet
    if errorlevel 1 (
        echo ---- Could not reach origin - skipping update check.
    ) else (
        for /f %%C in ('git rev-list HEAD..origin/main --count 2^>nul') do set "BEHIND_COUNT=%%C"
        if defined BEHIND_COUNT if not "!BEHIND_COUNT!"=="0" (
            echo.
            echo ---- !BEHIND_COUNT! update^(s^) available from origin/main - ask Claude to merge them:
            git log --oneline HEAD..origin/main
            echo.
        ) else (
            echo ---- Already up to date with origin/main.
        )
    )
)

REM ---- 4. Sync dependencies (no git pull - handled above) and start the UI ----
"%PY%" -m manager sync
if errorlevel 1 (
    echo.
    echo Setup failed - see output above.
    pause
    exit /b 1
)
"%PY%" -m manager launch
pause
exit /b 0

REM ---------------------------------------------------------------------------
REM Already-running branch
REM ---------------------------------------------------------------------------
:already_running
echo.
echo   AI Toolkit is already running at http://localhost:8675
echo.
echo   The launcher will not rebuild on top of a live stack, so there is
echo   nothing to start. Pick what you want to do:
echo.
echo     [O]  Open it in your browser        ^(default^)
echo     [R]  Restart - stop it, then start fresh
echo     [X]  Leave it running, close this window
echo.
set "CHOICE="
set /p "CHOICE=Choose [O/R/X]: "
if /i "!CHOICE!"=="R" goto :stop_running
if /i "!CHOICE!"=="X" exit /b 0
start http://localhost:8675
exit /b 0

:stop_running
echo.
echo ---- Stopping the running AI Toolkit...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop_ai_toolkit.ps1"
if errorlevel 1 (
    echo.
    echo Could not stop it - see above. Nothing was started.
    pause
    exit /b 1
)
echo.
goto :start_fresh
