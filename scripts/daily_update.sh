#!/usr/bin/env bash
# Daily cron entry point: fetch + fred + (mine/forecast/backtest if changed) + report.
# Runs under flock so an overrunning job can't overlap with the next day's trigger.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

LOCKFILE="/tmp/real_estate_pipeline.lock"
LOGFILE="data/pipeline.log"

# Invoke the in-project venv directly. cron's non-interactive PATH does not
# include ~/.local/bin, so calling `poetry` here fails with "command not found"
# and the whole run dies before it downloads anything.
PYTHON="$PROJECT_DIR/.venv/bin/python"

exec 200>"$LOCKFILE"
if ! flock -n 200; then
    echo "$(date -Iseconds) — previous run still in progress, skipping" >> "$LOGFILE"
    exit 0
fi

{
    echo "===== $(date -Iseconds) starting daily pipeline run ====="
    # No --force: `fetch` is change-detected (ETag/Last-Modified), so a daily
    # run costs a handful of HEAD requests on days nothing changed, and only
    # re-downloads a source that actually moved. `all` likewise skips the
    # expensive mine/forecast/backtest steps unless a bulk source changed.
    # A failed run is pushed to whatever channel is configured in
    # config.toml's [alerts] (see notify.py) before this script also exits
    # non-zero, so cron's own mail-on-failure (if configured) still fires too.
    "$PYTHON" pipeline.py all
    echo "===== $(date -Iseconds) finished ====="
} >> "$LOGFILE" 2>&1
