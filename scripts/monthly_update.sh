#!/usr/bin/env bash
# Monthly cron entry point: fetch + mine + forecast + report.
# Runs under flock so an overrunning job can't overlap with the next month's trigger.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

LOCKFILE="/tmp/real_estate_pipeline.lock"
LOGFILE="data/pipeline.log"

exec 200>"$LOCKFILE"
if ! flock -n 200; then
    echo "$(date -Iseconds) — previous run still in progress, skipping" >> "$LOGFILE"
    exit 0
fi

{
    echo "===== $(date -Iseconds) starting monthly pipeline run ====="
    # --force: Realtor.com republishes these filenames in place each month (not
    # incremental), so without --force `fetch` would see last month's file already
    # on disk and skip re-downloading it.
    poetry run python pipeline.py all --force
    echo "===== $(date -Iseconds) finished ====="
} >> "$LOGFILE" 2>&1
