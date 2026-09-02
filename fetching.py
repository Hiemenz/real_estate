#!/usr/bin/env python3
"""Change-detected downloading for the bulk housing sources.

The Realtor.com and Redfin exports are monthly publications republished in place
under a fixed filename. The old `fetch --force` re-downloaded ~2.4 GB on every
run regardless of whether anything changed, while plain `fetch` skipped any file
already on disk and so never picked up a new month at all.

Both origins are S3 and send ETag + Last-Modified + Content-Length. Polling those
with a HEAD costs about a kilobyte, so a daily job can check every source and
download only what actually moved.
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)


@dataclass
class FetchResult:
    name: str
    changed: bool
    reason: str
    bytes_downloaded: int = 0
    validators: dict = field(default_factory=dict)

    @property
    def downloaded(self) -> bool:
        return self.bytes_downloaded > 0


def _read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        log.warning(f"{path.name} is corrupt — treating every source as changed")
        return {}


def _write_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def probe(url: str, timeout: int = 30) -> dict:
    """HEAD a source and return whichever freshness validators it exposes."""
    r = requests.head(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    validators = {}
    for header, key in (("ETag", "etag"), ("Last-Modified", "last_modified"),
                        ("Content-Length", "content_length")):
        if header in r.headers:
            validators[key] = r.headers[header]
    return validators


def _decide(name: str, path: Path, seen: dict, current: dict,
            fetch_when_unverifiable: bool) -> tuple[bool, str]:
    """Compare freshly probed validators against the last-seen ones."""
    if not path.exists():
        return True, "no local copy"
    if not seen:
        return True, "no recorded state for this source"

    # ETag is the strongest signal; fall back through the weaker ones. Size alone
    # is last-resort — a same-size republish would slip past it, but it is better
    # than nothing when an origin sends no validators.
    for key, label in (("etag", "ETag"), ("last_modified", "Last-Modified"),
                       ("content_length", "size")):
        if key in current and key in seen:
            if current[key] != seen[key]:
                return True, f"{label} changed ({seen[key]} → {current[key]})"
            return False, f"{label} unchanged ({current[key]})"

    if fetch_when_unverifiable:
        return True, "source sent no comparable validators"
    return False, "source sent no comparable validators (configured to skip)"


def download(url: str, path: Path, timeout: int = 900) -> int:
    """Stream to a temp file and move into place, so an interrupted download
    can't leave a truncated file that later looks like a valid local copy."""
    tmp = path.with_suffix(path.suffix + ".part")
    log.info(f"GET {url}")
    written = 0
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    written += len(chunk)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    log.info(f"  saved {path}  ({written / 1e6:.1f} MB)")
    return written


def sync_source(name: str, src: dict, state_path: Path, cfg: dict,
                force: bool = False, validate=None) -> FetchResult:
    """Probe one source and download it only if it changed (or force=True)."""
    path: Path = src["path"]
    state = _read_state(state_path)
    seen = state.get(name, {})

    if force:
        current, changed, reason = {}, True, "forced"
        try:
            current = probe(src["url"], timeout=cfg["head_timeout"])
        except requests.RequestException as exc:
            log.warning(f"  HEAD failed for {name} ({exc}) — downloading anyway")
    else:
        try:
            current = probe(src["url"], timeout=cfg["head_timeout"])
        except requests.RequestException as exc:
            # A HEAD failure must not silently mean "unchanged" — that is the
            # exact way stale data goes unnoticed.
            log.warning(f"  HEAD failed for {name}: {exc}")
            if path.exists():
                return FetchResult(name, False, f"probe failed, keeping local copy: {exc}")
            raise
        changed, reason = _decide(name, path, seen, current,
                                  cfg.get("fetch_when_unverifiable", True))

    if not changed:
        log.info(f"Skip {name} — {reason}")
        return FetchResult(name, False, reason, validators=current)

    log.info(f"Fetch {name} — {reason}")
    written = download(src["url"], path, timeout=cfg["download_timeout"])

    if validate is not None:
        # Validate before recording state, so a schema break is re-detected on
        # the next run instead of being marked as successfully ingested.
        validate(path, src["required_columns"], src["sep"])

    state[name] = {**current, "fetched_at": datetime.now(timezone.utc).isoformat()}
    _write_state(state_path, state)
    return FetchResult(name, True, reason, bytes_downloaded=written, validators=current)
