"""fetching._decide is what turns a daily cron run cheap: it must correctly
tell 'unchanged, skip' from 'changed, download' using only HEAD validators."""
from pathlib import Path

from fetching import _decide


def test_no_local_file_always_fetches(tmp_path):
    changed, reason = _decide("x", tmp_path / "missing.csv", {}, {"etag": "abc"}, True)
    assert changed is True
    assert "no local copy" in reason


def test_no_recorded_state_fetches_even_if_file_exists(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("x")
    changed, reason = _decide("x", p, {}, {"etag": "abc"}, True)
    assert changed is True


def test_matching_etag_skips(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("x")
    changed, reason = _decide("x", p, {"etag": "abc"}, {"etag": "abc"}, True)
    assert changed is False
    assert "unchanged" in reason


def test_changed_etag_fetches(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("x")
    changed, reason = _decide("x", p, {"etag": "abc"}, {"etag": "def"}, True)
    assert changed is True
    assert "ETag changed" in reason


def test_falls_back_to_last_modified_when_no_etag(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("x")
    seen = {"last_modified": "Mon, 01 Jan 2026 00:00:00 GMT"}
    current = {"last_modified": "Tue, 02 Jan 2026 00:00:00 GMT"}
    changed, reason = _decide("x", p, seen, current, True)
    assert changed is True
    assert "Last-Modified" in reason


def test_no_comparable_validators_respects_fetch_when_unverifiable(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("x")
    seen = {"etag": "abc"}
    current = {}  # server sent nothing this time
    changed, _ = _decide("x", p, seen, current, True)
    assert changed is True
    changed, _ = _decide("x", p, seen, current, False)
    assert changed is False
