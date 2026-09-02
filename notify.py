#!/usr/bin/env python3
"""Failure alerting for pipeline runs.

Four consecutive cron failures went unnoticed for five weeks because the only
evidence was a line in data/pipeline.log. Any non-"none" channel configured in
config.toml's [alerts] turns that into a push.

Notification failures are swallowed deliberately: an unreachable ntfy server
must not take down an otherwise healthy pipeline run.
"""
import json
import logging
import subprocess

import requests

log = logging.getLogger(__name__)


def notify(cfg: dict, title: str, message: str, level: str = "error") -> bool:
    """Send an alert over the configured channel. Returns True if it went out."""
    channel = cfg.get("channel", "none")
    if channel == "none":
        return False

    try:
        if channel == "ntfy":
            topic = cfg.get("ntfy_topic", "").strip()
            if not topic:
                log.warning("alerts.channel is 'ntfy' but ntfy_topic is empty — no alert sent")
                return False
            server = cfg.get("ntfy_server", "https://ntfy.sh").rstrip("/")
            requests.post(
                f"{server}/{topic}",
                data=message.encode("utf-8"),
                headers={
                    "Title": title,
                    # ntfy maps these to a phone-level priority + icon.
                    "Priority": "high" if level == "error" else "default",
                    "Tags": "rotating_light" if level == "error" else "white_check_mark",
                },
                timeout=15,
            ).raise_for_status()

        elif channel == "webhook":
            url = cfg.get("webhook_url", "").strip()
            if not url:
                log.warning("alerts.channel is 'webhook' but webhook_url is empty — no alert sent")
                return False
            requests.post(
                url,
                json={"title": title, "message": message, "level": level},
                timeout=15,
            ).raise_for_status()

        elif channel == "command":
            cmd = cfg.get("command", "").strip()
            if not cmd:
                log.warning("alerts.channel is 'command' but command is empty — no alert sent")
                return False
            subprocess.run(
                cmd, shell=True, input=f"{title}\n\n{message}",
                text=True, timeout=30, check=True,
            )

        else:
            log.warning(f"Unknown alerts.channel {channel!r} — no alert sent")
            return False

    except Exception as exc:
        log.warning(f"Alert delivery failed ({channel}): {exc}")
        return False

    log.info(f"Alert sent via {channel}: {title}")
    return True


def notify_failure(cfg: dict, stage: str, exc: BaseException) -> None:
    notify(
        cfg,
        title=f"Real-estate pipeline failed: {stage}",
        message=f"{type(exc).__name__}: {exc}",
        level="error",
    )


def notify_success(cfg: dict, summary: dict) -> None:
    if not cfg.get("notify_on_success", False):
        return
    notify(
        cfg,
        title="Real-estate pipeline succeeded",
        message=json.dumps(summary, indent=2, default=str),
        level="info",
    )
