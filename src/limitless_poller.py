"""Background poller for Limitless API lifelogs.

Runs inside the FastAPI server process, reusing the already-loaded Pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from src.pipeline import Pipeline

logger = logging.getLogger(__name__)

API_BASE = "https://api.limitless.ai/v1"
STATE_FILE = Path("output/.limitless_state.json")
JST = timezone(timedelta(hours=9))


def _get_api_key() -> str | None:
    key = os.environ.get("LIMITLESS_API_KEY", "")
    if not key:
        for env_path in (Path(".env"), Path(__file__).resolve().parent.parent / ".env"):
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("LIMITLESS_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        break
            if key:
                break
    return key or None


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


async def _fetch_lifelogs(
    client: httpx.AsyncClient, api_key: str, date_str: str, limit: int = 50
) -> list[dict]:
    headers = {"X-API-Key": api_key}
    params = {
        "timezone": "Asia/Tokyo",
        "limit": limit,
        "direction": "asc",
        "includeMarkdown": "false",
        "includeHeadings": "false",
        "date": date_str,
    }
    all_logs: list[dict] = []
    cursor = None

    while True:
        if cursor:
            params["cursor"] = cursor
        resp = await client.get(f"{API_BASE}/lifelogs", headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        lifelogs = data.get("data", {}).get("lifelogs", [])
        if not lifelogs:
            break
        all_logs.extend(lifelogs)
        cursor = data.get("data", {}).get("nextCursor")
        if not cursor:
            break

    return all_logs


async def _download_audio(
    client: httpx.AsyncClient, api_key: str, start_ms: int, end_ms: int, out_path: Path
) -> None:
    headers = {"X-API-Key": api_key}
    params = {"startMs": start_ms, "endMs": end_ms}
    async with client.stream("GET", f"{API_BASE}/download-audio", headers=headers, params=params) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            async for chunk in resp.aiter_bytes():
                f.write(chunk)


async def _poll_once(pipeline: Pipeline, api_key: str) -> int:
    """Run one poll cycle. Returns number of processed lifelogs."""
    state = _load_state()
    last_processed_id = state.get("last_processed_id")
    last_processed_time = state.get("last_processed_time")

    now = datetime.now(JST)
    if last_processed_time:
        start_date = (datetime.fromisoformat(last_processed_time) - timedelta(hours=1)).date()
    else:
        start_date = (now - timedelta(days=1)).date()
    end_date = now.date()

    async with httpx.AsyncClient(timeout=120) as client:
        # Fetch day by day
        all_lifelogs: list[dict] = []
        current_date = start_date
        while current_date <= end_date:
            logs = await _fetch_lifelogs(client, api_key, current_date.strftime("%Y-%m-%d"))
            all_lifelogs.extend(logs)
            current_date += timedelta(days=1)

        # Deduplicate
        seen_ids: set[str] = set()
        unique: list[dict] = []
        for log in all_lifelogs:
            if log["id"] not in seen_ids:
                seen_ids.add(log["id"])
                unique.append(log)

        # Skip already-processed
        if last_processed_id:
            skip = True
            filtered = []
            for log in unique:
                if log["id"] == last_processed_id:
                    skip = False
                    continue
                if not skip:
                    filtered.append(log)
            unique = filtered

        if not unique:
            return 0

        logger.info("Found %d new lifelog(s)", len(unique))
        processed = 0

        for log in unique:
            log_id = log["id"]
            start_time = log.get("startTime", "")
            end_time = log.get("endTime", "")

            if not start_time or not end_time:
                continue

            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
            duration_sec = (end_dt - start_dt).total_seconds()

            if duration_sec < 5:
                continue

            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            try:
                await _download_audio(client, api_key, start_ms, end_ms, tmp_path)

                if tmp_path.stat().st_size < 100:
                    logger.warning("Audio too small for %s, skipping", log_id)
                    continue

                t0 = time.time()
                timestamp = start_dt.astimezone(JST)
                # Run blocking pipeline in thread to not block the event loop
                output_path = await asyncio.to_thread(
                    pipeline.process_file, tmp_path, timestamp
                )
                elapsed = time.time() - t0
                logger.info("Processed %s in %.1fs → %s", log_id, elapsed, output_path)
                processed += 1

            except Exception:
                logger.exception("Failed to process lifelog %s", log_id)
                continue
            finally:
                tmp_path.unlink(missing_ok=True)

            state["last_processed_id"] = log_id
            state["last_processed_time"] = datetime.now(JST).isoformat()
            _save_state(state)

    return processed


async def run_poller(pipeline: Pipeline, interval_sec: int = 300) -> None:
    """Long-running poller task. Call from FastAPI lifespan."""
    api_key = _get_api_key()
    if not api_key:
        logger.warning("LIMITLESS_API_KEY not set, poller disabled")
        return

    logger.info("Limitless poller started (interval=%ds)", interval_sec)
    while True:
        try:
            count = await _poll_once(pipeline, api_key)
            if count:
                logger.info("Poll cycle done: %d processed", count)
        except Exception:
            logger.exception("Poll cycle error")
        await asyncio.sleep(interval_sec)
