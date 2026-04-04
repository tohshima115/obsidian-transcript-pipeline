"""FastAPI server wrapping the ML pipeline.

Endpoints:
  POST /process    — full pipeline on conversation audio
  GET  /health     — health check

Background tasks:
  Limitless API poller — checks every 5 minutes for new lifelogs
"""

from __future__ import annotations

import asyncio
import struct
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from src.config import load_config
from src.limitless_poller import run_poller
from src.pipeline import Pipeline

_pipeline: Pipeline | None = None
_poller_task: asyncio.Task | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        config = load_config()
        _pipeline = Pipeline(config)
    return _pipeline


def pcm16_bytes_to_numpy(data: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Convert raw PCM16 little-endian bytes to float32 numpy array."""
    n_samples = len(data) // 2
    samples = struct.unpack(f"<{n_samples}h", data[: n_samples * 2])
    return np.array(samples, dtype=np.float32) / 32768.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _poller_task
    pipeline = get_pipeline()
    _poller_task = asyncio.create_task(run_poller(pipeline))
    yield
    _poller_task.cancel()
    try:
        await _poller_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Transcripts Pipeline API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/process")
async def process(
    request: Request,
    uid: str = Query(default="default"),
    start_at: str = Query(default=""),
    sample_rate: int = Query(default=16000),
):
    """Process finalized conversation audio through the full pipeline."""
    body = await request.body()
    if len(body) < 1000:
        return JSONResponse(
            {"error": "Audio too short"}, status_code=400
        )

    audio = pcm16_bytes_to_numpy(body, sample_rate)

    if start_at:
        try:
            timestamp = datetime.fromisoformat(start_at)
        except ValueError:
            timestamp = datetime.now()
    else:
        timestamp = datetime.now()

    pipeline = get_pipeline()
    output_path = pipeline.process_audio(audio, sample_rate, timestamp)

    return {
        "status": "ok",
        "output_path": str(output_path),
        "uid": uid,
    }
