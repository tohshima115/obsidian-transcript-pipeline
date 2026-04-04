"""FastAPI server wrapping the ML pipeline.

Endpoints:
  POST /vad-check  — quick speech detection on a chunk
  POST /process    — full pipeline on finalized conversation audio
  GET  /health     — health check
"""

from __future__ import annotations

import struct
from datetime import datetime

import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from src.config import load_config
from src.pipeline import Pipeline
from src.vad import VadProcessor

app = FastAPI(title="Transcripts Pipeline API")

# Lazy-loaded globals
_pipeline: Pipeline | None = None
_vad: VadProcessor | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        config = load_config()
        _pipeline = Pipeline(config)
    return _pipeline


def get_vad() -> VadProcessor:
    global _vad
    if _vad is None:
        config = load_config()
        _vad = VadProcessor(config.vad)
    return _vad


def pcm16_bytes_to_numpy(data: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Convert raw PCM16 little-endian bytes to float32 numpy array."""
    n_samples = len(data) // 2
    samples = struct.unpack(f"<{n_samples}h", data[: n_samples * 2])
    return np.array(samples, dtype=np.float32) / 32768.0


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/vad-check")
async def vad_check(
    request: Request,
    sample_rate: int = Query(default=16000),
):
    """Check if an audio chunk contains speech."""
    body = await request.body()
    if len(body) < 4:
        return JSONResponse({"has_speech": False})

    audio = pcm16_bytes_to_numpy(body, sample_rate)
    vad = get_vad()
    has_speech = vad.has_speech(audio, sample_rate)

    return {"has_speech": has_speech}


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
