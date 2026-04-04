#!/usr/bin/env python3
"""Simulate Omi app sending audio chunks to the webhook server.

Reads a WAV file, splits into 60-second PCM16 chunks,
and sends each to the webhook endpoint with a short delay.
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import requests
import soundfile as sf
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw PCM16 little-endian bytes."""
    clamped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clamped * 32767).astype(np.int16)
    return pcm16.tobytes()


def main():
    parser = argparse.ArgumentParser(description="Simulate Omi webhook")
    parser.add_argument("audio", help="Path to WAV file")
    parser.add_argument("--url", default="http://127.0.0.1:3000/audio")
    parser.add_argument("--api-key", default="dev-secret-key")
    parser.add_argument("--uid", default="test-user")
    parser.add_argument("--chunk-sec", type=int, default=60)
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between chunks (seconds)")
    args = parser.parse_args()

    audio, sr = sf.read(args.audio, dtype="float32")
    total_sec = len(audio) / sr
    chunk_samples = args.chunk_sec * sr
    print(f"Audio: {total_sec:.1f}s, {sr}Hz, sending in {args.chunk_sec}s chunks")

    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/octet-stream",
    }

    offset = 0
    chunk_idx = 0
    while offset < len(audio):
        chunk_audio = audio[offset : offset + chunk_samples]
        pcm_bytes = float32_to_pcm16_bytes(chunk_audio)

        chunk_sec = len(chunk_audio) / sr
        print(f"  Chunk {chunk_idx}: {chunk_sec:.1f}s ({len(pcm_bytes)} bytes)...", end=" ")

        resp = requests.post(
            args.url,
            params={"uid": args.uid, "sample_rate": str(sr)},
            headers=headers,
            data=pcm_bytes,
        )
        print(f"→ {resp.status_code} {resp.json()}")

        offset += chunk_samples
        chunk_idx += 1
        if offset < len(audio):
            time.sleep(args.delay)

    print(f"\nAll {chunk_idx} chunks sent.")
    print("Waiting for silence timeout to trigger finalization...")
    print("(Check node server logs: tail -f /tmp/node-server.log)")


if __name__ == "__main__":
    main()
