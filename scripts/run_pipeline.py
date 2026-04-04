#!/usr/bin/env python3
"""CLI entry point for running the transcription pipeline on audio files."""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcription pipeline")
    parser.add_argument("audio", help="Path to audio file (WAV, PCM16, 16kHz, mono)")
    parser.add_argument("--config", default="config.toml", help="Config file path")
    parser.add_argument(
        "--timestamp",
        help="Conversation timestamp (YYYY-MM-DD HH:MM). Defaults to now.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    timestamp = datetime.now()
    if args.timestamp:
        timestamp = datetime.strptime(args.timestamp, "%Y-%m-%d %H:%M")

    pipeline = Pipeline(config)

    print(f"\nProcessing: {args.audio}")
    t0 = time.time()
    output_path = pipeline.process_file(args.audio, timestamp)
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.1f}s")
    print(f"Output: {output_path}")
    print()
    print(output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
