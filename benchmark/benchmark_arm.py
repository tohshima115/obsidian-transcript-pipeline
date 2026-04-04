"""ARM CPU benchmark for transcripts-pipeline models.

Measures memory usage and STT processing speed under ARM64.
When run via QEMU emulation, memory numbers are accurate but
timing is 5-20x slower than native ARM hardware.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np


def get_rss_mb() -> float:
    """Current RSS in MB from /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    return 0.0


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target_sr mono via soundfile."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Resample if needed
    if sr != target_sr:
        import torch
        import torchaudio

        waveform = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        audio = waveform.squeeze().numpy()
    return audio


def pad_or_trim(audio: np.ndarray, target_seconds: float, sr: int = 16000) -> np.ndarray:
    """Pad (by tiling) or trim audio to target duration."""
    target_len = int(target_seconds * sr)
    if len(audio) >= target_len:
        return audio[:target_len]
    # Tile to reach target length
    repeats = (target_len // len(audio)) + 1
    return np.tile(audio, repeats)[:target_len]


def find_sample_audio() -> str | None:
    """Find a usable sample audio file."""
    candidates = [
        "sample.wav",
        "samples/sample_me_01.m4a",
        "samples/sample_me_02.m4a",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def benchmark_vad(results: dict) -> object:
    """Load Silero VAD and measure memory."""
    from src.config import VadConfig
    from src.vad import VadProcessor

    config = VadConfig(
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        window_size_samples=512,
        merge_max_gap_s=0.3,
    )
    before = get_rss_mb()
    vad = VadProcessor(config)
    after = get_rss_mb()
    results["vad_delta_mb"] = round(after - before, 1)
    results["after_vad_mb"] = round(after, 1)
    print(f"  VAD loaded: +{results['vad_delta_mb']} MB (total {after:.0f} MB)")
    return vad


def benchmark_speaker_id(results: dict) -> object:
    """Load ECAPA-TDNN speaker identification and measure memory."""
    from src.config import SpeakerIdConfig
    from src.speaker_id import SpeakerIdentifier
    from src.speaker_registry import SpeakerRegistry

    config = SpeakerIdConfig(
        model_name="speechbrain/spkrec-ecapa-voxceleb",
        known_threshold=0.55,
        unknown_threshold=0.25,
        min_segment_duration_s=0.5,
        device="cpu",
    )
    # Empty registry for benchmark (only need model load, not identification)
    registry = SpeakerRegistry.__new__(SpeakerRegistry)
    registry.speakers_dir = None
    registry.registry_file = None
    registry._embeddings = {}
    registry._entries = {}

    before = get_rss_mb()
    speaker_id = SpeakerIdentifier(config, registry)
    after = get_rss_mb()
    results["speaker_id_delta_mb"] = round(after - before, 1)
    results["after_speaker_id_mb"] = round(after, 1)
    print(f"  ECAPA-TDNN loaded: +{results['speaker_id_delta_mb']} MB (total {after:.0f} MB)")
    return speaker_id


def benchmark_stt(results: dict, dtype_str: str, audio: np.ndarray, sr: int = 16000) -> None:
    """Load STT model and benchmark transcription speed."""
    import gc

    import torch

    from src.config import SttConfig
    from src.stt import SttProcessor

    config = SttConfig(
        model_name="Qwen/Qwen3-ASR-1.7B",
        chunk_length_s=30,
        stride_length_s=(4, 4),
        language="ja",
        device="cpu",
        torch_dtype=dtype_str,
    )

    prefix = f"stt_{dtype_str}"

    print(f"\n  Loading STT (Qwen3-ASR-1.7B, {dtype_str})...")
    before = get_rss_mb()
    t0 = time.monotonic()

    try:
        stt = SttProcessor(config)
    except Exception as e:
        print(f"  FAILED to load Qwen3-ASR: {e}")
        results[f"{prefix}_load_error"] = str(e)
        # Try Whisper fallback
        print("  Trying Whisper fallback (openai/whisper-large-v3-turbo)...")
        config = SttConfig(
            model_name="openai/whisper-large-v3-turbo",
            chunk_length_s=30,
            stride_length_s=(4, 4),
            language="ja",
            device="cpu",
            torch_dtype=dtype_str,
        )
        prefix = f"whisper_{dtype_str}"
        try:
            stt = SttProcessor(config)
        except Exception as e2:
            print(f"  Whisper also FAILED: {e2}")
            results[f"{prefix}_load_error"] = str(e2)
            return

    load_time = time.monotonic() - t0
    after = get_rss_mb()
    results[f"{prefix}_delta_mb"] = round(after - before, 1)
    results[f"{prefix}_total_mb"] = round(after, 1)
    results[f"{prefix}_load_time_s"] = round(load_time, 1)
    print(f"  STT loaded: +{results[f'{prefix}_delta_mb']} MB (total {after:.0f} MB, {load_time:.0f}s)")

    # Benchmark inference
    audio_duration = len(audio) / sr
    print(f"  Running STT on {audio_duration:.0f}s audio...")
    t0 = time.monotonic()
    segments = stt.transcribe(audio, sr)
    elapsed = time.monotonic() - t0
    results[f"{prefix}_inference_s"] = round(elapsed, 1)
    results[f"{prefix}_audio_duration_s"] = round(audio_duration, 1)
    results[f"{prefix}_realtime_factor"] = round(elapsed / audio_duration, 2)
    results[f"{prefix}_peak_rss_mb"] = round(get_rss_mb(), 1)

    text_preview = segments[0].text[:100] if segments else "(no output)"
    print(f"  Inference: {elapsed:.1f}s for {audio_duration:.0f}s audio (RTF={elapsed / audio_duration:.2f}x)")
    print(f"  Output preview: {text_preview}")
    print(f"  Peak RSS: {get_rss_mb():.0f} MB")

    # Cleanup to test next dtype
    del stt
    gc.collect()
    if hasattr(torch, "cuda"):
        pass  # CPU only, no CUDA cleanup needed


def main() -> None:
    print("=" * 60)
    print("ARM CPU Benchmark — transcripts-pipeline")
    print("=" * 60)

    # Check if running under QEMU
    try:
        uname = os.uname()
        is_emulated = uname.machine == "aarch64" and os.path.exists("/proc/cpuinfo")
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        if "QEMU" in cpuinfo or "emulat" in cpuinfo.lower():
            print("\nWARNING: Running under QEMU emulation.")
            print("  - Memory measurements are ACCURATE")
            print("  - Timing is 5-20x SLOWER than native ARM")
            print("  - Divide realtime_factor by ~10 for native estimate\n")
    except Exception:
        pass

    results: dict = {}
    results["baseline_mb"] = round(get_rss_mb(), 1)
    print(f"\nBaseline RSS: {results['baseline_mb']} MB")

    # Find test audio
    audio_path = find_sample_audio()
    if not audio_path:
        print("ERROR: No sample audio found in samples/")
        sys.exit(1)
    print(f"Using audio: {audio_path}")
    audio = load_audio(audio_path)
    audio_60s = pad_or_trim(audio, 60.0)
    results["test_audio_original_s"] = round(len(audio) / 16000, 1)
    print(f"Original duration: {results['test_audio_original_s']}s, padded to 60s")

    # Step 1: VAD
    print("\n--- Silero VAD ---")
    benchmark_vad(results)

    # Step 2: Speaker ID
    print("\n--- ECAPA-TDNN ---")
    benchmark_speaker_id(results)

    # Step 3: STT (float16 — primary target for 8GB RAM)
    print("\n--- STT (float16) ---")
    benchmark_stt(results, "float16", audio_60s)

    # Step 4: STT (float32 — comparison)
    print("\n--- STT (float32) ---")
    benchmark_stt(results, "float32", audio_60s)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS (JSON)")
    print("=" * 60)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
