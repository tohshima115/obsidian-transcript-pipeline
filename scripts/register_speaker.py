#!/usr/bin/env python3
"""Register a speaker by providing audio samples.

Supports registering yourself ("me") or any named speaker.
Validates audio quality and embedding consistency before saving.

Usage:
    # Register yourself
    python scripts/register_speaker.py --name me sample1.wav sample2.wav sample3.wav

    # Register a friend
    python scripts/register_speaker.py --name tanaka sample1.wav sample2.wav

    # Record directly from microphone (requires sounddevice + soundfile)
    python scripts/register_speaker.py --name me --record 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.speaker_id import SpeakerIdentifier
from src.speaker_registry import SpeakerRegistry
from src.vad import VadProcessor

# ─── Recording tips ───────────────────────────────────────────
RECORDING_TIPS = """
╔══════════════════════════════════════════════════════════════╗
║                    録音のコツ / Recording Tips               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. 静かな場所で録音する                                      ║
║     エアコン・換気扇の音もなるべく避ける                        ║
║                                                              ║
║  2. 1サンプルあたり 30〜60秒                                  ║
║     短すぎると embedding 品質が落ちる                          ║
║                                                              ║
║  3. 3サンプル以上が理想                                       ║
║     異なる話題・トーンで録ると精度が上がる                      ║
║     例: (1) 普通の会話調 (2) 説明調 (3) 笑いながら            ║
║                                                              ║
║  4. マイクとの距離を一定に保つ                                 ║
║     20〜30cm くらいが理想                                     ║
║                                                              ║
║  5. 他の人の声が入らないようにする                             ║
║     1人だけの声が入っているのが大事                             ║
║                                                              ║
║  6. WAV 16kHz mono が最適（他の形式でも自動変換される）        ║
║                                                              ║
║  NG例:                                                       ║
║   - BGMが入っている                                           ║
║   - 複数人が同時に話している                                   ║
║   - 5秒未満の短い録音                                         ║
║   - ほぼ無音（咳や「あー」だけ）                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load audio from any common format. Returns (float32 mono array, sample_rate).

    Tries soundfile first (WAV/FLAC/OGG), falls back to pydub+ffmpeg for m4a/mp3/etc.
    """
    try:
        audio, sr = sf.read(str(path), dtype="float32")
    except Exception:
        try:
            from pydub import AudioSegment

            seg = AudioSegment.from_file(str(path))
            seg = seg.set_channels(1)
            sr = seg.frame_rate
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            audio = samples / (2 ** (seg.sample_width * 8 - 1))
            return audio, sr
        except Exception as e:
            print(f"  ✗ {path.name} を読み込めません: {e}")
            print("    対応形式: WAV, FLAC, OGG, MP3, M4A (M4A/MP3はffmpegが必要)")
            print("    ffmpegインストール: sudo apt install ffmpeg")
            raise

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio, sr


def validate_audio(path: Path, vad: VadProcessor) -> tuple[np.ndarray, int, dict]:
    """Validate an audio file and return quality metrics."""
    audio, sr = load_audio(path)

    # Resample to 16kHz if needed
    if sr != 16000:
        import torchaudio

        waveform = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(waveform).squeeze(0).numpy()
        sr = 16000

    duration_s = len(audio) / sr
    segments = vad.detect_speech(audio, sr)
    speech_duration = sum(s.end_s - s.start_s for s in segments)
    speech_ratio = speech_duration / duration_s if duration_s > 0 else 0

    metrics = {
        "duration_s": duration_s,
        "speech_duration_s": speech_duration,
        "speech_ratio": speech_ratio,
        "n_segments": len(segments),
        "rms": float(np.sqrt(np.mean(audio**2))),
    }

    return audio, sr, metrics


def print_quality_report(path: str, metrics: dict) -> list[str]:
    """Print quality report and return list of warnings."""
    warnings = []
    d = metrics

    print(f"\n  {Path(path).name}:")
    print(f"    長さ:     {d['duration_s']:.1f}s")
    print(f"    音声区間: {d['speech_duration_s']:.1f}s ({d['speech_ratio']*100:.0f}%)")
    print(f"    RMS音量:  {d['rms']:.4f}")

    if d["duration_s"] < 5:
        warnings.append("短すぎます（5秒未満）。30秒以上を推奨")
    elif d["duration_s"] < 15:
        warnings.append("やや短いです。30秒以上を推奨")

    if d["speech_ratio"] < 0.3:
        warnings.append("音声が少なすぎます（30%未満）。もっと話してください")

    if d["rms"] < 0.005:
        warnings.append("音量が小さすぎます。マイクに近づいてください")
    elif d["rms"] > 0.5:
        warnings.append("音量が大きすぎます（クリッピングの可能性）")

    if d["n_segments"] == 0:
        warnings.append("音声が検出されませんでした")

    if warnings:
        for w in warnings:
            print(f"    ⚠ {w}")
    else:
        print("    ✓ 品質OK")

    return warnings


def record_samples(n_samples: int, duration_s: int = 45) -> list[Path]:
    """Record audio samples from microphone."""
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice が必要です: pip install sounddevice")
        sys.exit(1)

    paths = []
    for i in range(n_samples):
        input(f"\n  サンプル {i+1}/{n_samples}: Enterを押すと{duration_s}秒間録音します...")
        print(f"  🎙 録音中... ({duration_s}秒)")
        audio = sd.rec(int(duration_s * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        audio = audio.squeeze()

        path = Path(f"/tmp/speaker_sample_{i+1}.wav")
        sf.write(str(path), audio, 16000)
        paths.append(path)
        print(f"  ✓ 保存: {path}")

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a speaker's voice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=RECORDING_TIPS,
    )
    parser.add_argument(
        "--name",
        required=True,
        help='Speaker name. Use "me" for yourself, or any name (e.g., "tanaka")',
    )
    parser.add_argument(
        "samples",
        nargs="*",
        help="Path(s) to WAV audio files of the speaker",
    )
    parser.add_argument(
        "--record",
        type=int,
        metavar="N",
        help="Record N samples from microphone instead of providing files",
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--tips", action="store_true", help="Show recording tips and exit"
    )
    args = parser.parse_args()

    if args.tips:
        print(RECORDING_TIPS)
        return

    if not args.samples and not args.record:
        print("エラー: 音声ファイルを指定するか、--record N で録音してください")
        parser.print_help()
        sys.exit(1)

    print(RECORDING_TIPS)

    config = load_config(args.config)
    name = args.name
    speaker_id = name if name == "me" else name.lower().replace(" ", "_")

    # Get audio files
    if args.record:
        sample_paths = record_samples(args.record)
    else:
        sample_paths = [Path(s) for s in args.samples]
        for p in sample_paths:
            if not p.exists():
                print(f"エラー: ファイルが見つかりません: {p}")
                sys.exit(1)

    # Initialize models
    print("\nモデルをロード中...")
    vad = VadProcessor(config.vad)
    registry = SpeakerRegistry(config.paths)
    registry.load()

    from src.speaker_id import SpeakerIdentifier

    identifier = SpeakerIdentifier(config.speaker_id, registry)

    # Validate and extract embeddings
    print("\n── 音声品質チェック ──")
    embeddings = []
    all_warnings = []

    for path in sample_paths:
        audio, sr, metrics = validate_audio(path, vad)
        warnings = print_quality_report(str(path), metrics)
        all_warnings.extend(warnings)

        if metrics["n_segments"] == 0:
            print(f"    ✗ スキップ（音声なし）")
            continue

        emb = identifier.extract_embedding(audio, sr)
        embeddings.append(emb)

    if not embeddings:
        print("\n✗ 有効な音声サンプルがありませんでした。")
        sys.exit(1)

    # Check embedding consistency
    print("\n── Embedding 一貫性チェック ──")
    if len(embeddings) >= 2:
        sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = F.cosine_similarity(
                    embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)
                ).item()
                sims.append(sim)
                print(f"  サンプル{i+1} ↔ サンプル{j+1}: similarity = {sim:.3f}")

        avg_sim = sum(sims) / len(sims)
        print(f"  平均 similarity: {avg_sim:.3f}")

        if avg_sim < 0.3:
            print("  ⚠ 一貫性が低いです。同一人物の音声であることを確認してください")
            print("    （異なる環境・マイクだと低くなることがあります）")
        elif avg_sim < 0.5:
            print("  △ やや低めですが許容範囲です")
        else:
            print("  ✓ 良好な一貫性です")

    # Average embedding
    avg_embedding = torch.stack(embeddings).mean(dim=0)
    print(f"\n  平均 embedding shape: {avg_embedding.shape}")

    # Save
    registry.save_embedding(speaker_id, avg_embedding)
    registry._ensure_registry_file()

    # Update registry.md for non-me speakers
    if speaker_id != "me":
        from datetime import datetime

        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Check if already in registry
        if speaker_id not in registry.entries:
            with open(config.paths.registry_file, "a", encoding="utf-8") as f:
                f.write(f"| {speaker_id} | {name} | {now} |\n")
        else:
            # Update existing entry name
            text = config.paths.registry_file.read_text(encoding="utf-8")
            import re

            text = re.sub(
                rf"^\|\s*{re.escape(speaker_id)}\s*\|.*\|.*\|",
                f"| {speaker_id} | {name} | {now} |",
                text,
                flags=re.MULTILINE,
            )
            config.paths.registry_file.write_text(text, encoding="utf-8")

    print(f"\n✓ 登録完了!")
    print(f"  ID:   {speaker_id}")
    print(f"  名前: {name}")
    print(f"  保存: {config.paths.speakers_dir}/{speaker_id}.pt")
    print(f"  サンプル数: {len(embeddings)}")

    if all_warnings:
        print(f"\n  ⚠ {len(all_warnings)}件の警告がありました。精度向上のため再録音も検討してください")


if __name__ == "__main__":
    main()
