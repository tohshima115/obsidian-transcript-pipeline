from __future__ import annotations

import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import soundfile as sf

from src.config import AppConfig
from src.hallucination_filter import filter_hallucination_loops
from src.llm import LlmProcessor
from src.markdown_writer import ConversationData, MarkdownWriter, TranscriptLine
from src.speaker_id import SpeakerIdentifier
from src.speaker_registry import SpeakerRegistry
from src.stt import SttProcessor, SttSegment
from src.vad import VadProcessor

logger = __import__("logging").getLogger(__name__)


class Pipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

        print("Loading VAD...")
        self.vad = VadProcessor(config.vad)

        print("Loading speaker registry...")
        self.registry = SpeakerRegistry(config.paths)
        self.registry.load()

        print("Loading speaker identifier...")
        self.speaker_id = SpeakerIdentifier(config.speaker_id, self.registry)

        print("Loading STT...")
        self.stt = SttProcessor(config.stt, speakers_dir=Path(config.paths.speakers_dir))

        print("Loading LLM processor...")
        self.llm = LlmProcessor(config.llm)

        self.writer = MarkdownWriter(config.paths)
        print("Pipeline ready.")

    def process_file(self, audio_path: str | Path, timestamp: datetime | None = None) -> Path:
        """Process a single audio file through the full pipeline.

        Supports WAV directly. For other formats (Ogg Opus, etc.),
        converts to WAV via ffmpeg automatically.
        """
        audio_path = Path(audio_path)
        if timestamp is None:
            timestamp = datetime.now()

        if audio_path.suffix.lower() in (".ogg", ".opus", ".mp3", ".m4a", ".flac"):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = Path(tmp.name)
            try:
                subprocess.run(
                    ["ffmpeg", "-i", str(audio_path), "-ar", "16000", "-ac", "1", "-f", "wav", str(wav_path), "-y"],
                    check=True, capture_output=True,
                )
                audio, sr = sf.read(str(wav_path), dtype="float32")
            finally:
                wav_path.unlink(missing_ok=True)
        else:
            audio, sr = sf.read(str(audio_path), dtype="float32")

        return self.process_audio(audio, sr, timestamp)

    def process_audio(
        self, audio: np.ndarray, sample_rate: int, timestamp: datetime
    ) -> Path:
        """Process raw audio array through the full pipeline."""
        # 1. VAD: detect speech segments, then merge close segments
        raw_segments = self.vad.detect_speech(audio, sample_rate)
        if not raw_segments:
            raise ValueError("No speech detected in audio")

        vad_segments = self.vad.merge_close_segments(
            raw_segments, self.config.vad.merge_max_gap_s
        )
        speech_parts = self.vad.extract_speech_audio(audio, vad_segments, sample_rate)

        # 2. STT per VAD segment + speaker ID
        stt_segments: list[SttSegment] = []
        transcript_lines: list[TranscriptLine] = []
        seen_speakers: dict[str, str] = {}
        has_unknown = False
        conv_start = vad_segments[0].start_s

        for seg, seg_audio in speech_parts:
            duration = seg.end_s - seg.start_s

            # STT on this segment
            stt_result = self.stt.transcribe_segment(
                seg_audio, sample_rate, seg.start_s, seg.end_s
            )
            if stt_result is None:
                continue

            stt_segments.append(stt_result)

        # 3. Hallucination filter
        stt_segments = filter_hallucination_loops(
            stt_segments, self.config.hallucination.max_consecutive_repeats
        )

        if not stt_segments:
            raise ValueError("No text segments after filtering")

        # 4. Speaker ID for each surviving segment
        for stt_seg in stt_segments:
            start_sample = int(stt_seg.start_s * sample_rate)
            end_sample = min(int(stt_seg.end_s * sample_rate), len(audio))
            seg_audio = audio[start_sample:end_sample]

            if len(seg_audio) / sample_rate >= self.config.speaker_id.min_segment_duration_s:
                speaker = self.speaker_id.identify(seg_audio, sample_rate)
                speaker_id = speaker.speaker_id
                label = speaker.label
            else:
                speaker_id = "unknown"
                label = "unknown"

            if speaker_id.startswith("spk_"):
                has_unknown = True

            seen_speakers[speaker_id] = label
            transcript_lines.append(
                TranscriptLine(
                    speaker_id=speaker_id,
                    label=label,
                    offset_s=max(0.0, stt_seg.start_s - conv_start),
                    text=stt_seg.text,
                )
            )

        # 5. Build conversation data
        conv_end = vad_segments[-1].end_s
        duration_sec = int(conv_end - conv_start)
        end_dt = timestamp + timedelta(seconds=duration_sec)

        conversation = ConversationData(
            date=timestamp.strftime("%Y-%m-%d"),
            start_time=timestamp.strftime("%H:%M"),
            end_time=end_dt.strftime("%H:%M"),
            duration_sec=duration_sec,
            speakers=list(seen_speakers.values()),
            lines=transcript_lines,
            has_unknown_speakers=has_unknown,
        )

        # 5.5. LLM metadata generation (title, summary, tags)
        try:
            transcript_text = "\n".join(
                f"{tl.label}: {tl.text}" for tl in transcript_lines
            )
            metadata = self.llm.generate_metadata(transcript_text, seen_speakers)
            conversation.title = metadata.title
            conversation.summary = metadata.summary
            conversation.tags = metadata.tags
        except Exception as e:
            logger.warning("LLM metadata generation failed, using fallback: %s", e)

        # 6. Write markdown
        return self.writer.write(conversation)
