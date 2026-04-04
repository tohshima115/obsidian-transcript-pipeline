from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.config import VadConfig


@dataclass
class SpeechSegment:
    start_s: float
    end_s: float


class VadProcessor:
    def __init__(self, config: VadConfig) -> None:
        self.config = config
        self.model, self.utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self._get_speech_timestamps = self.utils[0]

    def detect_speech(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> list[SpeechSegment]:
        """Detect speech segments in audio. Returns list of SpeechSegment with times in seconds."""
        tensor = torch.from_numpy(audio).float()
        if tensor.ndim > 1:
            tensor = tensor.squeeze()

        timestamps = self._get_speech_timestamps(
            tensor,
            self.model,
            sampling_rate=sample_rate,
            threshold=self.config.threshold,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            window_size_samples=self.config.window_size_samples,
        )

        return [
            SpeechSegment(
                start_s=ts["start"] / sample_rate,
                end_s=ts["end"] / sample_rate,
            )
            for ts in timestamps
        ]

    @staticmethod
    def merge_close_segments(
        segments: list[SpeechSegment], max_gap_s: float = 1.5
    ) -> list[SpeechSegment]:
        """Merge segments separated by less than max_gap_s into single segments."""
        if not segments:
            return []

        merged = [SpeechSegment(start_s=segments[0].start_s, end_s=segments[0].end_s)]
        for seg in segments[1:]:
            if seg.start_s - merged[-1].end_s <= max_gap_s:
                merged[-1].end_s = seg.end_s
            else:
                merged.append(SpeechSegment(start_s=seg.start_s, end_s=seg.end_s))
        return merged

    def extract_speech_audio(
        self, audio: np.ndarray, segments: list[SpeechSegment], sample_rate: int = 16000
    ) -> list[tuple[SpeechSegment, np.ndarray]]:
        """Extract audio arrays for each speech segment."""
        results = []
        for seg in segments:
            start_sample = int(seg.start_s * sample_rate)
            end_sample = int(seg.end_s * sample_rate)
            results.append((seg, audio[start_sample:end_sample]))
        return results

    def has_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Quick check: does this audio chunk contain any speech?"""
        return len(self.detect_speech(audio, sample_rate)) > 0
