from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.config import SttConfig

# Qwen3-ASR requires full language names
_LANG_TO_QWEN = {
    "ja": "Japanese",
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
}

TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@dataclass
class SttSegment:
    start_s: float
    end_s: float
    text: str


def _load_vocabulary(vocab_path: Path) -> str:
    """Load vocabulary.md and return comma-separated terms for ASR context."""
    if not vocab_path.exists():
        return ""
    terms: list[str] = []
    for line in vocab_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^-\s+(.+)$", line)
        if m:
            terms.append(m.group(1).strip())
    return ", ".join(terms)


class SttProcessor:
    def __init__(self, config: SttConfig, speakers_dir: Path | None = None) -> None:
        self.config = config
        self._context = ""
        if speakers_dir:
            self._context = _load_vocabulary(speakers_dir / "vocabulary.md")
        dtype = TORCH_DTYPES.get(config.torch_dtype, torch.float32)

        if "qwen" in config.model_name.lower():
            from qwen_asr import Qwen3ASRModel

            self._backend = "qwen"
            self._model = Qwen3ASRModel.from_pretrained(
                config.model_name,
                dtype=dtype,
                device_map="cpu" if config.device == "cpu" else (config.device + ":0" if ":" not in config.device else config.device),
                max_new_tokens=256,
            )
        else:
            from transformers import pipeline

            self._backend = "whisper"
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=config.model_name,
                device=config.device,
                dtype=dtype,
            )

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> list[SttSegment]:
        """Transcribe audio and return timestamped segments.

        For Qwen3-ASR: returns a single segment with full text (no internal timestamps).
        For Whisper: returns chunked segments with timestamps.
        """
        if self._backend == "qwen":
            return self._transcribe_qwen(audio, sample_rate)
        return self._transcribe_whisper(audio, sample_rate)

    def transcribe_segment(
        self, audio: np.ndarray, sample_rate: int, start_s: float, end_s: float
    ) -> SttSegment | None:
        """Transcribe a single pre-segmented audio chunk (e.g., from VAD).

        Returns None if no text is produced.
        """
        if self._backend == "qwen":
            results = self._model.transcribe(
                audio=[(audio.astype(np.float32), sample_rate)],
                language=_LANG_TO_QWEN.get(self.config.language, self.config.language),
                context=self._context,
            )
            text = results[0].text.strip() if results else ""
        else:
            result = self._pipe(
                {"raw": audio.astype(np.float32), "sampling_rate": sample_rate},
                generate_kwargs={"language": self.config.language},
            )
            text = result.get("text", "").strip()

        if not text:
            return None
        return SttSegment(start_s=start_s, end_s=end_s, text=text)

    def _transcribe_qwen(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[SttSegment]:
        results = self._model.transcribe(
            audio=[(audio.astype(np.float32), sample_rate)],
            language=_LANG_TO_QWEN.get(self.config.language, self.config.language),
            context=self._context,
        )
        text = results[0].text.strip() if results else ""
        if not text:
            return []
        duration = len(audio) / sample_rate
        return [SttSegment(start_s=0.0, end_s=duration, text=text)]

    def _transcribe_whisper(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[SttSegment]:
        result = self._pipe(
            {"raw": audio.astype(np.float32), "sampling_rate": sample_rate},
            chunk_length_s=self.config.chunk_length_s,
            stride_length_s=tuple(self.config.stride_length_s),
            return_timestamps=True,
            ignore_warning=True,
            generate_kwargs={"language": self.config.language},
        )

        segments = []
        for chunk in result.get("chunks", []):
            text = chunk["text"].strip()
            if not text:
                continue
            ts = chunk.get("timestamp", (0.0, 0.0))
            start = ts[0] if ts[0] is not None else 0.0
            end = ts[1] if ts[1] is not None else start
            segments.append(SttSegment(start_s=start, end_s=end, text=text))

        return segments
