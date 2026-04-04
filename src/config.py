from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VadConfig:
    threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    window_size_samples: int
    merge_max_gap_s: float


@dataclass(frozen=True)
class SttConfig:
    model_name: str
    chunk_length_s: int
    stride_length_s: tuple[int, int]
    language: str
    device: str
    torch_dtype: str


@dataclass(frozen=True)
class SpeakerIdConfig:
    model_name: str
    known_threshold: float
    unknown_threshold: float
    min_segment_duration_s: float
    device: str


@dataclass(frozen=True)
class PathsConfig:
    speakers_dir: Path
    output_dir: Path
    registry_file: Path


@dataclass(frozen=True)
class HallucinationConfig:
    max_consecutive_repeats: int


@dataclass(frozen=True)
class AppConfig:
    vad: VadConfig
    stt: SttConfig
    speaker_id: SpeakerIdConfig
    paths: PathsConfig
    hallucination: HallucinationConfig


def load_config(path: str | Path = "config.toml") -> AppConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    return AppConfig(
        vad=VadConfig(**raw["vad"]),
        stt=SttConfig(
            model_name=raw["stt"]["model_name"],
            chunk_length_s=raw["stt"]["chunk_length_s"],
            stride_length_s=tuple(raw["stt"]["stride_length_s"]),
            language=raw["stt"]["language"],
            device=raw["stt"]["device"],
            torch_dtype=raw["stt"]["torch_dtype"],
        ),
        speaker_id=SpeakerIdConfig(**raw["speaker_id"]),
        paths=PathsConfig(
            speakers_dir=Path(raw["paths"]["speakers_dir"]),
            output_dir=Path(raw["paths"]["output_dir"]),
            registry_file=Path(raw["paths"]["registry_file"]),
        ),
        hallucination=HallucinationConfig(**raw["hallucination"]),
    )
