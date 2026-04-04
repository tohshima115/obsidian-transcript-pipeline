from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.config import SpeakerIdConfig
from src.speaker_registry import SpeakerRegistry


@dataclass
class SpeakerResult:
    speaker_id: str
    label: str
    similarity: float
    is_new: bool


class SpeakerIdentifier:
    def __init__(self, config: SpeakerIdConfig, registry: SpeakerRegistry) -> None:
        self.config = config
        self.registry = registry

        from speechbrain.inference.speaker import EncoderClassifier

        self.classifier = EncoderClassifier.from_hparams(
            source=config.model_name,
            run_opts={"device": config.device},
        )

    def extract_embedding(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> torch.Tensor:
        """Extract speaker embedding from audio segment."""
        waveform = torch.from_numpy(audio).unsqueeze(0).float()
        embedding = self.classifier.encode_batch(waveform)
        return embedding.squeeze()

    def identify(self, audio: np.ndarray, sample_rate: int = 16000) -> SpeakerResult:
        """Identify the speaker of an audio segment using 2-tier thresholds."""
        duration_s = len(audio) / sample_rate
        if duration_s < self.config.min_segment_duration_s:
            return SpeakerResult(
                speaker_id="unknown",
                label="unknown",
                similarity=0.0,
                is_new=False,
            )

        seg_embedding = self.extract_embedding(audio, sample_rate)
        registered = self.registry.all_embeddings()

        if not registered:
            new_id = self.registry.register_unknown(seg_embedding)
            return SpeakerResult(
                speaker_id=new_id, label=new_id, similarity=0.0, is_new=True
            )

        best_id = ""
        best_sim = -1.0

        for spk_id, spk_emb in registered.items():
            sim = F.cosine_similarity(
                seg_embedding.unsqueeze(0), spk_emb.unsqueeze(0)
            ).item()
            if sim > best_sim:
                best_sim = sim
                best_id = spk_id

        if best_sim > self.config.known_threshold:
            label = self.registry.get_label(best_id)
            return SpeakerResult(
                speaker_id=best_id, label=label, similarity=best_sim, is_new=False
            )

        if best_sim > self.config.unknown_threshold:
            label = self.registry.get_label(best_id)
            return SpeakerResult(
                speaker_id=best_id, label=label, similarity=best_sim, is_new=False
            )

        # Completely new speaker
        new_id = self.registry.register_unknown(seg_embedding)
        return SpeakerResult(
            speaker_id=new_id, label=new_id, similarity=best_sim, is_new=True
        )
