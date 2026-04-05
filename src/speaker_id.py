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
        self._pending_unknowns: list[tuple[int, torch.Tensor]] = []

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

    def reset_pending(self) -> None:
        """Clear pending unknowns at the start of each conversation."""
        self._pending_unknowns.clear()

    def identify(
        self, audio: np.ndarray, sample_rate: int = 16000, segment_index: int = 0
    ) -> SpeakerResult:
        """Identify the speaker of an audio segment.

        Uses 3-tier thresholds and defers unknown registration until
        resolve_pending() is called.
        """
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
            self._pending_unknowns.append((segment_index, seg_embedding))
            return SpeakerResult(
                speaker_id="__pending__", label="__pending__",
                similarity=0.0, is_new=True,
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

        # Tier 1: high-confidence match
        if best_sim > self.config.known_threshold:
            if best_id.startswith("spk_"):
                self.registry.update_embedding(
                    best_id, seg_embedding, self.config.embedding_update_alpha
                )
            label = self.registry.get_label(best_id)
            return SpeakerResult(
                speaker_id=best_id, label=label, similarity=best_sim, is_new=False
            )

        # Tier 2: moderate match (known speakers)
        if best_sim > self.config.unknown_threshold:
            if best_id.startswith("spk_"):
                self.registry.update_embedding(
                    best_id, seg_embedding, self.config.embedding_update_alpha
                )
            label = self.registry.get_label(best_id)
            return SpeakerResult(
                speaker_id=best_id, label=label, similarity=best_sim, is_new=False
            )

        # Tier 3: relaxed match for existing spk_ entries
        if best_id.startswith("spk_") and best_sim > self.config.spk_match_threshold:
            self.registry.update_embedding(
                best_id, seg_embedding, self.config.embedding_update_alpha
            )
            label = self.registry.get_label(best_id)
            return SpeakerResult(
                speaker_id=best_id, label=label, similarity=best_sim, is_new=False
            )

        # No match — defer registration
        self._pending_unknowns.append((segment_index, seg_embedding))
        return SpeakerResult(
            speaker_id="__pending__", label="__pending__",
            similarity=best_sim, is_new=True,
        )

    def resolve_pending(self) -> dict[int, SpeakerResult]:
        """Cluster pending unknowns and register one spk_ per cluster.

        Returns a mapping from segment_index to the resolved SpeakerResult.
        """
        if not self._pending_unknowns:
            return {}

        # Greedy agglomerative clustering
        clusters: list[list[tuple[int, torch.Tensor]]] = []

        for seg_idx, emb in self._pending_unknowns:
            matched_cluster = None
            best_cluster_sim = -1.0

            for cluster in clusters:
                # Compare against cluster centroid (average of members)
                centroid = torch.stack([e for _, e in cluster]).mean(dim=0)
                sim = F.cosine_similarity(
                    emb.unsqueeze(0), centroid.unsqueeze(0)
                ).item()
                if sim > self.config.merge_threshold and sim > best_cluster_sim:
                    best_cluster_sim = sim
                    matched_cluster = cluster

            if matched_cluster is not None:
                matched_cluster.append((seg_idx, emb))
            else:
                clusters.append([(seg_idx, emb)])

        # Register one spk_ per cluster
        resolved: dict[int, SpeakerResult] = {}

        for cluster in clusters:
            avg_embedding = torch.stack([e for _, e in cluster]).mean(dim=0)
            avg_embedding = F.normalize(avg_embedding, dim=0)
            new_id = self.registry.register_unknown(avg_embedding)
            label = self.registry.get_label(new_id)

            for seg_idx, _ in cluster:
                resolved[seg_idx] = SpeakerResult(
                    speaker_id=new_id, label=label, similarity=0.0, is_new=True
                )

        self._pending_unknowns.clear()
        return resolved
