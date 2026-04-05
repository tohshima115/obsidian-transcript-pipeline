from __future__ import annotations

import re
import secrets
from datetime import datetime
from pathlib import Path

import torch

from src.config import PathsConfig

REGISTRY_HEADER = """# 話者レジストリ

| ID | 名前 | 初出 |
|---|---|---|
| me | 自分 | - |
"""


class SpeakerRegistry:
    def __init__(self, config: PathsConfig) -> None:
        self.speakers_dir = Path(config.speakers_dir)
        self.registry_file = Path(config.registry_file)
        self._entries: dict[str, str] = {}  # id -> display name
        self._embeddings: dict[str, torch.Tensor] = {}

    def load(self) -> None:
        """Load registry.md and all .pt embedding files."""
        self._entries.clear()
        self._embeddings.clear()

        # Parse registry.md
        if self.registry_file.exists():
            text = self.registry_file.read_text(encoding="utf-8")
            for match in re.finditer(
                r"^\|\s*(\S+)\s*\|\s*(.+?)\s*\|", text, re.MULTILINE
            ):
                sid, name = match.group(1), match.group(2).strip()
                if sid in ("ID", "---"):
                    continue
                self._entries[sid] = name
        else:
            self._entries["me"] = "自分"

        # Load embeddings
        if self.speakers_dir.exists():
            for pt_file in self.speakers_dir.glob("*.pt"):
                sid = pt_file.stem
                self._embeddings[sid] = torch.load(pt_file, weights_only=True, map_location="cpu")

    def get_label(self, speaker_id: str) -> str:
        """Get display name for a speaker ID."""
        name = self._entries.get(speaker_id, speaker_id)
        if name == "（未登録）":
            return speaker_id
        return name

    def get_embedding(self, speaker_id: str) -> torch.Tensor | None:
        return self._embeddings.get(speaker_id)

    def all_embeddings(self) -> dict[str, torch.Tensor]:
        return dict(self._embeddings)

    def register_unknown(self, embedding: torch.Tensor) -> str:
        """Register a new unknown speaker. Returns the new spk_XXXX id."""
        while True:
            new_id = f"spk_{secrets.token_hex(2)}"
            if new_id not in self._entries:
                break

        # Save embedding
        self.speakers_dir.mkdir(parents=True, exist_ok=True)
        torch.save(embedding, self.speakers_dir / f"{new_id}.pt")
        self._embeddings[new_id] = embedding

        # Update in-memory entries
        self._entries[new_id] = "（未登録）"

        # Append to registry.md
        self._ensure_registry_file()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(self.registry_file, "a", encoding="utf-8") as f:
            f.write(f"| {new_id} | （未登録） | {now} |\n")

        return new_id

    def _ensure_registry_file(self) -> None:
        if not self.registry_file.exists():
            self.speakers_dir.mkdir(parents=True, exist_ok=True)
            self.registry_file.write_text(REGISTRY_HEADER, encoding="utf-8")

    def update_embedding(
        self, speaker_id: str, new_embedding: torch.Tensor, alpha: float = 0.8
    ) -> None:
        """Update a speaker's embedding as a weighted average with the new one."""
        existing = self._embeddings.get(speaker_id)
        if existing is None:
            return
        blended = alpha * existing + (1 - alpha) * new_embedding
        blended = torch.nn.functional.normalize(blended, dim=0)
        torch.save(blended, self.speakers_dir / f"{speaker_id}.pt")
        self._embeddings[speaker_id] = blended

    def save_embedding(self, speaker_id: str, embedding: torch.Tensor) -> None:
        """Save an embedding for a known speaker (e.g., 'me')."""
        self.speakers_dir.mkdir(parents=True, exist_ok=True)
        torch.save(embedding, self.speakers_dir / f"{speaker_id}.pt")
        self._embeddings[speaker_id] = embedding

    @property
    def entries(self) -> dict[str, str]:
        return dict(self._entries)
