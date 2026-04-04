from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config import PathsConfig

SPEAKER_ICONS = {
    "me": "\U0001f64b",     # 🙋
    "known": "\U0001f464",  # 👤
    "unknown": "\u2753",    # ❓
}


@dataclass
class TranscriptLine:
    speaker_id: str
    label: str
    offset_s: float
    text: str


@dataclass
class ConversationData:
    date: str              # YYYY-MM-DD
    start_time: str        # HH:MM
    end_time: str          # HH:MM
    duration_sec: int
    speakers: list[str]
    lines: list[TranscriptLine]
    has_unknown_speakers: bool = False


def _icon_for(speaker_id: str) -> str:
    if speaker_id == "me":
        return SPEAKER_ICONS["me"]
    if speaker_id.startswith("spk_"):
        return SPEAKER_ICONS["unknown"]
    return SPEAKER_ICONS["known"]


class MarkdownWriter:
    def __init__(self, config: PathsConfig) -> None:
        self.output_dir = Path(config.output_dir)

    def write(self, conversation: ConversationData) -> Path:
        """Write conversation to a Markdown file. Returns the file path."""
        # Build directory
        day_dir = self.output_dir / conversation.date
        day_dir.mkdir(parents=True, exist_ok=True)

        # Filename: conversation_HHMMSS.md
        time_slug = conversation.start_time.replace(":", "") + "00"
        file_path = day_dir / f"conversation_{time_slug}.md"

        # Tags
        tags = ["omi", "auto-transcript"]
        if conversation.has_unknown_speakers:
            tags.append("unknown-speaker")

        # Frontmatter
        speakers_yaml = "[" + ", ".join(conversation.speakers) + "]"
        tags_yaml = "[" + ", ".join(tags) + "]"

        lines = [
            "---",
            f"date: {conversation.date}",
            f'start: "{conversation.start_time}"',
            f'end: "{conversation.end_time}"',
            f"duration_sec: {conversation.duration_sec}",
            "type: conversation",
            f"speakers: {speakers_yaml}",
            f"tags: {tags_yaml}",
            "---",
            "",
            f"# 会話 {conversation.start_time}\u301c{conversation.end_time}",
            "",
        ]

        # Transcript lines
        for tl in conversation.lines:
            icon = _icon_for(tl.speaker_id)
            lines.append(f"{icon} {tl.label} ({tl.offset_s:.1f}s): {tl.text}")

        content = "\n".join(lines) + "\n"
        file_path.write_text(content, encoding="utf-8")
        return file_path
