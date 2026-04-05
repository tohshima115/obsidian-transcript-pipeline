from __future__ import annotations

import re
import unicodedata
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
    title: str = ""
    summary: str = ""
    tags: list[str] = field(default_factory=list)


def _icon_for(speaker_id: str) -> str:
    if speaker_id == "me":
        return SPEAKER_ICONS["me"]
    if speaker_id.startswith("spk_"):
        return SPEAKER_ICONS["unknown"]
    return SPEAKER_ICONS["known"]


def _sanitize_filename(title: str) -> str:
    """Convert title to a safe filename component."""
    # Remove characters unsafe for filesystems
    sanitized = re.sub(r'[\\/:*?"<>|\n\r]', '', title)
    # Collapse whitespace to underscore
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    # Limit length to avoid filesystem issues
    if len(sanitized) > 80:
        sanitized = sanitized[:80]
    return sanitized


class MarkdownWriter:
    def __init__(self, config: PathsConfig) -> None:
        self.output_dir = Path(config.output_dir)

    def write(self, conversation: ConversationData) -> Path:
        """Write conversation to a Markdown file. Returns the file path."""
        # Build directory
        day_dir = self.output_dir / conversation.date
        day_dir.mkdir(parents=True, exist_ok=True)

        # Filename: HH-MM_{title}.md or HH-MM.md if no title
        time_slug = conversation.start_time.replace(":", "-")
        if conversation.title:
            safe_title = _sanitize_filename(conversation.title)
            file_path = day_dir / f"{time_slug}_{safe_title}.md"
        else:
            file_path = day_dir / f"{time_slug}.md"

        # Frontmatter
        speakers_yaml = "[" + ", ".join(conversation.speakers) + "]"
        tags_yaml = "[" + ", ".join(conversation.tags) + "]" if conversation.tags else "[]"

        # Heading
        heading = conversation.title or f"会話 {conversation.start_time}\u301c{conversation.end_time}"

        lines = [
            "---",
            f"date: {conversation.date}",
            f'start: "{conversation.start_time}"',
            f'end: "{conversation.end_time}"',
            f"duration_sec: {conversation.duration_sec}",
            f"speakers: {speakers_yaml}",
            f"tags: {tags_yaml}",
        ]
        if conversation.summary:
            escaped_summary = conversation.summary.replace('"', '\\"')
            lines.append(f'summary: "{escaped_summary}"')
        lines += [
            "---",
            "",
            f"# {heading}",
            "",
        ]

        # Summary section
        if conversation.summary:
            lines += [f"> {conversation.summary}", ""]

        # Transcript lines
        for tl in conversation.lines:
            icon = _icon_for(tl.speaker_id)
            lines.append(f"{icon} {tl.label} ({tl.offset_s:.1f}s): {tl.text}")

        content = "\n".join(lines) + "\n"
        file_path.write_text(content, encoding="utf-8")
        return file_path
