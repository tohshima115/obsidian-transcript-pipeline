from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from src.config import LlmConfig

logger = logging.getLogger(__name__)

TITLE_SUMMARY_PROMPT = """\
あなたは会話の文字起こしからタイトルと要約を生成するアシスタントです。

## 出力形式
必ず以下のJSON形式のみを出力してください。説明文は不要です。
```json
{"title": "...", "summary": "..."}
```

## title
- 会話内容を端的に表す日本語のタイトル（15文字以内目安）
- ファイル名に使うため、スラッシュ(/)やバックスラッシュ(\\)は使わない

## summary
- 会話内容の要約を2-3文で簡潔に記述\
"""

TAGS_PROMPT = """\
You are a tag generator for conversation transcripts.
Output ONLY a JSON array of tags. No explanation, no other text.

```json
["tag1", "tag2", "tag3"]
```

## Rules (STRICTLY ENFORCED)
- ASCII lowercase English ONLY (a-z, 0-9, hyphens, underscores, slashes)
- FORBIDDEN: Japanese, katakana, kanji, uppercase letters
- Good: "design", "3d-printer", "fashion" / Bad: "デザイン", "Design"
- Use hyphens (-), underscores (_), or slashes (/) to separate words
- Content tags only (topic/subject of the conversation)
- Forbidden: status tags (#done), time tags (#2023), location tags (#tokyo)
- Use singular form (#note, not #notes)
- Be specific and concise (#marketing-strategy, not #strategy)
- Only use widely recognized abbreviations (ai, ui)
- Use official names for proper nouns
- Maximum 5 tags
- Forbidden topics: todo, routine, daily-routine, journal, study, exercise\
"""


@dataclass
class ConversationMetadata:
    title: str
    summary: str
    tags: list[str]


def _build_speaker_context(profiles: dict[str, dict], speakers: dict[str, str]) -> str:
    """Build speaker profile context for the prompt."""
    lines = ["## 話者情報"]
    for speaker_id, label in speakers.items():
        profile = profiles.get(speaker_id, {})
        if profile:
            name = profile.get("name", label)
            role = profile.get("role", "")
            note = profile.get("note", "")
            parts = [f"- {name}"]
            if role:
                parts.append(f"（{role}）")
            if note:
                parts.append(f" - {note}")
            lines.append("".join(parts))
        else:
            lines.append(f"- {label}")
    return "\n".join(lines)


def _parse_md_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter and body from a markdown file."""
    m = re.match(r"^---\n(.+?)\n---\n?(.*)", text, re.DOTALL)
    if not m:
        return {}, text.strip()
    fm_raw, body = m.group(1), m.group(2).strip()
    fm: dict[str, str] = {}
    for line in fm_raw.split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            fm[key.strip()] = val.strip()
    return fm, body


def _load_profiles(speakers_dir: Path) -> dict[str, dict]:
    """Load speaker profiles from individual .md files in _speakers/."""
    profiles: dict[str, dict] = {}
    if not speakers_dir.is_dir():
        return profiles
    for md_file in speakers_dir.glob("*.md"):
        if md_file.name == "registry.md":
            continue
        speaker_id = md_file.stem
        content = md_file.read_text(encoding="utf-8")
        fm, body = _parse_md_frontmatter(content)
        profile = dict(fm)
        if body:
            profile["note"] = body
        profiles[speaker_id] = profile
    return profiles


def _extract_json(text: str, kind: str = "object") -> str:
    """Extract the first JSON object or array from LLM output.

    Handles markdown fences, preamble text, and trailing explanations.
    kind: "object" to find {...}, "array" to find [...]
    """
    if kind == "array":
        m = re.search(r"\[.*\]", text, re.DOTALL)
    else:
        m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return text.strip()


class LlmProcessor:
    def __init__(self, config: LlmConfig) -> None:
        self.config = config
        self.profiles = _load_profiles(config.speakers_dir)
        self.client = httpx.Client(timeout=120.0)

    def _call(self, system: str, user: str) -> str:
        """Send a chat completion request and return the content."""
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        resp = self.client.post(self.config.endpoint, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def generate_metadata(
        self,
        transcript_text: str,
        speakers: dict[str, str],
    ) -> ConversationMetadata:
        """Generate title, summary, and tags sequentially."""
        speaker_context = _build_speaker_context(self.profiles, speakers)
        user_content = f"{speaker_context}\n\n## 会話内容\n{transcript_text}"

        title_raw = self._call(TITLE_SUMMARY_PROMPT, user_content)
        tags_raw = self._call(TAGS_PROMPT, user_content)

        try:
            title_data = json.loads(_extract_json(title_raw, "object"))
        except json.JSONDecodeError:
            logger.warning("Failed to parse title/summary JSON, raw: %s", title_raw[:500])
            raise
        try:
            tags_list = json.loads(_extract_json(tags_raw, "array"))
        except json.JSONDecodeError:
            logger.warning("Failed to parse tags JSON, raw: %s", tags_raw[:500])
            raise

        # Enforce ASCII lowercase for tags
        tags = [t.lower().strip() for t in tags_list if t.isascii()][:5]

        return ConversationMetadata(
            title=title_data["title"],
            summary=title_data["summary"],
            tags=tags,
        )
