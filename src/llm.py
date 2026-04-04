from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from src.config import LlmConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
あなたは会話の文字起こしからメタデータを生成するアシスタントです。
以下の情報を日本語で生成してください。

## 出力形式
必ず以下のJSON形式のみを出力してください。説明文は不要です。
```json
{"title": "...", "summary": "...", "tags": ["...", "..."]}
```

## title
- 会話内容を端的に表す日本語のタイトル（15文字以内目安）
- ファイル名に使うため、スラッシュ(/)やバックスラッシュ(\\)は使わない

## summary
- 会話内容の要約を2-3文で簡潔に記述

## tags（必須ルール）
- 全て小文字
- スペース不可。ハイフン(-)、アンダースコア(_)、スラッシュ(/)で単語を区切る
- コンテンツタグのみ許可（トピック・主題を表すタグ）
- 禁止: ステータスタグ(#done等)、時間タグ(#2023等)、場所タグ(#tokyo等)
- 単数形を使用（#note, not #notes）
- 具体的かつ簡潔に（#marketing-strategy ○、#strategy ×）
- 略語は広く認知されているもののみ（ai, ui等）
- 固有名詞は公式名称を使用
- 最大5個
- 禁止タグ: todo, routine, daily-routine, journal, study, exercise 関連\
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
    # Simple YAML key: value parser (no nested structures needed)
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


def _parse_response(text: str) -> ConversationMetadata:
    """Parse LLM JSON response into ConversationMetadata."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    data = json.loads(cleaned)
    tags = [t.lower().strip() for t in data.get("tags", [])][:5]
    return ConversationMetadata(
        title=data["title"],
        summary=data["summary"],
        tags=tags,
    )


class LlmProcessor:
    def __init__(self, config: LlmConfig) -> None:
        self.config = config
        self.profiles = _load_profiles(config.speakers_dir)
        self.client = httpx.Client(timeout=120.0)

    def generate_metadata(
        self,
        transcript_text: str,
        speakers: dict[str, str],
    ) -> ConversationMetadata:
        """Generate title, summary, and tags from transcript text."""
        speaker_context = _build_speaker_context(self.profiles, speakers)
        user_content = f"{speaker_context}\n\n## 会話内容\n{transcript_text}"

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        resp = self.client.post(self.config.endpoint, json=payload)
        resp.raise_for_status()

        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        return _parse_response(content)
