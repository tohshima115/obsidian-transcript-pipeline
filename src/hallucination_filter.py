from __future__ import annotations

import re
from src.stt import SttSegment


def _normalize(text: str) -> str:
    """Normalize text for comparison: strip whitespace and punctuation."""
    return re.sub(r"[\s、。！？!?,.\-]+", "", text)


def filter_hallucination_loops(
    segments: list[SttSegment],
    max_repeats: int = 3,
) -> list[SttSegment]:
    """Remove hallucination loops: consecutive segments with identical text.

    If the same normalized text appears max_repeats or more times in a row,
    only the first occurrence is kept.
    """
    if not segments:
        return []

    result: list[SttSegment] = []
    run_start = 0

    while run_start < len(segments):
        current_norm = _normalize(segments[run_start].text)
        run_end = run_start + 1

        while run_end < len(segments) and _normalize(segments[run_end].text) == current_norm:
            run_end += 1

        run_length = run_end - run_start

        if run_length >= max_repeats:
            # Hallucination loop detected: keep only the first
            result.append(segments[run_start])
        else:
            result.extend(segments[run_start:run_end])

        run_start = run_end

    return result
