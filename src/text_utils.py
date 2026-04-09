# src/text_utils.py

import re

def clean_for_eval(text: str) -> str:
    """
    Strip obvious meta-chatter and keep only the first meaningful part
    for metric computation. This does NOT change the original logs,
    only what we feed into metrics.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1) Keep only first paragraph (before first blank line)
    parts = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    if parts:
        text = parts[0]

    # 2) Cut off at common meta-markers if they appear inside that paragraph
    stop_markers = [
        "note:",
        "i do not have enough information to answer this question",
        "let me know if this is acceptable",
        "here is the final answer",
        "here is the revised answer",
    ]
    lower = text.lower()
    cut_idx = len(text)
    for marker in stop_markers:
        idx = lower.find(marker)
        if idx != -1:
            cut_idx = min(cut_idx, idx)
    text = text[:cut_idx].strip()

    # 3) Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
