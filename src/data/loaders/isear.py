"""ISEAR loader (local CSV).

ISEAR (International Survey on Emotion Antecedents and Reactions) is not
redistributable, so the loader expects a CSV downloaded manually to
`data/raw/isear/isear.csv`. Standard column names: SIT (situation text),
EMOT (numeric or string emotion), with seven canonical emotions:
joy, fear, anger, sadness, disgust, shame, guilt.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from ..schema import EmotionExample, ISEAR_TO_PLUTCHIK
from .base import LoaderError, make_id, normalise_text


_DEFAULT_PATH = Path("data/raw/isear/isear.csv")

# ISEAR EMOT column is sometimes numeric (1-7) and sometimes a string;
# both forms map to the same canonical labels.
_NUMERIC_TO_LABEL = {
    "1": "joy",
    "2": "fear",
    "3": "anger",
    "4": "sadness",
    "5": "disgust",
    "6": "shame",
    "7": "guilt",
}


def _resolve_text_column(row: dict) -> str | None:
    for k in ("SIT", "situation", "Situation", "text", "content", "Content"):
        if k in row and row[k]:
            return row[k]
    return None


def _resolve_emot_column(row: dict) -> str | None:
    for k in ("EMOT", "emotion", "Emotion", "label", "sentiment", "Sentiment"):
        if k in row and row[k]:
            return str(row[k]).strip().lower()
    return None


def load(splits: list[str] | None = None, path: Path | None = None) -> Iterator[EmotionExample]:
    # ISEAR has no canonical splits; we treat the full file as "train" and
    # the contrastive builder later draws its own held-out subset.
    path = path or _DEFAULT_PATH
    if not path.exists():
        raise LoaderError(
            f"ISEAR CSV not found at {path}. "
            "Download from https://www.unige.ch/cisa/research/materials-and-online-research/research-material/ "
            "and place the CSV at the expected path."
        )

    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            raw_text = _resolve_text_column(row)
            raw_emot = _resolve_emot_column(row)
            if not raw_text or not raw_emot:
                continue
            text = normalise_text(raw_text)
            if not text:
                continue
            label_key = _NUMERIC_TO_LABEL.get(raw_emot, raw_emot)
            primary = ISEAR_TO_PLUTCHIK.get(label_key)
            if primary is None:
                continue
            yield EmotionExample(
                id=make_id("isear", i),
                text=text,
                source="isear",
                label_primary=primary,
                label_multi=[primary],
                source_labels={"raw": raw_emot},
            )
