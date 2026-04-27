"""DailyDialog loader.

Mirror: pixelsandpointers/better_daily_dialog (parquet).
Pre-flattened to one row per utterance with `utterance` (text) and `emotion`
(string code 0..6).
"""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from ..schema import EmotionExample, DAILY_DIALOG_TO_PLUTCHIK
from .base import make_id, normalise_text


_HF_ID = "pixelsandpointers/better_daily_dialog"


def load(splits: list[str] | None = None) -> Iterator[EmotionExample]:
    splits = splits or ["train", "validation", "test"]
    for split in splits:
        ds = load_dataset(_HF_ID, split=split)
        for i, row in enumerate(ds):
            text = normalise_text(row.get("utterance") or "")
            if not text:
                continue
            try:
                code = int(row["emotion"])
            except (KeyError, TypeError, ValueError):
                continue
            primary = DAILY_DIALOG_TO_PLUTCHIK.get(code)
            if primary is None:
                continue
            yield EmotionExample(
                id=make_id("daily_dialog", f"{split}-{row.get('dialog_id', i)}-{i}"),
                text=text,
                source="daily_dialog",
                label_primary=primary,
                label_multi=[primary],
                source_labels={"raw_code": code, "split": split},
            )
