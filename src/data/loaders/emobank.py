"""EmoBank loader — VAD supervision source (diagnostic role only).

Mirror: reallycarlaost/emobank_w (parquet) — columns are
`label1`, `label2`, `label3` (V/A/D in normalised [0, 1] range, stored as
strings) and `text`. We pass the raw triple through unchanged; downstream
consumers can rescale to the original 1-5 range if required.

Splits: only `train` and `test` are published; we treat them both as
collectable rows (the contrastive builder draws its own held-out subset).
"""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from ..schema import EmotionExample
from .base import make_id, normalise_text


_HF_ID = "reallycarlaost/emobank_w"


def load(splits: list[str] | None = None) -> Iterator[EmotionExample]:
    splits = splits or ["train", "test"]
    for split in splits:
        try:
            ds = load_dataset(_HF_ID, split=split)
        except ValueError:
            continue
        for i, row in enumerate(ds):
            text = normalise_text(row.get("text") or "")
            if not text:
                continue
            try:
                v = float(row["label1"])
                a = float(row["label2"])
                d = float(row["label3"])
            except (KeyError, TypeError, ValueError):
                continue
            yield EmotionExample(
                id=make_id("emobank", f"{split}-{i}"),
                text=text,
                source="emobank",
                label_primary="neutral",
                label_multi=[],
                vad=(v, a, d),
                source_labels={"split": split},
            )
