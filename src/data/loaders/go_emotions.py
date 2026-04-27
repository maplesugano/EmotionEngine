"""GoEmotions loader → EmotionExample.

Source: google-research-datasets/go_emotions (config="simplified").
Each example carries multiple integer label ids referencing the official
27-emotion + neutral list. We normalise via GO_EMOTIONS_TO_PLUTCHIK.

Multi-label handling: the canonical primary label is the *first* mapped Plutchik
category found among the example's labels (preserving annotator order).
Examples whose labels collapse entirely to None are dropped.
"""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from ..schema import EmotionExample, GO_EMOTIONS_TO_PLUTCHIK
from .base import make_id, normalise_text


_HF_ID = "google-research-datasets/go_emotions"
_CONFIG = "simplified"


def _load_label_names() -> list[str]:
    info = load_dataset(_HF_ID, _CONFIG, split="train", streaming=True)
    return info.features["labels"].feature.names  # type: ignore[index]


def load(splits: list[str] | None = None) -> Iterator[EmotionExample]:
    splits = splits or ["train", "validation", "test"]
    label_names = _load_label_names()

    for split in splits:
        ds = load_dataset(_HF_ID, _CONFIG, split=split)
        for i, row in enumerate(ds):
            text = normalise_text(row["text"])
            if not text:
                continue
            raw_labels = [label_names[idx] for idx in row["labels"]]
            mapped = [
                GO_EMOTIONS_TO_PLUTCHIK[name]
                for name in raw_labels
                if GO_EMOTIONS_TO_PLUTCHIK.get(name) is not None
            ]
            if not mapped:
                continue
            primary = mapped[0]
            multi = list(dict.fromkeys(mapped))   # dedupe, preserve order
            yield EmotionExample(
                id=make_id("go_emotions", f"{split}-{i}"),
                text=text,
                source="go_emotions",
                label_primary=primary,
                label_multi=multi,
                source_labels={"raw": raw_labels, "split": split},
            )
