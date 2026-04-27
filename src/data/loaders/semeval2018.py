"""SemEval-2018 Task 1 (Affect in Tweets), subtask E-c (English).

Mirror: vibhorag101/sem_eval_2018_task_1_english_cleaned_labels (parquet).
The 11 emotion columns are stored as the strings 'True' / 'False'.
"""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from ..schema import EmotionExample, SEMEVAL_TO_PLUTCHIK
from .base import make_id, normalise_text


_HF_ID = "vibhorag101/sem_eval_2018_task_1_english_cleaned_labels"

_EMOTION_COLS = (
    "anger", "anticipation", "disgust", "fear", "joy",
    "love", "optimism", "pessimism", "sadness", "surprise", "trust",
)


def _is_true(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).strip().lower() == "true"


def load(splits: list[str] | None = None) -> Iterator[EmotionExample]:
    splits = splits or ["train", "validation", "test"]
    for split in splits:
        ds = load_dataset(_HF_ID, split=split)
        for i, row in enumerate(ds):
            text = normalise_text(row.get("text") or "")
            if not text:
                continue
            active = [c for c in _EMOTION_COLS if _is_true(row.get(c))]
            mapped = [
                SEMEVAL_TO_PLUTCHIK[c]
                for c in active
                if SEMEVAL_TO_PLUTCHIK.get(c) is not None
            ]
            if not mapped:
                continue
            primary = mapped[0]
            multi = list(dict.fromkeys(mapped))
            yield EmotionExample(
                id=make_id("semeval2018", row.get("ID") or f"{split}-{i}"),
                text=text,
                source="semeval2018",
                label_primary=primary,
                label_multi=multi,
                source_labels={"raw": active, "split": split},
            )
