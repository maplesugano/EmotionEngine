"""Unified data schema and label-normalisation maps for Phase A.

All five source loaders emit `EmotionExample` records into Plutchik 8 + neutral.
Original source labels are preserved in metadata for traceability.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

# ── Canonical taxonomy ──────────────────────────────────────────────────────
PLUTCHIK = (
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
)
CATEGORIES = (*PLUTCHIK, "neutral")

# Plutchik dyad opposites — used for "opposite" negative sampling.
PLUTCHIK_OPPOSITE = {
    "joy": "sadness",
    "sadness": "joy",
    "trust": "disgust",
    "disgust": "trust",
    "fear": "anger",
    "anger": "fear",
    "surprise": "anticipation",
    "anticipation": "surprise",
    "neutral": None,
}

# ── Source-label → canonical map ────────────────────────────────────────────
# GoEmotions (Demszky 2020) 27 fine labels + neutral. Mapping follows the
# author-provided "ekman_mapping.json" extended to Plutchik via published
# crosswalks. Where a fine label has no clean Plutchik match it is dropped
# (returns None) and the example is excluded from contrastive mining.
GO_EMOTIONS_TO_PLUTCHIK = {
    # joy family
    "admiration": "trust",
    "amusement": "joy",
    "approval": "trust",
    "caring": "trust",
    "desire": "anticipation",
    "excitement": "joy",
    "gratitude": "joy",
    "joy": "joy",
    "love": "joy",
    "optimism": "anticipation",
    "pride": "joy",
    "relief": "joy",
    # surprise / anticipation
    "curiosity": "anticipation",
    "realization": "surprise",
    "surprise": "surprise",
    # sadness family
    "disappointment": "sadness",
    "embarrassment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "sadness": "sadness",
    # fear
    "fear": "fear",
    "nervousness": "fear",
    # anger / disgust
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "disgust",
    "disgust": "disgust",
    # neutral
    "confusion": None,   # no clean Plutchik match
    "neutral": "neutral",
}

# SemEval-2018 Affect-in-Tweets E-c multi-label set.
SEMEVAL_TO_PLUTCHIK = {
    "anger": "anger",
    "anticipation": "anticipation",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "love": "joy",
    "optimism": "anticipation",
    "pessimism": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "trust": "trust",
}

# DailyDialog (Li 2017) emotion codes 0..6.
DAILY_DIALOG_TO_PLUTCHIK = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "joy",
    5: "sadness",
    6: "surprise",
}

# ISEAR seven emotions (lower-cased label column).
ISEAR_TO_PLUTCHIK = {
    "joy": "joy",
    "fear": "fear",
    "anger": "anger",
    "sadness": "sadness",
    "disgust": "disgust",
    "shame": "sadness",
    "guilt": "sadness",
}


# ── Record schema ───────────────────────────────────────────────────────────
@dataclass
class EmotionExample:
    """One row in the unified contrastive dataset.

    Fields
    ------
    id            stable hash-based id ("<source>:<sourceid>")
    text          raw utterance (whitespace-stripped, no truncation)
    source        loader name, one of {go_emotions, semeval2018, daily_dialog, isear, emobank}
    label_primary canonical Plutchik category or "neutral"
    label_multi   canonical labels co-occurring in this example (multi-label sources)
    intensity     optional per-emotion intensity ∈ [0, 1] (SemEval EI-reg only for now)
    vad           optional (V, A, D) ∈ [1, 5]^3 (EmoBank only)
    source_labels original labels before normalisation (free-form per source)
    """

    id: str
    text: str
    source: str
    label_primary: str
    label_multi: list[str] = field(default_factory=list)
    intensity: Optional[float] = None
    vad: Optional[tuple[float, float, float]] = None
    source_labels: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.vad is not None:
            d["vad"] = list(self.vad)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EmotionExample":
        vad = d.get("vad")
        if vad is not None:
            vad = tuple(vad)
        return cls(
            id=d["id"],
            text=d["text"],
            source=d["source"],
            label_primary=d["label_primary"],
            label_multi=list(d.get("label_multi", [])),
            intensity=d.get("intensity"),
            vad=vad,
            source_labels=dict(d.get("source_labels", {})),
        )


def is_valid_category(label: str) -> bool:
    return label in CATEGORIES
