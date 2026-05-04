"""Module-level constants describing the model surface.

This file isolates the *shape* of the system from the implementation in
``mock_model.py`` / ``emotion_engine.py``. When the real activation-steering
model is wired in, these constants stay the same.
"""
from __future__ import annotations

from typing import Dict, List

BASIS_DIM: int = 64

MACRO_EMOTIONS: List[str] = [
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
]

PRESETS: List[str] = [
    "uncertainty",
    "self_doubt",
    "analytical_detachment",
    "addressivity",
    "warmth",
    "urgency",
]

# Mock-but-meaningful labels for a few basis components, derived from the
# qualitative findings of the EmotionEngine paper (b08 = addressivity /
# detachment axis, b11 = self-observation, b13 = indecision).
BASIS_LABELS: Dict[int, str] = {
    7: "addressivity / detachment",   # 0-indexed b08
    10: "self-observation",            # 0-indexed b11
    12: "indecision",                  # 0-indexed b13
}


def label_for(index: int) -> str:
    return BASIS_LABELS.get(index, "latent component")
