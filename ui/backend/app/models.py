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

# Per-component verbal labels for the k=64 / L=22 ICA basis used by the
# live UI backend. Generated from the per-axis judge experiment in
# `experiments/results/per_axis_judge_L22_k64/per_axis_summary.csv` —
# specifically the modal `top_other_label` (free-form, off-Plutchik) at
# α=+2 paired with the modal label at α=−2, giving a "+pos ↔ −neg"
# description of what the axis steers toward in each direction.
#
# Confidence varies: comments after each entry are
#   cnt = how many of 8 prompts agreed on the label at α=+2
#   uniq = number of distinct labels at α=+2 (lower = more agreement)
#   score = mean judge "off-Plutchik" confidence (higher = more lexical-gap-ish)
# Entries marked "?" had cnt=1 and score<0.35 — the label is one judge call,
# not a consensus, so treat it as a hint rather than a fact.
BASIS_LABELS: Dict[int, str] = {
     0: "helplessness ↔ curiosity",                # cnt=1 uniq=7 score=0.44
     1: "curiosity ↔ uncertainty",                 # cnt=2 uniq=3 score=0.30
     2: "self-care ↔ intellectual curiosity",      # ? cnt=1 uniq=4 score=0.20
     3: "frustration ↔ uncertainty",               # ? cnt=1 uniq=5 score=0.33
     4: "regret ↔ frustration",                    # cnt=1 uniq=5 score=0.38
     5: "anxiety ↔ uncertainty",                   # ? cnt=1 uniq=5 score=0.34
     6: "confusion ↔ curiosity",                   # cnt=2 uniq=5 score=0.41
     7: "commitment ↔ self-doubt",                 # cnt=1 uniq=5 score=0.36
     8: "frustration ↔ confusion",                 # cnt=2 uniq=4 score=0.39
     9: "uncertainty ↔ helplessness",              # cnt=1 uniq=6 score=0.36
    10: "none ↔ determination",                    # ? cnt=1 uniq=6 score=0.31
    11: "uncertainty ↔ contentment",               # ? cnt=1 uniq=5 score=0.30
    12: "determination ↔ curiosity",               # cnt=1 uniq=7 score=0.39
    13: "frustration ↔ stress",                    # cnt=2 uniq=5 score=0.45
    14: "determination ↔ frustration",             # cnt=1 uniq=6 score=0.41
    15: "indecision ↔ determination",              # cnt=1 uniq=6 score=0.40
    16: "uncertainty ↔ skepticism",                # cnt=2 uniq=4 score=0.33
    17: "frustration ↔ confusion",                 # ? cnt=1 uniq=4 score=0.25
    18: "uncertainty ↔ determination",             # ? cnt=1 uniq=3 score=0.17
    19: "confusion ↔ self-doubt",                  # ? cnt=1 uniq=4 score=0.25
    20: "repetitive desire ↔ frustration",         # cnt=2 uniq=4 score=0.38
    21: "motivation ↔ confusion",                  # cnt=1 uniq=5 score=0.38
    22: "unrequited affection ↔ confusion",        # cnt=1 uniq=7 score=0.39
    23: "indecision ↔ frustration",                # cnt=1 uniq=5 score=0.35
    24: "overwhelmed ↔ confusion",                 # ? cnt=1 uniq=5 score=0.26
    25: "excitement ↔ self-reflection",            # cnt=1 uniq=6 score=0.46
    26: "determination ↔ curiosity",               # ? cnt=1 uniq=5 score=0.26
    27: "self-doubt ↔ indecision",                 # ? cnt=1 uniq=4 score=0.29
    28: "frustration",                             # cnt=2 uniq=4 score=0.39
    29: "indecision ↔ confusion",                  # ? cnt=1 uniq=5 score=0.28
    30: "self-doubt ↔ repetitive behavior",        # ? cnt=1 uniq=5 score=0.30
    31: "frustration ↔ uncertainty",               # ? cnt=1 uniq=3 score=0.16
    32: "enthusiasm ↔ anxiety about preparation",  # ? cnt=1 uniq=4 score=0.33
    33: "self-doubt ↔ repetitive reassurance",     # cnt=1 uniq=6 score=0.46
    34: "frustration ↔ confusion",                 # cnt=1 uniq=6 score=0.41
    35: "curiosity ↔ indecision",                  # cnt=2 uniq=7 score=0.54
    36: "paranoid uncertainty ↔ contentment",      # ? cnt=1 uniq=3 score=0.26
    37: "self-doubt ↔ confusion",                  # ? cnt=1 uniq=5 score=0.30
    38: "paranoid hypervigilance ↔ frustration",   # cnt=1 uniq=5 score=0.38
    39: "uncertainty ↔ frustration",               # cnt=2 uniq=5 score=0.46
    40: "inner conflict ↔ eagerness",              # ? cnt=1 uniq=5 score=0.31
    41: "indecision ↔ confusion",                  # cnt=1 uniq=5 score=0.38
    42: "frustration ↔ uncertainty",               # cnt=1 uniq=8 score=0.56
    43: "cynicism ↔ confusion",                    # cnt=1 uniq=5 score=0.38
    44: "uncertainty ↔ confusion",                 # ? cnt=1 uniq=4 score=0.25
    45: "determination ↔ hunger",                  # cnt=1 uniq=5 score=0.35
    46: "fatigue ↔ apprehension",                  # cnt=1 uniq=6 score=0.40
    47: "frustration ↔ eagerness",                 # cnt=2 uniq=4 score=0.31
    48: "determination ↔ confusion",               # cnt=1 uniq=6 score=0.36
    49: "academic ambition ↔ uncertainty",         # ? cnt=1 uniq=2 score=0.15
    50: "enthusiasm ↔ uncertainty",                # ? cnt=1 uniq=4 score=0.24
    51: "uncertainty ↔ frustration",               # cnt=2 uniq=7 score=0.56
    52: "confusion ↔ encouragement",               # cnt=1 uniq=5 score=0.35
    53: "frustration ↔ self-doubt",                # cnt=1 uniq=5 score=0.35
    54: "frustration ↔ uncertainty",               # cnt=1 uniq=6 score=0.41
    55: "curiosity ↔ self-doubt",                  # cnt=2 uniq=5 score=0.39
    56: "determination ↔ confusion",               # ? cnt=1 uniq=3 score=0.20
    57: "uncertainty ↔ generosity",                # ? cnt=1 uniq=4 score=0.30
    58: "existential uncertainty ↔ relaxed acceptance",  # cnt=1 uniq=6 score=0.41
    59: "determination ↔ confusion",               # ? cnt=1 uniq=5 score=0.30
    60: "confusion ↔ disappointment",              # cnt=1 uniq=8 score=0.54
    61: "identity crisis ↔ excitement",            # ? cnt=1 uniq=3 score=0.22
    62: "self-doubt ↔ frustration",                # ? cnt=1 uniq=5 score=0.34
    63: "confusion ↔ disappointment",              # cnt=1 uniq=7 score=0.41
}


def label_for(index: int) -> str:
    return BASIS_LABELS.get(index, "latent component")
