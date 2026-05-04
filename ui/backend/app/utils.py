"""Small numerical / text helpers shared across the engine."""
from __future__ import annotations

import difflib
import hashlib
import re
from typing import List, Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def tanh_clamp(x: np.ndarray) -> np.ndarray:
    """Squash to (-1, 1) without hard clipping."""
    return np.tanh(x)


def seed_from_text(text: str) -> int:
    """Stable 32-bit seed derived from text contents."""
    h = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big", signed=False)


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def diff_words(a: str, b: str) -> List[Tuple[str, str]]:
    """Word-level diff returning ``(type, text)`` pairs.

    ``type`` is one of ``same`` | ``removed`` | ``added``.
    """
    # Split keeping whitespace so we can faithfully reassemble the text.
    pattern = re.compile(r"(\s+)")
    a_tokens = pattern.split(a)
    b_tokens = pattern.split(b)

    sm = difflib.SequenceMatcher(a=a_tokens, b=b_tokens, autojunk=False)
    out: List[Tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(("same", "".join(a_tokens[i1:i2])))
        elif tag == "delete":
            out.append(("removed", "".join(a_tokens[i1:i2])))
        elif tag == "insert":
            out.append(("added", "".join(b_tokens[j1:j2])))
        elif tag == "replace":
            out.append(("removed", "".join(a_tokens[i1:i2])))
            out.append(("added", "".join(b_tokens[j1:j2])))
    return out
