"""Cheap text-pathology metrics for steered generations.

Used by ``experiments/eval_basis_pathology.py`` to flag basis components that,
when injected, push the model into degenerate behaviours (token loops, n-gram
repetition, profanity / dirty language). All metrics are deterministic and
require no extra model.

Conventions
-----------
- All scores in [0, 1] except ``max_run`` (token count) and
  ``compress_ratio`` (raw zlib ratio in (0, ~1]).
- Higher repetition / toxicity values = worse.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from typing import Iterable

# A conservative profanity wordlist. Source: well-known LDNOOBW English list,
# trimmed to unambiguous obscenities / slurs commonly emitted by LLM token
# loops. Kept short on purpose — broader lists explode false positives on
# benign text. Match is whole-word, case-insensitive.
_PROFANITY: tuple[str, ...] = (
    "fuck", "fucking", "fucker", "fucked",
    "shit", "shitty", "bullshit",
    "bitch", "bitches",
    "asshole", "ass", "arsehole",
    "cunt", "dick", "pussy", "cock", "cocks",
    "bastard", "damn", "goddamn",
    "slut", "whore",
    "nigger", "nigga", "faggot", "fag", "retard", "retarded",
)
_PROFANITY_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _PROFANITY) + r")\b",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def ngram_repetition(tokens: list[str], n: int) -> float:
    """1 - (#unique n-grams / #n-grams). 0 = all unique, 1 = all identical."""
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def max_token_run(tokens: list[str]) -> int:
    """Longest run of an identical token (catches 'the the the …')."""
    if not tokens:
        return 0
    best = cur = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def compress_ratio(text: str) -> float:
    """zlib(text)/len(text). Lower => more redundant. ~0.3 for loops."""
    if not text:
        return 1.0
    raw = text.encode("utf-8", errors="ignore")
    if not raw:
        return 1.0
    comp = zlib.compress(raw, level=6)
    return len(comp) / len(raw)


def unique_token_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def toxicity_rate(text: str) -> float:
    """Fraction of word-tokens that match the profanity list."""
    toks = _tokens(text)
    if not toks:
        return 0.0
    hits = len(_PROFANITY_RE.findall(text))
    return hits / len(toks)


def toxicity_hits(text: str) -> list[str]:
    return [m.group(0).lower() for m in _PROFANITY_RE.finditer(text)]


def non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) > 127) / len(text)


def repetition_metrics(text: str) -> dict[str, float]:
    """Bundle of repetition-only metrics for one generation."""
    toks = _tokens(text)
    return {
        "n_tokens": float(len(toks)),
        "rep_2": ngram_repetition(toks, 2),
        "rep_3": ngram_repetition(toks, 3),
        "rep_4": ngram_repetition(toks, 4),
        "max_run": float(max_token_run(toks)),
        "max_run_norm": float(max_token_run(toks)) / max(len(toks), 1),
        "compress_ratio": compress_ratio(text),
        "unique_token_ratio": unique_token_ratio(toks),
    }


def pathology_metrics(text: str) -> dict[str, float]:
    """All cheap pathology metrics (repetition + toxicity + sanity)."""
    m = repetition_metrics(text)
    m["toxicity_rate"] = toxicity_rate(text)
    m["toxicity_hits"] = float(len(toxicity_hits(text)))
    m["non_ascii_ratio"] = non_ascii_ratio(text)
    return m


def composite_pathology_score(
    rep_4: float, max_run_norm: float, toxicity_rate: float,
    *, w_rep: float = 1.0, w_run: float = 1.0, w_tox: float = 2.0,
) -> float:
    """Weighted sum used to rank components. Toxicity weighted higher."""
    return w_rep * rep_4 + w_run * max_run_norm + w_tox * toxicity_rate


__all__ = [
    "ngram_repetition",
    "max_token_run",
    "compress_ratio",
    "unique_token_ratio",
    "toxicity_rate",
    "toxicity_hits",
    "non_ascii_ratio",
    "repetition_metrics",
    "pathology_metrics",
    "composite_pathology_score",
]
