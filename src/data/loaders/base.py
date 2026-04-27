"""Common helpers for source loaders."""

from __future__ import annotations


class LoaderError(RuntimeError):
    """Raised when a source dataset is unavailable or malformed."""


def normalise_text(text: str) -> str:
    """Strip whitespace and collapse internal blanks; preserve punctuation."""
    return " ".join(text.split())


def make_id(source: str, source_id) -> str:
    return f"{source}:{source_id}"
