"""Phase A unit tests — schema correctness and loader contract."""

from __future__ import annotations

import pytest

from src.data.schema import (
    CATEGORIES,
    DAILY_DIALOG_TO_PLUTCHIK,
    EmotionExample,
    GO_EMOTIONS_TO_PLUTCHIK,
    ISEAR_TO_PLUTCHIK,
    PLUTCHIK,
    PLUTCHIK_OPPOSITE,
    SEMEVAL_TO_PLUTCHIK,
    is_valid_category,
)


def test_categories_are_unique_and_canonical():
    assert len(set(CATEGORIES)) == len(CATEGORIES)
    assert "neutral" in CATEGORIES
    assert set(PLUTCHIK).issubset(CATEGORIES)


def test_label_maps_target_canonical_categories():
    for src_map in (GO_EMOTIONS_TO_PLUTCHIK, SEMEVAL_TO_PLUTCHIK,
                    DAILY_DIALOG_TO_PLUTCHIK, ISEAR_TO_PLUTCHIK):
        for v in src_map.values():
            assert v is None or is_valid_category(v), v


def test_plutchik_opposites_are_symmetric_or_none():
    for a, b in PLUTCHIK_OPPOSITE.items():
        if b is None:
            continue
        assert PLUTCHIK_OPPOSITE[b] == a, (a, b)


def test_emotion_example_roundtrip():
    ex = EmotionExample(
        id="t:1",
        text="hello",
        source="t",
        label_primary="joy",
        label_multi=["joy", "trust"],
        intensity=0.7,
        vad=(3.0, 4.0, 2.5),
        source_labels={"raw": "joy"},
    )
    d = ex.to_dict()
    assert d["vad"] == [3.0, 4.0, 2.5]
    back = EmotionExample.from_dict(d)
    assert back.vad == (3.0, 4.0, 2.5)
    assert back.label_multi == ["joy", "trust"]


@pytest.mark.parametrize("label", ["joy", "neutral", "anger"])
def test_is_valid_category_accepts_known(label):
    assert is_valid_category(label)


def test_is_valid_category_rejects_unknown():
    assert not is_valid_category("schadenfreude")
