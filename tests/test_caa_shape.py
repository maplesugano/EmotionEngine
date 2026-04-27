"""Smoke test for the CAA build pipeline.

Loads the saved ``caa.pt`` (skipped if missing) and asserts shape invariants.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


CAA_PATH = Path(__file__).resolve().parents[1] / "data" / "emotion_code" / "caa.pt"


@pytest.mark.skipif(not CAA_PATH.exists(), reason="run `python -m src.emotion_code.caa` first")
def test_caa_shape():
    payload = torch.load(CAA_PATH, map_location="cpu", weights_only=False)
    cats = payload["categories"]
    layers = payload["layers"]
    vec = payload["vectors"]
    assert vec.ndim == 3
    assert vec.shape[0] == len(cats), f"got {vec.shape}, cats={cats}"
    assert vec.shape[1] == len(layers), f"got {vec.shape}, layers={layers}"
    assert vec.shape[2] > 0
    # Plutchik 8 (no neutral row in pairs)
    assert len(cats) == 8
    # No NaN, non-trivial norms
    norms = vec.norm(dim=-1)
    assert torch.isfinite(norms).all()
    assert (norms > 0).all()


def test_make_split_deterministic():
    import pandas as pd
    from src.emotion_code.io import make_split

    meta = pd.DataFrame({
        "pair_id": list(range(100)),
        "category": ["joy"] * 50 + ["fear"] * 50,
    })
    a1, b1 = make_split(meta, train_frac=0.8, seed=0)
    a2, b2 = make_split(meta, train_frac=0.8, seed=0)
    assert (a1 == a2).all() and (b1 == b2).all()
    assert a1.sum() == 80
    assert b1.sum() == 20
    # Stratified: train should have ~40 from each cat
    cats = meta["category"].to_numpy()
    assert (cats[a1] == "joy").sum() == 40
    assert (cats[a1] == "fear").sum() == 40
