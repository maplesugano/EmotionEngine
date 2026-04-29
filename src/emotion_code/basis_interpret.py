"""Label-free interpretation of basis components via top-activating pairs.

Given a basis artifact (from :mod:`src.emotion_code.basis` or
:mod:`src.emotion_code.basis_sweep`) and the activation bundle it was fit on,
project per-pair Δ onto each basis vector ``b_j``:

    s_{i,j} = ⟨Δ_i, b_j⟩ / (||Δ_i|| · ||b_j||)

For each component j, report the top-N pairs by ``+s`` and ``-s``, joining in
the original positive / negative texts from ``pairs.parquet``. This is the
primary tool for asking "is this basis direction a real, sub-verbal emotion
primitive, or just a relabelling of a Plutchik category?".

Usage
-----
    uv run python -m src.emotion_code.basis_interpret \
        --basis data/emotion_code/basis_sweep/k016.pt \
        --which nmf --top-n 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.emotion_code.basis import _build_delta
from src.emotion_code.io import load_activations


def _project(delta: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Cosine-style projection. Returns [N, k]."""
    d_norm = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-12
    w_norm = np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
    return (delta / d_norm) @ (W / w_norm).T


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--basis", type=Path, required=True)
    p.add_argument("--which", default=None,
                   help="Decomposer key inside the artifact (nmf/pca/ica/dict). "
                        "Default = payload['decomposer'] or first matching block.")
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--activations-root", type=Path, default=Path("data/activations"))
    p.add_argument("--pairs", type=Path, default=Path("data/contrastive/pairs.parquet"))
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--output", type=Path, default=None,
                   help="Default: <basis>.interpret.<which>.json next to the artifact.")
    args = p.parse_args()

    payload = torch.load(args.basis, weights_only=False, map_location="cpu")
    which = args.which or payload.get("decomposer")
    if which is None:
        for cand in ("nmf", "pca", "ica", "dict"):
            if cand in payload:
                which = cand
                break
    if which is None or which not in payload:
        raise SystemExit(f"could not find decomposer block in {list(payload)}")
    W = payload[which]["W"].numpy()  # [k, D]
    layer = int(payload["layer"])
    train_mask_t = payload.get("train_mask")
    if train_mask_t is None:
        raise SystemExit("basis artifact missing 'train_mask'; refit with sweep CLI")
    train_mask = train_mask_t.numpy().astype(bool)

    cfg = yaml.safe_load(args.config.read_text())
    profile = payload.get("profile", cfg["active"])
    bundle = load_activations(profile=profile, root=args.activations_root)
    delta = _build_delta(bundle, layer, train_mask)

    pairs = pd.read_parquet(args.pairs)
    meta_train = bundle.meta.loc[train_mask].reset_index(drop=True)
    if "pair_id" not in meta_train.columns or "pair_id" not in pairs.columns:
        raise SystemExit("expected 'pair_id' in both meta and pairs")
    joined = meta_train.merge(
        pairs[["pair_id", "pos_text", "neg_text", "category", "provenance"]],
        on="pair_id", how="left", suffixes=("", "_pair"),
    )
    if len(joined) != len(meta_train):
        raise SystemExit("join produced unexpected row count")

    scores = _project(delta, W)  # [N, k]
    k = W.shape[0]
    top_n = args.top_n

    components = []
    for j in range(k):
        s = scores[:, j]
        top_idx = np.argsort(-s)[:top_n]
        bot_idx = np.argsort(s)[:top_n]

        def _rows(indices):
            out = []
            for i in indices:
                row = joined.iloc[int(i)]
                out.append({
                    "pair_id": int(row["pair_id"]),
                    "category": str(row["category"]),
                    "provenance": str(row["provenance"]),
                    "score": float(s[int(i)]),
                    "pos_text": str(row["pos_text"]),
                    "neg_text": str(row["neg_text"]),
                })
            return out

        # Plutchik-category histogram for this component (label-leak diagnostic).
        cat_hist_pos = (
            joined.iloc[top_idx]["category"].value_counts().to_dict()
        )
        components.append({
            "component": j,
            "category_hist_top": {str(k_): int(v) for k_, v in cat_hist_pos.items()},
            "top_positive": _rows(top_idx),
            "top_negative": _rows(bot_idx),
        })

    out_path = args.output or args.basis.with_suffix(f".interpret.{which}.json")
    out_path.write_text(json.dumps({
        "basis": str(args.basis),
        "which": which,
        "layer": layer,
        "k": int(k),
        "top_n": top_n,
        "components": components,
    }, indent=2, ensure_ascii=False))
    print(f"[interpret] wrote {out_path}  (k={k}, top_n={top_n})")


if __name__ == "__main__":
    main()
