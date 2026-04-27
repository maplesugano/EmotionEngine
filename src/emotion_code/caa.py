"""Contrastive Activation Addition (CAA) — per-category mean-difference vectors.

For each (category c, layer L):

    v_{c,L} = mean(pos[train,L]) - mean(neg[train,L])

The output ``caa.pt`` packs all categories × layers into a single tensor
``[n_categories, n_layers, d_model]`` plus index lists, mirroring the
formulation in Panickssery et al. 2024 (CAA, eq. 1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.emotion_code.io import ActivationBundle, load_activations, make_split


def compute_caa(
    bundle: ActivationBundle,
    train_mask: np.ndarray,
    categories: list[str] | None = None,
) -> dict:
    """Return CAA vectors and bookkeeping.

    Returns
    -------
    dict with keys:
        categories : list[str]            (length C, sorted)
        layers     : list[int]            (length L)
        vectors    : Tensor[C, L, d_model]
        norms      : Tensor[C, L]         (L2 norm per (cat, layer))
        n_per_cat  : Tensor[C]            train-side sample count per category
    """
    meta = bundle.meta
    layers = bundle.layers
    if categories is None:
        categories = sorted(meta["category"].unique().tolist())

    C, L, D = len(categories), len(layers), bundle.d_model
    vectors = torch.zeros(C, L, D, dtype=torch.float32)
    n_per_cat = torch.zeros(C, dtype=torch.long)

    cat_arr = meta["category"].to_numpy()
    for ci, cat in enumerate(categories):
        sel = train_mask & (cat_arr == cat)
        if not sel.any():
            raise RuntimeError(f"no train rows for category {cat!r}")
        idx = torch.from_numpy(np.flatnonzero(sel))
        n_per_cat[ci] = int(idx.numel())
        for li, layer in enumerate(layers):
            pos_mean = bundle.pos[layer].index_select(0, idx).mean(dim=0)
            neg_mean = bundle.neg[layer].index_select(0, idx).mean(dim=0)
            vectors[ci, li] = pos_mean - neg_mean

    norms = vectors.norm(dim=-1)
    return {
        "categories": categories,
        "layers": layers,
        "vectors": vectors,
        "norms": norms,
        "n_per_cat": n_per_cat,
    }


def cosine_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """Per-layer cosine similarity matrix between category vectors.

    vectors: [C, L, D]  ->  cos: [L, C, C]
    """
    C, L, _ = vectors.shape
    out = torch.zeros(L, C, C, dtype=torch.float32)
    for li in range(L):
        v = vectors[:, li, :]
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
        out[li] = v @ v.T
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument(
        "--activations-root", type=Path, default=Path("data/activations")
    )
    p.add_argument("--output-dir", type=Path, default=Path("data/emotion_code"))
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    profile = cfg["active"]
    print(f"[caa] profile={profile}")

    bundle = load_activations(profile=profile, root=args.activations_root)
    print(
        f"[caa] loaded {len(bundle.meta)} rows, "
        f"layers={bundle.layers}, d_model={bundle.d_model}"
    )

    train_mask, val_mask = make_split(
        bundle.meta, train_frac=args.train_frac, seed=args.seed
    )
    print(f"[caa] split: train={train_mask.sum()} val={val_mask.sum()}")

    out = compute_caa(bundle, train_mask)
    cos = cosine_matrix(out["vectors"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "categories": out["categories"],
        "layers": out["layers"],
        "vectors": out["vectors"],
        "norms": out["norms"],
        "n_per_cat": out["n_per_cat"],
        "profile": profile,
        "split_seed": args.seed,
        "train_frac": args.train_frac,
        "train_mask": torch.from_numpy(train_mask),
        "val_mask": torch.from_numpy(val_mask),
    }
    out_path = args.output_dir / "caa.pt"
    torch.save(payload, out_path)
    print(f"[caa] wrote {out_path}  vectors.shape={tuple(out['vectors'].shape)}")

    # Per-layer cosine matrix as long-form parquet for easy plotting.
    rows = []
    for li, layer in enumerate(out["layers"]):
        for i, ci in enumerate(out["categories"]):
            for j, cj in enumerate(out["categories"]):
                rows.append(
                    {
                        "layer": layer,
                        "cat_i": ci,
                        "cat_j": cj,
                        "cosine": float(cos[li, i, j]),
                    }
                )
    cos_path = args.output_dir / "caa_cos.parquet"
    pd.DataFrame.from_records(rows).to_parquet(cos_path, index=False)

    summary = {
        "profile": profile,
        "categories": out["categories"],
        "layers": out["layers"],
        "n_per_cat": {c: int(n) for c, n in zip(out["categories"], out["n_per_cat"])},
        "norms": {
            c: {str(l): float(out["norms"][i, j]) for j, l in enumerate(out["layers"])}
            for i, c in enumerate(out["categories"])
        },
    }
    (args.output_dir / "caa.summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[caa] wrote {cos_path} and caa.summary.json")


if __name__ == "__main__":
    main()
