"""Held-out reconstruction R² for sweep artifacts.

For each artifact under ``data/emotion_code/basis_sweep/`` we evaluate how
much of the *validation-side* per-pair Δ activation can be reconstructed by
projecting onto the basis ``W`` (least-squares coefficients, no refit).

This gives a label-free, VAD-free answer to "how big does k need to be?":
plot R²(k) per decomposer and look for the plateau.

    R² = 1 - ||Δ_val - Δ_val · W⁺ᵀ · W||² / ||Δ_val - Δ̄_train||²

Output: ``data/emotion_code/basis_sweep/reconstruction.csv``
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.emotion_code.basis import _build_delta
from src.emotion_code.io import load_activations, make_split

_ARTIFACT_RE = re.compile(r"^(?P<dec>[a-z]+)_k(?P<k>\d+)_seed(?P<seed>\d+)\.pt$")


def _r2(delta: np.ndarray, W: np.ndarray, mean_train: np.ndarray) -> float:
    # Least-squares coefficients in the basis: H = Δ · W⁺ᵀ
    coef, *_ = np.linalg.lstsq(W.T, delta.T, rcond=None)   # [k, N]
    recon = (W.T @ coef).T                                 # [N, D]
    ss_res = float(((delta - recon) ** 2).sum())
    ss_tot = float(((delta - mean_train) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", type=Path, default=Path("data/emotion_code/basis_sweep"))
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--activations-root", type=Path, default=Path("data/activations"))
    p.add_argument("--output", type=Path,
                   default=Path("data/emotion_code/basis_sweep/reconstruction.csv"))
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    profile = cfg["active"]
    bundle = load_activations(profile=profile, root=args.activations_root)

    delta_cache: dict[tuple, tuple] = {}
    rows: list[dict] = []
    for art_path in sorted(args.sweep_dir.glob("*_k*_seed*.pt")):
        m = _ARTIFACT_RE.match(art_path.name)
        if not m:
            continue
        dec = m.group("dec")
        payload = torch.load(art_path, weights_only=False, map_location="cpu")
        layer = int(payload["layer"])
        split_seed = int(payload.get("split_seed", 0))
        train_frac = float(payload.get("train_frac", 0.8))
        key = (layer, split_seed, train_frac)
        if key not in delta_cache:
            train_mask, val_mask = make_split(bundle.meta, train_frac=train_frac, seed=split_seed)
            delta_train = _build_delta(bundle, layer, train_mask)
            delta_val = _build_delta(bundle, layer, val_mask)
            delta_cache[key] = (delta_train, delta_val, delta_train.mean(axis=0, keepdims=True))
        delta_train, delta_val, mean_tr = delta_cache[key]

        W = payload[dec]["W"].numpy()
        r2_train = _r2(delta_train, W, mean_tr)
        r2_val = _r2(delta_val, W, mean_tr)
        row = {"decomposer": dec, "k": int(payload["k"]), "seed": int(payload.get("seed", 0)),
               "layer": layer, "r2_train": r2_train, "r2_val": r2_val,
               "n_train": int(delta_train.shape[0]), "n_val": int(delta_val.shape[0])}
        rows.append(row)
        print(f"[recon] {art_path.name}  R²_train={r2_train:.3f}  R²_val={r2_val:.3f}")

    df = pd.DataFrame(rows).sort_values(["decomposer", "k", "seed"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[recon] wrote {args.output}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
