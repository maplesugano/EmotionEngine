"""Pair-similarity vs additivity-residual analysis.

Tests whether the off-diagonal residual ratio in
``additivity_readout.csv`` correlates with how similar the two CAA
category vectors are.  Hypothesis: more orthogonal pairs should be
*more* additive (smaller residual ratio) because their subspaces don't
interact; near-aligned pairs should share variance and break additivity.

Outputs:
- ``pair_similarity.csv``  per-pair cosine of (a) raw CAA, (b) OLS
  reconstruction, (c) basis-weight vectors w_a vs w_b.
- ``residual_vs_similarity.csv``  per (cell × cell) residual ratio joined
  with similarities.
- ``summary.json``  Spearman/Pearson correlations.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--additivity-dir", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True,
                   help="Phase-1 OLS/NNLS/... weights .pt")
    p.add_argument("--basis", type=Path, required=True,
                   help="Basis .pt with W matrix")
    p.add_argument("--caa", type=Path,
                   default=Path("data/emotion_code/caa.pt"),
                   help="Phase B-5 CAA tensor (for raw v_c cosines)")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load basis + weights ----------
    bnd = torch.load(args.basis, weights_only=False)
    decomp = bnd.get("decomposer", "ica")
    inner = bnd.get(decomp, {})
    if "W" not in inner:
        raise SystemExit(f"basis matrix not found in {args.basis}; "
                         f"keys={list(bnd.keys())}, inner={list(inner.keys())}")
    W = np.asarray(inner["W"], dtype=np.float32)
    if W.shape[1] != 4096 and W.shape[0] == 4096:
        W = W.T
    print(f"[sim] basis W shape = {W.shape} (decomposer={decomp})")

    wpkg = torch.load(args.weights, weights_only=False)
    cats = list(wpkg["categories"])
    layer = int(wpkg["layer"])
    Wc = {c: np.asarray(wpkg["weights"][c]["ols"], dtype=np.float32) for c in cats}

    # Reconstructed CAA
    Vrecon = {c: Wc[c] @ W for c in cats}  # [d]

    # Raw CAA
    caa = torch.load(args.caa, weights_only=False)
    Vraw = {}
    try:
        cat_names = list(caa["categories"])
        layers = list(caa["layers"])
        vecs = np.asarray(caa["vectors"])  # [C, L, d]
        if layer in layers:
            li = layers.index(layer)
            for i, c in enumerate(cat_names):
                Vraw[c] = vecs[i, li].astype(np.float32)
    except Exception as e:
        print(f"[sim] could not load raw CAA: {e}")
    print(f"[sim] raw CAA loaded for {len(Vraw)} cats at layer {layer}")

    # ---------- Read additivity readout residuals ----------
    radd = pd.read_csv(args.additivity_dir / "additivity_readout.csv")
    pairs = sorted({(r.cat_a, r.cat_b) for r in radd.itertuples()})

    # ---------- Pair similarities ----------
    sim_rows = []
    for (a, b) in pairs:
        row = {"cat_a": a, "cat_b": b,
               "cos_recon": _cos(Vrecon[a], Vrecon[b]),
               "cos_w":     _cos(Wc[a], Wc[b])}
        if a in Vraw and b in Vraw:
            row["cos_caa_raw"] = _cos(Vraw[a], Vraw[b])
        else:
            row["cos_caa_raw"] = float("nan")
        sim_rows.append(row)
    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv(args.output_dir / "pair_similarity.csv", index=False)
    print("\n[sim] === pair similarities ===")
    print(sim_df.to_string(index=False))

    # ---------- Per-cell join ----------
    off = radd[(radd.alpha != 0) & (radd.beta != 0)].copy()
    off = off.merge(sim_df, on=["cat_a", "cat_b"], how="left")
    off["abs_alpha_beta"] = off["alpha"].abs() * off["beta"].abs()
    off["sign_alpha_beta"] = np.sign(off["alpha"] * off["beta"])
    off.to_csv(args.output_dir / "residual_vs_similarity.csv", index=False)

    # Per-pair median residual ratio
    pair_med = (off.groupby(["cat_a", "cat_b"])
                  .agg(median_resid_ratio=("median_resid_ratio", "median"),
                       mean_resid_ratio=("mean_resid_ratio", "mean"))
                  .reset_index()
                  .merge(sim_df, on=["cat_a", "cat_b"]))
    pair_med.to_csv(args.output_dir / "pair_residual_summary.csv", index=False)
    print("\n[sim] === per-pair median residual ratio vs similarity ===")
    print(pair_med.to_string(index=False))

    # ---------- Correlations ----------
    def _corr(x, y):
        x = np.asarray(x); y = np.asarray(y)
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < 3:
            return {"n": int(m.sum()), "pearson_r": None, "pearson_p": None,
                    "spearman_r": None, "spearman_p": None}
        pr, pp = pearsonr(x[m], y[m])
        sr, sp = spearmanr(x[m], y[m])
        return {"n": int(m.sum()),
                "pearson_r": float(pr), "pearson_p": float(pp),
                "spearman_r": float(sr), "spearman_p": float(sp)}

    corrs = {}
    for sim_col in ("cos_recon", "cos_w", "cos_caa_raw"):
        # cell-level (32 datapoints)
        corrs[f"cell_{sim_col}"] = _corr(off[sim_col].values,
                                         off["median_resid_ratio"].values)
        # pair-level (4 datapoints)
        corrs[f"pair_{sim_col}"] = _corr(pair_med[sim_col].values,
                                         pair_med["median_resid_ratio"].values)

    summary = {
        "additivity_dir": str(args.additivity_dir),
        "n_pairs": len(pairs),
        "n_offdiag_cells": int(len(off)),
        "pair_similarity": sim_df.to_dict(orient="records"),
        "pair_residual": pair_med.to_dict(orient="records"),
        "correlations": corrs,
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[sim] === correlations (residual ratio vs similarity) ===")
    for k, v in corrs.items():
        if v["pearson_r"] is None:
            print(f"  {k:25s} n={v['n']} (insufficient)")
        else:
            print(f"  {k:25s} n={v['n']}  pearson r={v['pearson_r']:+.3f} "
                  f"(p={v['pearson_p']:.3f})  spearman r={v['spearman_r']:+.3f} "
                  f"(p={v['spearman_p']:.3f})")

    print(f"\n[sim] wrote {args.output_dir}/")


if __name__ == "__main__":
    main()
