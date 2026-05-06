"""Verify that excluding pathological basis components retains expressiveness.

After ``eval_basis_pathology`` flags some components as repetition / dirty-
language drivers, we want to know: do the *remaining* components still span
the emotional content of the corpus? This script answers that without any
GPU work — pure linear algebra on the cached basis payload.

Metrics (computed for the active vs full basis):
- ``H @ W`` Frobenius retention      → per-pair reconstruction in residual space
- per-category loading L2 retention  → how much of each Plutchik direction survives
- category centroid cosine matrix    → pairwise distinguishability of the 8 categories
- ``W_macro[:, keep]`` condition #   → stability of the slider→basis pinv path
- per-category fraction of loading energy that lived on the excluded axes

Usage
-----
    uv run python -m experiments.eval_basis_pathology_coverage \
        --basis data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt \
        --exclude data/emotion_code/basis_sweep_L22/ica_k064_seed0.exclude.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _select_block(payload: dict, which: str | None) -> tuple[str, dict]:
    if which is None:
        which = payload.get("decomposer")
        if which is None:
            for cand in ("ica", "nmf", "pca", "dict"):
                if cand in payload:
                    which = cand
                    break
    return which, payload[which]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--basis", type=Path, required=True)
    p.add_argument("--exclude", type=Path, default=None,
                   help="Path to the exclude.json sidecar. Defaults to "
                        "<basis>.exclude.json.")
    p.add_argument("--which", default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    payload = torch.load(args.basis, weights_only=False, map_location="cpu")
    which, block = _select_block(payload, args.which)
    W = block["W"].numpy().astype(np.float64)            # [k, D]
    H = block["H"].numpy().astype(np.float64)            # [N, k]
    cat_load = block["category_loadings"].numpy().astype(np.float64)  # [C, k]
    categories: list[str] = list(payload["categories"])
    layer = int(payload["layer"])
    k, D = W.shape
    C = cat_load.shape[0]

    excl_path = args.exclude or args.basis.with_suffix(".exclude.json")
    if not excl_path.exists():
        raise FileNotFoundError(
            f"exclude sidecar not found at {excl_path}. Run "
            "eval_basis_pathology.py first."
        )
    excluded = sorted({int(i) for i in json.loads(excl_path.read_text())["exclude"]})
    keep = sorted(set(range(k)) - set(excluded))

    out_root = args.output_dir or (
        Path("experiments/results/basis_pathology") / args.basis.stem
    )
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[coverage] basis={args.basis.name} which={which} layer={layer} k={k}")
    print(f"[coverage] excluded={len(excluded)} keep={len(keep)}  "
          f"({len(keep)/k:.0%} of axes)")

    # --------------------------------------------------------------- 1. H@W reconstruction
    #   Δ̂_full   = H @ W                       (rank ≤ k)
    #   Δ̂_keep   = H_keep @ W_keep             (= H @ M @ W where M zeros excluded)
    #   retention = ||Δ̂_keep||_F / ||Δ̂_full||_F
    HW_full = H @ W
    HW_keep = H[:, keep] @ W[keep, :]
    fro_full = float(np.linalg.norm(HW_full))
    fro_keep = float(np.linalg.norm(HW_keep))
    fro_lost = float(np.linalg.norm(HW_full - HW_keep))
    fro_ret = fro_keep / max(fro_full, 1e-12)
    print(f"[coverage] HW Frobenius: full={fro_full:.3f}  keep={fro_keep:.3f}  "
          f"retention={fro_ret:.3f}  lost_norm={fro_lost:.3f}")

    # --------------------------------------------------------------- 2. category loading L2
    #   per-category L2 of cat_load[c, :] before vs after masking.
    cat_full = np.linalg.norm(cat_load, axis=1)
    cat_keep = np.linalg.norm(cat_load[:, keep], axis=1)
    cat_lost = np.linalg.norm(cat_load[:, excluded] if excluded else np.zeros((C, 0)), axis=1)
    cat_df = pd.DataFrame({
        "category": categories,
        "loading_l2_full": cat_full,
        "loading_l2_keep": cat_keep,
        "loading_l2_excluded": cat_lost,
        "retention": cat_keep / np.clip(cat_full, 1e-12, None),
        "fraction_lost": cat_lost / np.clip(cat_full, 1e-12, None),
    })
    print("\n[coverage] per-category loading L2 retention:")
    print(cat_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # --------------------------------------------------------------- 3. category-pair separability
    #   cosine matrix on cat_load rows (full vs keep). Off-diagonal
    #   max-cosine summarises distinguishability.
    def _cosmat(M: np.ndarray) -> np.ndarray:
        n = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)
        return n @ n.T

    cos_full = _cosmat(cat_load)
    cos_keep = _cosmat(cat_load[:, keep])
    off = ~np.eye(C, dtype=bool)
    print(f"\n[coverage] off-diag cosine(full)  mean={cos_full[off].mean():+.3f}  "
          f"max={cos_full[off].max():+.3f}")
    print(f"[coverage] off-diag cosine(keep)  mean={cos_keep[off].mean():+.3f}  "
          f"max={cos_keep[off].max():+.3f}  "
          f"(higher = categories collapsing toward each other)")

    # --------------------------------------------------------------- 4. macro → basis pinv stability
    #   The UI's slider path uses pinv(W_macro). When we drop columns the
    #   problem becomes more under-determined and the pinv solution norm /
    #   condition number can blow up.
    s_full = np.linalg.svd(cat_load, compute_uv=False)
    s_keep = np.linalg.svd(cat_load[:, keep], compute_uv=False)
    cond_full = s_full[0] / max(s_full[-1], 1e-12)
    cond_keep = s_keep[0] / max(s_keep[-1], 1e-12)
    pinv_norm_full = float(np.linalg.norm(np.linalg.pinv(cat_load)))
    pinv_norm_keep = float(np.linalg.norm(np.linalg.pinv(cat_load[:, keep])))
    print(f"\n[coverage] W_macro singular values:")
    print(f"           full: {np.array2string(s_full, precision=3)}")
    print(f"           keep: {np.array2string(s_keep, precision=3)}")
    print(f"[coverage] cond(W_macro): full={cond_full:.2f}  keep={cond_keep:.2f}")
    print(f"[coverage] ||pinv||_F   : full={pinv_norm_full:.2f}  keep={pinv_norm_keep:.2f}  "
          f"(↑ = sliders need bigger basis edits to hit the same macro target)")

    # --------------------------------------------------------------- 5. residual rank check
    #   Did we drop a singular axis from W itself? Check the rank.
    sW_full = np.linalg.svd(W, compute_uv=False)
    sW_keep = np.linalg.svd(W[keep, :], compute_uv=False)
    eff_rank_full = float((sW_full / sW_full[0] > 1e-3).sum())
    eff_rank_keep = float((sW_keep / sW_keep[0] > 1e-3).sum())
    print(f"\n[coverage] W effective rank (sv > 1e-3 * sv[0]): "
          f"full={eff_rank_full:.0f}/{k}  keep={eff_rank_keep:.0f}/{len(keep)}")

    # --------------------------------------------------------------- write artefacts
    summary = {
        "basis": str(args.basis),
        "exclude": str(excl_path),
        "k": int(k),
        "n_excluded": len(excluded),
        "n_keep": len(keep),
        "excluded": excluded,
        "frobenius_retention": fro_ret,
        "frobenius_lost_norm": fro_lost,
        "category_loading_retention_min": float(cat_df["retention"].min()),
        "category_loading_retention_mean": float(cat_df["retention"].mean()),
        "offdiag_cosine_mean_full": float(cos_full[off].mean()),
        "offdiag_cosine_mean_keep": float(cos_keep[off].mean()),
        "cond_W_macro_full": float(cond_full),
        "cond_W_macro_keep": float(cond_keep),
        "pinv_W_macro_norm_full": pinv_norm_full,
        "pinv_W_macro_norm_keep": pinv_norm_keep,
        "W_effective_rank_full": eff_rank_full,
        "W_effective_rank_keep": eff_rank_keep,
    }
    (out_root / "coverage.summary.json").write_text(json.dumps(summary, indent=2))
    cat_df.to_csv(out_root / "coverage_per_category.csv", index=False)
    pd.DataFrame(cos_full, index=categories, columns=categories).to_csv(
        out_root / "coverage_cosine_full.csv"
    )
    pd.DataFrame(cos_keep, index=categories, columns=categories).to_csv(
        out_root / "coverage_cosine_keep.csv"
    )
    print(f"\n[coverage] wrote {out_root}/coverage.summary.json")
    print(f"[coverage] wrote {out_root}/coverage_per_category.csv")
    print(f"[coverage] wrote {out_root}/coverage_cosine_{{full,keep}}.csv")

    # --------------------------------------------------------------- verdict
    bad = []
    if fro_ret < 0.85:
        bad.append(f"low HW retention {fro_ret:.2f} (<0.85)")
    if cat_df["retention"].min() < 0.7:
        worst = cat_df.iloc[int(cat_df["retention"].idxmin())]
        bad.append(f"category '{worst['category']}' only retains "
                   f"{worst['retention']:.2f} of its loading L2")
    if cos_keep[off].max() > cos_full[off].max() + 0.1:
        bad.append("category cosine max grew >0.1 (categories collapsing)")
    if cond_keep > 3 * cond_full:
        bad.append(f"cond(W_macro) tripled: {cond_full:.1f} → {cond_keep:.1f}")
    if bad:
        print("\n[coverage] ⚠ EXPRESSIVENESS WARNINGS:")
        for b in bad:
            print(f"  - {b}")
    else:
        print("\n[coverage] ✅ remaining basis preserves expressiveness on all checks.")


if __name__ == "__main__":
    main()
