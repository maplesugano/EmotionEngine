"""Decompose per-category CAA steering vectors as weighted sums of basis vectors.

Tests the hypothesis that each Plutchik-category CAA vector lies (approximately)
in the span of a small number of basis directions discovered by Phase C:

    v_caa[c, L] ≈ Σ_k w_{c,k} · b_k[L]

For every basis artifact under the supplied sweep directories we fit
``w_c`` to ``v_caa[c, layer_of_artifact]`` using three solvers — OLS, NNLS,
and LASSO — and record reconstruction R², cosine, residual L2, and weight
sparsity. A 3-axis VAD baseline is fit per CAA layer for headline comparison.

Outputs land under ``experiments/results/caa_basis_decomposition/``:

* ``decomposition.csv`` — one row per (artifact, category, fit_method[, lasso_alpha])
* ``vad_baseline.csv`` — one row per (layer, category)
* ``weights/{artifact_stem}.pt`` — dict[category → {ols, nnls, lasso_α}] weight tensors
* ``summary.json`` — aggregated medians grouped by (decomposer, k, layer, fit)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls
from sklearn.linear_model import Lasso

_ARTIFACT_RE = re.compile(r"^(?P<dec>[a-z]+)_k(?P<k>\d+)_seed(?P<seed>\d+)\.pt$")
_LASSO_ALPHAS = (1e-3, 1e-2, 1e-1)
_SPARSE_TOL = 1e-4


def _metrics(target: np.ndarray, recon: np.ndarray, weights: np.ndarray) -> dict:
    resid = target - recon
    ss_res = float((resid ** 2).sum())
    ss_tot = float((target ** 2).sum())
    cos = float(target @ recon / (np.linalg.norm(target) * np.linalg.norm(recon) + 1e-12))
    return {
        "r2": 1.0 - ss_res / max(ss_tot, 1e-12),
        "cosine": cos,
        "residual_l2": float(np.sqrt(ss_res)),
        "weight_l1": float(np.abs(weights).sum()),
        "weight_l2": float(np.linalg.norm(weights)),
        "weight_l0": int((np.abs(weights) > _SPARSE_TOL).sum()),
    }


def _fit_ols(W: np.ndarray, t: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(W.T, t, rcond=None)
    return coef


def _fit_nnls(W: np.ndarray, t: np.ndarray) -> np.ndarray:
    coef, _ = nnls(W.T, t, maxiter=10 * W.shape[0])
    return coef


def _fit_lasso(W: np.ndarray, t: np.ndarray, alpha_rel: float) -> np.ndarray:
    # Scale alpha by target magnitude so the same alpha_rel is comparable
    # across categories/layers with different ||t||.
    alpha = alpha_rel * float(np.linalg.norm(t)) / np.sqrt(W.shape[1])
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=20000, tol=1e-6)
    model.fit(W.T, t)
    return model.coef_


def _decompose_one(
    W: np.ndarray, target: np.ndarray
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Return list of metric rows (one per fit method) and weight dict."""
    rows: list[dict] = []
    weights: dict[str, np.ndarray] = {}

    w_ols = _fit_ols(W, target)
    rows.append({"fit": "ols", "lasso_alpha": None,
                 **_metrics(target, W.T @ w_ols, w_ols)})
    weights["ols"] = w_ols

    w_nnls = _fit_nnls(W, target)
    rows.append({"fit": "nnls", "lasso_alpha": None,
                 **_metrics(target, W.T @ w_nnls, w_nnls)})
    weights["nnls"] = w_nnls

    for a in _LASSO_ALPHAS:
        w = _fit_lasso(W, target, a)
        rows.append({"fit": "lasso", "lasso_alpha": a,
                     **_metrics(target, W.T @ w, w)})
        weights[f"lasso_{a:g}"] = w

    return rows, weights


def _vad_baseline(W_vad: np.ndarray, target: np.ndarray) -> dict:
    w, *_ = np.linalg.lstsq(W_vad.T, target, rcond=None)
    return {"fit": "vad_ols", **_metrics(target, W_vad.T @ w, w),
            "weights": w.tolist()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dirs", type=Path, nargs="+",
                   default=[Path("data/emotion_code/basis_sweep"),
                            Path("data/emotion_code/basis_sweep_L13"),
                            Path("data/emotion_code/basis_sweep_L16"),
                            Path("data/emotion_code/basis_sweep_L19"),
                            Path("data/emotion_code/basis_sweep_L22")])
    p.add_argument("--caa", type=Path, default=Path("data/emotion_code/caa.pt"))
    p.add_argument("--vad", type=Path, default=Path("data/emotion_code/vad_mapping.pt"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/results/caa_basis_decomposition"))
    p.add_argument("--limit-artifacts", type=int, default=None,
                   help="smoke-test cap on number of artifacts processed")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = args.output_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    # Load CAA -----------------------------------------------------------------
    caa = torch.load(args.caa, weights_only=False, map_location="cpu")
    caa_vecs = caa["vectors"].numpy().astype(np.float32)   # [C, L, D]
    categories: list[str] = list(caa["categories"])
    caa_layers: list[int] = list(caa["layers"])
    print(f"[decomp] CAA: {caa_vecs.shape}, layers={caa_layers}, "
          f"categories={categories}")

    # Load VAD baseline --------------------------------------------------------
    vad = torch.load(args.vad, weights_only=False, map_location="cpu")
    W_vad = vad["W"].numpy().astype(np.float32)            # [3, D]
    vad_layer = int(vad["layer"])
    print(f"[decomp] VAD: W={W_vad.shape}, layer={vad_layer}")

    # Collect artifacts (dedupe by absolute path) ------------------------------
    seen: set[Path] = set()
    artifacts: list[Path] = []
    for d in args.sweep_dirs:
        if not d.exists():
            print(f"[decomp] skip missing dir {d}")
            continue
        for p_ in sorted(d.glob("*_k*_seed*.pt")):
            ap = p_.resolve()
            if ap in seen:
                continue
            seen.add(ap)
            artifacts.append(p_)
    if args.limit_artifacts:
        artifacts = artifacts[: args.limit_artifacts]
    print(f"[decomp] {len(artifacts)} artifacts queued")

    # Per-artifact decomposition ----------------------------------------------
    rows: list[dict] = []
    for art in artifacts:
        m = _ARTIFACT_RE.match(art.name)
        if not m:
            print(f"[decomp] skip unparseable {art.name}")
            continue
        payload = torch.load(art, weights_only=False, map_location="cpu")
        dec = payload.get("decomposer", m.group("dec"))
        layer = int(payload["layer"])
        if layer not in caa_layers:
            print(f"[decomp] skip {art.name}: layer {layer} not in CAA layers")
            continue
        layer_idx = caa_layers.index(layer)
        W = payload[dec]["W"].numpy().astype(np.float32)   # [k, D]
        k = int(payload.get("k", W.shape[0]))
        seed = int(payload.get("seed", m.group("seed")))
        sweep_tag = art.parent.name

        artifact_weights: dict[str, dict[str, np.ndarray]] = {}
        for ci, cat in enumerate(categories):
            target = caa_vecs[ci, layer_idx]
            cat_rows, cat_weights = _decompose_one(W, target)
            artifact_weights[cat] = cat_weights
            base = {
                "sweep": sweep_tag,
                "artifact": art.name,
                "decomposer": dec,
                "k": k,
                "seed": seed,
                "layer": layer,
                "category": cat,
                "target_norm": float(np.linalg.norm(target)),
            }
            for r in cat_rows:
                rows.append({**base, **r})

        wpath = weights_dir / f"{sweep_tag}__{art.stem}.pt"
        torch.save(
            {"categories": categories, "layer": layer, "decomposer": dec,
             "k": k, "seed": seed,
             "weights": {c: {k_: torch.from_numpy(v) for k_, v in d.items()}
                         for c, d in artifact_weights.items()}},
            wpath,
        )
        print(f"[decomp] {sweep_tag}/{art.name}  L={layer} k={k}  -> "
              f"R²(ols)={np.median([r['r2'] for r in rows[-len(categories) * (2 + len(_LASSO_ALPHAS)):] if r['fit'] == 'ols']):.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "decomposition.csv", index=False)
    print(f"[decomp] wrote decomposition.csv  ({len(df)} rows)")

    # VAD baseline ------------------------------------------------------------
    vad_rows: list[dict] = []
    for ci, cat in enumerate(categories):
        for li, layer in enumerate(caa_layers):
            target = caa_vecs[ci, li]
            res = _vad_baseline(W_vad, target)
            vad_rows.append({
                "layer": layer, "category": cat,
                "target_norm": float(np.linalg.norm(target)),
                **res,
            })
    vad_df = pd.DataFrame(vad_rows)
    vad_df.to_csv(args.output_dir / "vad_baseline.csv", index=False)
    print(f"[decomp] wrote vad_baseline.csv  ({len(vad_df)} rows)")

    # Aggregate summary -------------------------------------------------------
    summary: dict = {
        "n_artifacts": len(artifacts),
        "n_categories": len(categories),
        "categories": categories,
        "caa_layers": caa_layers,
        "vad_baseline": {
            "median_r2": float(vad_df["r2"].median()),
            "median_cosine": float(vad_df["cosine"].median()),
            "per_layer_median_r2": {
                str(L): float(vad_df[vad_df.layer == L]["r2"].median())
                for L in caa_layers
            },
        },
        "by_fit_method": {},
        "by_decomposer_k_layer": [],
    }
    for fit in df["fit"].unique():
        sub = df[df.fit == fit]
        summary["by_fit_method"][fit] = {
            "median_r2": float(sub["r2"].median()),
            "median_cosine": float(sub["cosine"].median()),
            "median_l0": float(sub["weight_l0"].median()),
        }
    for (dec, k, layer, fit), sub in df.groupby(["decomposer", "k", "layer", "fit"]):
        summary["by_decomposer_k_layer"].append({
            "decomposer": dec, "k": int(k), "layer": int(layer), "fit": fit,
            "median_r2": float(sub["r2"].median()),
            "mean_r2": float(sub["r2"].mean()),
            "median_cosine": float(sub["cosine"].median()),
            "median_l0": float(sub["weight_l0"].median()),
            "n_rows": int(len(sub)),
        })
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[decomp] wrote summary.json")
    print(f"[decomp] median basis R² (OLS): "
          f"{summary['by_fit_method'].get('ols', {}).get('median_r2'):.3f}")
    print(f"[decomp] median VAD R²: {summary['vad_baseline']['median_r2']:.3f}")


if __name__ == "__main__":
    main()
