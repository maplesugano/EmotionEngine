"""Score basis components on language-independence and stability.

Given basis artifacts produced by :mod:`src.emotion_code.basis_sweep`, compute
five diagnostics per component ``b_j`` so we can rank "candidate sub-verbal
emotion primitives":

  • MI(component, category)        ─ low ⇒ component is decoupled from labels
  • linear_sep_acc                  ─ 8-way logistic on s_{·,j}; ≈ 1/8 = chance
  • vad_explained                   ─ ||proj_{V,A,D}(b_j)||² / ||b_j||²
  • category_top1_dominance         ─ max class share among top-N by score
  • silhouette (per-artifact)       ─ silhouette of H rows w.r.t. categories

Stability across seeds (Hungarian + cosine on W rows) is computed when
multiple seeds exist for the same (decomposer, k).

Outputs ``<artifact>.metrics.json`` next to each artifact and a pooled
``metrics.summary.csv`` in the sweep directory.

Usage
-----
    uv run python -m src.emotion_code.basis_metrics \
        --sweep-dir data/emotion_code/basis_sweep \
        --vad data/emotion_code/vad_mapping.pt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold

from src.emotion_code.basis import _build_delta
from src.emotion_code.io import load_activations, make_split


_ARTIFACT_RE = re.compile(r"^(?P<dec>[a-z]+)_k(?P<k>\d+)_seed(?P<seed>\d+)\.pt$")


def _project(delta: np.ndarray, W: np.ndarray) -> np.ndarray:
    d_norm = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-12
    w_norm = np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
    return (delta / d_norm) @ (W / w_norm).T


def _vad_explained(W: np.ndarray, vad_W: np.ndarray) -> np.ndarray:
    """Fraction of each b_j's energy captured by the V/A/D subspace."""
    # Orthonormalise the V/A/D basis (3, D).
    Q, _ = np.linalg.qr(vad_W.T)        # (D, 3)
    proj = W @ Q                         # (k, 3)
    num = (proj ** 2).sum(axis=1)
    den = (W ** 2).sum(axis=1) + 1e-12
    return num / den


def _linear_sep_acc(scores: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Stratified 5-fold logistic regression on a single component score."""
    if len(np.unique(y)) < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for tr, va in skf.split(scores.reshape(-1, 1), y):
        clf = LogisticRegression(max_iter=200)
        clf.fit(scores[tr].reshape(-1, 1), y[tr])
        accs.append(clf.score(scores[va].reshape(-1, 1), y[va]))
    return float(np.mean(accs))


def _stability(W_a: np.ndarray, W_b: np.ndarray) -> float:
    """Mean cosine of best-matched components between two W matrices (k×D)."""
    Wa = W_a / (np.linalg.norm(W_a, axis=1, keepdims=True) + 1e-12)
    Wb = W_b / (np.linalg.norm(W_b, axis=1, keepdims=True) + 1e-12)
    sim = np.abs(Wa @ Wb.T)             # k×k
    cost = -sim
    row, col = linear_sum_assignment(cost)
    return float(sim[row, col].mean())


def _metrics_for_artifact(
    art_path: Path, decomposer: str, delta: np.ndarray, cat_arr: np.ndarray,
    categories: list[str], vad_W: np.ndarray | None,
) -> dict:
    payload = torch.load(art_path, weights_only=False, map_location="cpu")
    W = payload[decomposer]["W"].numpy()
    H = payload[decomposer]["H"].numpy()
    k = W.shape[0]

    scores = _project(delta, W)         # [N, k]
    y = np.array([categories.index(c) for c in cat_arr], dtype=np.int64)

    mi = mutual_info_classif(scores, y, discrete_features=False, random_state=0)
    sep = np.array([_linear_sep_acc(scores[:, j], y) for j in range(k)])

    # Top-N category dominance (label-leak indicator).
    top_n = 8
    dominance = np.zeros(k)
    for j in range(k):
        top_idx = np.argsort(-scores[:, j])[:top_n]
        labels, counts = np.unique(cat_arr[top_idx], return_counts=True)
        dominance[j] = counts.max() / top_n

    vad_explained = _vad_explained(W, vad_W) if vad_W is not None else np.full(k, np.nan)

    # Silhouette of H rows wrt categories (single number per artifact).
    try:
        sil = float(silhouette_score(H, y, metric="cosine", sample_size=min(2000, len(y))))
    except Exception:
        sil = float("nan")

    per_component = []
    for j in range(k):
        per_component.append({
            "component": int(j),
            "mi": float(mi[j]),
            "linear_sep_acc": float(sep[j]),
            "category_top1_dominance": float(dominance[j]),
            "vad_explained": float(vad_explained[j]),
        })
    return {
        "artifact": str(art_path),
        "decomposer": decomposer,
        "k": int(k),
        "seed": int(payload.get("seed", 0)),
        "layer": int(payload.get("layer", -1)),
        "silhouette_cosine_H": sil,
        "chance_acc": 1.0 / max(1, len(categories)),
        "per_component": per_component,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", type=Path, default=Path("data/emotion_code/basis_sweep"))
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--activations-root", type=Path, default=Path("data/activations"))
    p.add_argument("--vad", type=Path, default=Path("data/emotion_code/vad_mapping.pt"))
    p.add_argument("--summary-csv", type=Path,
                   default=Path("data/emotion_code/basis_sweep/metrics.summary.csv"))
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    profile = cfg["active"]
    bundle = load_activations(profile=profile, root=args.activations_root)

    vad_W = None
    if args.vad.exists():
        vad_payload = torch.load(args.vad, weights_only=False, map_location="cpu")
        vad_W = vad_payload["W"].numpy()  # [3, D]
        print(f"[metrics] loaded VAD mapping (layer={vad_payload.get('layer')})")
    else:
        print(f"[metrics] WARNING: {args.vad} not found; vad_explained=NaN")

    artifacts = sorted(args.sweep_dir.glob("*_k*_seed*.pt"))
    if not artifacts:
        raise SystemExit(f"no artifacts under {args.sweep_dir}; run basis_sweep first")
    print(f"[metrics] scoring {len(artifacts)} artifacts")

    # Cache (split, Δ) per (split_seed, train_frac, layer).
    delta_cache: dict[tuple, tuple] = {}
    pooled_rows: list[dict] = []
    grouped_W: dict[tuple, list[tuple[int, np.ndarray]]] = {}

    for art_path in artifacts:
        m = _ARTIFACT_RE.match(art_path.name)
        if m is None:
            print(f"[metrics] skipping {art_path.name} (unrecognised)")
            continue
        decomposer = m.group("dec")
        payload_meta = torch.load(art_path, weights_only=False, map_location="cpu")
        layer = int(payload_meta["layer"])
        split_seed = int(payload_meta.get("split_seed", 0))
        train_frac = float(payload_meta.get("train_frac", 0.8))
        key = (layer, split_seed, train_frac)
        if key not in delta_cache:
            train_mask, _ = make_split(bundle.meta, train_frac=train_frac, seed=split_seed)
            delta = _build_delta(bundle, layer, train_mask)
            cat_arr = bundle.meta["category"].to_numpy()[train_mask]
            categories = sorted(bundle.meta["category"].unique().tolist())
            delta_cache[key] = (delta, cat_arr, categories)
        delta, cat_arr, categories = delta_cache[key]

        result = _metrics_for_artifact(
            art_path, decomposer, delta, cat_arr, categories, vad_W,
        )
        out_path = art_path.with_suffix(".metrics.json")
        out_path.write_text(json.dumps(result, indent=2))

        # Aggregate per-component rows for the pooled CSV.
        for comp in result["per_component"]:
            pooled_rows.append({
                "decomposer": decomposer,
                "k": result["k"],
                "seed": result["seed"],
                "layer": result["layer"],
                "silhouette_cosine_H": result["silhouette_cosine_H"],
                **comp,
            })

        grouped_W.setdefault(
            (decomposer, result["k"]), []
        ).append((result["seed"], payload_meta[decomposer]["W"].numpy()))

        print(f"[metrics] {art_path.name}  silhouette={result['silhouette_cosine_H']:.3f}")

    # Stability across seeds.
    stability_rows = []
    for (decomposer, k), entries in grouped_W.items():
        if len(entries) < 2:
            continue
        entries.sort()
        sims = []
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                sims.append(_stability(entries[i][1], entries[j][1]))
        stability_rows.append({
            "decomposer": decomposer, "k": k,
            "n_seeds": len(entries),
            "mean_pairwise_stability": float(np.mean(sims)),
            "min_pairwise_stability": float(np.min(sims)),
        })

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pooled_rows).to_csv(args.summary_csv, index=False)
    print(f"[metrics] wrote {args.summary_csv}  ({len(pooled_rows)} rows)")
    if stability_rows:
        stab_path = args.summary_csv.with_name("stability.summary.csv")
        pd.DataFrame(stability_rows).to_csv(stab_path, index=False)
        print(f"[metrics] wrote {stab_path}  ({len(stability_rows)} rows)")


if __name__ == "__main__":
    main()
