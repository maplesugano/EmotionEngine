"""Phase C-3 (B): Weight matrix W (8 cats x k components) structure analysis.

Loads OLS weights from `eval_caa_basis_decomposition.py`, then for the chosen
basis artifact:
  - W in R^{8 x k} = stacked per-category OLS weights.
  - Renders heatmap (PNG).
  - Computes per-component statistics:
      L1 mass, L_inf, n_active (|w|>tau * max), participation ratio,
      sign-balance, top categories.
  - Classifies each component into:
      'pan'        : loaded broadly across categories (PR >= 4)
      'cat_specific': dominated by one or two categories (PR <= 2)
      'lexical_gap': low absolute mass everywhere (max |w| < gap_thresh * global max)
      'mixed'      : else
  - Writes summary.json + classification.csv.

Outputs go to experiments/results/caa_basis_weight_structure/<tag>/.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_W(weights_path: Path, fit_method: str = "ols") -> tuple[np.ndarray, list[str], dict]:
    payload = torch.load(weights_path, weights_only=False)
    cats = list(payload["categories"])
    k = int(payload["k"])
    W = np.zeros((len(cats), k), dtype=np.float64)
    for i, c in enumerate(cats):
        W[i] = payload["weights"][c][fit_method].cpu().numpy().astype(np.float64)
    meta = {
        "layer": int(payload["layer"]),
        "decomposer": str(payload["decomposer"]),
        "k": k,
        "seed": int(payload["seed"]),
        "fit_method": fit_method,
    }
    return W, cats, meta


def _per_component_stats(W: np.ndarray, cats: list[str]) -> pd.DataFrame:
    # W: [C, K]
    C, K = W.shape
    abs_W = np.abs(W)
    rows = []
    global_max = float(abs_W.max())
    for k in range(K):
        col = W[:, k]
        a = np.abs(col)
        l1 = float(a.sum())
        linf = float(a.max())
        # participation ratio (effective # of cats loaded): (sum a)^2 / sum a^2
        pr = float((a.sum() ** 2) / (np.square(a).sum() + 1e-12))
        # sign balance: fraction of pos mass
        pos_mass = float(col[col > 0].sum())
        neg_mass = float(-col[col < 0].sum())
        sign_bal = pos_mass / (pos_mass + neg_mass + 1e-12)
        order = np.argsort(-a)
        top_cats = [cats[i] for i in order[:3]]
        top_vals = [float(col[i]) for i in order[:3]]
        rows.append({
            "component": k,
            "l1": l1, "linf": linf,
            "participation_ratio": pr,
            "sign_balance_pos": sign_bal,
            "rel_max": linf / (global_max + 1e-12),
            "top1_cat": top_cats[0], "top1_w": top_vals[0],
            "top2_cat": top_cats[1], "top2_w": top_vals[1],
            "top3_cat": top_cats[2], "top3_w": top_vals[2],
        })
    return pd.DataFrame(rows)


def _classify(stats: pd.DataFrame, gap_thresh: float = 0.20,
              pan_pr: float = 4.0, specific_pr: float = 2.0) -> pd.DataFrame:
    def label(row):
        if row["rel_max"] < gap_thresh:
            return "lexical_gap"
        if row["participation_ratio"] >= pan_pr:
            return "pan"
        if row["participation_ratio"] <= specific_pr:
            return "cat_specific"
        return "mixed"
    out = stats.copy()
    out["class"] = stats.apply(label, axis=1)
    return out


def _heatmap(W: np.ndarray, cats: list[str], out_path: Path,
             title: str, classes: list[str] | None = None):
    C, K = W.shape
    vmax = float(np.abs(W).max())
    fig, ax = plt.subplots(figsize=(max(8, K * 0.5), max(3, C * 0.45)))
    im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(C)); ax.set_yticklabels(cats)
    if classes is not None:
        labs = [f"b{k}\n[{classes[k][:3]}]" for k in range(K)]
    else:
        labs = [f"b{k}" for k in range(K)]
    ax.set_xticks(range(K)); ax.set_xticklabels(labs, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path,
                   default=Path("experiments/results/caa_basis_decomposition/weights/basis_sweep__ica_k016_seed0.pt"))
    p.add_argument("--fit-method", type=str, default="ols",
                   choices=["ols", "nnls", "lasso_0.001", "lasso_0.01", "lasso_0.1"])
    p.add_argument("--gap-thresh", type=float, default=0.20,
                   help="rel_max below this => 'lexical_gap'")
    p.add_argument("--pan-pr", type=float, default=4.0,
                   help="participation_ratio >= this => 'pan'")
    p.add_argument("--specific-pr", type=float, default=2.0,
                   help="participation_ratio <= this => 'cat_specific'")
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/results/caa_basis_weight_structure"))
    args = p.parse_args()

    W, cats, meta = _load_W(args.weights, fit_method=args.fit_method)
    tag = f"{meta['decomposer']}_k{meta['k']:03d}_seed{meta['seed']}_L{meta['layer']}_{args.fit_method}"
    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Wstruct] W shape = {W.shape} (cats x k); tag = {tag}")

    stats = _per_component_stats(W, cats)
    classified = _classify(stats, gap_thresh=args.gap_thresh,
                           pan_pr=args.pan_pr, specific_pr=args.specific_pr)
    classified.to_csv(out_dir / "component_classification.csv", index=False)

    classes_by_idx = classified.set_index("component")["class"].to_dict()
    classes = [classes_by_idx[k] for k in range(W.shape[1])]
    _heatmap(W, cats, out_dir / "W_heatmap.png",
             title=f"OLS weights W ({tag})", classes=classes)

    # Sort components by PR for a second heatmap (visual grouping)
    order = np.argsort(-stats["participation_ratio"].to_numpy())
    W_sorted = W[:, order]
    classes_sorted = [classes[i] for i in order]
    _heatmap(W_sorted, cats, out_dir / "W_heatmap_sorted_by_PR.png",
             title=f"W sorted by participation ratio ({tag})",
             classes=classes_sorted)

    # Per-class summary
    summary = {
        "tag": tag, **meta,
        "W_shape": list(W.shape),
        "global_max_abs": float(np.abs(W).max()),
        "thresholds": {"gap_thresh": args.gap_thresh,
                       "pan_pr": args.pan_pr,
                       "specific_pr": args.specific_pr},
        "n_components_by_class": classified["class"].value_counts().to_dict(),
        "lexical_gap_components": classified[classified["class"] == "lexical_gap"]["component"].tolist(),
        "pan_components": classified[classified["class"] == "pan"]["component"].tolist(),
        "cat_specific_components": classified[classified["class"] == "cat_specific"]["component"].tolist(),
        "components_sorted_by_PR": [
            {"component": int(stats.iloc[i]["component"]),
             "PR": float(stats.iloc[i]["participation_ratio"]),
             "rel_max": float(stats.iloc[i]["rel_max"]),
             "class": classes[int(stats.iloc[i]["component"])],
             "top1": stats.iloc[i]["top1_cat"]}
            for i in np.argsort(-stats["participation_ratio"].to_numpy())
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[Wstruct] class counts:", summary["n_components_by_class"])
    print("[Wstruct] lexical_gap components:", summary["lexical_gap_components"])
    print("[Wstruct] pan components:", summary["pan_components"])
    print(f"[Wstruct] wrote {out_dir}")


if __name__ == "__main__":
    main()
