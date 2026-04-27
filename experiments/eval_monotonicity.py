"""Monotonicity: Spearman ρ between α and target-class probability.

Reuses the cache produced by ``_gen_cache.py`` and the classifier from
``eval_shift_accuracy.py``.  For each category, we collect the classifier's
probability mass on the target class for every α and compute Spearman
correlation between α and that probability across all (prompt, α) pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.eval_shift_accuracy import HARTMANN_TO_PLUTCHIK, UNMEASURABLE, _classify


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path,
                   default=Path("experiments/results/_gen_cache.parquet"))
    p.add_argument("--cls-cache", type=Path,
                   default=Path("experiments/results/_cls_cache.parquet"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/results/monotonicity.csv"))
    args = p.parse_args()

    df = pd.read_parquet(args.cache)
    if args.cls_cache.exists():
        cls = pd.read_parquet(args.cls_cache)
    else:
        cls = _classify(df["generation"].astype(str).tolist())
        args.cls_cache.parent.mkdir(parents=True, exist_ok=True)
        cls.to_parquet(args.cls_cache, index=False)
    df = pd.concat([df.reset_index(drop=True), cls.reset_index(drop=True)], axis=1)

    # Inverse map Plutchik -> Hartmann column name (one per measurable cat).
    plut_to_hart = {v: k for k, v in HARTMANN_TO_PLUTCHIK.items() if v != "neutral"}

    rows = []
    for cat, sub in df.groupby("category"):
        if cat in UNMEASURABLE or cat not in plut_to_hart:
            rows.append({"category": cat, "rho": np.nan, "p_value": np.nan,
                         "note": "no classifier label"})
            continue
        col = plut_to_hart[cat]
        rho, pval = spearmanr(sub["alpha_unit"].to_numpy(), sub[col].to_numpy())
        rows.append({
            "category": cat,
            "rho": float(rho),
            "p_value": float(pval),
            "n": int(len(sub)),
            "note": "",
        })

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    measurable = out.dropna(subset=["rho"])
    summary = {
        "mean_rho": float(measurable["rho"].mean()) if len(measurable) else None,
        "min_rho": float(measurable["rho"].min()) if len(measurable) else None,
        "per_category": rows,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"[monotonicity] mean_rho={summary['mean_rho']}  -> {args.output}")


if __name__ == "__main__":
    main()
