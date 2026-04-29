"""Cross-layer consistency of basis components.

Given basis artifacts at multiple layers (same decomposer & k), match
components between layer pairs by Hungarian assignment on |cosine| of the
right-singular subspace projections — but since W is in the residual-stream
space which is approximately the same across consecutive layers in a
pre-norm transformer, raw cosine on W rows is the standard, intuitive metric.

For each (decomposer, k) we report:
  • per-pair mean cosine   (how much of the basis is preserved layer→layer?)
  • per-component cosine   (which b_j survives across layers?)

"Layer-stable" components are strong candidates for genuine emotion atoms,
independent of any verbal label.

Usage
-----
    # First sweep at multiple layers:
    for L in 13 16 19 22; do
        uv run python -m src.emotion_code.basis_sweep --layer $L \
            --decomposers ica --ks 16 --n-seeds 1 \
            --output-dir data/emotion_code/basis_sweep_L${L}
    done
    uv run python -m experiments.eval_basis_layerconsistency \
        --sweep-dirs data/emotion_code/basis_sweep_L*
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

_ARTIFACT_RE = re.compile(r"^(?P<dec>[a-z]+)_k(?P<k>\d+)_seed(?P<seed>\d+)\.pt$")


def _match(W_a: np.ndarray, W_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Wa = W_a / (np.linalg.norm(W_a, axis=1, keepdims=True) + 1e-12)
    Wb = W_b / (np.linalg.norm(W_b, axis=1, keepdims=True) + 1e-12)
    sim = np.abs(Wa @ Wb.T)
    row, col = linear_sum_assignment(-sim)
    return row, col, sim[row, col]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--decomposer", default=None,
                   help="Restrict to one decomposer (default: every one found).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-pair", type=Path,
                   default=Path("data/emotion_code/basis_sweep/layer_pair_consistency.csv"))
    p.add_argument("--output-component", type=Path,
                   default=Path("data/emotion_code/basis_sweep/layer_component_consistency.csv"))
    args = p.parse_args()

    # Index artifacts per (decomposer, k, seed, layer).
    by_key: dict[tuple, dict[int, np.ndarray]] = {}
    for d in args.sweep_dirs:
        for art_path in sorted(d.glob("*_k*_seed*.pt")):
            m = _ARTIFACT_RE.match(art_path.name)
            if not m:
                continue
            dec = m.group("dec")
            if args.decomposer and dec != args.decomposer:
                continue
            seed = int(m.group("seed"))
            if seed != args.seed:
                continue
            payload = torch.load(art_path, weights_only=False, map_location="cpu")
            k = int(payload["k"])
            layer = int(payload["layer"])
            W = payload[dec]["W"].numpy()
            by_key.setdefault((dec, k, seed), {})[layer] = W

    pair_rows: list[dict] = []
    comp_rows: list[dict] = []
    for (dec, k, seed), layer_W in by_key.items():
        layers = sorted(layer_W)
        if len(layers) < 2:
            print(f"[layerC] {dec} k={k} seed={seed}: only {len(layers)} layers, skip")
            continue
        for i, la in enumerate(layers):
            for lb in layers[i + 1:]:
                Wa, Wb = layer_W[la], layer_W[lb]
                row_idx, col_idx, sims = _match(Wa, Wb)
                pair_rows.append({
                    "decomposer": dec, "k": k, "seed": seed,
                    "layer_a": la, "layer_b": lb,
                    "mean_cosine": float(sims.mean()),
                    "min_cosine": float(sims.min()),
                    "median_cosine": float(np.median(sims)),
                })
                for ra, cb, s in zip(row_idx, col_idx, sims):
                    comp_rows.append({
                        "decomposer": dec, "k": k, "seed": seed,
                        "layer_a": la, "layer_b": lb,
                        "component_a": int(ra), "component_b": int(cb),
                        "abs_cosine": float(s),
                    })
        print(f"[layerC] {dec} k={k} seed={seed}: layers={layers}, "
              f"pairs={len(layers)*(len(layers)-1)//2}")

    pd.DataFrame(pair_rows).to_csv(args.output_pair, index=False)
    pd.DataFrame(comp_rows).to_csv(args.output_component, index=False)
    print(f"[layerC] wrote {args.output_pair} and {args.output_component}")


if __name__ == "__main__":
    main()
