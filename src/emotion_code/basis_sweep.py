"""Sweep basis rank ``k`` (and decomposer) on per-pair Δ activations.

Additive exploration on top of :mod:`src.emotion_code.basis` — the existing
single-k pipeline (``data/emotion_code/basis.pt``) is left untouched.

For every (decomposer, k, seed) we refit on the *same* per-pair Δ matrix
(no category averaging) and write one artifact per (decomposer, k, seed)
under ``data/emotion_code/basis_sweep/``.

Usage
-----
    uv run python -m src.emotion_code.basis_sweep \
        --ks 8 16 32 64 --decomposers nmf pca ica --n-seeds 2 --max-iter 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.emotion_code.basis import _build_delta
from src.emotion_code.decompose import DECOMPOSERS, category_loadings
from src.emotion_code.io import load_activations, make_split


def _artifact_path(out_dir: Path, decomposer: str, k: int, seed: int) -> Path:
    return out_dir / f"{decomposer}_k{k:03d}_seed{seed}.pt"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--activations-root", type=Path, default=Path("data/activations"))
    p.add_argument("--output-dir", type=Path, default=Path("data/emotion_code/basis_sweep"))
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--decomposers", nargs="+", default=["nmf", "pca"],
                   choices=sorted(DECOMPOSERS.keys()))
    p.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--n-seeds", type=int, default=1,
                   help="Refit each (decomposer, k) with this many seeds (0..n-1).")
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--dict-alpha", type=float, default=1.0,
                   help="Sparsity penalty for `dict` decomposer.")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.8)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    profile = cfg["active"]
    bundle = load_activations(profile=profile, root=args.activations_root)
    layer = args.layer if args.layer is not None else bundle.layers[len(bundle.layers) // 2]
    if layer not in bundle.layers:
        raise ValueError(f"layer {layer} not in {bundle.layers}")

    train_mask, _ = make_split(bundle.meta, train_frac=args.train_frac, seed=args.split_seed)
    delta = _build_delta(bundle, layer, train_mask)
    categories = sorted(bundle.meta["category"].unique().tolist())
    cat_arr = bundle.meta["category"].to_numpy()[train_mask]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] profile={profile} layer={layer} Δ.shape={delta.shape}")
    print(f"[sweep] decomposers={args.decomposers} ks={args.ks} n_seeds={args.n_seeds}")

    summary_rows: list[dict] = []
    for decomposer in args.decomposers:
        fitter = DECOMPOSERS[decomposer]
        for k in args.ks:
            for seed in range(args.n_seeds):
                kwargs: dict = {"max_iter": args.max_iter, "seed": seed}
                if decomposer == "dict":
                    kwargs["alpha"] = args.dict_alpha
                if decomposer == "pca":
                    kwargs.pop("max_iter", None)
                result = fitter(delta, k=k, **kwargs)
                W = result["W"]
                H = result["H"]
                loadings = category_loadings(delta, categories, cat_arr, W)

                payload: dict = {
                    "decomposer": decomposer,
                    "profile": profile,
                    "layer": layer,
                    "k": k,
                    "seed": seed,
                    "categories": categories,
                    "split_seed": args.split_seed,
                    "train_frac": args.train_frac,
                    "train_mask": torch.from_numpy(train_mask),
                    decomposer: {
                        "W": torch.from_numpy(W),
                        "H": torch.from_numpy(H),
                        "category_loadings": torch.from_numpy(loadings),
                    },
                }
                for extra in ("reconstruction_err", "n_iter", "converged",
                              "explained_variance_ratio"):
                    if extra in result:
                        payload[decomposer][extra] = result[extra]

                out_path = _artifact_path(args.output_dir, decomposer, k, seed)
                torch.save(payload, out_path)

                row: dict = {"decomposer": decomposer, "k": k, "seed": seed,
                             "layer": layer, "profile": profile, "artifact": str(out_path)}
                if "reconstruction_err" in result:
                    row["reconstruction_err"] = result["reconstruction_err"]
                if "n_iter" in result:
                    row["n_iter"] = result["n_iter"]
                if "converged" in result:
                    row["converged"] = result["converged"]
                if "explained_variance_ratio" in result:
                    row["explained_variance"] = float(sum(result["explained_variance_ratio"]))
                summary_rows.append(row)
                desc = " ".join(f"{kk}={vv}" for kk, vv in row.items()
                                if kk not in {"artifact", "profile"})
                print(f"[sweep] {desc}")

    (args.output_dir / "sweep.summary.json").write_text(
        json.dumps({"layer": layer, "profile": profile, "rows": summary_rows}, indent=2)
    )
    print(f"[sweep] wrote {len(summary_rows)} artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
