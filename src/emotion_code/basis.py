"""Decompose per-pair difference activations into a low-rank emotion basis.

For a chosen layer L we form per-pair differences

    Δ_i = pos[i, L] - neg[i, L]      ∈ R^{d_model},  i ∈ train

NMF requires non-negative inputs, so we sign-split the matrix:

    Δ̃ = [ relu(Δ) | relu(-Δ) ]      ∈ R^{N × 2·d_model}

and learn ``NMF(n_components=k, init="nndsvd")``. The k basis vectors live in
the doubled space; we recover signed prototypes in d_model by

    W_signed[k, :] = W_pos - W_neg

PCA on the raw Δ is also computed for comparison.

`category_loadings[c, k]` is the cosine of the per-category mean Δ with each
basis vector — it tells us how each Plutchik category projects onto the
learned basis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.decomposition import NMF, PCA

from src.emotion_code.io import load_activations, make_split


def _build_delta(bundle, layer: int, mask: np.ndarray) -> np.ndarray:
    pos = bundle.pos[layer].numpy()
    neg = bundle.neg[layer].numpy()
    return (pos - neg)[mask].astype(np.float32)


def _signed_nmf(delta: np.ndarray, k: int, max_iter: int = 400) -> dict:
    """Sign-split NMF.  Returns dict with W_signed [k, d_model] and loadings."""
    pos = np.maximum(delta, 0.0)
    neg = np.maximum(-delta, 0.0)
    augmented = np.concatenate([pos, neg], axis=1)  # [N, 2D]
    nmf = NMF(
        n_components=k,
        init="nndsvd",
        max_iter=max_iter,
        tol=1e-4,
        solver="cd",
        beta_loss="frobenius",
    )
    H = nmf.fit_transform(augmented)            # [N, k]
    W = nmf.components_                         # [k, 2D]
    d = delta.shape[1]
    W_signed = W[:, :d] - W[:, d:]              # [k, D]
    return {
        "W_signed": W_signed.astype(np.float32),
        "H": H.astype(np.float32),
        "reconstruction_err": float(nmf.reconstruction_err_),
        "n_iter": int(nmf.n_iter_),
    }


def _pca(delta: np.ndarray, k: int) -> dict:
    pca = PCA(n_components=k, svd_solver="auto", random_state=0)
    H = pca.fit_transform(delta)                # [N, k]
    return {
        "W_signed": pca.components_.astype(np.float32),  # [k, D]
        "H": H.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
    }


def _category_loadings(
    delta: np.ndarray,
    categories: list[str],
    cat_arr: np.ndarray,
    W_signed: np.ndarray,
) -> np.ndarray:
    """Cosine of per-category mean Δ with each basis vector. Returns [C, k]."""
    W = W_signed
    W_norm = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    out = np.zeros((len(categories), W.shape[0]), dtype=np.float32)
    for ci, cat in enumerate(categories):
        sel = cat_arr == cat
        if not sel.any():
            continue
        mu = delta[sel].mean(axis=0)
        mu_n = mu / (np.linalg.norm(mu) + 1e-12)
        out[ci] = W_norm @ mu_n
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--steering-config", type=Path, default=Path("configs/steering.yaml"))
    p.add_argument("--activations-root", type=Path, default=Path("data/activations"))
    p.add_argument("--output-dir", type=Path, default=Path("data/emotion_code"))
    p.add_argument("--layer", type=int, default=None,
                   help="Layer to decompose. Default = middle of hook_layers.")
    p.add_argument("--method", choices=["nmf", "pca", "both"], default="both")
    p.add_argument("--k", type=int, default=None,
                   help="Override n_basis from steering.yaml.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.8)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    profile = cfg["active"]
    sc = yaml.safe_load(args.steering_config.read_text())
    k = args.k or int(sc["emotion_code"]["n_basis"])

    bundle = load_activations(profile=profile, root=args.activations_root)
    layer = args.layer if args.layer is not None else bundle.layers[len(bundle.layers) // 2]
    if layer not in bundle.layers:
        raise ValueError(f"layer {layer} not in {bundle.layers}")
    print(f"[basis] profile={profile} layer={layer} k={k} method={args.method}")

    train_mask, _ = make_split(bundle.meta, train_frac=args.train_frac, seed=args.seed)
    delta = _build_delta(bundle, layer, train_mask)
    print(f"[basis] Δ.shape = {delta.shape}")

    categories = sorted(bundle.meta["category"].unique().tolist())
    cat_arr = bundle.meta["category"].to_numpy()[train_mask]

    payload: dict = {
        "profile": profile,
        "layer": layer,
        "k": k,
        "categories": categories,
        "split_seed": args.seed,
        "train_frac": args.train_frac,
    }

    if args.method in ("nmf", "both"):
        print("[basis] fitting NMF (sign-split)...")
        nmf = _signed_nmf(delta, k=k)
        load_nmf = _category_loadings(delta, categories, cat_arr, nmf["W_signed"])
        payload["nmf"] = {
            "W": torch.from_numpy(nmf["W_signed"]),
            "category_loadings": torch.from_numpy(load_nmf),
            "reconstruction_err": nmf["reconstruction_err"],
            "n_iter": nmf["n_iter"],
        }
        print(f"[basis] NMF reconstruction_err={nmf['reconstruction_err']:.4f} "
              f"iters={nmf['n_iter']}")

    if args.method in ("pca", "both"):
        print("[basis] fitting PCA...")
        pca = _pca(delta, k=k)
        load_pca = _category_loadings(delta, categories, cat_arr, pca["W_signed"])
        payload["pca"] = {
            "W": torch.from_numpy(pca["W_signed"]),
            "category_loadings": torch.from_numpy(load_pca),
            "explained_variance_ratio": pca["explained_variance_ratio"],
        }
        print(f"[basis] PCA explained_var={sum(pca['explained_variance_ratio']):.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "basis.pt"
    torch.save(payload, out_path)
    print(f"[basis] wrote {out_path}")

    summary = {
        "profile": profile,
        "layer": layer,
        "k": k,
        "categories": categories,
    }
    if "nmf" in payload:
        summary["nmf"] = {
            "reconstruction_err": payload["nmf"]["reconstruction_err"],
            "category_loadings": payload["nmf"]["category_loadings"].tolist(),
        }
    if "pca" in payload:
        summary["pca"] = {
            "explained_variance_ratio": payload["pca"]["explained_variance_ratio"],
            "category_loadings": payload["pca"]["category_loadings"].tolist(),
        }
    (args.output_dir / "basis.summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
