"""Decompose per-pair Δ activations using one of several factorisations.

Public functions are pure (numpy in / numpy out) so :mod:`basis_sweep` and
:mod:`basis_metrics` can both reuse them. Each function returns a dict with at
least ``W`` ([k, D] in the original signed space) and ``H`` ([N, k] mixing
coefficients on the *training* sample).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def fit_nmf(delta: np.ndarray, k: int, *, max_iter: int = 2000,
            tol: float = 1e-5, seed: int = 0) -> dict[str, Any]:
    """Sign-split NMF: concat [relu(Δ) | relu(-Δ)] and recover signed prototypes."""
    from sklearn.decomposition import NMF

    pos = np.maximum(delta, 0.0)
    neg = np.maximum(-delta, 0.0)
    aug = np.concatenate([pos, neg], axis=1)
    nmf = NMF(
        n_components=k, init="nndsvda", max_iter=max_iter, tol=tol,
        solver="cd", beta_loss="frobenius", random_state=seed,
    )
    H = nmf.fit_transform(aug)
    W = nmf.components_
    d = delta.shape[1]
    W_signed = (W[:, :d] - W[:, d:]).astype(np.float32)
    return {
        "W": W_signed, "H": H.astype(np.float32),
        "reconstruction_err": float(nmf.reconstruction_err_),
        "n_iter": int(nmf.n_iter_),
        "converged": int(nmf.n_iter_) < max_iter,
    }


def fit_pca(delta: np.ndarray, k: int, *, seed: int = 0, **_: Any) -> dict[str, Any]:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=k, svd_solver="auto", random_state=seed)
    H = pca.fit_transform(delta)
    return {
        "W": pca.components_.astype(np.float32),
        "H": H.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
    }


def fit_ica(delta: np.ndarray, k: int, *, max_iter: int = 2000, tol: float = 1e-4,
            seed: int = 0) -> dict[str, Any]:
    """FastICA on Δ. Returns components in original space (whitened internally)."""
    from sklearn.decomposition import FastICA

    ica = FastICA(
        n_components=k, whiten="unit-variance", max_iter=max_iter, tol=tol,
        random_state=seed, algorithm="parallel",
    )
    H = ica.fit_transform(delta)            # [N, k]
    W = ica.components_                     # [k, D]  (un-mixing matrix wrt whitened space)
    return {
        "W": W.astype(np.float32),
        "H": H.astype(np.float32),
        "n_iter": int(ica.n_iter_),
        "converged": int(ica.n_iter_) < max_iter,
    }


def fit_dict(delta: np.ndarray, k: int, *, max_iter: int = 2000,
             alpha: float = 1.0, seed: int = 0) -> dict[str, Any]:
    """Sparse dictionary learning (mini-batch). Encourages sparse loadings."""
    from sklearn.decomposition import MiniBatchDictionaryLearning

    dl = MiniBatchDictionaryLearning(
        n_components=k, alpha=alpha, max_iter=max_iter,
        batch_size=256, random_state=seed, fit_algorithm="cd",
        transform_algorithm="lasso_cd", transform_alpha=alpha,
    )
    H = dl.fit_transform(delta)
    return {
        "W": dl.components_.astype(np.float32),
        "H": H.astype(np.float32),
        "n_iter": int(getattr(dl, "n_iter_", max_iter)),
    }


DECOMPOSERS = {
    "nmf": fit_nmf,
    "pca": fit_pca,
    "ica": fit_ica,
    "dict": fit_dict,
}


def category_loadings(
    delta: np.ndarray, categories: list[str], cat_arr: np.ndarray, W: np.ndarray,
) -> np.ndarray:
    """Cosine of per-category mean Δ with each basis vector. [C, k]."""
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    out = np.zeros((len(categories), W.shape[0]), dtype=np.float32)
    for ci, cat in enumerate(categories):
        sel = cat_arr == cat
        if not sel.any():
            continue
        mu = delta[sel].mean(axis=0)
        mu_n = mu / (np.linalg.norm(mu) + 1e-12)
        out[ci] = Wn @ mu_n
    return out
