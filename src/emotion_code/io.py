"""I/O helpers for activation shards and train/val splitting.

Activations were written by ``src.activations.collect`` as one safetensors
file per (shard, side) with key ``layer_{L}`` of shape ``[shard_size, d_model]``.
``index.parquet`` carries row-aligned metadata
(``shard``, ``row``, ``pair_id``, ``category``, ``provenance``) — its row order
matches the per-shard concatenation order, so a global stack of shards in
shard-then-row order yields the same N-row table as ``index.parquet``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from safetensors.torch import load_file


@dataclass
class ActivationBundle:
    """Stacked activations for one model profile.

    Attributes
    ----------
    pos, neg : dict[int, Tensor]
        ``layer -> tensor[N, d_model]`` (float32, on CPU). Row order matches
        ``meta``.
    meta : DataFrame
        Per-row metadata with at least ``pair_id``, ``category``, ``provenance``.
    layers : list[int]
        Hook layers present in ``pos``/``neg``.
    d_model : int
    profile : str
    """

    pos: dict[int, torch.Tensor]
    neg: dict[int, torch.Tensor]
    meta: pd.DataFrame
    layers: list[int]
    d_model: int
    profile: str


def _activations_dir(profile: str, root: Path) -> Path:
    return root / profile


def load_activations(
    profile: str | None = None,
    root: Path | str = Path("data/activations"),
    model_config: Path | str = Path("configs/model.yaml"),
    layers: list[int] | None = None,
) -> ActivationBundle:
    """Load all shards for ``profile`` (default = active profile in config).

    Parameters
    ----------
    profile : str or None
        Which subdirectory under ``root`` to read. If None, uses
        ``active`` from ``model_config``.
    layers : list[int] or None
        Subset of layers to materialise. None = all layers in manifest.
    """
    root = Path(root)
    if profile is None:
        cfg = yaml.safe_load(Path(model_config).read_text())
        profile = cfg["active"]

    adir = _activations_dir(profile, root)
    manifest = json.loads((adir / "manifest.json").read_text())
    all_layers: list[int] = list(manifest["hook_layers"])
    if layers is None:
        layers = all_layers
    else:
        missing = set(layers) - set(all_layers)
        if missing:
            raise ValueError(f"layers {sorted(missing)} not in manifest {all_layers}")

    meta = pd.read_parquet(adir / "index.parquet")
    n_shards = int(manifest["n_shards"])

    pos_chunks: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    neg_chunks: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    for s in range(n_shards):
        pos_path = adir / f"shard_{s:05d}_pos.safetensors"
        neg_path = adir / f"shard_{s:05d}_neg.safetensors"
        pos_t = load_file(str(pos_path))
        neg_t = load_file(str(neg_path))
        for l in layers:
            key = f"layer_{l}"
            pos_chunks[l].append(pos_t[key].to(torch.float32))
            neg_chunks[l].append(neg_t[key].to(torch.float32))

    pos = {l: torch.cat(pos_chunks[l], dim=0).contiguous() for l in layers}
    neg = {l: torch.cat(neg_chunks[l], dim=0).contiguous() for l in layers}

    n_rows = pos[layers[0]].shape[0]
    if n_rows != len(meta):
        raise RuntimeError(
            f"row count mismatch: activations={n_rows}, index.parquet={len(meta)}"
        )

    return ActivationBundle(
        pos=pos,
        neg=neg,
        meta=meta.reset_index(drop=True),
        layers=layers,
        d_model=int(manifest["d_model"]),
        profile=profile,
    )


def make_split(
    meta: pd.DataFrame,
    train_frac: float = 0.8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train/val split, stratified by ``category`` and grouped
    by ``pair_id`` (the same pair never straddles the split).

    Returns
    -------
    train_mask, val_mask : np.ndarray[bool] of length ``len(meta)``
    """
    rng = np.random.default_rng(seed)
    n = len(meta)
    train_mask = np.zeros(n, dtype=bool)

    for _, idx in meta.groupby("category", sort=True).groups.items():
        idx_arr = np.asarray(list(idx))
        # one row = one pair_id here (each pair_id appears once in index.parquet)
        rng.shuffle(idx_arr)
        cut = int(round(len(idx_arr) * train_frac))
        train_mask[idx_arr[:cut]] = True

    val_mask = ~train_mask
    return train_mask, val_mask
