"""Learn a Valence/Arousal/Dominance linear map from EmoBank activations.

EmoBank (Buechel & Hahn 2017) provides ~10k English sentences scored on 3
continuous scales V/A/D ∈ [1, 5]. We collect Llama last-token residuals at
the chosen layer and fit Ridge regression for each axis.

Output: ``data/emotion_code/vad_mapping.pt``
    W : Tensor[3, d_model]   bias-absorbed weights for [V, A, D]
    b : Tensor[3]
    r2 : {"V": float, "A": float, "D": float}
    layer : int
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from src.activations._runtime import collect_batch, load_model, load_profile


def _load_emobank(hf_id: str, limit: int | None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (texts, vad[N,3], split_idx) where split_idx==0 train, 1 val, 2 test."""
    import pandas as pd

    csv_path = Path("data/raw/emobank.csv")
    if not csv_path.exists():
        from datasets import load_dataset

        ds = load_dataset(hf_id)
        texts: list[str] = []
        vad: list[list[float]] = []
        split: list[int] = []
        name_to_id = {"train": 0, "validation": 1, "test": 2}
        for split_name, sid in name_to_id.items():
            if split_name not in ds:
                continue
            for row in ds[split_name]:
                v, a, d = float(row["V"]), float(row["A"]), float(row["D"])
                texts.append(row["text"])
                vad.append([v, a, d])
                split.append(sid)
    else:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["text", "V", "A", "D"])
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 0]
        name_to_id = {"train": 0, "dev": 1, "validation": 1, "test": 2}
        texts = df["text"].tolist()
        vad = df[["V", "A", "D"]].values.astype(float).tolist()
        split = [name_to_id.get(s, 0) for s in df["split"].astype(str).tolist()]
    vad_arr = np.asarray(vad, dtype=np.float32)
    split_arr = np.asarray(split, dtype=np.int64)
    if limit and limit > 0 and limit < len(texts):
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(texts))[:limit]
        texts = [texts[i] for i in idx]
        vad_arr = vad_arr[idx]
        split_arr = split_arr[idx]
    return texts, vad_arr, split_arr


def _embed(model, device, texts: list[str], layer: int, batch_size: int) -> np.ndarray:
    out = []
    for s in tqdm(range(0, len(texts), batch_size), desc="[vad] embed"):
        chunk = texts[s : s + batch_size]
        cap = collect_batch(model, chunk, [layer], device)
        out.append(cap[layer].numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument("--output-dir", type=Path, default=Path("data/emotion_code"))
    p.add_argument("--layer", type=int, default=None,
                   help="Default = middle of hook_layers in profile.")
    p.add_argument("--limit", type=int, default=2000,
                   help="Cap on EmoBank rows (regression saturates fast).")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--cache", type=Path, default=Path("data/emotion_code/_vad_cache.npz"))
    args = p.parse_args()

    profile, prof_name = load_profile(args.config)
    layer = args.layer if args.layer is not None else profile["hook_layers"][len(profile["hook_layers"]) // 2]
    print(f"[vad] profile={prof_name} layer={layer} limit={args.limit}")

    dcfg = yaml.safe_load(args.data_config.read_text())
    eb = next(d for d in dcfg["source_datasets"] if d["name"] == "emobank")
    hf_id = eb["hf_id"]

    texts, vad, split = _load_emobank(hf_id, args.limit)
    print(f"[vad] loaded {len(texts)} EmoBank rows  (V/A/D ranges: "
          f"{vad.min(0).tolist()} .. {vad.max(0).tolist()})")

    if args.cache.exists():
        z = np.load(args.cache)
        if (
            z["texts_hash"].item() == hash(tuple(texts))
            and int(z["layer"]) == layer
        ):
            X = z["X"]
            print(f"[vad] reused embedding cache {args.cache}")
        else:
            X = None
    else:
        X = None
    if X is None:
        model, device, _ = load_model(profile)
        X = _embed(model, device, texts, layer, args.batch_size)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.cache,
            X=X,
            layer=np.int64(layer),
            texts_hash=np.int64(hash(tuple(texts))),
        )
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"[vad] X.shape={X.shape}")

    # Use EmoBank's own train/val/test split where available, else 80/20.
    if (split == 0).any() and (split != 0).any():
        train_idx = np.flatnonzero(split == 0)
        val_idx = np.flatnonzero(split != 0)
    else:
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(X))
        cut = int(0.8 * len(X))
        train_idx, val_idx = perm[:cut], perm[cut:]

    axes = ["V", "A", "D"]
    W = np.zeros((3, X.shape[1]), dtype=np.float32)
    b = np.zeros(3, dtype=np.float32)
    r2 = {}
    for k, name in enumerate(axes):
        ridge = Ridge(alpha=args.ridge_alpha, fit_intercept=True)
        ridge.fit(X[train_idx], vad[train_idx, k])
        pred = ridge.predict(X[val_idx])
        r2[name] = float(r2_score(vad[val_idx, k], pred))
        W[k] = ridge.coef_.astype(np.float32)
        b[k] = float(ridge.intercept_)
        print(f"[vad] axis={name}  R²={r2[name]:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "vad_mapping.pt"
    torch.save(
        {
            "W": torch.from_numpy(W),
            "b": torch.from_numpy(b),
            "r2": r2,
            "axes": axes,
            "layer": layer,
            "profile": prof_name,
            "ridge_alpha": args.ridge_alpha,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
        },
        out_path,
    )
    (args.output_dir / "vad_mapping.summary.json").write_text(
        json.dumps(
            {"layer": layer, "r2": r2, "n_train": int(len(train_idx)),
             "n_val": int(len(val_idx))}, indent=2,
        )
    )
    print(f"[vad] wrote {out_path}")


if __name__ == "__main__":
    main()
