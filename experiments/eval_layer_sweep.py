"""Layer sweep: per-layer CAA train/val linear-separability accuracy.

For each hook layer we fit a logistic regression on (pos vs neg) train
activations and report held-out accuracy. The layer with the highest val
accuracy is the most linearly separable and thus the natural choice for
steering injection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.emotion_code.io import load_activations, make_split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--activations-root", type=Path, default=Path("data/activations"))
    p.add_argument("--output", type=Path, default=Path("experiments/results/layer_sweep.json"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    bundle = load_activations(root=args.activations_root)
    train_mask, val_mask = make_split(bundle.meta, seed=args.seed)
    print(f"[sweep] layers={bundle.layers} train={train_mask.sum()} val={val_mask.sum()}")

    results = {}
    for layer in bundle.layers:
        pos = bundle.pos[layer].numpy()
        neg = bundle.neg[layer].numpy()
        X = np.concatenate([pos, neg], axis=0)
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        # Mask repeated for both halves
        m_train = np.concatenate([train_mask, train_mask])
        m_val = np.concatenate([val_mask, val_mask])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[m_train], y[m_train])
        acc = float(clf.score(X[m_val], y[m_val]))
        results[layer] = {"val_acc": acc}
        print(f"[sweep] layer={layer}  val_acc={acc:.4f}")

    best_layer = max(results, key=lambda k: results[k]["val_acc"])
    out = {
        "profile": bundle.profile,
        "layers": bundle.layers,
        "per_layer": {str(k): v for k, v in results.items()},
        "best_layer": best_layer,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[sweep] best_layer={best_layer}  -> {args.output}")


if __name__ == "__main__":
    main()
