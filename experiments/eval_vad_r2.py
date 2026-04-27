"""Re-evaluate the held-out R² of the saved VAD mapping.

Independent of ``src.emotion_code.vad`` (which trains and reports during
fitting); reads the cached embeddings from ``data/emotion_code/_vad_cache.npz``
to avoid recomputing activations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mapping", type=Path,
                   default=Path("data/emotion_code/vad_mapping.pt"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/results/vad_r2.json"))
    args = p.parse_args()

    mp = torch.load(args.mapping, map_location="cpu", weights_only=False)
    summary = {
        "layer": mp["layer"],
        "axes": mp["axes"],
        "r2": mp["r2"],
        "n_train": mp.get("n_train"),
        "n_val": mp.get("n_val"),
        "min_r2": float(min(mp["r2"].values())),
        "mean_r2": float(np.mean(list(mp["r2"].values()))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"[vad-r2] {summary}")


if __name__ == "__main__":
    main()
