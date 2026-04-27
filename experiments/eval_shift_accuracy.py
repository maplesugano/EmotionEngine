"""Shift accuracy: does +α push the classifier toward the target category?

Loads the generation cache from ``_gen_cache.py``, classifies each
generation with ``j-hartmann/emotion-english-distilroberta-base`` (7-class
Ekman + neutral), maps to Plutchik via the same crosswalk used in Phase A,
and reports per-category and average shift accuracy at α=+max.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# j-hartmann labels -> Plutchik (best-effort; "neutral"/"surprise" pass through)
HARTMANN_TO_PLUTCHIK = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "neutral": "neutral",
    "sadness": "sadness",
    "surprise": "surprise",
}
# Categories the classifier cannot directly score; mark "unmeasurable".
UNMEASURABLE = {"trust", "anticipation"}


def _classify(texts: list[str], batch_size: int = 32) -> pd.DataFrame:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "j-hartmann/emotion-english-distilroberta-base"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(device).eval()
    id2label = mdl.config.id2label

    rows = []
    with torch.inference_mode():
        for s in tqdm(range(0, len(texts), batch_size), desc="[shift] classify"):
            chunk = texts[s : s + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            logits = mdl(**enc).logits
            prob = torch.softmax(logits, dim=-1).cpu().numpy()
            for p in prob:
                top = int(np.argmax(p))
                row = {id2label[i]: float(p[i]) for i in range(p.shape[0])}
                row["pred_label"] = id2label[top]
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path,
                   default=Path("experiments/results/_gen_cache.parquet"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/results/shift_accuracy.csv"))
    p.add_argument("--alpha-pos", type=float, default=2.0)
    args = p.parse_args()

    df = pd.read_parquet(args.cache)
    cls = _classify(df["generation"].astype(str).tolist())
    df = pd.concat([df.reset_index(drop=True), cls], axis=1)

    # Map predicted label -> Plutchik
    df["pred_plutchik"] = df["pred_label"].map(HARTMANN_TO_PLUTCHIK).fillna("other")

    # Per-category: hit rate at α=+alpha_pos vs at α=0 (baseline).
    target_alpha = args.alpha_pos
    rows = []
    for cat, sub in df.groupby("category"):
        if cat in UNMEASURABLE:
            rows.append({"category": cat, "n": 0, "shift_acc": np.nan,
                         "baseline_acc": np.nan, "delta": np.nan,
                         "note": "no classifier label"})
            continue
        pos = sub[sub["alpha_unit"] == target_alpha]
        base = sub[sub["alpha_unit"] == 0.0]
        if len(pos) == 0 or len(base) == 0:
            continue
        shift_acc = float((pos["pred_plutchik"] == cat).mean())
        baseline_acc = float((base["pred_plutchik"] == cat).mean())
        rows.append({
            "category": cat, "n": int(len(pos)),
            "shift_acc": shift_acc, "baseline_acc": baseline_acc,
            "delta": shift_acc - baseline_acc, "note": "",
        })

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    measurable = out.dropna(subset=["shift_acc"])
    summary = {
        "alpha_pos": target_alpha,
        "mean_shift_acc": float(measurable["shift_acc"].mean()) if len(measurable) else None,
        "mean_baseline_acc": float(measurable["baseline_acc"].mean()) if len(measurable) else None,
        "mean_delta": float(measurable["delta"].mean()) if len(measurable) else None,
        "per_category": rows,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"[shift] wrote {args.output}  mean_shift_acc={summary['mean_shift_acc']}")


if __name__ == "__main__":
    main()
