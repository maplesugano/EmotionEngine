"""Quality filter: discard examples where an external emotion classifier
strongly disagrees with the source label.

We use `j-hartmann/emotion-english-distilroberta-base`, which outputs 7 Ekman
labels (anger, disgust, fear, joy, neutral, sadness, surprise). For Plutchik
labels with a direct Ekman counterpart we keep an example iff
    P(classifier == counterpart) >= min_agreement.
For Plutchik labels with no clean Ekman counterpart (`trust`, `anticipation`),
we keep the example unconditionally — these classes carry the most signal we
explicitly want to preserve, and the classifier cannot adjudicate them.

EmoBank rows (`source == "emobank"`) bypass the filter entirely; they are used
only for the diagnostic V/A/D regressor.

Usage
-----
    uv run python -m src.data.quality_filter \
        --input  data/unified/examples.parquet \
        --output data/unified/examples.filtered.parquet \
        --batch-size 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CLASSIFIER_ID = "j-hartmann/emotion-english-distilroberta-base"

# Plutchik → Ekman counterpart used by the DistilRoBERTa classifier.
# `None` means the classifier cannot adjudicate — we keep the example.
PLUTCHIK_TO_EKMAN: dict[str, str | None] = {
    "joy": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
    "neutral": "neutral",
    "trust": None,
    "anticipation": None,
}


@torch.inference_mode()
def _score_batch(model, tokenizer, texts: list[str], device) -> np.ndarray:
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=256
    ).to(device)
    logits = model(**enc).logits
    return torch.softmax(logits, dim=-1).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/unified/examples.parquet"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/unified/examples.filtered.parquet"),
    )
    p.add_argument("--min-agreement", type=float, default=0.50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    print(f"[qf] loaded {len(df)} rows from {args.input}")

    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_ID)
    model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_ID).to(
        args.device
    )
    model.eval()
    label_to_idx = {v.lower(): k for k, v in model.config.id2label.items()}
    print(f"[qf] classifier labels: {sorted(label_to_idx)}")

    n = len(df)
    target_idx = np.full(n, -1, dtype=np.int64)
    bypass = np.zeros(n, dtype=bool)
    for i, (src, lab) in enumerate(zip(df["source"].values, df["label_primary"].values)):
        if src == "emobank":
            bypass[i] = True
            continue
        ek = PLUTCHIK_TO_EKMAN.get(lab)
        if ek is None:
            bypass[i] = True
            continue
        target_idx[i] = label_to_idx[ek]

    # Score every row (cheap on a 67M-param model; we need argmax + score anyway).
    qf_score = np.zeros(n, dtype=np.float32)
    qf_argmax = np.empty(n, dtype=object)
    texts = df["text"].astype(str).tolist()

    bs = args.batch_size
    for start in range(0, n, bs):
        batch = texts[start : start + bs]
        probs = _score_batch(model, tokenizer, batch, args.device)
        argmax = probs.argmax(axis=1)
        idx = np.arange(probs.shape[0])
        # Score against target label for non-bypass rows; for bypass rows we
        # still record the argmax label for diagnostics but score = NaN.
        ti = target_idx[start : start + len(batch)]
        scores = np.where(ti >= 0, probs[idx, np.where(ti < 0, 0, ti)], np.nan)
        qf_score[start : start + len(batch)] = scores
        for j, am in enumerate(argmax):
            qf_argmax[start + j] = model.config.id2label[am].lower()
        if (start // bs) % 50 == 0:
            print(f"[qf] {start + len(batch)}/{n}")

    keep = bypass | (qf_score >= args.min_agreement)
    df["qf_score"] = qf_score
    df["qf_argmax"] = qf_argmax
    df["qf_bypass"] = bypass
    df["qf_keep"] = keep

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_kept = df[keep].reset_index(drop=True)
    df_kept.to_parquet(args.output, index=False)

    stats = {
        "input_rows": int(n),
        "kept_rows": int(keep.sum()),
        "bypass_rows": int(bypass.sum()),
        "min_agreement": args.min_agreement,
        "kept_by_source": (
            df_kept.groupby("source").size().to_dict() if len(df_kept) else {}
        ),
        "kept_by_category": (
            df_kept.groupby("label_primary").size().to_dict() if len(df_kept) else {}
        ),
    }
    stats_path = args.output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, default=int))
    print(f"[qf] kept {keep.sum()}/{n} → {args.output}")
    print(f"[qf] stats → {stats_path}")


if __name__ == "__main__":
    main()
