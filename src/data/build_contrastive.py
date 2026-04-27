"""Build contrastive (positive, negative) text pairs per Plutchik category.

For each category c we emit `pairs_per_category` pairs split into three
provenance buckets controlled by `contrastive.composition` in configs/data.yaml:

  • mined  (default 0.60): positive = real example with label == c.
                            negative = real example whose label is either
                            "neutral" or `PLUTCHIK_OPPOSITE[c]`, controlled by
                            `negative_strategy` (`mixed` mixes both half/half).
  • llm_swap (default 0.30): emitted as PLACEHOLDER rows with provenance =
                            "llm_swap" but text drawn from the mined pool;
                            an offline rewriting step (Phase A.5) fills them
                            in. They are still useful for sampling diagnostics.
  • template (default 0.10): synthesised from a small prompt template per
                            category, paired with a neutral template.

Output schema (parquet):
  pair_id, category, provenance, pos_id, pos_text, neg_id, neg_text,
  neg_label, source_pos, source_neg

Usage
-----
    uv run python -m src.data.build_contrastive \
        --input  data/unified/examples.filtered.parquet \
        --config configs/data.yaml \
        --output data/contrastive/pairs.parquet \
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from .schema import PLUTCHIK, PLUTCHIK_OPPOSITE


# Small fallback templates used to fill the `template` bucket. Kept short on
# purpose — templates exist to anchor each category, not to dominate.
TEMPLATES: dict[str, list[str]] = {
    "joy":          ["I feel wonderful right now.", "Everything turned out great."],
    "trust":        ["I know I can rely on you.", "She has always been honest with me."],
    "fear":         ["I am terrified of what will happen next.", "Something feels deeply wrong."],
    "surprise":     ["I cannot believe what just happened.", "That came completely out of nowhere."],
    "sadness":      ["I feel hollow and tired.", "Nothing seems to matter anymore."],
    "disgust":      ["This is revolting.", "I can barely stand to look at it."],
    "anger":        ["I am furious about this.", "How dare they do that to us."],
    "anticipation": ["I cannot wait for tomorrow.", "Something exciting is just around the corner."],
}
NEUTRAL_TEMPLATES = [
    "The package was delivered this afternoon.",
    "The report covers the second quarter results.",
    "She walked to the office at the usual time.",
    "The form must be submitted by Friday.",
]


def _pick_negative_label(category: str, strategy: str, rng: random.Random) -> str:
    """Resolve the desired label for a negative example."""
    opp = PLUTCHIK_OPPOSITE.get(category)
    if strategy == "neutral" or opp is None:
        return "neutral"
    if strategy == "opposite":
        return opp
    # mixed: 50/50
    return "neutral" if rng.random() < 0.5 else opp


def _sample_one(pool: list[dict], rng: random.Random) -> dict | None:
    if not pool:
        return None
    return pool[rng.randrange(len(pool))]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/unified/examples.filtered.parquet"),
    )
    p.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/contrastive/pairs.parquet"),
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["contrastive"]
    n_per_cat = int(cfg["pairs_per_category"])
    comp = cfg["composition"]
    f_mined = float(comp["mined_fraction"])
    f_llm = float(comp["llm_swap_fraction"])
    f_tmpl = float(comp["template_fraction"])
    strategy = cfg.get("negative_strategy", "mixed")
    min_tokens = int(cfg.get("min_text_tokens", 3))
    max_tokens = int(cfg.get("max_text_tokens", 80))

    df = pd.read_parquet(args.input)
    print(f"[cp] loaded {len(df)} filtered rows")

    # Per-category indices (drop very short / very long texts).
    def _ok(text: str) -> bool:
        n = len(text.split())
        return min_tokens <= n <= max_tokens

    pools: dict[str, list[dict]] = defaultdict(list)
    for row in df.itertuples(index=False):
        if not _ok(row.text):
            continue
        rec = {"id": row.id, "text": row.text, "source": row.source,
               "label": row.label_primary}
        pools[row.label_primary].append(rec)
    for cat, pool in pools.items():
        print(f"[cp] pool[{cat}] = {len(pool)}")

    rng = random.Random(args.seed)
    out: list[dict] = []
    pid = 0

    n_mined = int(round(n_per_cat * f_mined))
    n_llm = int(round(n_per_cat * f_llm))
    n_tmpl = max(0, n_per_cat - n_mined - n_llm)

    for cat in PLUTCHIK:
        pos_pool = pools.get(cat, [])
        if not pos_pool:
            print(f"[cp] WARNING no positives for {cat}; skipping")
            continue

        # mined
        for _ in range(n_mined):
            pos = _sample_one(pos_pool, rng)
            neg_label = _pick_negative_label(cat, strategy, rng)
            neg = _sample_one(pools.get(neg_label, []), rng)
            if neg is None:
                neg = _sample_one(pools.get("neutral", []), rng)
                neg_label = "neutral" if neg else neg_label
            if neg is None:
                continue
            out.append({
                "pair_id": pid, "category": cat, "provenance": "mined",
                "pos_id": pos["id"], "pos_text": pos["text"], "source_pos": pos["source"],
                "neg_id": neg["id"], "neg_text": neg["text"], "source_neg": neg["source"],
                "neg_label": neg_label,
            })
            pid += 1

        # llm_swap placeholders (text duplicated from mined pool; rewrite later)
        for _ in range(n_llm):
            pos = _sample_one(pos_pool, rng)
            neg_label = _pick_negative_label(cat, strategy, rng)
            neg = _sample_one(pools.get(neg_label, []), rng) or _sample_one(
                pools.get("neutral", []), rng
            )
            if neg is None:
                continue
            out.append({
                "pair_id": pid, "category": cat, "provenance": "llm_swap",
                "pos_id": pos["id"], "pos_text": pos["text"], "source_pos": pos["source"],
                "neg_id": neg["id"], "neg_text": neg["text"], "source_neg": neg["source"],
                "neg_label": neg_label,
            })
            pid += 1

        # template
        tmpls = TEMPLATES.get(cat, [])
        for _ in range(n_tmpl):
            if not tmpls:
                break
            pos_text = rng.choice(tmpls)
            neg_text = rng.choice(NEUTRAL_TEMPLATES)
            out.append({
                "pair_id": pid, "category": cat, "provenance": "template",
                "pos_id": f"tmpl-{cat}-{pid}", "pos_text": pos_text, "source_pos": "template",
                "neg_id": f"tmpl-neutral-{pid}", "neg_text": neg_text, "source_neg": "template",
                "neg_label": "neutral",
            })
            pid += 1

    pairs = pd.DataFrame.from_records(out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(args.output, index=False)

    stats = {
        "total_pairs": int(len(pairs)),
        "by_category": pairs.groupby("category").size().to_dict() if len(pairs) else {},
        "by_provenance": pairs.groupby("provenance").size().to_dict() if len(pairs) else {},
        "config": {
            "pairs_per_category": n_per_cat,
            "mined": n_mined, "llm_swap": n_llm, "template": n_tmpl,
            "negative_strategy": strategy,
            "min_text_tokens": min_tokens, "max_text_tokens": max_tokens,
            "seed": args.seed,
        },
    }
    stats_path = args.output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, default=int))
    print(f"[cp] wrote {len(pairs)} pairs → {args.output}")
    print(f"[cp] stats → {stats_path}")


if __name__ == "__main__":
    main()
