"""LLM-as-judge for lexical-gap steering generations.

For each generation in lexical_gap_steering/generations.parquet, ask GPT
to score (0-1) how well the text expresses each Plutchik primary emotion,
plus a free-form ``other_label`` and ``other_score`` for any state that is
*not* well captured by Plutchik (lexical-gap detector).

Aggregates:
  - mean per-target Plutchik max-score      (low => not Plutchik)
  - mean per-target other_score             (high => lexical-gap)
  - mean per-target Plutchik entropy        (high => mixed)
  - top other_labels by frequency           (qualitative inventory)

Outputs:
  judgments.parquet      one row per generation
  summary_by_target.csv  aggregated per target × alpha
  other_labels.csv       (label, count) sorted by frequency
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm

PLUTCHIK = ["anger", "anticipation", "disgust", "fear",
            "joy", "sadness", "surprise", "trust"]

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "scores": {
            "type": "object",
            "additionalProperties": False,
            "properties": {p: {"type": "number"} for p in PLUTCHIK},
            "required": PLUTCHIK,
        },
        "other_label": {
            "type": "string",
            "description": "Short phrase (1-4 words) for any state NOT well "
                           "captured by Plutchik (e.g., 'self-individuation', "
                           "'paranoid hypervigilance', 'metaphysical wonder'). "
                           "Empty string if none.",
        },
        "other_score": {
            "type": "number",
            "description": "0..1: confidence that the text expresses a state "
                           "outside the 8 Plutchik categories.",
        },
        "rationale": {"type": "string"},
    },
    "required": ["scores", "other_label", "other_score", "rationale"],
}

SYSTEM = (
    "You are an emotion-analysis judge. Given a short text passage, you must "
    "(a) score each of the 8 Plutchik primary emotions (anger, anticipation, "
    "disgust, fear, joy, sadness, surprise, trust) on a 0..1 scale of how "
    "strongly the text expresses that state, (b) name in 1-4 words any state "
    "the text expresses that is NOT well captured by Plutchik, and rate that "
    "state on 0..1 (other_score). Be calibrated: 0.0 means absent, 0.5 means "
    "noticeably present, 1.0 means dominant. If the text is purely neutral / "
    "informational, all scores should be near 0."
)

USER_TMPL = "Text:\n\"\"\"\n{text}\n\"\"\""


def _judge_one(client: OpenAI, model: str, text: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(text=text)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "emotion_judge", "schema": JUDGE_SCHEMA, "strict": True},
        },
        temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=Path,
                   default=Path("experiments/results/lexical_gap_steering/generations.parquet"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/results/lexical_gap_judge"))
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Skip rows already present in judgments.parquet")
    p.add_argument("--sleep", type=float, default=0.0)
    args = p.parse_args()

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set (checked .env)")
    client = OpenAI(api_key=api_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "judgments.parquet"

    df = pd.read_parquet(args.gens)
    if args.limit:
        df = df.head(args.limit).copy()
    print(f"[judge] {len(df)} generations to score with {args.model}")

    done_keys: set[tuple] = set()
    prev_rows: list[dict] = []
    if args.resume and out_path.exists():
        prev = pd.read_parquet(out_path)
        prev_rows = prev.to_dict(orient="records")
        done_keys = {(r["target"], float(r["alpha_unit"]), int(r["prompt_id"]))
                     for r in prev_rows}
        print(f"[judge] resume: {len(done_keys)} prior judgments")

    rows: list[dict] = list(prev_rows)
    pbar = tqdm(total=len(df), initial=len(done_keys), desc="[judge]")
    n_new = 0
    for r in df.to_dict(orient="records"):
        key = (r["target"], float(r["alpha_unit"]), int(r["prompt_id"]))
        if key in done_keys:
            continue
        text = r["generation"] or ""
        try:
            j = _judge_one(client, args.model, text)
        except Exception as e:
            print(f"[judge] error key={key}: {e}", flush=True)
            time.sleep(2.0)
            continue
        scores = j["scores"]
        rows.append({
            "target": r["target"],
            "alpha_unit": float(r["alpha_unit"]),
            "prompt_id": int(r["prompt_id"]),
            "prompt": r["prompt"],
            "generation": text,
            **{f"s_{k}": float(scores[k]) for k in PLUTCHIK},
            "other_label": str(j.get("other_label") or "").strip().lower(),
            "other_score": float(j.get("other_score") or 0.0),
            "rationale": j.get("rationale", ""),
        })
        done_keys.add(key)
        n_new += 1
        pbar.update(1)
        if n_new % 16 == 0:
            pd.DataFrame(rows).to_parquet(out_path, index=False)
        if args.sleep > 0:
            time.sleep(args.sleep)
    pbar.close()
    judg = pd.DataFrame(rows)
    judg.to_parquet(out_path, index=False)
    print(f"[judge] wrote {out_path} ({len(judg)} rows)")

    # Aggregate -------------------------------------------------------------
    s_cols = [f"s_{k}" for k in PLUTCHIK]
    judg["plutchik_max"] = judg[s_cols].max(axis=1)
    judg["plutchik_sum"] = judg[s_cols].sum(axis=1)

    def _ent(row):
        v = np.array([row[c] for c in s_cols], dtype=float)
        if v.sum() <= 1e-9:
            return 0.0
        p = v / v.sum()
        p = np.clip(p, 1e-12, 1.0)
        return float(-(p * np.log(p)).sum())
    judg["plutchik_entropy"] = judg.apply(_ent, axis=1)

    agg_rows = []
    for (tname, au), sub in judg.groupby(["target", "alpha_unit"]):
        d = {
            "target": tname, "alpha_unit": float(au), "n": int(len(sub)),
            "mean_plutchik_max": float(sub["plutchik_max"].mean()),
            "mean_plutchik_sum": float(sub["plutchik_sum"].mean()),
            "mean_plutchik_entropy": float(sub["plutchik_entropy"].mean()),
            "mean_other_score": float(sub["other_score"].mean()),
            "frac_other_dominant": float(
                (sub["other_score"] > sub["plutchik_max"]).mean()
            ),
        }
        for c in s_cols:
            d[f"mean_{c}"] = float(sub[c].mean())
        agg_rows.append(d)
    agg = pd.DataFrame(agg_rows).sort_values(["alpha_unit", "target"])
    agg.to_csv(args.output_dir / "summary_by_target.csv", index=False)

    print("[judge] === per-target (alpha=+max) ===")
    pos = agg[agg["alpha_unit"] == agg["alpha_unit"].max()]
    cols = ["target", "mean_plutchik_max", "mean_other_score",
            "frac_other_dominant", "mean_plutchik_entropy"]
    print(pos[cols].to_string(index=False))

    # Other-label inventory at α=+max
    pos_judg = judg[judg["alpha_unit"] == judg["alpha_unit"].max()].copy()
    pos_judg = pos_judg[pos_judg["other_label"] != ""]
    label_counts = (pos_judg.groupby(["target", "other_label"]).size()
                    .reset_index(name="count")
                    .sort_values(["target", "count"], ascending=[True, False]))
    label_counts.to_csv(args.output_dir / "other_labels.csv", index=False)
    print(f"[judge] wrote {args.output_dir}/other_labels.csv "
          f"({len(label_counts)} unique label-rows)")


if __name__ == "__main__":
    main()
