"""Phase C-3 follow-up: per-axis judge labelling for an entire basis.

For every component b_k (k = 0..K-1) of a basis artifact, generate text
with single-axis steering (α ∈ {-α_pos, 0, +α_pos}) over a pool of
neutral prompts, then ask GPT to:
  (a) score the 8 Plutchik primaries (0..1 each), and
  (b) emit a 1-4 word free-form ``other_label`` for any state that is
      *not* well captured by Plutchik.

Per axis we then aggregate:
  - top free-form labels by frequency at α=+/-α_pos
  - mean Plutchik scores, max, entropy
  - mean ``other_score`` and frac of rows where other > max(plutchik)
  - "best matching word" = mode of other_label at the active α

Outputs (under --output-dir):
  generations.parquet         one row per (axis, alpha_unit, prompt_id)
  judgments.parquet           same rows + judge fields
  per_axis_summary.csv        per (axis, alpha_unit) aggregates
  per_axis_top_labels.csv     top-N other_labels per axis with counts
  per_axis_top_labels.md      human-readable summary of best label per axis

Reuses the same JUDGE_SCHEMA / SYSTEM as eval_lexical_gap_judge so the
``other_label`` namespace is consistent and can later be embedded with
the §3.X.3 cluster pipeline (eval_lexical_gap_cluster.py).

Example:
  uv run python -m experiments.eval_basis_per_axis_judge \
      --basis data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt \
      --alpha-pos 2.0 --n-prompts 8 --max-new-tokens 48 \
      --output-dir experiments/results/per_axis_judge_L22_k64
"""
from __future__ import annotations
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm

from experiments._gen_cache import load_neutral_prompts
from experiments.eval_lexical_gap_judge import (
    JUDGE_SCHEMA,
    PLUTCHIK,
    SYSTEM,
    USER_TMPL,
)
from src.activations._runtime import load_model, load_profile
from src.steering.generate import steered_generate


# ---------- generation ------------------------------------------------------
def _generate_all(args, B, layer, scale, median_caa_norm, prompts):
    """Generate steered texts for every (axis, alpha_unit, prompt_id)."""
    K = B.shape[0]
    axes = list(range(K)) if args.axes is None else list(args.axes)
    alphas = sorted({float(a) for a in args.alphas})

    profile, _ = load_profile(args.config)
    sc = yaml.safe_load(args.steering_config.read_text())
    apply_to = sc["caa"].get("apply_to", "generation")
    model, _device, _ = load_model(profile)

    gen_path = args.output_dir / "generations.parquet"
    prev_rows: list[dict] = []
    done: set[tuple] = set()
    if args.resume and gen_path.exists():
        prev = pd.read_parquet(gen_path)
        prev_rows = prev.to_dict(orient="records")
        done = {
            (int(r["axis"]), float(r["alpha_unit"]), int(r["prompt_id"]))
            for r in prev_rows
        }
        print(f"[gen] resume: {len(done)} prior generations")

    rows: list[dict] = list(prev_rows)
    total = len(axes) * len(alphas) * len(prompts)
    pbar = tqdm(total=total, initial=len(done), desc="[gen]")
    for k in axes:
        vec = B[k].astype(np.float32)
        v_raw_norm = float(np.linalg.norm(vec))
        if args.alpha_mode == "caa_match":
            v_use = vec * (median_caa_norm / (v_raw_norm + 1e-12))
        else:
            v_use = vec
        v_norm = float(np.linalg.norm(v_use))
        v = torch.from_numpy(v_use.astype(np.float32))
        added = 0
        for au in alphas:
            alpha = au * scale * v_norm
            for pi, prompt in enumerate(prompts):
                key = (int(k), float(au), int(pi))
                if key in done:
                    continue
                out = steered_generate(
                    model, prompt, v,
                    alpha=alpha if au != 0.0 else 0.0,
                    layers=[layer], apply_to=apply_to,
                    max_new_tokens=args.max_new_tokens,
                )
                tail = (out[len(prompt):].strip()
                        if out.startswith(prompt) else out.strip())
                rows.append({
                    "axis": int(k),
                    "target": f"b{k}",
                    "alpha_unit": float(au),
                    "alpha": float(alpha),
                    "vector_norm": v_norm,
                    "layer": int(layer),
                    "prompt_id": int(pi),
                    "prompt": prompt,
                    "generation": tail,
                })
                done.add(key)
                added += 1
                pbar.update(1)
        if added:
            pd.DataFrame(rows).to_parquet(gen_path, index=False)
    pbar.close()
    df = pd.DataFrame(rows)
    df.to_parquet(gen_path, index=False)
    print(f"[gen] wrote {gen_path} ({len(df)} rows)")
    return df


# ---------- judge -----------------------------------------------------------
def _judge_one(client: OpenAI, model: str, text: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(text=text)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "emotion_judge",
                            "schema": JUDGE_SCHEMA, "strict": True},
        },
        temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content)


def _judge_all(args, gens: pd.DataFrame) -> pd.DataFrame:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set (checked .env)")
    client = OpenAI(api_key=api_key)

    out_path = args.output_dir / "judgments.parquet"
    prev_rows: list[dict] = []
    done: set[tuple] = set()
    if args.resume and out_path.exists():
        prev = pd.read_parquet(out_path)
        prev_rows = prev.to_dict(orient="records")
        done = {(int(r["axis"]), float(r["alpha_unit"]), int(r["prompt_id"]))
                for r in prev_rows}
        print(f"[judge] resume: {len(done)} prior judgments")

    rows: list[dict] = list(prev_rows)
    pbar = tqdm(total=len(gens), initial=len(done), desc="[judge]")
    n_new = 0
    for r in gens.to_dict(orient="records"):
        key = (int(r["axis"]), float(r["alpha_unit"]), int(r["prompt_id"]))
        if key in done:
            continue
        text = r["generation"] or ""
        try:
            j = _judge_one(client, args.judge_model, text)
        except Exception as e:
            print(f"[judge] error key={key}: {e}", flush=True)
            time.sleep(2.0)
            continue
        scores = j["scores"]
        rows.append({
            "axis": int(r["axis"]),
            "target": str(r["target"]),
            "alpha_unit": float(r["alpha_unit"]),
            "prompt_id": int(r["prompt_id"]),
            "prompt": r["prompt"],
            "generation": text,
            **{f"s_{k}": float(scores[k]) for k in PLUTCHIK},
            "other_label": str(j.get("other_label") or "").strip().lower(),
            "other_score": float(j.get("other_score") or 0.0),
            "rationale": j.get("rationale", ""),
        })
        done.add(key)
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
    return judg


# ---------- aggregation -----------------------------------------------------
def _aggregate(args, judg: pd.DataFrame):
    s_cols = [f"s_{k}" for k in PLUTCHIK]
    j = judg.copy()
    j["plutchik_max"] = j[s_cols].max(axis=1)
    j["plutchik_argmax"] = j[s_cols].idxmax(axis=1).str[2:]
    j["plutchik_sum"] = j[s_cols].sum(axis=1)

    def _ent(row):
        v = np.array([row[c] for c in s_cols], dtype=float)
        if v.sum() <= 1e-9:
            return 0.0
        p = v / v.sum()
        p = np.clip(p, 1e-12, 1.0)
        return float(-(p * np.log(p)).sum())
    j["plutchik_entropy"] = j.apply(_ent, axis=1)

    # per (axis, alpha_unit)
    agg_rows = []
    for (axis, au), sub in j.groupby(["axis", "alpha_unit"]):
        labels = [l for l in sub["other_label"].tolist() if l]
        cnt = Counter(labels)
        top_label, top_count = (cnt.most_common(1)[0] if cnt else ("", 0))
        top_plut = sub["plutchik_argmax"].mode()
        d = {
            "axis": int(axis),
            "alpha_unit": float(au),
            "n": int(len(sub)),
            "mean_plutchik_max": float(sub["plutchik_max"].mean()),
            "mean_plutchik_entropy": float(sub["plutchik_entropy"].mean()),
            "mean_other_score": float(sub["other_score"].mean()),
            "frac_other_dominant": float(
                (sub["other_score"] > sub["plutchik_max"]).mean()
            ),
            "top_other_label": top_label,
            "top_other_count": int(top_count),
            "top_plutchik": str(top_plut.iloc[0]) if len(top_plut) else "",
            "n_unique_other_labels": int(len(cnt)),
        }
        for c in s_cols:
            d[f"mean_{c}"] = float(sub[c].mean())
        agg_rows.append(d)
    agg = pd.DataFrame(agg_rows).sort_values(["axis", "alpha_unit"])
    agg.to_csv(args.output_dir / "per_axis_summary.csv", index=False)

    # top-N labels per (axis, alpha) -- long format
    top_rows = []
    for (axis, au), sub in j.groupby(["axis", "alpha_unit"]):
        cnt = Counter(l for l in sub["other_label"].tolist() if l)
        for rank, (lab, c) in enumerate(cnt.most_common(args.top_n), start=1):
            top_rows.append({
                "axis": int(axis),
                "alpha_unit": float(au),
                "rank": rank,
                "label": lab,
                "count": int(c),
            })
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(args.output_dir / "per_axis_top_labels.csv", index=False)

    # markdown summary -- "best word" per axis at +max alpha
    pos_alpha = float(max(args.alphas))
    neg_alpha = float(min(args.alphas))
    lines = ["# Per-axis best-matching word (judge other_label mode)", ""]
    lines.append(
        f"basis: `{args.basis}` | layer={int(j['axis'].iloc[0]) if False else ''}"
    )
    lines.append(f"alphas: pos={pos_alpha}, neg={neg_alpha}, n_prompts={args.n_prompts}")
    lines.append("")
    lines.append("| axis | top@+α (count) | top@−α (count) | top@0 | top Plutchik@+α | mean other@+α |")
    lines.append("|---:|---|---|---|---|---:|")
    axes_sorted = sorted(j["axis"].unique())
    for axis in axes_sorted:
        def _row(au):
            sub = j[(j["axis"] == axis) & (j["alpha_unit"] == au)]
            cnt = Counter(l for l in sub["other_label"].tolist() if l)
            return (cnt.most_common(1)[0] if cnt else ("", 0)), sub
        (pos_lab, pos_n), pos_sub = _row(pos_alpha)
        (neg_lab, neg_n), _ = _row(neg_alpha)
        (zero_lab, zero_n), _ = _row(0.0)
        top_plut = (pos_sub["plutchik_argmax"].mode()
                    if len(pos_sub) else pd.Series([""]))
        mean_other = float(pos_sub["other_score"].mean()) if len(pos_sub) else 0.0
        lines.append(
            f"| b{axis} "
            f"| {pos_lab or '—'} ({pos_n}) "
            f"| {neg_lab or '—'} ({neg_n}) "
            f"| {zero_lab or '—'} ({zero_n}) "
            f"| {top_plut.iloc[0] if len(top_plut) else ''} "
            f"| {mean_other:.2f} |"
        )
    (args.output_dir / "per_axis_top_labels.md").write_text("\n".join(lines))
    print(f"[agg] wrote per_axis_summary.csv, per_axis_top_labels.csv/.md")
    print("[agg] === best label @ +α (head) ===")
    head = agg[agg["alpha_unit"] == pos_alpha].sort_values("axis").head(20)
    print(head[["axis", "top_other_label", "top_other_count",
                "top_plutchik", "mean_other_score"]].to_string(index=False))


# ---------- main ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basis", type=Path,
                   default=Path("data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt"))
    p.add_argument("--caa", type=Path, default=Path("data/emotion_code/caa.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--steering-config", type=Path,
                   default=Path("configs/steering.yaml"))
    p.add_argument("--axes", type=int, nargs="+", default=None,
                   help="Subset of basis indices (default: all 0..k-1)")
    p.add_argument("--alpha-pos", type=float, default=2.0,
                   help="Magnitude of the steering grid; alphas = {-a, 0, +a}.")
    p.add_argument("--alphas", type=float, nargs="+", default=None,
                   help="Override the alpha grid (e.g. -3 -2 0 2 3).")
    p.add_argument("--alpha-mode", type=str, default="caa_match",
                   choices=["caa_match", "unit_v"])
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    p.add_argument("--top-n", type=int, default=5,
                   help="Top-N labels stored per (axis, alpha_unit).")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/results/per_axis_judge"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-generate", action="store_true",
                   help="Reuse generations.parquet in output-dir; only judge+agg.")
    p.add_argument("--skip-judge", action="store_true",
                   help="Reuse judgments.parquet in output-dir; only aggregate.")
    args = p.parse_args()

    if args.alphas is None:
        a = float(args.alpha_pos)
        args.alphas = [-a, 0.0, a]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load basis -------------------------------------------------------------
    bp = torch.load(args.basis, weights_only=False, map_location="cpu")
    layer = int(bp["layer"])
    decomposer = bp["decomposer"]
    B = bp[decomposer]["W"].numpy().astype(np.float32)
    print(f"[main] basis: {decomposer} k={B.shape[0]} layer={layer}")

    # CAA-norm scale ---------------------------------------------------------
    caa = torch.load(args.caa, weights_only=False, map_location="cpu")
    caa_layers = list(caa["layers"])
    li_caa = caa_layers.index(layer)
    caa_vecs = caa["vectors"].numpy().astype(np.float32)
    caa_norms = np.linalg.norm(caa_vecs[:, li_caa, :], axis=-1)
    scale = float(1.0 / np.median(caa_norms))
    median_caa_norm = float(np.median(caa_norms))
    print(f"[main] alpha scale={scale:.5f}  median||CAA||={median_caa_norm:.3f}  "
          f"alpha_mode={args.alpha_mode}")

    prompts = load_neutral_prompts(n=args.n_prompts, seed=args.seed)

    # Generate ---------------------------------------------------------------
    gen_path = args.output_dir / "generations.parquet"
    if args.skip_generate:
        if not gen_path.exists():
            raise SystemExit(f"--skip-generate but {gen_path} missing")
        gens = pd.read_parquet(gen_path)
        print(f"[main] reuse {gen_path} ({len(gens)} rows)")
    else:
        gens = _generate_all(args, B, layer, scale, median_caa_norm, prompts)

    # Judge ------------------------------------------------------------------
    judg_path = args.output_dir / "judgments.parquet"
    if args.skip_judge:
        if not judg_path.exists():
            raise SystemExit(f"--skip-judge but {judg_path} missing")
        judg = pd.read_parquet(judg_path)
        print(f"[main] reuse {judg_path} ({len(judg)} rows)")
    else:
        judg = _judge_all(args, gens)

    # Aggregate --------------------------------------------------------------
    _aggregate(args, judg)

    # Final summary.json -----------------------------------------------------
    smry = {
        "basis": str(args.basis),
        "decomposer": decomposer,
        "k": int(B.shape[0]),
        "layer": layer,
        "n_prompts": args.n_prompts,
        "alphas": args.alphas,
        "alpha_mode": args.alpha_mode,
        "judge_model": args.judge_model,
        "n_generations": int(len(gens)),
        "n_judgments": int(len(judg)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(smry, indent=2))


if __name__ == "__main__":
    main()
