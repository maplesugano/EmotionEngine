"""Phase C-3 (C): Lexical-gap steering — does basis support emotions outside Plutchik?

Builds steering targets directly from individual basis components (and
small combinations), runs them through `steered_generate`, classifies with
the Hartmann-Plutchik mapping, and produces a side-by-side qualitative
comparison.

Components are split into three groups (set via --gap, --pan, --extra):
  * 'gap'   — lexical-gap candidates (default: b1, b11, b13 from Wstruct)
  * 'pan'   — broadly-loaded reference components (default: b0, b5, b8)
  * 'combo' — pairwise sums of two gap components

Each component-vector v_k is a row of basis B (∈ ℝ^{k × D}). The alpha
scale uses the same convention as `_gen_cache.py`:
    alpha = alpha_unit * scale * ||v_k||
where scale = 1 / median(||CAA_layer||).

Outputs:
  generations.parquet, classifier.parquet, summary_by_target.csv,
  qualitative.md  (human-readable samples)

Per-target metrics:
  - mean classifier max-prob (low = "outside Plutchik")
  - entropy of mean Plutchik distribution (high = uncertain → potentially gap)
  - top predicted Plutchik categories (if any)
  - generation samples (one per prompt at α=+2)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from experiments._gen_cache import load_neutral_prompts
from experiments.eval_shift_accuracy import (
    HARTMANN_TO_PLUTCHIK,
    UNMEASURABLE,
    _classify,
)
from src.activations._runtime import load_model, load_profile
from src.steering.generate import steered_generate


def _build_targets(B: np.ndarray, gap: list[int], pan: list[int],
                   include_combos: bool) -> dict[str, np.ndarray]:
    """Return mapping target_name -> steering vector ([D])."""
    targets: dict[str, np.ndarray] = {}
    for k in gap:
        targets[f"gap_b{k}"] = B[k].astype(np.float32)
    for k in pan:
        targets[f"pan_b{k}"] = B[k].astype(np.float32)
    if include_combos:
        for i in range(len(gap)):
            for j in range(i + 1, len(gap)):
                a, b = gap[i], gap[j]
                v = B[a] + B[b]
                targets[f"combo_b{a}+b{b}"] = v.astype(np.float32)
    return targets


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basis", type=Path,
                   default=Path("data/emotion_code/basis_sweep/ica_k016_seed0.pt"))
    p.add_argument("--caa", type=Path, default=Path("data/emotion_code/caa.pt"),
                   help="Used only for the alpha-scale convention.")
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--steering-config", type=Path,
                   default=Path("configs/steering.yaml"))
    p.add_argument("--gap", type=int, nargs="+", default=[1, 11, 13],
                   help="Indices of lexical-gap candidate components")
    p.add_argument("--pan", type=int, nargs="+", default=[0, 5, 8],
                   help="Indices of pan / broadly-loaded reference components")
    p.add_argument("--no-combos", action="store_true",
                   help="Skip pairwise combos of gap components")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[-2.0, 0.0, 2.0])
    p.add_argument("--alpha-mode", type=str, default="caa_match",
                   choices=["caa_match", "unit_v"],
                   help="caa_match: normalize v to median(||CAA||) so the "
                        "effective injected magnitude matches the CAA scale "
                        "(recommended for tiny basis components). "
                        "unit_v: use ||v|| directly (matches Phase 2 script).")
    p.add_argument("--n-prompts", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/results/lexical_gap_steering"))
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load basis -------------------------------------------------------------
    bp = torch.load(args.basis, weights_only=False, map_location="cpu")
    layer = int(bp["layer"])
    decomposer = bp["decomposer"]
    B = bp[decomposer]["W"].numpy().astype(np.float32)   # [k, D]
    print(f"[gap] basis: {decomposer} k={B.shape[0]} layer={layer}")

    # CAA: only used for the alpha-scale convention -------------------------
    caa = torch.load(args.caa, weights_only=False, map_location="cpu")
    caa_layers = list(caa["layers"])
    li_caa = caa_layers.index(layer)
    caa_vecs = caa["vectors"].numpy().astype(np.float32)
    caa_norms_layer = np.linalg.norm(caa_vecs[:, li_caa, :], axis=-1)
    scale = float(1.0 / np.median(caa_norms_layer))
    median_caa_norm = float(np.median(caa_norms_layer))
    print(f"[gap] alpha scale = {scale:.5f}  median||CAA||={median_caa_norm:.3f}")
    print(f"[gap] alpha-mode = {args.alpha_mode}")

    targets = _build_targets(B, args.gap, args.pan, include_combos=not args.no_combos)
    print(f"[gap] targets: {list(targets.keys())}")

    # Load model -------------------------------------------------------------
    profile, _ = load_profile(args.config)
    sc = yaml.safe_load(args.steering_config.read_text())
    apply_to = sc["caa"].get("apply_to", "generation")
    model, _device, _ = load_model(profile)

    prompts = load_neutral_prompts(n=args.n_prompts, seed=args.seed)
    print(f"[gap] {len(prompts)} prompts × {len(targets)} targets × "
          f"{len(args.alphas)} alphas")

    # Generate ---------------------------------------------------------------
    gen_path = args.output_dir / "generations.parquet"
    done: set[tuple] = set()
    prev_rows: list[dict] = []
    if args.resume and gen_path.exists():
        prev = pd.read_parquet(gen_path)
        prev_rows = prev.to_dict(orient="records")
        done = {
            (r["target"], float(r["alpha_unit"]), int(r["prompt_id"]))
            for r in prev_rows
        }
        print(f"[gap] resume: {len(done)} prior gens loaded")

    rows: list[dict] = list(prev_rows)
    total = len(targets) * len(args.alphas) * len(prompts)
    pbar = tqdm(total=total, desc="[gap] gen", initial=len(done))
    for tname, vec in targets.items():
        v_raw_norm = float(np.linalg.norm(vec))
        if args.alpha_mode == "caa_match":
            # rescale v to median CAA norm so effective magnitude matches CAA
            v_use = vec * (median_caa_norm / (v_raw_norm + 1e-12))
            v_norm = float(np.linalg.norm(v_use))
        else:
            v_use = vec
            v_norm = v_raw_norm
        v = torch.from_numpy(v_use.astype(np.float32))
        cell_added = 0
        for alpha_unit in args.alphas:
            alpha = alpha_unit * scale * v_norm
            for pi, prompt in enumerate(prompts):
                key = (tname, float(alpha_unit), pi)
                if key in done:
                    continue
                out = steered_generate(
                    model, prompt, v,
                    alpha=alpha if alpha_unit != 0.0 else 0.0,
                    layers=[layer], apply_to=apply_to,
                    max_new_tokens=args.max_new_tokens,
                )
                tail = out[len(prompt):].strip() if out.startswith(prompt) else out.strip()
                rows.append({
                    "target": tname, "alpha_unit": alpha_unit, "alpha": alpha,
                    "vector_norm": v_norm, "layer": layer,
                    "prompt_id": pi, "prompt": prompt, "generation": tail,
                })
                done.add(key)
                cell_added += 1
                pbar.update(1)
        if cell_added:
            pd.DataFrame(rows).to_parquet(gen_path, index=False)
    pbar.close()
    df = pd.DataFrame(rows)
    df.to_parquet(gen_path, index=False)
    print(f"[gap] wrote {gen_path} ({len(df)} rows)")

    # Classify ---------------------------------------------------------------
    cls_path = args.output_dir / "classifier.parquet"
    print("[gap] classifying...")
    texts = df["generation"].fillna("").tolist()
    cls_raw = _classify(texts, batch_size=32)  # DataFrame: hartmann label cols + pred_label
    plutchik_labels = sorted(set(HARTMANN_TO_PLUTCHIK.values()) - {""})
    measurable_labels = [p for p in plutchik_labels if p not in UNMEASURABLE]
    hartmann_cols = [c for c in cls_raw.columns if c != "pred_label"]
    rows_c = []
    for r, (_, row) in zip(df.to_dict("records"), cls_raw.iterrows()):
        plutchik_probs = {p: 0.0 for p in plutchik_labels}
        for hlabel in hartmann_cols:
            mapped = HARTMANN_TO_PLUTCHIK.get(hlabel, "")
            if mapped:
                plutchik_probs[mapped] += float(row[hlabel])
        measurable = {k: plutchik_probs[k] for k in measurable_labels}
        s = sum(measurable.values()) or 1.0
        measurable = {k: v / s for k, v in measurable.items()}
        max_prob = max(measurable.values())
        ent = _entropy(np.array(list(measurable.values())))
        rows_c.append({
            **{k: r[k] for k in ("target", "alpha_unit", "prompt_id")},
            "pred_hartmann": row["pred_label"],
            "pred_score": float(row[row["pred_label"]]),
            "max_plutchik_prob": float(max_prob),
            "plutchik_entropy": float(ent),
            **{f"p_{k}": v for k, v in measurable.items()},
        })
    cls_df = pd.DataFrame(rows_c)
    cls_df.to_parquet(cls_path, index=False)

    # Summary per target -----------------------------------------------------
    summary_rows = []
    for tname, sub in cls_df.groupby("target"):
        for au, sub2 in sub.groupby("alpha_unit"):
            d = {
                "target": tname,
                "alpha_unit": float(au),
                "n": int(len(sub2)),
                "mean_max_plutchik_prob": float(sub2["max_plutchik_prob"].mean()),
                "mean_plutchik_entropy": float(sub2["plutchik_entropy"].mean()),
            }
            for k in plutchik_labels:
                if k in UNMEASURABLE:
                    continue
                col = f"p_{k}"
                if col in sub2.columns:
                    d[f"mean_{col}"] = float(sub2[col].mean())
            summary_rows.append(d)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary_by_target.csv", index=False)
    print("[gap] === per-target classifier summary (alpha=+2) ===")
    print(summary[summary["alpha_unit"] == max(args.alphas)]
          [["target", "mean_max_plutchik_prob", "mean_plutchik_entropy"]]
          .to_string(index=False))

    # Qualitative MD ---------------------------------------------------------
    md_path = args.output_dir / "qualitative.md"
    lines = ["# Lexical-gap steering — qualitative samples", ""]
    pos_alpha = max(args.alphas)
    for tname in targets:
        lines.append(f"## {tname}  (α=+{pos_alpha})")
        sub = df[(df["target"] == tname) & (df["alpha_unit"] == pos_alpha)]
        sub_c = cls_df[(cls_df["target"] == tname) & (cls_df["alpha_unit"] == pos_alpha)]
        merged = sub.merge(sub_c[["prompt_id", "max_plutchik_prob", "plutchik_entropy", "pred_hartmann"]],
                           on="prompt_id")
        for _, r in merged.head(8).iterrows():
            lines.append(f"- **prompt**: {r['prompt']!r}")
            lines.append(f"  - **gen**: {r['generation']}")
            lines.append(f"  - max_p={r['max_plutchik_prob']:.2f} "
                         f"H={r['plutchik_entropy']:.2f} "
                         f"hart={r['pred_hartmann']}")
        lines.append("")
    md_path.write_text("\n".join(lines))
    print(f"[gap] wrote {md_path}")

    # Final summary.json
    smry = {
        "layer": layer, "decomposer": decomposer, "k": int(B.shape[0]),
        "gap_components": args.gap, "pan_components": args.pan,
        "n_prompts": args.n_prompts, "alphas": args.alphas,
        "per_target_at_max_alpha": summary[summary["alpha_unit"] == pos_alpha]
            .to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(smry, indent=2))


if __name__ == "__main__":
    main()
