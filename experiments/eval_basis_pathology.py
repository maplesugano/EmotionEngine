"""Identify pathological basis components (repetition / dirty language).

For each component of a basis-sweep artifact, sweep over signed alphas and
neutral prompts, generate continuations, and score each generation with the
cheap deterministic metrics in ``src/steering/_pathology_lexicon.py``. We
then aggregate per (component, alpha) and per component (worst-α), rank
components by a composite pathology score, and emit a recommended exclude
list for the UI palette.

Usage
-----
    uv run python -m experiments.eval_basis_pathology \
        --basis data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt \
        --alphas -3 -2 -1 0 1 2 3 \
        --n-prompts 8 --max-new-tokens 96

Outputs (under ``experiments/results/basis_pathology/<artifact_stem>/``):
    generations.parquet     one row per (component, alpha, prompt)
    per_cell.csv            mean metrics per (component, alpha)
    per_component.csv       worst-α aggregate, ranked by pathology score
    summary.json            metadata + recommended exclude list
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
from src.activations._runtime import load_model, load_profile
from src.steering._pathology_lexicon import (
    composite_pathology_score,
    pathology_metrics,
    toxicity_hits,
)
from src.steering.generate import steered_generate


def _select_W(payload: dict, which: str | None) -> tuple[np.ndarray, str]:
    if which is None:
        which = payload.get("decomposer")
        if which is None:
            for cand in ("ica", "nmf", "pca", "dict"):
                if cand in payload:
                    which = cand
                    break
    return payload[which]["W"].numpy(), which


def _category_loading_l2(payload: dict, which: str) -> np.ndarray | None:
    block = payload.get(which, {})
    cl = block.get("category_loadings")
    if cl is None:
        return None
    cl = cl.numpy() if hasattr(cl, "numpy") else np.asarray(cl)
    # cl: [C, k]  -> per-component L2 across categories
    return np.linalg.norm(cl, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--basis", type=Path, required=True)
    p.add_argument("--which", default=None,
                   help="Decomposer key inside the payload (default: payload['decomposer']).")
    p.add_argument("--components", type=int, nargs="*", default=None,
                   help="Components to evaluate (default: all).")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                   help="Alphas in std-norm units (1.0 ≈ median ||b||).")
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--prompts-seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--steering-config", type=Path,
                   default=Path("configs/steering.yaml"))
    p.add_argument("--output-dir", type=Path, default=None)
    # Exclude-list thresholds (applied to per-component worst-α aggregates)
    p.add_argument("--rep4-thresh", type=float, default=0.45,
                   help="Components with rep_4 ≥ this are flagged.")
    p.add_argument("--tox-thresh", type=float, default=0.01,
                   help="Components with toxicity_rate ≥ this are flagged.")
    p.add_argument("--maxrun-thresh", type=float, default=0.25,
                   help="Components with max_run_norm ≥ this are flagged.")
    args = p.parse_args()

    # ------------------------------------------------------------------ load basis
    payload = torch.load(args.basis, weights_only=False, map_location="cpu")
    W, which = _select_W(payload, args.which)
    layer = int(payload["layer"])
    k, D = W.shape
    norms = np.linalg.norm(W, axis=1)
    scale = 1.0 / float(np.median(norms))
    cat_l2 = _category_loading_l2(payload, which)  # [k] or None

    components = list(range(k)) if args.components is None else list(args.components)

    out_root = args.output_dir or (
        Path("experiments/results/basis_pathology") / args.basis.stem
    )
    out_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ model + prompts
    profile, _ = load_profile(args.config)
    sc = yaml.safe_load(args.steering_config.read_text())
    apply_to = sc["caa"].get("apply_to", "generation")
    prompts = load_neutral_prompts(n=args.n_prompts, seed=args.prompts_seed)
    model, _, _ = load_model(profile)

    print(f"[pathology] basis={args.basis.name} which={which} layer={layer} "
          f"k={k} D={D} components={len(components)} alphas={args.alphas} "
          f"prompts={len(prompts)} apply_to={apply_to}")

    # ------------------------------------------------------------------ sweep
    rows: list[dict] = []
    total = len(components) * len(args.alphas) * len(prompts)
    pbar = tqdm(total=total, desc="gen", smoothing=0.05)
    for c in components:
        b_t = torch.from_numpy(W[c]).to(torch.float32)
        b_norm = float(norms[c])
        for alpha_unit in args.alphas:
            if alpha_unit == 0.0:
                alpha = 0.0
                vec = torch.zeros(D)
            else:
                alpha = float(alpha_unit) * scale * b_norm
                vec = b_t
            for pi, prompt in enumerate(prompts):
                out = steered_generate(
                    model, prompt, vector=vec, alpha=alpha,
                    layers=[layer], apply_to=apply_to,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    repetition_penalty=1.0, no_repeat_ngram_size=0,
                )
                tail = out[len(prompt):] if out.startswith(prompt) else out
                tail = tail.strip()
                m = pathology_metrics(tail)
                rows.append({
                    "component": c,
                    "alpha_unit": float(alpha_unit),
                    "prompt_idx": pi,
                    "prompt": prompt,
                    "text": tail,
                    "tox_words": ",".join(toxicity_hits(tail)),
                    **m,
                })
                pbar.update(1)
    pbar.close()

    gen_df = pd.DataFrame(rows)
    gen_df.to_parquet(out_root / "generations.parquet", index=False)

    # ------------------------------------------------------------------ per-cell aggregate
    metric_cols = [
        "n_tokens", "rep_2", "rep_3", "rep_4", "max_run", "max_run_norm",
        "compress_ratio", "unique_token_ratio",
        "toxicity_rate", "toxicity_hits", "non_ascii_ratio",
    ]
    per_cell = (
        gen_df.groupby(["component", "alpha_unit"])[metric_cols].mean().reset_index()
    )
    per_cell.to_csv(out_root / "per_cell.csv", index=False)

    # ------------------------------------------------------------------ per-component (worst-α)
    # For each component we report the worst alpha for each pathology axis,
    # plus a single composite score (worst over alphas).
    nonzero = per_cell[per_cell["alpha_unit"] != 0.0].copy()
    nonzero["pathology_score"] = [
        composite_pathology_score(r.rep_4, r.max_run_norm, r.toxicity_rate)
        for r in nonzero.itertuples()
    ]
    idx_worst = nonzero.groupby("component")["pathology_score"].idxmax()
    worst = nonzero.loc[idx_worst].rename(columns={"alpha_unit": "worst_alpha"})

    per_comp = worst[[
        "component", "worst_alpha", "pathology_score",
        "rep_4", "max_run_norm", "toxicity_rate",
        "compress_ratio", "unique_token_ratio", "non_ascii_ratio",
    ]].sort_values("pathology_score", ascending=False).reset_index(drop=True)

    if cat_l2 is not None:
        per_comp["category_loading_l2"] = cat_l2[per_comp["component"].to_numpy()]

    # Attach a short example: the worst-scoring single generation per component.
    examples: list[str] = []
    for c, a in zip(per_comp["component"], per_comp["worst_alpha"]):
        cell = gen_df[(gen_df["component"] == c) & (gen_df["alpha_unit"] == a)]
        if cell.empty:
            examples.append("")
            continue
        cell = cell.assign(
            _score=[
                composite_pathology_score(r.rep_4, r.max_run_norm, r.toxicity_rate)
                for r in cell.itertuples()
            ]
        )
        ex = cell.sort_values("_score", ascending=False).iloc[0]["text"]
        examples.append(ex[:240].replace("\n", " "))
    per_comp["example"] = examples

    per_comp.to_csv(out_root / "per_component.csv", index=False)

    # ------------------------------------------------------------------ exclude list
    flagged = per_comp[
        (per_comp["rep_4"] >= args.rep4_thresh)
        | (per_comp["toxicity_rate"] >= args.tox_thresh)
        | (per_comp["max_run_norm"] >= args.maxrun_thresh)
    ]
    exclude = sorted(int(c) for c in flagged["component"].tolist())

    summary = {
        "basis": str(args.basis),
        "which": which,
        "layer": layer,
        "k": int(k),
        "alphas": list(args.alphas),
        "n_prompts": int(args.n_prompts),
        "max_new_tokens": int(args.max_new_tokens),
        "thresholds": {
            "rep_4": args.rep4_thresh,
            "toxicity_rate": args.tox_thresh,
            "max_run_norm": args.maxrun_thresh,
        },
        "n_flagged": len(exclude),
        "exclude": exclude,
        "top10": per_comp.head(10).to_dict(orient="records"),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    # Sidecar next to the artifact, for the UI to consume.
    sidecar = args.basis.with_suffix(".exclude.json")
    sidecar.write_text(json.dumps({
        "basis": args.basis.name,
        "layer": layer,
        "exclude": exclude,
        "thresholds": summary["thresholds"],
    }, indent=2))

    print(f"[pathology] wrote {out_root}/generations.parquet")
    print(f"[pathology] wrote {out_root}/per_component.csv  (top: {exclude[:10]} …)")
    print(f"[pathology] wrote {out_root}/summary.json")
    print(f"[pathology] sidecar: {sidecar}  ({len(exclude)}/{k} flagged)")


if __name__ == "__main__":
    main()
