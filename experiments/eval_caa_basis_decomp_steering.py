"""Behavioral validation for CAA → basis decomposition.

Tests whether a CAA steering vector reconstructed as a weighted sum of basis
vectors (Phase 1 of the experiment) actually produces the same emotional
shift in generation as the original CAA vector.

For each (winner_artifact, category) we steer with several vector variants:

    * ``caa``      — original CAA vector (gold reference)
    * ``ols``      — Σ w_ols · b_k                      (high-R² ceiling)
    * ``nnls``     — Σ w_nnls · b_k        (non-negative)
    * ``lasso``    — Σ w_lasso · b_k       (sparse, single chosen alpha)
    * ``vad``      — VAD-only reconstruction Σ w_vad · b_VAD   (3-axis baseline)
    * ``random``   — basis combo with random weights matched to ols ||w||₂
                     (sanity floor — same magnitude, no semantic content)

The steering, classification, and shift-accuracy logic mirrors
``experiments/_gen_cache.py`` and ``experiments/eval_shift_accuracy.py``
so results are directly comparable to the production CAA numbers.
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

DEFAULT_LASSO_ALPHA = 1e-3  # least-aggressive of the three Phase 1 alphas


def _build_variants(
    W: np.ndarray,                     # [k, D] basis
    W_vad: np.ndarray,                 # [3, D] VAD axes
    caa_vec: np.ndarray,               # [D]
    weights_payload: dict,             # per-category fit weights from Phase 1
    category: str,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Return mapping variant_name -> steering vector ([D])."""
    fits = weights_payload["weights"][category]
    variants: dict[str, np.ndarray] = {"caa": caa_vec.astype(np.float32)}

    w_ols = fits["ols"].numpy()
    variants["ols"] = (W.T @ w_ols).astype(np.float32)

    w_nnls = fits["nnls"].numpy()
    variants["nnls"] = (W.T @ w_nnls).astype(np.float32)

    lasso_key = next(
        (k for k in fits if k.startswith("lasso_") and float(k.split("_", 1)[1]) == DEFAULT_LASSO_ALPHA),
        None,
    )
    if lasso_key is None:
        lasso_key = sorted(k for k in fits if k.startswith("lasso_"))[0]
    variants["lasso"] = (W.T @ fits[lasso_key].numpy()).astype(np.float32)

    # VAD reconstruction: fit caa ≈ W_vad.T @ w_vad, then v_recon = W_vad.T @ w_vad
    w_vad, *_ = np.linalg.lstsq(W_vad.T, caa_vec, rcond=None)
    variants["vad"] = (W_vad.T @ w_vad).astype(np.float32)

    # Random sanity floor: Gaussian weights rescaled to match ||w_ols||₂
    rand_w = rng.standard_normal(W.shape[0]).astype(np.float32)
    rand_w *= float(np.linalg.norm(w_ols)) / (float(np.linalg.norm(rand_w)) + 1e-12)
    variants["random"] = (W.T @ rand_w).astype(np.float32)

    return variants


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path,
                   default=Path("experiments/results/caa_basis_decomposition/"
                                "weights/basis_sweep__ica_k016_seed0.pt"),
                   help="Per-artifact weights file from Phase 1")
    p.add_argument("--basis", type=Path,
                   default=Path("data/emotion_code/basis_sweep/ica_k016_seed0.pt"))
    p.add_argument("--caa", type=Path, default=Path("data/emotion_code/caa.pt"))
    p.add_argument("--vad", type=Path, default=Path("data/emotion_code/vad_mapping.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--steering-config", type=Path,
                   default=Path("configs/steering.yaml"))
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[-2.0, 0.0, 2.0])
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variants", type=str, nargs="+", default=None,
                   help="Subset of {caa, ols, nnls, lasso, vad, random}. "
                        "Default: run all.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/results/caa_basis_decomp_steering"))
    p.add_argument("--resume", action="store_true",
                   help="Skip (variant, category, alpha, prompt) combos already "
                        "present in generations.parquet")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Load weights, basis, CAA, VAD ------------------------------------------
    weights_payload = torch.load(args.weights, weights_only=False, map_location="cpu")
    layer = int(weights_payload["layer"])
    decomposer = weights_payload["decomposer"]
    categories = list(weights_payload["categories"])

    basis_payload = torch.load(args.basis, weights_only=False, map_location="cpu")
    if int(basis_payload["layer"]) != layer:
        raise ValueError(
            f"basis layer {basis_payload['layer']} != weights layer {layer}"
        )
    W = basis_payload[decomposer]["W"].numpy().astype(np.float32)   # [k, D]
    print(f"[steer] basis: {decomposer} k={W.shape[0]} layer={layer}")

    caa = torch.load(args.caa, weights_only=False, map_location="cpu")
    if list(caa["categories"]) != categories:
        raise ValueError("category order mismatch between weights and CAA")
    caa_layers = list(caa["layers"])
    if layer not in caa_layers:
        raise ValueError(f"layer {layer} not in CAA layers {caa_layers}")
    caa_vecs = caa["vectors"].numpy().astype(np.float32)
    li_caa = caa_layers.index(layer)

    vad = torch.load(args.vad, weights_only=False, map_location="cpu")
    W_vad = vad["W"].numpy().astype(np.float32)

    # Load model + steering config -------------------------------------------
    profile, _ = load_profile(args.config)
    sc = yaml.safe_load(args.steering_config.read_text())
    apply_to = sc["caa"].get("apply_to", "generation")
    model, _device, _ = load_model(profile)

    prompts = load_neutral_prompts(n=args.n_prompts, seed=args.seed)
    print(f"[steer] {len(prompts)} prompts × {len(categories)} cats × "
          f"{len(args.alphas)} alphas")

    # Build variants per category --------------------------------------------
    per_cat_variants: dict[str, dict[str, np.ndarray]] = {}
    for ci, cat in enumerate(categories):
        per_cat_variants[cat] = _build_variants(
            W, W_vad, caa_vecs[ci, li_caa], weights_payload, cat, rng,
        )

    # Alpha scale: same convention as _gen_cache.py (1 / median ||caa||)
    caa_norms_layer = np.linalg.norm(caa_vecs[:, li_caa, :], axis=-1)
    scale = float(1.0 / np.median(caa_norms_layer))
    print(f"[steer] alpha scale = {scale:.5f}")

    # Generate ----------------------------------------------------------------
    gen_path = args.output_dir / "generations.parquet"
    done: set[tuple] = set()
    prev_rows: list[dict] = []
    if args.resume and gen_path.exists():
        prev = pd.read_parquet(gen_path)
        prev_rows = prev.to_dict(orient="records")
        done = {
            (r["category"], r["variant"], float(r["alpha_unit"]), int(r["prompt_id"]))
            for r in prev_rows
        }
        print(f"[steer] resume: {len(done)} prior generations loaded from {gen_path}")

    rows: list[dict] = list(prev_rows)
    variant_names = list(next(iter(per_cat_variants.values())).keys())
    if args.variants:
        keep = set(args.variants)
        unknown = keep - set(variant_names)
        if unknown:
            raise ValueError(f"unknown variants: {sorted(unknown)}; "
                             f"available: {variant_names}")
        variant_names = [v for v in variant_names if v in keep]
    print(f"[steer] variants: {variant_names}")

    total = len(categories) * len(variant_names) * len(args.alphas) * len(prompts)
    pbar = tqdm(total=total, desc="[steer] gen", initial=len(done & {
        (c, v, a, pi)
        for c in categories for v in variant_names
        for a in args.alphas for pi in range(len(prompts))
    }))
    for cat in categories:
        for variant in variant_names:
            v = torch.from_numpy(per_cat_variants[cat][variant]).to(torch.float32)
            v_norm = float(np.linalg.norm(per_cat_variants[cat][variant]))
            cell_added = 0
            for alpha_unit in args.alphas:
                alpha = alpha_unit * scale * v_norm
                for pi, prompt in enumerate(prompts):
                    key = (cat, variant, float(alpha_unit), pi)
                    if key in done:
                        continue
                    if alpha_unit == 0.0:
                        out = steered_generate(
                            model, prompt, v, alpha=0.0, layers=[layer],
                            apply_to=apply_to, max_new_tokens=args.max_new_tokens,
                        )
                    else:
                        out = steered_generate(
                            model, prompt, v, alpha=alpha, layers=[layer],
                            apply_to=apply_to, max_new_tokens=args.max_new_tokens,
                        )
                    tail = out[len(prompt):].strip() if out.startswith(prompt) else out.strip()
                    rows.append({
                        "category": cat, "variant": variant,
                        "alpha_unit": alpha_unit, "alpha": alpha,
                        "vector_norm": v_norm,
                        "layer": layer, "prompt_id": pi, "prompt": prompt,
                        "generation": tail,
                    })
                    done.add(key)
                    cell_added += 1
                    pbar.update(1)
            # Incremental checkpoint after each (cat, variant) cell
            if cell_added:
                pd.DataFrame(rows).to_parquet(gen_path, index=False)
    pbar.close()
    df = pd.DataFrame(rows)
    df.to_parquet(gen_path, index=False)
    print(f"[steer] wrote {gen_path} ({len(df)} rows)")

    # Classify ---------------------------------------------------------------
    cls = _classify(df["generation"].astype(str).tolist())
    df = pd.concat([df.reset_index(drop=True), cls], axis=1)
    df["pred_plutchik"] = df["pred_label"].map(HARTMANN_TO_PLUTCHIK).fillna("other")
    df.to_parquet(args.output_dir / "generations_classified.parquet", index=False)

    # Shift-accuracy aggregation per (variant, category) ---------------------
    target_alpha = max(args.alphas)
    rows_metric: list[dict] = []
    for (variant, cat), sub in df.groupby(["variant", "category"]):
        if cat in UNMEASURABLE:
            rows_metric.append({"variant": variant, "category": cat,
                                "shift_acc": np.nan, "baseline_acc": np.nan,
                                "delta": np.nan, "n": 0,
                                "note": "no classifier label"})
            continue
        pos = sub[sub["alpha_unit"] == target_alpha]
        base = sub[sub["alpha_unit"] == 0.0]
        if len(pos) == 0 or len(base) == 0:
            continue
        shift_acc = float((pos["pred_plutchik"] == cat).mean())
        baseline_acc = float((base["pred_plutchik"] == cat).mean())
        rows_metric.append({
            "variant": variant, "category": cat, "n": int(len(pos)),
            "shift_acc": shift_acc, "baseline_acc": baseline_acc,
            "delta": shift_acc - baseline_acc, "note": "",
        })
    metric_df = pd.DataFrame(rows_metric)
    metric_df.to_csv(args.output_dir / "shift_by_variant.csv", index=False)

    # Summary: mean shift-acc per variant, plus retention vs CAA --------------
    measurable = metric_df.dropna(subset=["shift_acc"])
    per_variant = (
        measurable.groupby("variant")
        .agg(mean_shift_acc=("shift_acc", "mean"),
             mean_baseline_acc=("baseline_acc", "mean"),
             mean_delta=("delta", "mean"))
        .reset_index()
    )
    caa_acc = float(per_variant.loc[per_variant.variant == "caa", "mean_shift_acc"].iloc[0]) \
        if (per_variant.variant == "caa").any() else None
    if caa_acc and caa_acc > 0:
        per_variant["retention_vs_caa"] = per_variant["mean_shift_acc"] / caa_acc
    per_variant.to_csv(args.output_dir / "summary_by_variant.csv", index=False)

    summary = {
        "layer": layer,
        "decomposer": decomposer,
        "k": int(W.shape[0]),
        "alpha_target": target_alpha,
        "n_prompts": args.n_prompts,
        "alphas": args.alphas,
        "lasso_alpha_used": DEFAULT_LASSO_ALPHA,
        "per_variant": per_variant.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[steer] === per-variant mean shift accuracy ===")
    print(per_variant.to_string(index=False))


if __name__ == "__main__":
    main()
