"""Additivity test for OLS-reconstructed CAA category vectors.

Background
----------
Phase 1/2 of the CAA → basis decomposition (eval_caa_basis_decomposition.py +
eval_caa_basis_decomp_steering.py) showed that each Plutchik CAA vector is
reconstructible as a dense linear combination of basis components,

    v_c ≈ Σ_k w_k^c · b_k          (R² ≈ 0.96, retention ≈ 0.93 at k=64/L=22)

If the basis truly behaves as a set of *atoms*, then arbitrary linear
combinations of CAA reconstructions should also behave linearly: steering with
``α v_A + β v_B`` should produce the same readout / behavioural shift as the
sum of the marginal effects of ``α v_A`` and ``β v_B`` independently.

This script tests that hypothesis directly by:

1. Building reconstructed steering vectors ``v_A``, ``v_B`` for each requested
   category pair using the OLS weights stored in the Phase 1 ``weights/`` file.
2. Generating text under joint steering for ``(α, β) ∈ alphas × alphas``
   (typically ``{-2, 0, +2}``) over N neutral prompts.
3. Re-encoding the generations at the basis layer and projecting onto the
   basis to obtain readouts ``r(α, β) ∈ R^k``.
4. Classifying generations with the same Hartmann classifier used by
   ``eval_shift_accuracy.py`` to measure shift accuracy to category A and B.
5. Computing additivity residuals in *both* spaces:
   - readout space: ``resid = r(α,β) − [r(α,0) + r(0,β) − r(0,0)]``
   - shift-acc space: same predictor applied to the per-cell A/B shift rates.

Outputs (in ``--output-dir``):
- ``generations.parquet`` — raw generations per (pair, α, β, prompt)
- ``generations_classified.parquet`` — generations + classifier labels
- ``additivity_readout.csv`` — per-cell readout residual stats
- ``additivity_shift.csv`` — per-cell shift-acc and additivity residuals
- ``summary.json`` — overall medians and config

Usage
-----
    uv run python -m experiments.eval_caa_basis_additivity \
        --weights experiments/results/caa_basis_decomposition_L22/weights/basis_sweep_L22__ica_k064_seed0.pt \
        --basis data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt \
        --cat-pairs joy,anger joy,sadness anger,fear trust,fear \
        --alphas -2 0 2 --n-prompts 8 --max-new-tokens 32 \
        --output-dir experiments/results/caa_basis_additivity_L22_k64
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
from src.activations._runtime import collect_batch, load_model, load_profile
from src.steering.generate import steered_generate


def _parse_pair(s: str) -> tuple[str, str]:
    a, b = s.split(",")
    return a.strip(), b.strip()


def _project_all(h: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Cosine projection of each row of h onto each row of W -> [N, k]."""
    h_n = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    w_n = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    return h_n @ w_n.T


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True,
                   help="Per-category weights file from Phase 1 "
                        "(basis_sweep_*__ica_k*_seed*.pt)")
    p.add_argument("--basis", type=Path, required=True)
    p.add_argument("--caa", type=Path, default=Path("data/emotion_code/caa.pt"))
    p.add_argument("--cat-pairs", type=_parse_pair, nargs="+", required=True,
                   help="Category pairs as 'A,B' (e.g. joy,anger).")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[-2.0, 0.0, 2.0])
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--prompts-seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--steering-config", type=Path,
                   default=Path("configs/steering.yaml"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # -------- Load weights, basis, CAA --------
    weights_payload = torch.load(args.weights, weights_only=False, map_location="cpu")
    layer = int(weights_payload["layer"])
    decomposer = weights_payload["decomposer"]
    categories = list(weights_payload["categories"])
    cat_set = set(categories)
    for a, b in args.cat_pairs:
        for c in (a, b):
            if c not in cat_set:
                raise ValueError(f"category {c!r} not in {categories}")

    basis_payload = torch.load(args.basis, weights_only=False, map_location="cpu")
    if int(basis_payload["layer"]) != layer:
        raise ValueError(
            f"basis layer {basis_payload['layer']} != weights layer {layer}"
        )
    W = basis_payload[decomposer]["W"].numpy().astype(np.float32)   # [k, D]
    k = W.shape[0]
    print(f"[add] basis: {decomposer} k={k} layer={layer}")

    caa = torch.load(args.caa, weights_only=False, map_location="cpu")
    caa_layers = list(caa["layers"])
    if layer not in caa_layers:
        raise ValueError(f"layer {layer} not in CAA layers {caa_layers}")
    li_caa = caa_layers.index(layer)
    caa_vecs = caa["vectors"].numpy().astype(np.float32)            # [C, L, D]
    cat_idx = {c: i for i, c in enumerate(categories)}

    # -------- Build OLS-reconstructed CAA per category --------
    recons: dict[str, np.ndarray] = {}
    for cat in cat_set.intersection({c for ab in args.cat_pairs for c in ab}):
        w_ols = weights_payload["weights"][cat]["ols"].numpy().astype(np.float32)
        recons[cat] = (W.T @ w_ols).astype(np.float32)              # [D]

    # Alpha scale: same convention as eval_caa_basis_decomp_steering.py
    caa_norms = np.linalg.norm(caa_vecs[:, li_caa, :], axis=-1)
    scale = float(1.0 / np.median(caa_norms))
    print(f"[add] alpha scale = {scale:.5f}")

    # -------- Load model + steering config --------
    profile, _ = load_profile(args.config)
    sc = yaml.safe_load(args.steering_config.read_text())
    apply_to = sc["caa"].get("apply_to", "generation")
    model, device, _ = load_model(profile)

    prompts = load_neutral_prompts(n=args.n_prompts, seed=args.prompts_seed)
    print(f"[add] {len(args.cat_pairs)} pairs × "
          f"{len(args.alphas)}×{len(args.alphas)} alphas × "
          f"{len(prompts)} prompts")

    # -------- Generate --------
    gen_path = args.output_dir / "generations.parquet"
    done: set[tuple] = set()
    rows: list[dict] = []
    if args.resume and gen_path.exists():
        prev = pd.read_parquet(gen_path)
        rows = prev.to_dict(orient="records")
        done = {
            (r["cat_a"], r["cat_b"], float(r["alpha"]), float(r["beta"]),
             int(r["prompt_id"]))
            for r in rows
        }
        print(f"[add] resume: {len(done)} prior rows")

    total_target = (len(args.cat_pairs) * len(args.alphas) * len(args.alphas)
                    * len(prompts))
    pbar = tqdm(total=total_target, desc="[add] gen", initial=len(done))
    for (cat_a, cat_b) in args.cat_pairs:
        v_a_np = recons[cat_a]
        v_b_np = recons[cat_b]
        norm_a = float(np.linalg.norm(v_a_np))
        norm_b = float(np.linalg.norm(v_b_np))
        b_a = torch.from_numpy(v_a_np).to(torch.float32)
        b_b = torch.from_numpy(v_b_np).to(torch.float32)
        cell_added = 0
        for a in args.alphas:
            for b in args.alphas:
                # Match CAA-decomp convention: each axis contributes
                # (alpha_unit * scale * ||v||) magnitude independently.
                coef_a = float(a) * scale * norm_a
                coef_b = float(b) * scale * norm_b
                vec = coef_a * b_a + coef_b * b_b
                if torch.allclose(vec, torch.zeros_like(vec)):
                    vec_use = torch.zeros_like(b_a)
                    alpha_use = 0.0
                else:
                    vec_use = vec
                    alpha_use = 1.0
                for pi, prompt in enumerate(prompts):
                    key = (cat_a, cat_b, float(a), float(b), pi)
                    if key in done:
                        continue
                    out = steered_generate(
                        model, prompt, vector=vec_use, alpha=alpha_use,
                        layers=[layer], apply_to=apply_to,
                        max_new_tokens=args.max_new_tokens,
                    )
                    tail = out[len(prompt):].strip() if out.startswith(prompt) else out.strip()
                    rows.append({
                        "cat_a": cat_a, "cat_b": cat_b,
                        "alpha": float(a), "beta": float(b),
                        "coef_a": coef_a, "coef_b": coef_b,
                        "layer": layer, "prompt_id": pi, "prompt": prompt,
                        "generation": tail,
                    })
                    done.add(key)
                    cell_added += 1
                    pbar.update(1)
        if cell_added:
            pd.DataFrame(rows).to_parquet(gen_path, index=False)
    pbar.close()
    df = pd.DataFrame(rows)
    df.to_parquet(gen_path, index=False)
    print(f"[add] wrote {gen_path} ({len(df)} rows)")

    # -------- Re-encode + project onto basis --------
    # Group by (cat_a, cat_b, alpha, beta) to encode in batches per cell.
    print("[add] encoding generations and projecting onto basis ...")
    df = df.sort_values(["cat_a", "cat_b", "alpha", "beta", "prompt_id"]).reset_index(drop=True)
    proj_rows: list[dict] = []
    # Cache mean-readout per cell for additivity prediction.
    cell_readouts: dict[tuple[str, str, float, float], np.ndarray] = {}
    for (cat_a, cat_b, a, b), sub in tqdm(
        df.groupby(["cat_a", "cat_b", "alpha", "beta"], sort=False),
        desc="[add] encode",
    ):
        texts = sub["generation"].astype(str).tolist()
        h = collect_batch(model, texts, [layer], device)[layer].numpy()
        proj = _project_all(h, W)                                   # [N, k]
        cell_readouts[(cat_a, cat_b, float(a), float(b))] = proj
        for offset, (_, row) in enumerate(sub.iterrows()):
            proj_rows.append({
                "cat_a": cat_a, "cat_b": cat_b,
                "alpha": float(a), "beta": float(b),
                "prompt_id": int(row["prompt_id"]),
                "readout": proj[offset].tolist(),
            })
    proj_df = pd.DataFrame(proj_rows)
    proj_df.to_parquet(args.output_dir / "readouts.parquet", index=False)

    # -------- Readout-space additivity --------
    readout_rows: list[dict] = []
    for (cat_a, cat_b) in args.cat_pairs:
        r00 = cell_readouts[(cat_a, cat_b, 0.0, 0.0)]               # [N, k]
        for a in args.alphas:
            for b in args.alphas:
                if a == 0.0 and b == 0.0:
                    continue
                actual = cell_readouts[(cat_a, cat_b, float(a), float(b))]
                marg_a = cell_readouts[(cat_a, cat_b, float(a), 0.0)]
                marg_b = cell_readouts[(cat_a, cat_b, 0.0, float(b))]
                predicted = marg_a + marg_b - r00
                resid = actual - predicted
                num = np.linalg.norm(resid, axis=1)                 # [N]
                # Denominator: norm of marginal joint effect (excl. baseline).
                den = np.linalg.norm(predicted - r00, axis=1) + 1e-12
                ratio = num / den
                readout_rows.append({
                    "cat_a": cat_a, "cat_b": cat_b,
                    "alpha": float(a), "beta": float(b),
                    "mean_resid_norm": float(np.mean(num)),
                    "mean_marg_norm": float(np.mean(den)),
                    "mean_resid_ratio": float(np.mean(ratio)),
                    "median_resid_ratio": float(np.median(ratio)),
                })
    readout_df = pd.DataFrame(readout_rows)
    readout_df.to_csv(args.output_dir / "additivity_readout.csv", index=False)

    # -------- Classify + shift-acc additivity --------
    print("[add] classifying generations ...")
    cls = _classify(df["generation"].astype(str).tolist())
    df_cls = pd.concat([df.reset_index(drop=True), cls], axis=1)
    df_cls["pred_plutchik"] = df_cls["pred_label"].map(HARTMANN_TO_PLUTCHIK).fillna("other")
    df_cls.to_parquet(args.output_dir / "generations_classified.parquet", index=False)

    # Shift-accuracy per cell to A and B.
    shift_rows: list[dict] = []
    cell_shift: dict[tuple[str, str, float, float], dict[str, float]] = {}
    for (cat_a, cat_b, a, b), sub in df_cls.groupby(
        ["cat_a", "cat_b", "alpha", "beta"], sort=False,
    ):
        rec = {}
        for cat_target, key in [(cat_a, "to_a"), (cat_b, "to_b")]:
            if cat_target in UNMEASURABLE:
                rec[key] = float("nan")
            else:
                rec[key] = float((sub["pred_plutchik"] == cat_target).mean())
        cell_shift[(cat_a, cat_b, float(a), float(b))] = rec

    for (cat_a, cat_b) in args.cat_pairs:
        s00 = cell_shift[(cat_a, cat_b, 0.0, 0.0)]
        for a in args.alphas:
            for b in args.alphas:
                if a == 0.0 and b == 0.0:
                    continue
                actual = cell_shift[(cat_a, cat_b, float(a), float(b))]
                marg_a = cell_shift[(cat_a, cat_b, float(a), 0.0)]
                marg_b = cell_shift[(cat_a, cat_b, 0.0, float(b))]
                row = {"cat_a": cat_a, "cat_b": cat_b,
                       "alpha": float(a), "beta": float(b)}
                for key in ("to_a", "to_b"):
                    pred = marg_a[key] + marg_b[key] - s00[key]
                    row[f"{key}_actual"] = actual[key]
                    row[f"{key}_marg_a"] = marg_a[key]
                    row[f"{key}_marg_b"] = marg_b[key]
                    row[f"{key}_baseline"] = s00[key]
                    row[f"{key}_predicted"] = pred
                    row[f"{key}_resid"] = actual[key] - pred
                shift_rows.append(row)
    shift_df = pd.DataFrame(shift_rows)
    shift_df.to_csv(args.output_dir / "additivity_shift.csv", index=False)

    # -------- Summary --------
    offdiag = readout_df[(readout_df.alpha != 0) & (readout_df.beta != 0)]
    shift_off = shift_df[(shift_df.alpha != 0) & (shift_df.beta != 0)]
    summary = {
        "layer": layer,
        "decomposer": decomposer,
        "k": int(k),
        "cat_pairs": [list(p) for p in args.cat_pairs],
        "alphas": list(args.alphas),
        "n_prompts": args.n_prompts,
        "max_new_tokens": args.max_new_tokens,
        "readout_offdiag": {
            "median_resid_ratio": float(offdiag["mean_resid_ratio"].median())
            if not offdiag.empty else None,
            "mean_resid_ratio": float(offdiag["mean_resid_ratio"].mean())
            if not offdiag.empty else None,
        },
        "shift_offdiag": {
            "abs_resid_to_a_median": float(shift_off["to_a_resid"].abs().median())
            if not shift_off.empty else None,
            "abs_resid_to_b_median": float(shift_off["to_b_resid"].abs().median())
            if not shift_off.empty else None,
            "abs_resid_to_a_mean": float(shift_off["to_a_resid"].abs().mean())
            if not shift_off.empty else None,
            "abs_resid_to_b_mean": float(shift_off["to_b_resid"].abs().mean())
            if not shift_off.empty else None,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("[add] === readout off-diagonal additivity ===")
    print(f"  median |resid|/|marg| = "
          f"{summary['readout_offdiag']['median_resid_ratio']:.3f}")
    print("[add] === shift-acc off-diagonal additivity ===")
    print(f"  median |resid_to_a| = {summary['shift_offdiag']['abs_resid_to_a_median']:.3f}")
    print(f"  median |resid_to_b| = {summary['shift_offdiag']['abs_resid_to_b_median']:.3f}")
    print(f"[add] wrote {args.output_dir}/")


if __name__ == "__main__":
    main()
