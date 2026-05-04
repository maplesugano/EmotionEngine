"""Basis-native additivity test, with output schema feeding meta-emotion projection.

Background
----------
§3.X.7–§3.X.8 established a two-layer story for *CAA-pair* joint steering:
    joint = (linear sum of marginals) + (nonlinear residual)
where the residual was large in basis-readout space (median |resid|/|marg|
≈ 0.81) yet near zero in shift-acc space, *and* did not project onto any of
the 7 LLM-judge meta-emotion clusters (§3.X.8(c)).

That negative result is consistent with the §3.X.2 finding that
*basis-native* combos (e.g. ``b1 + b11``) are the actual triggers of
"meta-emotions" like *過警戒的内省 / hyper-vigilant introspection* —
combinations the existing CAA-pair sweep cannot reach because reconstructed
CAA vectors are dense pan mixtures of all basis components.

This script closes the gap by running the *basis-native* version of the
additivity test:

    α b_i + β b_j           with (α, β) ∈ alphas × alphas

It saves generations in a schema compatible with
``eval_caa_additivity_metaemotion.py`` (``cat_a``, ``cat_b``, ``alpha``,
``beta``, ``prompt_id``, ``generation``), so the meta-emotion projection
pipeline can be reused without changes.

Pipeline
--------
1. Load basis (decomposer artifact from ``data/emotion_code/basis_sweep*``).
2. For each component pair ``(i, j)`` and each ``(α, β)`` cell, steer
   ``α b_i + β b_j`` with ``caa_match`` rescaling (each axis is rescaled to
   ``median(||CAA||)`` so the effective injected magnitude matches the CAA
   convention used by every other steering script in this project).
3. Re-encode every generation at the basis layer and project onto the basis
   to get readouts ``r(α, β) ∈ R^k``.
4. Compute readout-space additivity residuals
       resid = r(α, β) − [r(α, 0) + r(0, β) − r(0, 0)]
   and per-cell summaries (``mean_resid_ratio``, etc.).
5. Save ``generations.parquet`` so the user can run
       uv run python -m experiments.eval_caa_additivity_metaemotion \\
           --additivity-dir <output_dir> \\
           --cluster-dir   experiments/results/lexical_gap_judge \\
           --output-dir    <output_dir>/metaemotion
   and obtain the meta-emotion projections of joint / marginal / residual
   embedding vectors. The projection script is schema-agnostic to whether
   ``cat_a`` / ``cat_b`` are Plutchik names or basis indices like ``"b1"``.

Usage
-----
    # default: replicates §3.X.2 combos plus self-pair sanity
    uv run python -m experiments.eval_basis_additivity_metaemotion \\
        --basis data/emotion_code/basis_sweep/ica_k016_seed0.pt \\
        --pairs 1,11 11,13 1,13 8,4 8,7 \\
        --alphas -2 0 2 --n-prompts 8 --max-new-tokens 32 \\
        --output-dir experiments/results/basis_additivity_metaemotion_L19_k16
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from experiments._gen_cache import load_neutral_prompts
from src.activations._runtime import collect_batch, load_model, load_profile
from src.steering.generate import steered_generate


def _project_all(h: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Cosine projection of each row of h onto each row of W -> [N, k]."""
    h_n = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    w_n = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    return h_n @ w_n.T


def _parse_pair(s: str) -> tuple[int, int]:
    a, b = s.split(",")
    return int(a), int(b)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--basis", type=Path, required=True)
    p.add_argument("--which", default=None,
                   help="Decomposer key inside the basis payload "
                        "(default: payload['decomposer']).")
    p.add_argument("--pairs", type=_parse_pair, nargs="+", required=True,
                   help="Basis component pairs as 'i,j' (space-separated).")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[-2.0, 0.0, 2.0])
    p.add_argument("--alpha-mode", choices=["caa_match", "unit_v"],
                   default="caa_match",
                   help="caa_match: rescale each basis row to median(||CAA||) "
                        "before applying alpha (matches §3.X.2 lexical-gap "
                        "convention; recommended). "
                        "unit_v: use raw ||b|| (matches the legacy "
                        "eval_basis_additivity.py — yields tiny effective "
                        "magnitudes for ICA basis).")
    p.add_argument("--caa", type=Path, default=Path("data/emotion_code/caa.pt"),
                   help="Used only for the alpha-scale convention "
                        "(median(||CAA||) at the basis layer).")
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--prompts-seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--steering-config", type=Path,
                   default=Path("configs/steering.yaml"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-metaemotion", action="store_true",
                   help="After generation+readout additivity, automatically "
                        "invoke eval_caa_additivity_metaemotion on the "
                        "generations.parquet. Requires OPENAI_API_KEY in env.")
    p.add_argument("--cluster-dir", type=Path,
                   default=Path("experiments/results/lexical_gap_judge"),
                   help="Passed through to eval_caa_additivity_metaemotion "
                        "when --run-metaemotion is set.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # -------- Load basis --------
    payload = torch.load(args.basis, weights_only=False, map_location="cpu")
    which = args.which or payload.get("decomposer")
    if which is None:
        for cand in ("ica", "nmf", "pca", "dict"):
            if cand in payload:
                which = cand
                break
    W = payload[which]["W"].numpy().astype(np.float32)              # [k, D]
    layer = int(payload["layer"])
    k = W.shape[0]
    raw_norms = np.linalg.norm(W, axis=1)
    print(f"[bam] basis={args.basis.name} which={which} layer={layer} k={k}")
    print(f"[bam] median ||b||={float(np.median(raw_norms)):.3f}  "
          f"min/max={float(raw_norms.min()):.3f}/{float(raw_norms.max()):.3f}")

    # CAA-norm scale (same convention as every other steering script) ------
    caa = torch.load(args.caa, weights_only=False, map_location="cpu")
    caa_layers = list(caa["layers"])
    if layer not in caa_layers:
        raise ValueError(f"basis layer {layer} not in CAA layers {caa_layers}")
    li_caa = caa_layers.index(layer)
    caa_norms_layer = np.linalg.norm(
        caa["vectors"].numpy().astype(np.float32)[:, li_caa, :], axis=-1
    )
    caa_scale = float(1.0 / np.median(caa_norms_layer))
    median_caa_norm = float(np.median(caa_norms_layer))
    print(f"[bam] alpha-mode={args.alpha_mode}  "
          f"median||CAA||={median_caa_norm:.3f}  caa_scale={caa_scale:.5f}")

    # Per-component effective steering rows (after caa_match rescaling).
    # We keep *both* (a) the original W rows for projection and (b) the
    # rescaled rows used as injection vectors so that α=1 corresponds to one
    # CAA-norm-equivalent unit along that axis (matching shift-acc / lexical-
    # gap conventions).
    if args.alpha_mode == "caa_match":
        steer_rows = W * (median_caa_norm / (raw_norms[:, None] + 1e-12))
    else:
        steer_rows = W.copy()
    steer_norms = np.linalg.norm(steer_rows, axis=1)

    # -------- Load model --------
    profile, _ = load_profile(args.config)
    sc = yaml.safe_load(args.steering_config.read_text())
    apply_to = sc["caa"].get("apply_to", "generation")
    model, device, _ = load_model(profile)

    prompts = load_neutral_prompts(n=args.n_prompts, seed=args.prompts_seed)
    print(f"[bam] {len(args.pairs)} pairs × "
          f"{len(args.alphas)}×{len(args.alphas)} alphas × "
          f"{len(prompts)} prompts")

    # -------- Generate --------
    gen_path = args.output_dir / "generations.parquet"
    rows: list[dict] = []
    done: set[tuple] = set()
    if args.resume and gen_path.exists():
        prev = pd.read_parquet(gen_path)
        rows = prev.to_dict(orient="records")
        done = {
            (r["cat_a"], r["cat_b"], float(r["alpha"]), float(r["beta"]),
             int(r["prompt_id"]))
            for r in rows
        }
        print(f"[bam] resume: {len(done)} prior rows")

    total = len(args.pairs) * len(args.alphas) * len(args.alphas) * len(prompts)
    pbar = tqdm(total=total, desc="[bam] gen", initial=len(done))
    for (i, j) in args.pairs:
        cat_a, cat_b = f"b{i}", f"b{j}"
        b_i = torch.from_numpy(steer_rows[i].astype(np.float32))
        b_j = torch.from_numpy(steer_rows[j].astype(np.float32))
        n_i = float(steer_norms[i])
        n_j = float(steer_norms[j])
        cell_added = 0
        for a in args.alphas:
            for b in args.alphas:
                # alpha (in vector-norm units) → raw coefficient that
                # `steered_generate` will multiply against the (pre-scaled)
                # vector. The combined vector has both axes mixed, so we set
                # alpha_use=1.0 and fold the magnitudes into the vector.
                coef_a = float(a) * caa_scale * n_i
                coef_b = float(b) * caa_scale * n_j
                vec = coef_a * b_i + coef_b * b_j
                if torch.allclose(vec, torch.zeros_like(vec)):
                    vec_use = torch.zeros_like(b_i)
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
                    tail = (out[len(prompt):].strip()
                            if out.startswith(prompt) else out.strip())
                    rows.append({
                        "cat_a": cat_a, "cat_b": cat_b,
                        "i": int(i), "j": int(j),
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
    print(f"[bam] wrote {gen_path} ({len(df)} rows)")

    # -------- Re-encode + project onto basis (readouts) --------
    print("[bam] encoding generations and projecting onto basis ...")
    df = df.sort_values(["cat_a", "cat_b", "alpha", "beta", "prompt_id"]
                        ).reset_index(drop=True)
    cell_readouts: dict[tuple[str, str, float, float], np.ndarray] = {}
    proj_rows: list[dict] = []
    for (cat_a, cat_b, a, b), sub in tqdm(
        df.groupby(["cat_a", "cat_b", "alpha", "beta"], sort=False),
        desc="[bam] encode",
    ):
        texts = sub["generation"].astype(str).tolist()
        h = collect_batch(model, texts, [layer], device)[layer].numpy()
        proj = _project_all(h, W)                                   # [N, k]
        cell_readouts[(cat_a, cat_b, float(a), float(b))] = proj
        for offset, (_, row) in enumerate(sub.iterrows()):
            proj_rows.append({
                "cat_a": cat_a, "cat_b": cat_b,
                "i": int(row["i"]), "j": int(row["j"]),
                "alpha": float(a), "beta": float(b),
                "prompt_id": int(row["prompt_id"]),
                "readout": proj[offset].tolist(),
            })
    pd.DataFrame(proj_rows).to_parquet(
        args.output_dir / "readouts.parquet", index=False
    )

    # -------- Readout-space additivity --------
    readout_rows: list[dict] = []
    detail_rows: list[dict] = []
    for (i, j) in args.pairs:
        cat_a, cat_b = f"b{i}", f"b{j}"
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
                num = np.linalg.norm(resid, axis=1)
                den = np.linalg.norm(predicted - r00, axis=1) + 1e-12
                ratio = num / den

                # Per-axis self-readout error (the §3.X.7 main metric).
                err_i = np.abs(actual[:, i] - predicted[:, i])
                err_j = np.abs(actual[:, j] - predicted[:, j])
                for pi in range(actual.shape[0]):
                    detail_rows.append({
                        "cat_a": cat_a, "cat_b": cat_b,
                        "i": int(i), "j": int(j),
                        "alpha": float(a), "beta": float(b),
                        "prompt_id": int(pi),
                        "actual_i": float(actual[pi, i]),
                        "actual_j": float(actual[pi, j]),
                        "pred_i": float(predicted[pi, i]),
                        "pred_j": float(predicted[pi, j]),
                        "err_i": float(err_i[pi]),
                        "err_j": float(err_j[pi]),
                        "resid_norm": float(num[pi]),
                        "marg_norm": float(den[pi]),
                        "resid_ratio": float(ratio[pi]),
                    })
                readout_rows.append({
                    "cat_a": cat_a, "cat_b": cat_b,
                    "i": int(i), "j": int(j),
                    "alpha": float(a), "beta": float(b),
                    "mean_err_i": float(np.mean(err_i)),
                    "mean_err_j": float(np.mean(err_j)),
                    "mean_resid_norm": float(np.mean(num)),
                    "mean_marg_norm": float(np.mean(den)),
                    "mean_resid_ratio": float(np.mean(ratio)),
                    "median_resid_ratio": float(np.median(ratio)),
                })
    readout_df = pd.DataFrame(readout_rows)
    readout_df.to_csv(args.output_dir / "additivity_readout.csv", index=False)
    pd.DataFrame(detail_rows).to_parquet(
        args.output_dir / "additivity_readout_detail.parquet", index=False
    )

    # -------- Summary --------
    offdiag = readout_df[(readout_df.alpha != 0) & (readout_df.beta != 0)]
    summary = {
        "basis": str(args.basis),
        "decomposer": which,
        "layer": layer,
        "k": int(k),
        "pairs": [list(p) for p in args.pairs],
        "alphas": list(args.alphas),
        "alpha_mode": args.alpha_mode,
        "n_prompts": args.n_prompts,
        "max_new_tokens": args.max_new_tokens,
        "median_caa_norm": median_caa_norm,
        "readout_offdiag": {
            "median_resid_ratio": float(offdiag["mean_resid_ratio"].median())
            if not offdiag.empty else None,
            "mean_resid_ratio": float(offdiag["mean_resid_ratio"].mean())
            if not offdiag.empty else None,
            "median_err_i": float(offdiag["mean_err_i"].median())
            if not offdiag.empty else None,
            "median_err_j": float(offdiag["mean_err_j"].median())
            if not offdiag.empty else None,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("[bam] === readout off-diagonal additivity ===")
    if not offdiag.empty:
        print(f"  median |resid|/|marg| = "
              f"{summary['readout_offdiag']['median_resid_ratio']:.3f}")
        print(f"  median err_i = {summary['readout_offdiag']['median_err_i']:.4f}, "
              f"err_j = {summary['readout_offdiag']['median_err_j']:.4f}")
    print(f"[bam] wrote {args.output_dir}/")

    # -------- Optionally invoke meta-emotion projection --------
    if args.run_metaemotion:
        meta_out = args.output_dir / "metaemotion"
        cmd = [
            sys.executable, "-m", "experiments.eval_caa_additivity_metaemotion",
            "--additivity-dir", str(args.output_dir),
            "--cluster-dir", str(args.cluster_dir),
            "--output-dir", str(meta_out),
        ]
        print("[bam] launching meta-emotion projection:")
        print("       " + " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print("\n[bam] next step (meta-emotion projection):")
        print(
            "  uv run python -m experiments.eval_caa_additivity_metaemotion \\\n"
            f"      --additivity-dir {args.output_dir} \\\n"
            f"      --cluster-dir   experiments/results/lexical_gap_judge \\\n"
            f"      --output-dir    {args.output_dir}/metaemotion"
        )


if __name__ == "__main__":
    main()
