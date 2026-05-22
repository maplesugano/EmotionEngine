"""Discover candidate *primitive affective units* from a learned basis.

A primitive affective unit, in this project, is a basis direction ``b_j``
that satisfies four properties simultaneously:

  1. **label-independent** — its activation is not dominated by any single
     Plutchik category (low ``category_top1_dominance``, low MI).
  2. **causally steerable** — injecting α · b_j at the basis layer shifts
     the model's *own* readout of b_j proportionally to α (sign-correct,
     monotone Spearman ρ across α).
  3. **self-consistent under encode→steer→re-encode** — the readout shift
     concentrates on b_j and not on other components (low cross-talk).
  4. **not reducible to VAD** — the component direction has small overlap
     with the V/A/D affine subspace (low ``vad_explained``).

Components passing these tests are then probed with strong-α qualitative
steering so a human can read what each unit *does*.

This script does not perform LLM judging or meta-emotion clustering;
those are optional follow-ups (see ``eval_basis_per_axis_judge.py`` and
``eval_lexical_gap_*``). The PrimitiveScore is a transparent weighted sum
of behavioural and structural metrics.

Outputs (under ``--output-dir`` / ``<basis_stem>``):
  primitive_scores.csv         per-component scores + score breakdown
  primitive_scores.json        same as CSV plus run config
  selfcons_readouts.parquet    raw encode-steer-re-encode readouts
  strong_generations.parquet   strong-α generations for top-k components
  top_candidates.md            human-readable shortlist with examples
  config.json                  exact CLI/run config

Usage
-----
    uv run python -m experiments.eval_primitive_affective_units \
        --basis data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt \
        --top-k 10 --n-prompts 8 --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from experiments._gen_cache import load_neutral_prompts
from src.activations._runtime import collect_batch, load_model, load_profile
from src.steering.generate import steered_generate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _project_all(h: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Cosine projection of activations h[N,D] onto each row of W[k,D]."""
    if h.ndim == 1:
        h = h[None, :]
    h_n = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    W_n = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    return h_n @ W_n.T


def _parse_alpha_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _load_basis(path: Path, which: str | None
                ) -> tuple[np.ndarray, str, int, int]:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    decomposer = which or payload.get("decomposer")
    if decomposer is None or decomposer not in payload:
        for cand in ("ica", "nmf", "pca", "dict"):
            if cand in payload:
                decomposer = cand
                break
    if decomposer is None:
        raise SystemExit(f"Cannot find a decomposer block in {path}")
    W = payload[decomposer]["W"].numpy().astype(np.float32)
    layer = int(payload["layer"])
    k = int(W.shape[0])
    return W, decomposer, layer, k


def _load_metrics_sibling(basis_path: Path) -> dict[int, dict[str, float]]:
    """Return {component_index: {metric: value}} from the *.metrics.json
    sibling, if present."""
    metrics_path = basis_path.with_suffix(".metrics.json")
    if not metrics_path.exists():
        # Try stem + ".metrics.json"
        alt = basis_path.parent / f"{basis_path.stem}.metrics.json"
        metrics_path = alt if alt.exists() else metrics_path
    if not metrics_path.exists():
        print(f"[primitive] no metrics sibling at {metrics_path}")
        return {}
    payload = json.loads(metrics_path.read_text())
    out: dict[int, dict[str, float]] = {}
    for r in payload.get("per_component", []):
        out[int(r["component"])] = {
            "mi": float(r.get("mi", float("nan"))),
            "linear_sep_acc": float(r.get("linear_sep_acc", float("nan"))),
            "category_top1_dominance": float(
                r.get("category_top1_dominance", float("nan"))
            ),
            "vad_explained": float(r.get("vad_explained", float("nan"))),
        }
    print(f"[primitive] loaded label/VAD metrics from {metrics_path}")
    return out


def _caa_median_norm(caa_path: Path, layer: int) -> float:
    """Median ||CAA vector|| at the given layer, used by ``caa_match`` mode."""
    caa = torch.load(caa_path, weights_only=False, map_location="cpu")
    layers = list(caa["layers"])
    if layer not in layers:
        raise SystemExit(
            f"--caa layer={layer} not in {layers}; pass a different --caa "
            f"or run build_caa for that layer."
        )
    li = layers.index(layer)
    norms = caa["vectors"][:, li, :].norm(dim=-1).numpy()
    return float(np.median(norms))


def _scaled_vector(
    vec: np.ndarray,
    alpha_mode: str,
    median_caa_norm: float | None,
) -> tuple[np.ndarray, float]:
    """Apply alpha_mode normalisation and return (vector, ||vector||)."""
    raw_norm = float(np.linalg.norm(vec))
    if alpha_mode == "caa_match":
        if median_caa_norm is None:
            raise SystemExit("--alpha-mode=caa_match requires --caa.")
        v = vec * (median_caa_norm / (raw_norm + 1e-12))
    elif alpha_mode == "unit_v":
        v = vec
    else:
        raise SystemExit(f"unknown --alpha-mode={alpha_mode}")
    return v.astype(np.float32), float(np.linalg.norm(v))


# ---------------------------------------------------------------------------
# Self-consistency pass
# ---------------------------------------------------------------------------
def _run_selfconsistency(
    args,
    model,
    device,
    W: np.ndarray,
    layer: int,
    components: list[int],
    alphas: list[float],
    prompts: list[str],
    apply_to: str,
    scale: float,
    median_caa_norm: float | None,
    out_dir: Path,
) -> pd.DataFrame:
    """Encode→steer→re-encode loop for every (component, alpha, prompt)."""
    raw_path = out_dir / "selfcons_readouts.parquet"
    done: set[tuple[int, float, int]] = set()
    rows: list[dict] = []
    if args.resume and raw_path.exists():
        prev = pd.read_parquet(raw_path)
        rows = prev.to_dict(orient="records")
        done = {
            (int(r["component"]), float(r["alpha_unit"]), int(r["prompt_id"]))
            for r in rows
        }
        print(f"[selfcons] resume: {len(done)} prior readouts")

    # 1. Baseline generations at α=0 (shared across all components).
    print("[selfcons] baseline generations...")
    baseline_gen: list[str] = []
    for prompt in tqdm(prompts, desc="[selfcons] baseline"):
        out = steered_generate(
            model, prompt, vector=torch.zeros(W.shape[1]),
            alpha=0.0, layers=[layer], apply_to=apply_to,
            max_new_tokens=args.max_new_tokens,
        )
        baseline_gen.append(out)
    baseline_h = collect_batch(
        model, baseline_gen, [layer], device
    )[layer].numpy()
    baseline_proj = _project_all(baseline_h, W)  # [N_prompt, k]

    k = W.shape[0]
    for j in components:
        vec = W[j].astype(np.float32)
        v_use, v_norm = _scaled_vector(vec, args.alpha_mode, median_caa_norm)
        b_t = torch.from_numpy(v_use)
        added = 0
        for au in alphas:
            keys = [(j, float(au), pi) for pi in range(len(prompts))]
            if all(k_ in done for k_ in keys):
                continue
            if au == 0.0:
                gens = baseline_gen
            else:
                alpha = float(au) * scale * v_norm
                gens = []
                for prompt in tqdm(
                    prompts, desc=f"[selfcons] b{j} α={au:+}", leave=False
                ):
                    gens.append(steered_generate(
                        model, prompt, vector=b_t, alpha=alpha,
                        layers=[layer], apply_to=apply_to,
                        max_new_tokens=args.max_new_tokens,
                    ))
            h = collect_batch(model, gens, [layer], device)[layer].numpy()
            proj = _project_all(h, W)
            delta = proj - baseline_proj
            for pi in range(len(prompts)):
                key = (j, float(au), pi)
                if key in done:
                    continue
                for l in range(k):
                    rows.append({
                        "component": int(j),
                        "alpha_unit": float(au),
                        "prompt_id": int(pi),
                        "readout_component": int(l),
                        "delta_cosine": float(delta[pi, l]),
                        "abs_cosine": float(proj[pi, l]),
                        "is_self": int(l == j),
                    })
                done.add(key)
                added += 1
        if added:
            pd.DataFrame(rows).to_parquet(raw_path, index=False)
    df = pd.DataFrame(rows)
    df.to_parquet(raw_path, index=False)
    print(f"[selfcons] wrote {raw_path} ({len(df)} rows)")
    return df


def _aggregate_selfcons(df: pd.DataFrame,
                        components: list[int],
                        alphas: list[float],
                        ) -> dict[int, dict[str, float]]:
    """Per-component: self_delta_pos/neg, sign_correct, self_rho, specificity,
    causal_strength, cross_talk."""
    pos_a = max(alphas)
    neg_a = min(alphas)
    out: dict[int, dict[str, float]] = {}
    for j in components:
        sub = df[df["component"] == j]
        rec: dict[str, float] = {}

        def _self_mean(au: float) -> float:
            s = sub[(sub["alpha_unit"] == au) & (sub["is_self"] == 1)]
            return float(s["delta_cosine"].mean()) if len(s) else float("nan")

        def _other_absmean(au: float) -> float:
            s = sub[(sub["alpha_unit"] == au) & (sub["is_self"] == 0)]
            return float(s["delta_cosine"].abs().mean()) if len(s) else float("nan")

        rec["self_delta_pos"] = _self_mean(pos_a)
        rec["self_delta_neg"] = _self_mean(neg_a)
        rec["sign_correct"] = float(
            (rec["self_delta_pos"] > 0) and (rec["self_delta_neg"] < 0)
        )

        # Spearman ρ: per-prompt then averaged.
        self_sub = sub[sub["is_self"] == 1]
        rhos: list[float] = []
        for pi in sorted(self_sub["prompt_id"].unique()):
            s = self_sub[self_sub["prompt_id"] == pi].sort_values("alpha_unit")
            if len(s) < 3:
                continue
            r, _ = spearmanr(s["alpha_unit"], s["delta_cosine"])
            if not np.isnan(r):
                rhos.append(float(r))
        rec["self_rho"] = float(np.mean(rhos)) if rhos else float("nan")

        # specificity at extremes (self − mean|other|).
        spec_pos = rec["self_delta_pos"] - _other_absmean(pos_a)
        spec_neg = (-rec["self_delta_neg"]) - _other_absmean(neg_a)
        rec["specificity"] = float(np.nanmean([spec_pos, spec_neg]))

        # causal strength: mean |self_delta| at α±extreme.
        rec["causal_strength"] = float(np.nanmean([
            abs(rec["self_delta_pos"]),
            abs(rec["self_delta_neg"]),
        ]))

        # cross-talk: mean |other_delta| at α±extreme.
        rec["cross_talk"] = float(np.nanmean([
            _other_absmean(pos_a), _other_absmean(neg_a)
        ]))
        out[j] = rec
    return out


# ---------------------------------------------------------------------------
# PrimitiveScore
# ---------------------------------------------------------------------------
SCORE_WEIGHTS: dict[str, float] = {
    "self_rho": 1.0,
    "sign_correct": 1.0,
    "causal_strength": 0.5,
    "label_dominance": -0.5,    # category_top1_dominance
    "vad_explained": -0.5,
    "cross_talk": -0.25,
}


def _primitive_score(rec: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Weighted sum, NaN-safe (missing terms are dropped)."""
    contribs: dict[str, float] = {}
    total = 0.0
    used = False
    for key, w in SCORE_WEIGHTS.items():
        v = rec.get(key, float("nan"))
        if v is None or (isinstance(v, float) and np.isnan(v)):
            contribs[key] = float("nan")
            continue
        contrib = w * float(v)
        contribs[key] = contrib
        total += contrib
        used = True
    return (total if used else float("nan")), contribs


# ---------------------------------------------------------------------------
# Strong-α qualitative generation
# ---------------------------------------------------------------------------
def _run_strong_generation(
    args,
    model,
    W: np.ndarray,
    layer: int,
    decomposer: str,
    components: list[int],
    alphas: list[float],
    prompts: list[str],
    apply_to: str,
    scale: float,
    median_caa_norm: float | None,
    out_dir: Path,
) -> pd.DataFrame:
    gen_path = out_dir / "strong_generations.parquet"
    rows: list[dict] = []
    done: set[tuple[int, float, int]] = set()
    if args.resume and gen_path.exists():
        prev = pd.read_parquet(gen_path)
        rows = prev.to_dict(orient="records")
        done = {
            (int(r["component"]), float(r["alpha_unit"]), int(r["prompt_id"]))
            for r in rows
        }
        print(f"[strong] resume: {len(done)} prior generations")

    pbar = tqdm(
        total=len(components) * len(alphas) * len(prompts),
        initial=len(done), desc="[strong]",
    )
    for j in components:
        vec = W[j].astype(np.float32)
        v_use, v_norm = _scaled_vector(vec, args.alpha_mode, median_caa_norm)
        v_t = torch.from_numpy(v_use)
        added = 0
        for au in alphas:
            alpha = 0.0 if au == 0.0 else float(au) * scale * v_norm
            for pi, prompt in enumerate(prompts):
                key = (int(j), float(au), int(pi))
                if key in done:
                    continue
                out = steered_generate(
                    model, prompt, v_t, alpha=alpha,
                    layers=[layer], apply_to=apply_to,
                    max_new_tokens=args.max_new_tokens,
                )
                tail = (out[len(prompt):].strip()
                        if out.startswith(prompt) else out.strip())
                rows.append({
                    "component": int(j),
                    "alpha": float(alpha),
                    "alpha_unit": float(au),
                    "effective_alpha": float(alpha),
                    "vector_norm": float(v_norm),
                    "prompt_id": int(pi),
                    "prompt": prompt,
                    "generation": tail,
                    "basis_path": str(args.basis),
                    "decomposer": decomposer,
                    "layer": int(layer),
                    "k": int(W.shape[0]),
                })
                done.add(key)
                added += 1
                pbar.update(1)
        if added:
            pd.DataFrame(rows).to_parquet(gen_path, index=False)
    pbar.close()
    df = pd.DataFrame(rows)
    df.to_parquet(gen_path, index=False)
    print(f"[strong] wrote {gen_path} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _write_top_md(
    out_path: Path,
    ranked: pd.DataFrame,
    gens: pd.DataFrame | None,
    strong_alphas: list[float],
    basis_path: Path,
    decomposer: str,
    layer: int,
    k: int,
    top_k: int,
) -> None:
    extreme_neg = min(strong_alphas) if strong_alphas else None
    extreme_pos = max(strong_alphas) if strong_alphas else None
    sample_alphas: list[float] = []
    for au in (extreme_neg, 0.0, extreme_pos):
        if au is not None and au not in sample_alphas:
            sample_alphas.append(au)

    lines: list[str] = []
    lines.append("# Primitive Affective Unit Candidates")
    lines.append("")
    lines.append(f"- basis: `{basis_path}`")
    lines.append(f"- decomposer: `{decomposer}`  layer={layer}  k={k}")
    lines.append(f"- top_k: {top_k}")
    lines.append(f"- strong_alphas: {strong_alphas}")
    lines.append("")
    lines.append("Each candidate is a basis direction `b_j` that scored well "
                 "on the encode→steer→re-encode test, has low Plutchik / VAD "
                 "dominance, and produces a coherent qualitative shift under "
                 "strong α steering. Fill in `proposed_name` after reading "
                 "the example generations.")
    lines.append("")
    for _, row in ranked.head(top_k).iterrows():
        j = int(row["component"])
        lines.append(f"## b{j} — PrimitiveScore = {row['primitive_score']:.3f}")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        lines.append(f"| self_rho | {row['self_rho']:.3f} |")
        lines.append(f"| sign_correct | {int(row['sign_correct'])} |")
        lines.append(f"| self_delta_pos | {row['self_delta_pos']:.3f} |")
        lines.append(f"| self_delta_neg | {row['self_delta_neg']:.3f} |")
        lines.append(f"| causal_strength | {row['causal_strength']:.3f} |")
        lines.append(f"| cross_talk | {row['cross_talk']:.3f} |")
        lines.append(f"| specificity | {row['specificity']:.3f} |")
        lines.append(f"| label_dominance | {row.get('label_dominance', float('nan')):.3f} |")
        lines.append(f"| mi | {row.get('mi', float('nan')):.3f} |")
        lines.append(f"| linear_sep_acc | {row.get('linear_sep_acc', float('nan')):.3f} |")
        lines.append(f"| vad_explained | {row.get('vad_explained', float('nan')):.3f} |")
        lines.append("")
        lines.append("**proposed_name:** _____")
        lines.append("")
        if gens is not None and len(gens):
            sub = gens[gens["component"] == j]
            for au in sample_alphas:
                ssub = sub[sub["alpha_unit"] == au].sort_values("prompt_id")
                if not len(ssub):
                    continue
                lines.append(f"### α = {au:+.1f}")
                lines.append("")
                for _, g in ssub.head(3).iterrows():
                    p = str(g["prompt"]).replace("\n", " ")
                    t = str(g["generation"]).replace("\n", " ")
                    lines.append(f"- prompt: `{p}`")
                    lines.append(f"  → {t}")
                lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"[primitive] wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--basis", type=Path, required=True)
    p.add_argument("--which", default=None,
                   help="Decomposer block (auto-detected if omitted).")
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/results/primitive_affective_units"))
    p.add_argument("--components", type=int, nargs="+", default=None,
                   help="Subset of component indices (default: all).")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--self-alphas", type=str, default="-2,0,2",
                   help="Comma-separated α grid for self-consistency.")
    p.add_argument("--strong-alphas", type=str, default="-6,-3,0,3,6",
                   help="Comma-separated α grid for qualitative generations.")
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layer", type=int, default=None,
                   help="Override basis layer (otherwise read from artifact).")
    p.add_argument("--alpha-mode", type=str, default="caa_match",
                   choices=["caa_match", "unit_v"])
    p.add_argument("--caa", type=Path, default=Path("data/emotion_code/caa.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--steering-config", type=Path,
                   default=Path("configs/steering.yaml"))
    p.add_argument("--skip-generation", action="store_true",
                   help="Skip strong-α qualitative generations.")
    p.add_argument("--skip-selfconsistency", action="store_true",
                   help="Skip the encode→steer→re-encode pass (requires --resume "
                        "with prior selfcons_readouts.parquet).")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    self_alphas = _parse_alpha_list(args.self_alphas)
    strong_alphas = _parse_alpha_list(args.strong_alphas)

    out_dir = args.output_dir / args.basis.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    W, decomposer, layer_artifact, k = _load_basis(args.basis, args.which)
    layer = args.layer if args.layer is not None else layer_artifact
    components = (args.components if args.components is not None
                  else list(range(k)))
    print(f"[primitive] basis={args.basis.name} which={decomposer} "
          f"layer={layer} k={k} components={len(components)}")

    label_metrics = _load_metrics_sibling(args.basis)

    median_caa_norm: float | None = None
    if args.alpha_mode == "caa_match":
        median_caa_norm = _caa_median_norm(args.caa, layer)
        scale = 1.0 / median_caa_norm
        print(f"[primitive] caa_match: median ||CAA||={median_caa_norm:.3f} "
              f"scale={scale:.5f}")
    else:
        norms = np.linalg.norm(W, axis=1)
        scale = 1.0 / float(np.median(norms))
        print(f"[primitive] unit_v: median ||b||={np.median(norms):.3f} "
              f"scale={scale:.5f}")

    sc = yaml.safe_load(args.steering_config.read_text())
    apply_to = sc["caa"].get("apply_to", "generation")

    config = {
        "basis": str(args.basis),
        "decomposer": decomposer,
        "layer": layer,
        "k": k,
        "components": components,
        "self_alphas": self_alphas,
        "strong_alphas": strong_alphas,
        "n_prompts": args.n_prompts,
        "max_new_tokens": args.max_new_tokens,
        "alpha_mode": args.alpha_mode,
        "caa": str(args.caa),
        "median_caa_norm": median_caa_norm,
        "scale": scale,
        "score_weights": SCORE_WEIGHTS,
        "seed": args.seed,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    profile, _ = load_profile(args.config)
    prompts = load_neutral_prompts(n=args.n_prompts, seed=args.seed)

    # ------------------------------------------------------------------
    # Self-consistency
    # ------------------------------------------------------------------
    selfcons_path = out_dir / "selfcons_readouts.parquet"
    if args.skip_selfconsistency:
        if not selfcons_path.exists():
            raise SystemExit(
                f"--skip-selfconsistency but {selfcons_path} not found"
            )
        sc_df = pd.read_parquet(selfcons_path)
        print(f"[selfcons] reused {selfcons_path} ({len(sc_df)} rows)")
        model = None
        device = None
    else:
        model, device, _ = load_model(profile)
        sc_df = _run_selfconsistency(
            args, model, device, W, layer, components, self_alphas,
            prompts, apply_to, scale, median_caa_norm, out_dir,
        )

    sc_aggr = _aggregate_selfcons(sc_df, components, self_alphas)

    # ------------------------------------------------------------------
    # PrimitiveScore
    # ------------------------------------------------------------------
    score_rows: list[dict[str, Any]] = []
    for j in components:
        rec = dict(sc_aggr.get(j, {}))
        lm = label_metrics.get(j, {})
        rec["mi"] = lm.get("mi", float("nan"))
        rec["linear_sep_acc"] = lm.get("linear_sep_acc", float("nan"))
        rec["label_dominance"] = lm.get("category_top1_dominance",
                                        float("nan"))
        rec["vad_explained"] = lm.get("vad_explained", float("nan"))

        score, contribs = _primitive_score(rec)
        rec["primitive_score"] = score
        for key, val in contribs.items():
            rec[f"contrib_{key}"] = val
        rec["component"] = int(j)
        score_rows.append(rec)

    ranked = pd.DataFrame(score_rows).sort_values(
        "primitive_score", ascending=False
    ).reset_index(drop=True)

    csv_path = out_dir / "primitive_scores.csv"
    json_path = out_dir / "primitive_scores.json"
    ranked.to_csv(csv_path, index=False)
    ranked.to_json(json_path, orient="records", indent=2)
    print(f"[primitive] wrote {csv_path} and {json_path}")

    # ------------------------------------------------------------------
    # Strong-α qualitative generation for top candidates
    # ------------------------------------------------------------------
    top_components = [int(j) for j in ranked["component"].head(args.top_k)]
    gens_df: pd.DataFrame | None = None
    if not args.skip_generation:
        if model is None:
            model, device, _ = load_model(profile)
        gens_df = _run_strong_generation(
            args, model, W, layer, decomposer, top_components, strong_alphas,
            prompts, apply_to, scale, median_caa_norm, out_dir,
        )
    else:
        gen_path = out_dir / "strong_generations.parquet"
        if gen_path.exists():
            gens_df = pd.read_parquet(gen_path)

    _write_top_md(
        out_dir / "top_candidates.md",
        ranked, gens_df, strong_alphas,
        args.basis, decomposer, layer, k, args.top_k,
    )

    print("[primitive] done.")


if __name__ == "__main__":
    main()
