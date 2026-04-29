"""Encode-steer-re-encode self-consistency for basis components.

For each chosen component ``b_j`` and each scalar α we:
  1. Inject α · b_j at the basis layer during generation (greedy, no sampling).
  2. Re-encode the generated text and read its last-token residual at the
     same layer.
  3. Project that residual onto every basis vector ``b_l`` (cosine):
        ŝ_{l}(α; j) = ⟨h(gen) − h_baseline, b_l⟩ / (||·|| · ||b_l||)
     where the baseline is the α=0 generation for the same prompt.

Two derived metrics:
  • specificity_at_alpha :  ŝ_j(α) − mean_{l≠j} ŝ_l(α)
  • monotonicity_self    :  Spearman ρ between α and ŝ_j(α)  (per prompt, then
                            averaged)

This is the central "axis exists" test: if injecting α along b_j makes the
model's *own* readout of b_j move monotonically with α, while leaving other
b_l roughly untouched, then b_j functions as an axis — without ever invoking
external labels or VAD.

Usage
-----
    uv run python -m experiments.eval_basis_selfconsistency \
        --basis data/emotion_code/basis_sweep/ica_k016_seed0.pt \
        --components 13 2 8 \
        --alphas -2 -1 0 1 2 \
        --n-prompts 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from experiments._gen_cache import load_neutral_prompts
from src.activations._runtime import collect_batch, load_model, load_profile
from src.steering.generate import steered_generate


def _project_all(h: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Cosine projection of activation vectors h[N,D] onto each row of W[k,D]."""
    if h.ndim == 1:
        h = h[None, :]
    h_n = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    W_n = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    return h_n @ W_n.T


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--basis", type=Path, required=True)
    p.add_argument("--which", default=None)
    p.add_argument("--caa-layer", type=int, default=None,
                   help="When --basis points at caa.pt, pick which layer's vectors "
                        "to use as the pseudo-basis (default = middle).")
    p.add_argument("--components", type=int, nargs="+", default=None,
                   help="Component indices to test (default: all).")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[-2.0, -1.0, 0.0, 1.0, 2.0])
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--steering-config", type=Path, default=Path("configs/steering.yaml"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/results/basis_selfconsistency"))
    p.add_argument("--prompts-seed", type=int, default=0)
    args = p.parse_args()

    payload = torch.load(args.basis, weights_only=False, map_location="cpu")
    which = args.which or payload.get("decomposer")
    if which is None or which not in payload:
        for cand in ("nmf", "pca", "ica", "dict"):
            if cand in payload:
                which = cand; break

    if which is None and "vectors" in payload:
        # CAA artifact: vectors[C, L, D] + categories + layers. Treat as a basis
        # where each row is one category vector at the chosen layer.
        layers = list(payload["layers"])
        layer = args.caa_layer if args.caa_layer is not None else layers[len(layers) // 2]
        if layer not in layers:
            raise SystemExit(f"--caa-layer {layer} not in {layers}")
        li = layers.index(layer)
        W = payload["vectors"][:, li, :].numpy()              # [C, D]
        categories = list(payload["categories"])
        which = "caa"
        comp_labels = categories
    else:
        W = payload[which]["W"].numpy()                       # [k, D]
        layer = int(payload["layer"])
        comp_labels = [f"b{j}" for j in range(W.shape[0])]
    k = W.shape[0]
    components = args.components if args.components else list(range(k))
    print(f"[self] basis={args.basis.name} which={which} layer={layer} "
          f"k={k} components={[comp_labels[j] for j in components]}")

    sc = yaml.safe_load(args.steering_config.read_text())
    apply_to = sc["caa"].get("apply_to", "generation")

    # Steering scale: align with existing pipeline (1 / median ||b||).
    norms = np.linalg.norm(W, axis=1)
    scale = 1.0 / float(np.median(norms))
    print(f"[self] median ||b||={np.median(norms):.3f}  scale={scale:.5f}")

    profile, _ = load_profile(args.config)
    prompts = load_neutral_prompts(n=args.n_prompts, seed=args.prompts_seed)
    print(f"[self] {len(prompts)} prompts × {len(components)} components × "
          f"{len(args.alphas)} alphas = "
          f"{len(prompts) * len(components) * len(args.alphas)} generations")

    model, device, _ = load_model(profile)

    # 1. Baseline (α=0) generations & their re-encoded residuals.
    print("[self] baseline generations...")
    baseline_gen: list[str] = []
    for prompt in tqdm(prompts, desc="[self] baseline"):
        out = steered_generate(
            model, prompt, vector=torch.zeros(W.shape[1]),
            alpha=0.0, layers=[layer], apply_to=apply_to,
            max_new_tokens=args.max_new_tokens,
        )
        baseline_gen.append(out)
    baseline_h = collect_batch(model, baseline_gen, [layer], device)[layer].numpy()
    baseline_proj = _project_all(baseline_h, W)              # [N_prompt, k]

    rows: list[dict] = []
    for j in components:
        b_j_t = torch.from_numpy(W[j]).to(torch.float32)
        for alpha_unit in args.alphas:
            if alpha_unit == 0.0:
                gens = baseline_gen
            else:
                alpha = float(alpha_unit) * scale * float(norms[j])
                gens = []
                for prompt in tqdm(prompts, desc=f"[self] b{j} α={alpha_unit:+}",
                                   leave=False):
                    out = steered_generate(
                        model, prompt, vector=b_j_t, alpha=alpha,
                        layers=[layer], apply_to=apply_to,
                        max_new_tokens=args.max_new_tokens,
                    )
                    gens.append(out)
            h = collect_batch(model, gens, [layer], device)[layer].numpy()
            proj = _project_all(h, W)                         # [N_prompt, k]
            delta_proj = proj - baseline_proj                 # cosines diff
            for pi in range(len(prompts)):
                for l in range(k):
                    rows.append({
                        "component": j, "alpha_unit": alpha_unit,
                        "prompt_id": pi, "readout_component": l,
                        "delta_cosine": float(delta_proj[pi, l]),
                        "abs_cosine": float(proj[pi, l]),
                        "is_self": int(l == j),
                    })

    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"{args.basis.stem}__readouts.parquet"
    df.to_parquet(raw_path, index=False)

    # Aggregate per (component, alpha): self vs other readout.
    summary: list[dict] = []
    for j in components:
        sub = df[df["component"] == j]
        for alpha_unit in sorted(sub["alpha_unit"].unique()):
            s = sub[sub["alpha_unit"] == alpha_unit]
            self_mean = float(s.loc[s["is_self"] == 1, "delta_cosine"].mean())
            other_mean = float(s.loc[s["is_self"] == 0, "delta_cosine"].mean())
            other_abs = float(s.loc[s["is_self"] == 0, "delta_cosine"].abs().mean())
            summary.append({
                "component": j, "alpha_unit": alpha_unit,
                "self_delta_cosine": self_mean,
                "other_delta_cosine_mean": other_mean,
                "other_delta_cosine_absmean": other_abs,
                "specificity": self_mean - other_abs,
            })

    # Monotonicity per component: ρ(α, self_delta) per prompt, averaged.
    mono_rows: list[dict] = []
    for j in components:
        sub = df[(df["component"] == j) & (df["is_self"] == 1)]
        per_prompt = []
        for pi in sorted(sub["prompt_id"].unique()):
            s = sub[sub["prompt_id"] == pi].sort_values("alpha_unit")
            if len(s) < 3:
                continue
            r, _ = spearmanr(s["alpha_unit"], s["delta_cosine"])
            if not np.isnan(r):
                per_prompt.append(r)
        mono_rows.append({
            "component": j,
            "spearman_self_mean": float(np.mean(per_prompt)) if per_prompt else float("nan"),
            "spearman_self_min":  float(np.min(per_prompt))  if per_prompt else float("nan"),
            "n_prompts": len(per_prompt),
        })

    summary_path = args.output_dir / f"{args.basis.stem}__summary.csv"
    mono_path    = args.output_dir / f"{args.basis.stem}__monotonicity.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    pd.DataFrame(mono_rows).to_csv(mono_path, index=False)

    overall = {
        "basis": str(args.basis),
        "which": which, "layer": layer, "k": k,
        "components": components,
        "alphas": args.alphas,
        "n_prompts": len(prompts),
        "monotonicity_mean": float(np.nanmean([r["spearman_self_mean"] for r in mono_rows])),
        "specificity_at_extreme": float(np.mean([
            r["specificity"] for r in summary
            if r["alpha_unit"] in (max(args.alphas), min(args.alphas))
        ])),
    }
    (args.output_dir / f"{args.basis.stem}__overall.json").write_text(
        json.dumps(overall, indent=2)
    )
    print(f"[self] wrote {raw_path}, {summary_path}, {mono_path}")
    print(f"[self] OVERALL  monotonicity_mean={overall['monotonicity_mean']:.3f}  "
          f"specificity_at_extreme={overall['specificity_at_extreme']:.3f}")


if __name__ == "__main__":
    main()
