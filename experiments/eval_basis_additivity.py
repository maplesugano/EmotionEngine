"""Additivity test for basis components.

For a pair of components (i, j) and grid (α, β) ∈ alphas × alphas, generate
text under the joint steering ``α·b_i + β·b_j`` and re-encode the residual
into k cosine readouts ``r(α, β) ∈ R^k``.

If the basis is acting linearly, the joint readout should equal the sum of
the marginal effects:

    predicted(α, β) = r(α, 0) + r(0, β) − r(0, 0)

We report:
  • per-component (i and j) self-readout: actual vs predicted
  • full-vector residual norm and its ratio to the predicted norm
  • per-cell rows in a parquet for downstream plotting

Usage
-----
    uv run python -m experiments.eval_basis_additivity \
        --basis data/emotion_code/basis_sweep/ica_k016_seed0.pt \
        --pairs 8,4 8,7 4,7 \
        --alphas -2 0 2 --n-prompts 4 --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.activations._runtime import collect_batch, load_model, load_profile
from src.steering.generate import steered_generate
from experiments._gen_cache import load_neutral_prompts


def _project_all(h: np.ndarray, W: np.ndarray) -> np.ndarray:
    h_n = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    w_n = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    return h_n @ w_n.T


def _parse_pair(s: str) -> tuple[int, int]:
    a, b = s.split(",")
    return int(a), int(b)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--basis", type=Path, required=True)
    p.add_argument("--which", default=None)
    p.add_argument("--pairs", type=_parse_pair, nargs="+", required=True,
                   help="Component index pairs as i,j (space-separated).")
    p.add_argument("--alphas", type=float, nargs="+", default=[-2.0, 0.0, 2.0])
    p.add_argument("--n-prompts", type=int, default=4)
    p.add_argument("--prompts-seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--steering-config", type=Path, default=Path("configs/steering.yaml"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/results/basis_additivity"))
    args = p.parse_args()

    payload = torch.load(args.basis, weights_only=False, map_location="cpu")
    which = args.which or payload.get("decomposer")
    if which is None:
        for cand in ("ica", "nmf", "pca", "dict"):
            if cand in payload:
                which = cand
                break
    W = payload[which]["W"].numpy().astype(np.float32)
    layer = int(payload["layer"])
    norms = np.linalg.norm(W, axis=1)
    scale = 1.0 / float(np.median(norms))
    print(f"[add] basis={args.basis.name} which={which} layer={layer} "
          f"k={W.shape[0]} pairs={args.pairs}")
    print(f"[add] median ||b||={np.median(norms):.3f}  scale={scale:.5f}")

    sc = yaml.safe_load(args.steering_config.read_text())
    apply_to = sc["caa"].get("apply_to", "generation")

    profile, _ = load_profile(args.config)
    prompts = load_neutral_prompts(n=args.n_prompts, seed=args.prompts_seed)
    model, device, _ = load_model(profile)

    # Cache encoded readouts r[(i,j,α,β)] of shape (N_prompt, k).
    # We use (i,j,α,β) keying; for marginal cells β=0 or α=0 we still encode
    # because the joint cell (α,β) pair-specific noise must be matched.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    summary: list[dict] = []

    for (i, j) in args.pairs:
        b_i = torch.from_numpy(W[i]).to(torch.float32)
        b_j = torch.from_numpy(W[j]).to(torch.float32)
        # Generate cells r[α,β] for α,β in alphas
        readouts: dict[tuple[float, float], np.ndarray] = {}
        for a in args.alphas:
            for b in args.alphas:
                alpha_i = float(a) * scale * float(norms[i])
                alpha_j = float(b) * scale * float(norms[j])
                # Build joint vector: α·b_i + β·b_j with alpha=1.0 (we fold scale in).
                vec = alpha_i * b_i + alpha_j * b_j
                if torch.allclose(vec, torch.zeros_like(vec)):
                    vec_use = torch.zeros_like(b_i)
                    alpha_use = 0.0
                else:
                    vec_use = vec
                    alpha_use = 1.0
                gens = []
                for prompt in tqdm(prompts,
                                   desc=f"[add] (b{i},b{j}) α={a:+} β={b:+}",
                                   leave=False):
                    out = steered_generate(
                        model, prompt, vector=vec_use, alpha=alpha_use,
                        layers=[layer], apply_to=apply_to,
                        max_new_tokens=args.max_new_tokens,
                    )
                    gens.append(out)
                h = collect_batch(model, gens, [layer], device)[layer].numpy()
                proj = _project_all(h, W)                   # [N, k]
                readouts[(a, b)] = proj

        # baseline cell
        r00 = readouts[(0.0, 0.0)]
        for a in args.alphas:
            for b in args.alphas:
                actual = readouts[(a, b)]                   # [N, k]
                marg_a = readouts[(a, 0.0)]
                marg_b = readouts[(0.0, b)]
                predicted = marg_a + marg_b - r00           # additive null
                resid = actual - predicted
                # per-prompt norms
                num = np.linalg.norm(resid, axis=1)
                den = np.linalg.norm(predicted - r00, axis=1) + 1e-12
                ratio = num / den
                # self readouts at i and j
                for pi in range(actual.shape[0]):
                    rows.append({
                        "i": i, "j": j, "alpha": a, "beta": b, "prompt_id": pi,
                        "actual_i": float(actual[pi, i]),
                        "actual_j": float(actual[pi, j]),
                        "pred_i":   float(predicted[pi, i]),
                        "pred_j":   float(predicted[pi, j]),
                        "resid_norm": float(num[pi]),
                        "marg_norm":  float(den[pi]),
                        "resid_ratio": float(ratio[pi]),
                    })
                # cell summary (skip cells where α=0 and β=0 trivially additive)
                if a == 0.0 and b == 0.0:
                    continue
                summary.append({
                    "i": i, "j": j, "alpha": a, "beta": b,
                    "mean_resid_ratio": float(np.mean(ratio)),
                    "median_resid_ratio": float(np.median(ratio)),
                    "mean_actual_i": float(np.mean(actual[:, i])),
                    "mean_pred_i":   float(np.mean(predicted[:, i])),
                    "mean_actual_j": float(np.mean(actual[:, j])),
                    "mean_pred_j":   float(np.mean(predicted[:, j])),
                })

    df = pd.DataFrame(rows)
    sm = pd.DataFrame(summary)
    raw_path = args.output_dir / f"{args.basis.stem}__additivity.parquet"
    sum_path = args.output_dir / f"{args.basis.stem}__additivity_summary.csv"
    df.to_parquet(raw_path, index=False)
    sm.to_csv(sum_path, index=False)

    overall = {
        "n_pairs": len(args.pairs),
        "n_alphas": len(args.alphas),
        "n_prompts": args.n_prompts,
        "median_resid_ratio_offdiag": float(
            sm[(sm.alpha != 0) & (sm.beta != 0)]["mean_resid_ratio"].median()
        ) if not sm.empty else None,
    }
    (args.output_dir / f"{args.basis.stem}__additivity_overall.json").write_text(
        json.dumps(overall, indent=2)
    )
    print(f"[add] wrote {raw_path}, {sum_path}")
    print(f"[add] OVERALL  median resid_ratio (off-diag cells) = "
          f"{overall['median_resid_ratio_offdiag']}")


if __name__ == "__main__":
    main()
