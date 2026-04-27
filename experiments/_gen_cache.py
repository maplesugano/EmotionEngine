"""Shared neutral-prompt set and steered-generation cache for B-5 evals.

Generates one (category × alpha) generation for every neutral prompt and
caches results to ``experiments/results/_gen_cache.parquet`` so multiple
evaluation scripts (shift accuracy, monotonicity) can share the same
expensive generation pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from src.activations._runtime import load_model, load_profile
from src.steering.generate import steered_generate


def load_neutral_prompts(
    unified_path: Path = Path("data/unified/examples.filtered.parquet"),
    n: int = 50,
    seed: int = 0,
    min_chars: int = 20,
    max_chars: int = 200,
) -> list[str]:
    df = pd.read_parquet(unified_path)
    df = df[df["label_primary"] == "neutral"]
    df = df[df["text"].str.len().between(min_chars, max_chars)]
    df = df[df["source"] == "daily_dialog"]
    df = df.sample(n=n, random_state=seed)
    return df["text"].tolist()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--steering-config", type=Path, default=Path("configs/steering.yaml"))
    p.add_argument("--caa", type=Path, default=Path("data/emotion_code/caa.pt"))
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument(
        "--alphas", type=float, nargs="+",
        default=[-2.0, -1.0, 0.0, 1.0, 2.0],
        help="Multipliers (in units of vector L2 norm; see --normalize).",
    )
    p.add_argument(
        "--alpha-scale", type=float, default=None,
        help="If set, multiplies each alpha by this scalar (raw weight). "
             "Default: scale = 1 / mean ||v|| so alpha is in 'std-norm' units.",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--output", type=Path,
                   default=Path("experiments/results/_gen_cache.parquet"))
    args = p.parse_args()

    profile, prof_name = load_profile(args.config)
    sc = yaml.safe_load(args.steering_config.read_text())
    inject_layers = list(sc["caa"]["inject_layers"])
    apply_to = sc["caa"].get("apply_to", "generation")

    caa = torch.load(args.caa, map_location="cpu", weights_only=False)
    cats: list[str] = caa["categories"]
    layers: list[int] = caa["layers"]
    vectors: torch.Tensor = caa["vectors"]  # [C, L, D]

    # Pick the row of `vectors` matching the inject_layers (use first if list).
    layer = inject_layers[0]
    if layer not in layers:
        raise ValueError(
            f"inject_layers[0]={layer} not in caa layers {layers}; "
            "rerun build_caa or update steering.yaml."
        )
    li = layers.index(layer)

    # Auto-scale: median ||v||  (so alpha=1 ≈ one std of CAA direction).
    norms = vectors[:, li, :].norm(dim=-1)
    scale = args.alpha_scale if args.alpha_scale is not None else float(1.0 / norms.median())
    print(f"[gen] inject layer={layer}  median ||v||={norms.median():.2f}  scale={scale:.5f}")

    prompts = load_neutral_prompts(n=args.n_prompts)
    print(f"[gen] {len(prompts)} prompts × {len(cats)} cats × {len(args.alphas)} alphas "
          f"= {len(prompts) * len(cats) * len(args.alphas)} generations")

    model, _device, _ = load_model(profile)

    rows = []
    for ci, cat in enumerate(cats):
        v = vectors[ci, li]
        for alpha_unit in args.alphas:
            alpha = alpha_unit * scale * float(norms[ci])
            for pi, prompt in enumerate(tqdm(
                prompts, desc=f"[gen] {cat:>12s} α={alpha_unit:+.1f}", leave=False,
            )):
                if alpha_unit == 0.0:
                    # Baseline: skip hook entirely.
                    out = steered_generate(
                        model, prompt, v, alpha=0.0, layers=[layer],
                        apply_to=apply_to, max_new_tokens=args.max_new_tokens,
                    )
                else:
                    out = steered_generate(
                        model, prompt, v, alpha=alpha, layers=[layer],
                        apply_to=apply_to, max_new_tokens=args.max_new_tokens,
                    )
                rows.append({
                    "category": cat, "alpha_unit": alpha_unit, "alpha": alpha,
                    "layer": layer, "prompt_id": pi, "prompt": prompt,
                    "generation": out,
                })

    df = pd.DataFrame.from_records(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[gen] wrote {args.output}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
