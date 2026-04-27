"""Perplexity guardrail: largest α whose generation PPL ≤ ratio·baseline PPL.

We sweep α ∈ {0, 1, 2, 3, 4, 5} (in std-norm units) and measure perplexity
of the model's own generations on a fixed set of neutral prompts, per
category. The largest α whose ``PPL(α) / PPL(0) ≤ max_perplexity_ratio``
is recorded; this informs ``configs/steering.yaml::caa.alpha``.
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

from src.activations._runtime import load_model, load_profile
from src.steering.generate import steered_generate
from src.steering.hook import steering_hooks


@torch.inference_mode()
def _ppl(model, device, text: str) -> float:
    tokens = model.to_tokens(text, prepend_bos=True).to(device)
    if tokens.shape[1] < 2:
        return float("nan")
    logits = model(tokens)
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = tokens[:, 1:]
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(torch.exp(nll.mean()))


def _gen(model, prompt: str, vec, alpha, layer, max_new_tokens=64):
    return steered_generate(
        model, prompt, vec, alpha=alpha, layers=[layer],
        apply_to="generation", max_new_tokens=max_new_tokens, temperature=0.0,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--steering-config", type=Path, default=Path("configs/steering.yaml"))
    p.add_argument("--caa", type=Path, default=Path("data/emotion_code/caa.pt"))
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument("--alpha-units", type=float, nargs="+",
                   default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    p.add_argument("--output", type=Path,
                   default=Path("experiments/results/perplexity_alpha.csv"))
    args = p.parse_args()

    profile, _ = load_profile(args.config)
    sc = yaml.safe_load(args.steering_config.read_text())
    layer = int(sc["caa"]["inject_layers"][0])
    max_ratio = float(sc["caa"]["max_perplexity_ratio"])

    caa = torch.load(args.caa, map_location="cpu", weights_only=False)
    cats = caa["categories"]
    li = caa["layers"].index(layer)
    vectors = caa["vectors"]
    norms = vectors[:, li, :].norm(dim=-1)
    scale = float(1.0 / norms.median())

    from experiments._gen_cache import load_neutral_prompts
    prompts = load_neutral_prompts(n=args.n_prompts)

    model, device, _ = load_model(profile)

    rows = []
    for ci, cat in enumerate(cats):
        v = vectors[ci, li]
        for alpha_unit in args.alpha_units:
            alpha = alpha_unit * scale * float(norms[ci])
            ppls = []
            for prompt in tqdm(prompts, desc=f"[ppl] {cat:>12s} α={alpha_unit:+.1f}", leave=False):
                gen = _gen(model, prompt, v, alpha, layer)
                if gen.strip():
                    ppls.append(_ppl(model, device, gen))
            mean_ppl = float(np.nanmean(ppls)) if ppls else float("nan")
            rows.append({"category": cat, "alpha_unit": alpha_unit,
                         "alpha": alpha, "mean_ppl": mean_ppl})

    df = pd.DataFrame.from_records(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    # Per-category max α with ratio ≤ max_ratio
    chosen = []
    for cat, sub in df.groupby("category"):
        base = sub.loc[sub["alpha_unit"] == 0.0, "mean_ppl"].iloc[0]
        ok = sub[sub["mean_ppl"] / base <= max_ratio]
        amax = float(ok["alpha_unit"].max()) if len(ok) else 0.0
        chosen.append({"category": cat, "baseline_ppl": base,
                       "max_alpha_unit": amax})
    pd.DataFrame(chosen).to_csv(
        args.output.with_name("perplexity_chosen.csv"), index=False
    )
    args.output.with_suffix(".json").write_text(json.dumps(
        {"max_perplexity_ratio": max_ratio,
         "chosen": chosen,
         "median_max_alpha_unit": float(np.median([c["max_alpha_unit"] for c in chosen]))},
        indent=2,
    ))
    print(f"[ppl] -> {args.output}")


if __name__ == "__main__":
    main()
