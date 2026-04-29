"""Qualitative inspection: print steered generations for selected basis components.

Generates text under α∈{−2, 0, +2} for each requested component of a basis
artifact and prints them side by side. Use this to *read* what an axis does.

Usage
-----
    uv run python -m experiments.eval_basis_qualitative \
        --basis data/emotion_code/basis_sweep/ica_k016_seed0.pt \
        --components 8 4 7 --alphas -2 0 2 --n-prompts 5 --max-new-tokens 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.activations._runtime import load_model, load_profile
from src.steering.generate import steered_generate
from experiments._gen_cache import load_neutral_prompts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--basis", type=Path, required=True)
    p.add_argument("--which", default=None)
    p.add_argument("--components", type=int, nargs="+", required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[-2.0, 0.0, 2.0])
    p.add_argument("--n-prompts", type=int, default=4)
    p.add_argument("--prompts-seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--no-repeat-ngram-size", type=int, default=0)
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--steering-config", type=Path, default=Path("configs/steering.yaml"))
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    payload = torch.load(args.basis, weights_only=False, map_location="cpu")
    which = args.which or payload.get("decomposer")
    if which is None:
        for cand in ("ica", "nmf", "pca", "dict"):
            if cand in payload:
                which = cand
                break
    W = payload[which]["W"].numpy()
    layer = int(payload["layer"])
    norms = np.linalg.norm(W, axis=1)
    scale = 1.0 / float(np.median(norms))

    sc = yaml.safe_load(args.steering_config.read_text())
    apply_to = sc["caa"].get("apply_to", "generation")

    profile, _ = load_profile(args.config)
    prompts = load_neutral_prompts(n=args.n_prompts, seed=args.prompts_seed)
    model, _, _ = load_model(profile)

    lines: list[str] = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"# Qualitative steering — basis={args.basis.name} which={which} "
         f"layer={layer}")
    emit(f"# components={args.components} alphas={args.alphas} "
         f"n_prompts={args.n_prompts} max_new_tokens={args.max_new_tokens}")
    emit("")

    for j in args.components:
        emit(f"\n{'='*88}")
        emit(f"=== component b{j}  ||b||={norms[j]:.3f}  layer={layer}")
        emit(f"{'='*88}")
        b_j_t = torch.from_numpy(W[j]).to(torch.float32)
        for pi, prompt in enumerate(prompts):
            emit(f"\n--- prompt {pi}: {prompt!r}")
            for alpha_unit in args.alphas:
                if alpha_unit == 0.0:
                    alpha = 0.0
                    vec = torch.zeros(W.shape[1])
                else:
                    alpha = float(alpha_unit) * scale * float(norms[j])
                    vec = b_j_t
                out = steered_generate(
                    model, prompt, vector=vec, alpha=alpha,
                    layers=[layer], apply_to=apply_to,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )
                # strip leading prompt if echoed
                tail = out[len(prompt):] if out.startswith(prompt) else out
                emit(f"  α={alpha_unit:+.1f} : {tail.strip()}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines))
        print(f"\n[qual] wrote {args.output}")


if __name__ == "__main__":
    main()
