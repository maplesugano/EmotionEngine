"""Estimate ΔVAD induced by each basis component from threshold_rewrites texts.

Approach A: for each of the 64 ICA components, take the already-generated
``generation_positive`` / ``generation_negative`` texts saved in
``experiments/results/threshold_rewrites/ica_k064_L22/b{i:02d}.json``,
run a forward pass through the model, extract the last-token residual at the
VAD-mapping layer, and apply the linear map W·h + b to obtain a VAD triple.

    ΔVAD[component, prompt] = VAD(generation_positive) - VAD(generation_negative)

Then average across all 64 components and all prompts.

Outputs (under ``experiments/results/threshold_vad/``):
  per_component_prompt.parquet — one row per (component, prompt_id, axis)
  component_means.csv          — per-component mean ΔVAD across prompts
  summary.json                 — grand-mean and std of ΔVAD per axis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.activations._runtime import collect_batch, load_model, load_profile

_REWRITES_DIR = Path("experiments/results/threshold_rewrites/ica_k064_L22")
_DEFAULT_VAD = Path("data/emotion_code/vad_mapping.pt")
_DEFAULT_OUT = Path("experiments/results/threshold_vad")
_AXES = ["V", "A", "D"]


def _vad_from_residuals(h: torch.Tensor, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """h: [N, d_model] float32 numpy → returns [N, 3] VAD scores."""
    return h @ W.T + b  # [N, 3]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rewrites-dir", type=Path, default=_REWRITES_DIR)
    p.add_argument("--vad", type=Path, default=_DEFAULT_VAD)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--batch-size", type=int, default=16,
                   help="Number of texts per forward-pass batch")
    args = p.parse_args()

    # ── Load VAD mapping ─────────────────────────────────────────────────────
    vad_mp = torch.load(args.vad, map_location="cpu", weights_only=False)
    W_vad: np.ndarray = vad_mp["W"].numpy().astype(np.float32)   # [3, d_model]
    b_vad: np.ndarray = vad_mp["b"].numpy().astype(np.float32)   # [3]
    vad_layer: int = int(vad_mp["layer"])
    print(f"[threshold_vad] VAD layer={vad_layer}  "
          f"R²={vad_mp['r2']}")

    # ── Collect all texts ────────────────────────────────────────────────────
    json_files = sorted(args.rewrites_dir.glob("b??.json"))
    if not json_files:
        raise FileNotFoundError(f"No b??.json found in {args.rewrites_dir}")
    print(f"[threshold_vad] Found {len(json_files)} component files")

    records: list[dict] = []   # (component, prompt_id, pos_text, neg_text)
    for jf in json_files:
        data = json.loads(jf.read_text())
        comp = int(data["component"])
        for entry in data["prompts"]:
            records.append({
                "component": comp,
                "prompt_id": int(entry["prompt_id"]),
                "text_pos": entry["generation_positive"],
                "text_neg": entry["generation_negative"],
            })

    n_total = len(records)
    print(f"[threshold_vad] {n_total} (component, prompt) pairs")

    # ── Load model ───────────────────────────────────────────────────────────
    profile, _ = load_profile(args.config)
    model, device, _ = load_model(profile)

    # ── Forward passes in batches ────────────────────────────────────────────
    # We interleave pos/neg so each batch of size B has B/2 pos + B/2 neg
    # from the same block of records.
    rows: list[dict] = []
    bs = args.batch_size
    for start in tqdm(range(0, n_total, bs // 2), desc="batches"):
        chunk = records[start : start + bs // 2]
        texts_pos = [r["text_pos"] for r in chunk]
        texts_neg = [r["text_neg"] for r in chunk]
        texts_all = texts_pos + texts_neg   # pos first, then neg

        caps = collect_batch(model, texts_all, hook_layers=[vad_layer], device=device)
        H = caps[vad_layer].numpy()         # [2*len(chunk), d_model]

        vad_all = _vad_from_residuals(H, W_vad, b_vad)  # [2*len(chunk), 3]
        n = len(chunk)
        vad_pos = vad_all[:n]   # [n, 3]
        vad_neg = vad_all[n:]   # [n, 3]
        delta = vad_pos - vad_neg               # [n, 3]

        for i, r in enumerate(chunk):
            for j, ax in enumerate(_AXES):
                rows.append({
                    "component": r["component"],
                    "prompt_id": r["prompt_id"],
                    "axis": ax,
                    "vad_pos": float(vad_pos[i, j]),
                    "vad_neg": float(vad_neg[i, j]),
                    "delta": float(delta[i, j]),
                })

    # ── Save results ─────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(args.output_dir / "per_component_prompt.parquet", index=False)

    # Per-component mean Δ per axis
    comp_means = (
        df.groupby(["component", "axis"])["delta"]
        .mean()
        .unstack("axis")
        .reset_index()
    )
    comp_means.columns.name = None
    comp_means.to_csv(args.output_dir / "component_means.csv", index=False)

    # Grand summary across all 64 components
    summary: dict = {"vad_layer": vad_layer, "n_components": len(json_files),
                     "n_prompts_per_component": n_total // len(json_files)}
    for ax in _AXES:
        sub = df[df["axis"] == ax]["delta"]
        summary[f"mean_delta_{ax}"] = float(sub.mean())
        summary[f"std_delta_{ax}"] = float(sub.std())
        summary[f"mean_abs_delta_{ax}"] = float(sub.abs().mean())
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n[threshold_vad] Grand-mean ΔVAD (averaged over 64 components × prompts):")
    for ax in _AXES:
        print(f"  Δ{ax} = {summary[f'mean_delta_{ax}']:+.4f}  "
              f"(|Δ| = {summary[f'mean_abs_delta_{ax}']:.4f})")
    print(f"\n[threshold_vad] Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
