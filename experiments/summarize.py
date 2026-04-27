"""Aggregate B-5 evaluation results into experiments/results/SUMMARY.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GO_BARS = {
    "shift_acc": 0.40,
    "monotonicity_rho": 0.70,
    "perplexity_ratio": 2.0,
    "vad_r2_min": 0.50,
}


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    p.add_argument("--output", type=Path, default=Path("experiments/results/SUMMARY.md"))
    args = p.parse_args()

    sweep = _load(args.results_dir / "layer_sweep.json")
    shift = _load(args.results_dir / "shift_accuracy.json")
    mono = _load(args.results_dir / "monotonicity.json")
    ppl = _load(args.results_dir / "perplexity_alpha.json")
    vad = _load(args.results_dir / "vad_r2.json")

    lines: list[str] = ["# Phase B-5 Evaluation Summary\n"]

    def _row(name, val, target, op=">="):
        if val is None:
            return f"| {name} | _missing_ | {op} {target} | ⚠️ |"
        ok = (val >= target) if op == ">=" else (val <= target)
        return f"| {name} | {val:.3f} | {op} {target} | {'✅' if ok else '❌'} |"

    lines.append("| Metric | Value | Target | Pass |")
    lines.append("|---|---|---|---|")
    lines.append(_row("Shift accuracy (mean, α=+2)",
                      shift and shift.get("mean_shift_acc"),
                      GO_BARS["shift_acc"]))
    lines.append(_row("Monotonicity ρ (min over cats)",
                      mono and mono.get("min_rho"),
                      GO_BARS["monotonicity_rho"]))
    lines.append(_row("Median max alpha_unit @ ratio≤2",
                      ppl and ppl.get("median_max_alpha_unit"),
                      1.0))
    lines.append(_row("VAD R² (min over V/A/D)",
                      vad and vad.get("min_r2"),
                      GO_BARS["vad_r2_min"]))

    if sweep:
        lines.append("\n## Layer sweep\n")
        lines.append(f"- best_layer: **{sweep['best_layer']}**")
        for k, v in sweep["per_layer"].items():
            lines.append(f"  - layer {k}: val_acc={v['val_acc']:.3f}")

    if shift:
        lines.append("\n## Shift accuracy (per category)\n")
        lines.append("| category | shift_acc | baseline | Δ | note |")
        lines.append("|---|---|---|---|---|")
        for r in shift["per_category"]:
            sa = "n/a" if r["shift_acc"] is None or r["shift_acc"] != r["shift_acc"] else f"{r['shift_acc']:.2f}"
            ba = "n/a" if r["baseline_acc"] is None or r["baseline_acc"] != r["baseline_acc"] else f"{r['baseline_acc']:.2f}"
            d = "n/a" if r["delta"] is None or r["delta"] != r["delta"] else f"{r['delta']:+.2f}"
            lines.append(f"| {r['category']} | {sa} | {ba} | {d} | {r.get('note','')} |")

    if mono:
        lines.append("\n## Monotonicity (Spearman ρ per category)\n")
        lines.append("| category | rho | p | n | note |")
        lines.append("|---|---|---|---|---|")
        for r in mono["per_category"]:
            rho = "n/a" if r.get("rho") is None or r.get("rho") != r.get("rho") else f"{r['rho']:.2f}"
            pv = "n/a" if r.get("p_value") is None or r.get("p_value") != r.get("p_value") else f"{r['p_value']:.2g}"
            lines.append(f"| {r['category']} | {rho} | {pv} | {r.get('n','')} | {r.get('note','')} |")

    if ppl:
        lines.append("\n## Perplexity guardrail\n")
        for r in ppl["chosen"]:
            lines.append(f"- {r['category']}: max α (≤×{ppl['max_perplexity_ratio']} baseline) = "
                         f"**{r['max_alpha_unit']:.1f}** (baseline PPL={r['baseline_ppl']:.2f})")

    if vad:
        lines.append("\n## VAD regression (held-out)\n")
        lines.append(f"- layer: {vad['layer']}")
        for ax, r2 in vad["r2"].items():
            ok = "✅" if r2 >= GO_BARS["vad_r2_min"] else "❌"
            lines.append(f"- {ax}: R² = **{r2:.3f}** {ok}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"[summary] wrote {args.output}")


if __name__ == "__main__":
    main()
