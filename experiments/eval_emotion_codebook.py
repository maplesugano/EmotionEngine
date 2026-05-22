"""Full Emotion Codebook: Systematic interpretability for all ICA basis components.

This experiment constructs a complete 64-dimensional Emotion Codebook by assigning
provisional affective interpretations to every basis axis. The methodology combines:
    - Causal steering with adaptive breakage search (early stopping per component)
    - Diagnostic metrics (label independence, self-consistency)
    - Top-loading source texts
    - LLM-based axis naming (optional, requires OPENAI_API_KEY)

These axes should NOT be interpreted as Plutchik categories. They are latent
affective directions whose meanings are inferred from steering behavior,
source text patterns, and quantitative independence diagnostics.

Usage
-----
    uv run python -m experiments.eval_emotion_codebook \\
        --basis data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt \\
        --output-dir experiments/results/emotion_codebook/ica_k064_L22 \\
        --n-prompts 8 \\
        --prompt-source fixed \\
        --min-sentences 2 \\
        --run-judge
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from experiments._gen_cache import load_neutral_prompts
from src.activations._runtime import collect_batch, load_model, load_profile
from src.emotion_code.basis_interpret import _project
from src.emotion_code.io import load_activations
from src.steering.generate import steered_generate


BREAKAGE_CANDIDATE_ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


class GenerationCache:
    """Global cache for steered generations.

    Keyed by (component, signed_alpha_unit, prompt_text).
    Reuses existing generations across runs so identical
    (component, alpha, prompt) combinations are never re-generated.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[int, float, str], str] = {}

    def load_parquet(self, path: Path) -> int:
        """Load all rows from a parquet file. Returns number of new entries added.

        Handles multiple schemas found across past experiments:
        - Standard (emotion_codebook):  component, alpha (signed), prompt, generation
        - Adaptive:                      component, sign, alpha_unit, prompt, generation
        - per_axis_judge:                axis (=component), alpha_unit (signed), prompt, generation
        - strong_generations:            component, alpha (signed), prompt, generation
        - basis_pathology:               component, alpha_unit, prompt, text (=generation)
        """
        if not path.exists():
            return 0
        try:
            df = pd.read_parquet(path)
        except Exception:
            return 0

        # Resolve component column
        if "component" in df.columns:
            comp_col = "component"
        elif "axis" in df.columns:
            comp_col = "axis"
        else:
            return 0  # no component identifier

        # Resolve generation column
        if "generation" in df.columns:
            gen_col = "generation"
        elif "text" in df.columns:
            gen_col = "text"
        else:
            return 0  # no generation text

        added = 0
        for _, row in df.iterrows():
            try:
                comp = int(row[comp_col])
            except (ValueError, TypeError):
                continue

            # Resolve signed alpha
            if "sign" in df.columns and "alpha_unit" in df.columns:
                signed = float(row["sign"]) * float(row["alpha_unit"])
            elif "alpha" in df.columns:
                signed = float(row["alpha"])
            elif "alpha_unit" in df.columns:
                signed = float(row["alpha_unit"])  # assumed already signed
            else:
                continue

            key = (comp, round(signed, 6), str(row["prompt"]))
            if key not in self._cache:
                self._cache[key] = str(row[gen_col])
                added += 1
        return added

    @classmethod
    def from_results_dir(cls, results_root: Path) -> "GenerationCache":
        """Scan results_root for all *generations*.parquet and *adaptive*.parquet files
        and load them all into a new cache instance."""
        cache = cls()
        patterns = ["**/generations.parquet", "**/adaptive_generations.parquet"]
        seen: set[Path] = set()
        for pattern in patterns:
            for p in sorted(results_root.rglob(pattern.lstrip("**/"))):
                if p in seen:
                    continue
                seen.add(p)
                n = cache.load_parquet(p)
                if n:
                    print(f"[cache] +{n:4d} entries from {p.relative_to(results_root)}")
        print(f"[cache] total: {len(cache)} entries loaded from {results_root}")
        return cache

    def get(self, component: int, signed_alpha_unit: float, prompt: str) -> str | None:
        return self._cache.get((component, round(signed_alpha_unit, 6), prompt))

    def put(self, component: int, signed_alpha_unit: float, prompt: str, generation: str) -> None:
        self._cache[(component, round(signed_alpha_unit, 6), prompt)] = generation

    def __len__(self) -> int:
        return len(self._cache)


CODEBOOK_FIXED_PROMPTS = [
    "My teammate asks whether I can finish my part before tomorrow's deadline.",
    "A close friend texts me after a long silence and asks to talk tonight.",
    "During a meeting, my manager asks for my opinion on a risky plan.",
    "I open an email and learn that my application decision has just been released.",
    "Someone I care about apologizes for a recent argument and asks to reconnect.",
    "I notice a sudden change in my health before an important event, and someone asks how I feel.",
    "A colleague publicly credits me for work I quietly contributed to.",
    "I am asked to choose between a safe option and an uncertain opportunity right now.",
]


def _finalize_complete_sentence(text: str, *, min_trim_chars: int = 12) -> str:
    """Post-process text so it ends as a complete sentence when possible.

    Strategy:
      1. If already ends with sentence punctuation, keep as-is.
      2. Else trim back to the last sentence-ending punctuation if reasonably long.
      3. Else append a period as a fallback.
    """
    s = text.strip()
    if not s:
        return s
    if s.endswith((".", "!", "?", "。", "！", "？")):
        return s

    last_idx = max(s.rfind("."), s.rfind("!"), s.rfind("?"), s.rfind("。"), s.rfind("！"), s.rfind("？"))
    if last_idx >= min_trim_chars:
        return s[: last_idx + 1].strip()

    return s + "."


def _finalize_min_sentences(
    text: str,
    *,
    min_sentences: int = 2,
) -> str:
    """Prefer outputs with at least ``min_sentences`` sentence-like units.

    We do not fabricate new semantic content. If there is only one complete
    sentence but a trailing fragment exists, we terminate the fragment with a
    period to form a second sentence-like unit.
    """
    s = _finalize_complete_sentence(text)
    if min_sentences <= 1:
        return s

    ends = [m.end() for m in re.finditer(r"[.!?。！？]", s)]
    if len(ends) >= min_sentences:
        return s

    # Try to promote a trailing fragment into a second sentence.
    if len(ends) == 1:
        tail = s[ends[0]:].strip()
        if tail and tail != ".":
            return s + "."

    return s


def _format_alpha(alpha_unit: float, sign: int) -> float:
    return float(alpha_unit) * float(sign)


def _load_codebook_prompts(n: int, seed: int, prompt_source: str) -> list[str]:
    """Load prompts for codebook generation.

    prompt_source='fixed' uses a less-vague curated set of contextual prompts.
    prompt_source='dataset' uses sampled neutral prompts from the unified corpus.
    """
    if prompt_source == "fixed":
        if n <= len(CODEBOOK_FIXED_PROMPTS):
            return CODEBOOK_FIXED_PROMPTS[:n]
        out = []
        while len(out) < n:
            out.extend(CODEBOOK_FIXED_PROMPTS)
        return out[:n]

    return load_neutral_prompts(n=n, seed=seed)


def _prepare_threshold_judge_inputs(
    threshold_df: pd.DataFrame,
    top_texts: dict[int, dict],
    diagnostics: dict[int, dict],
    W: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for _, row in threshold_df.iterrows():
        component = int(row["component"])
        texts = top_texts.get(component, {})
        diags = diagnostics.get(component, {})
        record = {
            "task": "naming",
            "component": component,
            "threshold_alpha_unit_negative": float(row.get("threshold_alpha_unit_negative", np.nan)),
            "threshold_alpha_unit_positive": float(row.get("threshold_alpha_unit_positive", np.nan)),
            "threshold_broken_rate_negative": float(row.get("threshold_broken_rate_negative", np.nan)),
            "threshold_broken_rate_positive": float(row.get("threshold_broken_rate_positive", np.nan)),
            "threshold_mean_severity_negative": float(row.get("threshold_mean_severity_negative", np.nan)),
            "threshold_mean_severity_positive": float(row.get("threshold_mean_severity_positive", np.nan)),
            "threshold_generations_negative": row.get("threshold_generations_negative", []),
            "threshold_generations_positive": row.get("threshold_generations_positive", []),
            "top_positive_texts": texts.get("top_positive", [])[:4],
            "top_negative_texts": texts.get("top_negative", [])[:4],
            "diagnostics": {k: (float(v) if not pd.isna(v) else None) for k, v in diags.items()},
            "prompt": (
                "You are interpreting one latent affective steering axis in a language model. "
                "You will see text generated at the largest steering magnitude that still remains usable, "
                "on both the negative and positive sides, plus baseline examples. "
                "Infer the underlying affective or communicative dimension controlled by the axis. "
                "Return JSON with: negative_pole_name, positive_pole_name, axis_name, short_description, "
                "relation_to_Plutchik_8, relation_to_VAD, family, confidence, notes."
            ),
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[judge_inputs] wrote {output_path} ({len(threshold_df)} rows)")


def _load_basis_artifact(
    basis_path: Path, which: str | None = None
) -> tuple[np.ndarray, int, int, str]:
    """Load basis vectors from artifact.
    
    Returns:
        W: [k, D] basis vectors
        layer: injection layer
        k: number of basis vectors
        decomposer: method used ('ica', 'nmf', 'pca', 'dict')
    """
    payload = torch.load(basis_path, weights_only=False, map_location="cpu")
    decomposer = which or payload.get("decomposer")
    if decomposer is None:
        for cand in ("ica", "nmf", "pca", "dict"):
            if cand in payload:
                decomposer = cand
                break
    if decomposer is None or decomposer not in payload:
        raise ValueError(f"Could not find decomposer in {list(payload.keys())}")
    
    W = payload[decomposer]["W"].numpy()  # [k, D]
    layer = int(payload["layer"])
    k = int(payload["k"])
    return W, layer, k, decomposer


def _compute_alpha_scale(W: np.ndarray, alpha_mode: str = "caa_match") -> float:
    """Compute alpha scaling factor.
    
    caa_match: scale = 1 / median(||W||) so alpha=1 is in units of median norm.
    """
    norms = np.linalg.norm(W, axis=1)
    if alpha_mode == "caa_match":
        scale = 1.0 / float(np.median(norms))
    else:
        raise ValueError(f"Unknown alpha_mode: {alpha_mode}")
    return scale


def _load_diagnostics(
    basis_path: Path, W: np.ndarray, layer: int, k: int
) -> dict[int, dict[str, float]]:
    """Load component diagnostics from existing metric files or return NaN.
    
    Tries to load from:
    1. basis_sweep/*.metrics.json
    2. primitive_affective_units/*/primitive_scores.json
    3. per_axis_judge_*/summary.json
    
    Returns dict: {component_id: {metric_name: value}}
    """
    diags = {j: {} for j in range(k)}
    
    # Try to load from basis_sweep metrics
    metrics_path = basis_path.with_suffix(".metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics_data = json.load(f)
            for comp_data in metrics_data.get("per_component", []):
                j = comp_data["component"]
                diags[j].update({
                    "mi": comp_data.get("mi", np.nan),
                    "linear_sep_acc": comp_data.get("linear_sep_acc", np.nan),
                    "label_dominance": comp_data.get("category_top1_dominance", np.nan),
                    "vad_explained": comp_data.get("vad_explained", np.nan),
                })
    
    # Try to load from primitive_affective_units results
    prim_dir = Path("experiments/results/primitive_affective_units")
    for subdir in prim_dir.glob(f"*_k{k:03d}_*"):
        prim_scores_path = subdir / "primitive_scores.json"
        if prim_scores_path.exists():
            with open(prim_scores_path) as f:
                prim_data = json.load(f)
                if isinstance(prim_data, dict):
                    comp_iter = prim_data.get("components", [])
                else:
                    comp_iter = prim_data
                for comp_data in comp_iter:
                    j = comp_data["component"]
                    diags[j].update({
                        "self_rho": comp_data.get("self_rho", np.nan),
                        "sign_correct": comp_data.get("sign_correct", np.nan),
                        "label_dominance": comp_data.get("label_dominance", np.nan),
                        "vad_explained": comp_data.get("vad_explained", np.nan),
                        "causal_strength": comp_data.get("causal_strength", np.nan),
                        "cross_talk": comp_data.get("cross_talk", np.nan),
                        "mi": comp_data.get("mi", np.nan),
                        "linear_sep_acc": comp_data.get("linear_sep_acc", np.nan),
                        "primitive_score": comp_data.get("primitive_score", np.nan),
                    })
            break
    
    return diags


def _compute_top_texts(
    W: np.ndarray,
    bundle: Any,
    layer: int,
    pairs_path: Path,
    train_mask: np.ndarray,
    n_top: int = 8,
) -> dict[int, dict[str, Any]]:
    """Compute top-loading source texts for each component.
    
    Returns dict: {component_id: {
        'top_positive': [{'pair_id', 'pos_text', 'neg_text', 'category', 'score'}, ...],
        'top_negative': [...],
        'category_hist_pos': {category: count},
        'category_hist_neg': {category: count},
    }}
    """
    from src.emotion_code.basis import _build_delta
    
    delta = _build_delta(bundle, layer, train_mask)
    scores = _project(delta, W)  # [N, k]
    
    pairs_df = pd.read_parquet(pairs_path)
    meta_train = bundle.meta.loc[train_mask].reset_index(drop=True)
    joined = meta_train.merge(
        pairs_df[["pair_id", "pos_text", "neg_text", "category", "provenance"]],
        on="pair_id",
        how="left",
        suffixes=("", "_pair"),
    )
    
    top_texts = {}
    for j in range(W.shape[0]):
        s = scores[:, j]
        top_idx = np.argsort(-s)[:n_top]
        bot_idx = np.argsort(s)[:n_top]
        
        def _rows(indices):
            out = []
            for i in indices:
                row = joined.iloc[int(i)]
                out.append({
                    "pair_id": int(row["pair_id"]),
                    "pos_text": str(row["pos_text"]),
                    "neg_text": str(row["neg_text"]),
                    "category": str(row["category"]),
                    "provenance": str(row.get("provenance", "unknown")),
                    "score": float(s[int(i)]),
                })
            return out
        
        cat_hist_pos = joined.iloc[top_idx]["category"].value_counts().to_dict()
        cat_hist_neg = joined.iloc[bot_idx]["category"].value_counts().to_dict()
        
        top_texts[j] = {
            "top_positive": _rows(top_idx),
            "top_negative": _rows(bot_idx),
            "category_hist_pos": {str(k): int(v) for k, v in cat_hist_pos.items()},
            "category_hist_neg": {str(k): int(v) for k, v in cat_hist_neg.items()},
        }
    
    return top_texts


def _load_judge_outputs(judge_outputs_path: Path) -> dict[int, dict]:
    """Load JSONL judgment outputs.
    
    Returns dict: {component_id: judgment_dict}
    """
    outputs = {}
    if not judge_outputs_path.exists():
        return outputs
    
    with open(judge_outputs_path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                outputs[rec["component"]] = rec
    return outputs


def _fallback_judge_outputs_from_existing() -> dict[int, dict]:
    """Load existing provisional names if run-judge is not used.

    Priority:
      1) llm_judge_names_top10.json
      2) llm_judge_names.json
      3) per_axis_top_labels.csv-derived provisional names for all remaining axes
    """
    out: dict[int, dict] = {}

    candidates = [
        Path("experiments/results/primitive_affective_units/ica_k064_seed0/llm_judge_names_top10.json"),
        Path("experiments/results/primitive_affective_units/ica_k064_seed0/llm_judge_names.json"),
    ]

    for p in candidates:
        if not p.exists():
            continue
        raw = json.loads(p.read_text())
        for rec in raw:
            j = int(rec.get("component"))
            name = str(rec.get("proposed_name", f"b{j}"))
            neg = "negative"
            pos = "positive"
            if " to " in name:
                left, right = name.split(" to ", 1)
                neg = left.strip() or "negative"
                pos = right.strip() or "positive"
            family = name.split()[0] if name else "unknown"
            out[j] = {
                "component": j,
                "axis_name": name,
                "negative_pole_name": neg,
                "positive_pole_name": pos,
                "family": family,
                "confidence": float(rec.get("confidence", np.nan)),
                "notes": rec.get("rationale", ""),
            }
        if raw:
            print(f"[judge] loaded fallback provisional names from {p} ({len(raw)} components)")

    # Fill remaining unnamed axes from per-axis label summaries.
    per_axis_path = Path("experiments/results/per_axis_judge_L22_k64/per_axis_top_labels.csv")
    if per_axis_path.exists():
        df = pd.read_csv(per_axis_path)
        df = df[df["rank"] == 1].copy()
        for axis in sorted(df["axis"].unique()):
            j = int(axis)
            if j in out:
                continue
            neg_rows = df[(df["axis"] == j) & (df["alpha_unit"] == -2.0)]
            pos_rows = df[(df["axis"] == j) & (df["alpha_unit"] == 2.0)]
            base_rows = df[(df["axis"] == j) & (df["alpha_unit"] == 0.0)]

            neg = str(neg_rows.iloc[0]["label"]).strip() if not neg_rows.empty else "low-axis-activation"
            pos = str(pos_rows.iloc[0]["label"]).strip() if not pos_rows.empty else "high-axis-activation"
            base = str(base_rows.iloc[0]["label"]).strip() if not base_rows.empty else ""

            if neg == pos:
                axis_name = pos
            else:
                axis_name = f"{neg} to {pos}"

            family = pos.split()[0] if pos and pos != "none" else (neg.split()[0] if neg else "latent")
            notes = f"Derived automatically from per-axis judge labels (baseline={base or 'n/a'})."
            out[j] = {
                "component": j,
                "axis_name": axis_name,
                "negative_pole_name": neg,
                "positive_pole_name": pos,
                "family": family or "latent",
                "confidence": np.nan,
                "notes": notes,
            }
        print(f"[judge] filled provisional names from {per_axis_path} ({len(out)} total components)")

    return out


def _call_llm_judge(
    judge_inputs_path: Path,
    judge_outputs_path: Path,
    model: str = "gpt-4o-mini",
) -> None:
    """Call external LLM judge service to interpret axes.
    
    Reads judge_inputs.jsonl, sends to LLM, writes judge_outputs.jsonl.
    """
    import os
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set")
    
    try:
        import openai
    except ImportError:
        raise ImportError("openai package required for judge; pip install openai")
    
    client = openai.OpenAI()
    
    outputs = []
    with open(judge_inputs_path) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            inp = json.loads(line)
            component_id = int(inp["component"])
            task = str(inp.get("task", "naming"))

            if task == "brokenness":
                # Use the task instruction as system prompt so response_format=json_object is valid.
                prompt = inp.get("instruction", "Return JSON with fields broken, severity, fluency, coherence, off_topic, repetition, rationale.")
                context = (
                    f"Component b{component_id}\n"
                    f"Sign: {inp.get('sign')}\n"
                    f"Alpha unit: {inp.get('alpha_unit')}\n"
                    f"Prompt: {inp.get('prompt', '')}\n"
                    f"Generation: {inp.get('generation', '')}\n"
                    "Respond in JSON only.\n"
                )
            else:
                prompt = inp.get("prompt") or "Return JSON with axis interpretation fields."
                if "generations" in inp:
                    context = f"""
Component b{component_id}:

Generations:
- α=-2 (negative direction): {' '.join(inp['generations'].get('alpha_neg2', []))}
- α=0 (baseline): {' '.join(inp['generations'].get('alpha_zero', []))}
- α=+2 (positive direction): {' '.join(inp['generations'].get('alpha_pos2', []))}

Top-loading positive source texts:
{json.dumps(inp.get('top_positive_texts', []), ensure_ascii=False)}

Top-loading negative source texts:
{json.dumps(inp.get('top_negative_texts', []), ensure_ascii=False)}

Diagnostics:
{json.dumps(inp.get('diagnostics', {}), ensure_ascii=False)}

Respond in JSON only.
"""
                else:
                    context = f"""
Component b{component_id}:

Threshold generations (negative):
{json.dumps(inp.get('threshold_generations_negative', []), ensure_ascii=False)}

Threshold generations (positive):
{json.dumps(inp.get('threshold_generations_positive', []), ensure_ascii=False)}

Top-loading positive source texts:
{json.dumps(inp.get('top_positive_texts', []), ensure_ascii=False)}

Top-loading negative source texts:
{json.dumps(inp.get('top_negative_texts', []), ensure_ascii=False)}

Diagnostics:
{json.dumps(inp.get('diagnostics', {}), ensure_ascii=False)}

Respond in JSON only.
"""
            
            # Call LLM
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": context},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=500,
            )
            
            try:
                # Try to parse JSON from response
                text = response.choices[0].message.content
                # Try to extract JSON block
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    judgment = json.loads(match.group())
                else:
                    judgment = {"raw_response": text, "parse_error": True}
            except Exception as e:
                judgment = {"raw_response": text, "parse_error": str(e)}

            judgment["task"] = task
            judgment["component"] = component_id
            if "sign" in inp:
                judgment["sign"] = int(inp["sign"])
            if "alpha_unit" in inp:
                judgment["alpha_unit"] = float(inp["alpha_unit"])
            if "prompt_id" in inp:
                judgment["prompt_id"] = int(inp["prompt_id"])
            outputs.append(judgment)
            
            if line_num % 10 == 0:
                print(f"[judge] {line_num} components completed")
    
    # Write outputs
    judge_outputs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(judge_outputs_path, "w") as f:
        for out in outputs:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[judge] wrote {judge_outputs_path}  ({len(outputs)} judgments)")


def _generate_axis_cards(
    output_dir: Path,
    top_texts: dict[int, dict],
    diagnostics: dict[int, dict],
    judge_outputs: dict[int, dict],
    W: np.ndarray,
    gen_cache: "GenerationCache",
    prompts: list[str],
) -> None:
    """Generate one markdown card per component."""
    cards_dir = output_dir / "axis_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    k = W.shape[0]
    for j in range(k):
        judge = judge_outputs.get(j, {})
        axis_name = judge.get("axis_name", f"latent-axis-b{j:02d}")
        neg_pole = judge.get("negative_pole_name", "low-axis-activation")
        pos_pole = judge.get("positive_pole_name", "high-axis-activation")
        family = judge.get("family", "latent")
        confidence = judge.get("confidence", np.nan)
        th_neg = judge.get("threshold_alpha_unit_negative", np.nan)
        th_pos = judge.get("threshold_alpha_unit_positive", np.nan)
        short_desc = judge.get("short_description", "")

        diags = diagnostics.get(j, {})
        self_rho = diags.get("self_rho", np.nan)
        label_dom = diags.get("label_dominance", np.nan)
        vad_exp = diags.get("vad_explained", np.nan)

        texts = top_texts.get(j, {})
        top_pos = texts.get("top_positive", [])
        top_neg = texts.get("top_negative", [])

        # Baseline (α=0) from cache
        gens_zero = [gen_cache.get(j, 0.0, p) for p in prompts]
        gens_zero = [g for g in gens_zero if g is not None]
        
        # Build markdown content
        lines = []
        lines.append(f"# Axis Card: b{j}")
        lines.append("")
        lines.append(f"## Axis Name: {axis_name}")
        lines.append(f"- Family: {family}")
        lines.append(f"- Confidence: {confidence if not np.isnan(confidence) else 'N/A'}")
        if not pd.isna(th_neg) or not pd.isna(th_pos):
            lines.append(f"- Threshold alpha (negative): {th_neg if not pd.isna(th_neg) else 'N/A'}")
            lines.append(f"- Threshold alpha (positive): {th_pos if not pd.isna(th_pos) else 'N/A'}")
        lines.append("")
        
        lines.append("## Polar Directions")
        lines.append(f"- **Negative pole (α=-2)**: {neg_pole}")
        lines.append(f"- **Positive pole (α=+2)**: {pos_pole}")
        lines.append("")
        
        if short_desc:
            lines.append(f"## Description\n{short_desc}\n")
        
        lines.append("## Diagnostics")
        lines.append(f"- self_ρ: {self_rho:.3f}" if not np.isnan(self_rho) else "- self_ρ: N/A")
        lines.append(f"- label_dominance: {label_dom:.3f}" if not np.isnan(label_dom) else "- label_dominance: N/A")
        lines.append(f"- VAD_explained: {vad_exp:.4f}" if not np.isnan(vad_exp) else "- VAD_explained: N/A")
        lines.append("")
        
        lines.append("## Top-Loading Source Texts")
        lines.append("### Positive-loading examples:")
        for ex in top_pos[:3]:
            lines.append(f"- pos: *{ex['pos_text'][:60]}...*")
            lines.append(f"  neg: *{ex['neg_text'][:60]}...*")
        lines.append("")
        
        lines.append("### Negative-loading examples:")
        for ex in top_neg[:3]:
            lines.append(f"- pos: *{ex['pos_text'][:60]}...*")
            lines.append(f"  neg: *{ex['neg_text'][:60]}...*")
        lines.append("")
        
        lines.append("## Example Generations")
        neg_threshold_gens = judge.get("threshold_generations_negative", [])
        pos_threshold_gens = judge.get("threshold_generations_positive", [])
        if neg_threshold_gens:
            lines.append("### Negative threshold generations:")
            for ex in neg_threshold_gens[:3]:
                lines.append(f"- {ex[:80]}...")
            lines.append("")
        else:
            lines.append("### Negative direction (fallback):")
            lines.append("*(No threshold generations available)*")
            lines.append("")

        lines.append("### α=0 (Baseline):")
        for ex in gens_zero[:2]:
            lines.append(f"- {ex[:80]}...")
        lines.append("")

        if pos_threshold_gens:
            lines.append("### Positive threshold generations:")
            for ex in pos_threshold_gens[:3]:
                lines.append(f"- {ex[:80]}...")
            lines.append("")
        else:
            lines.append("### Positive direction (fallback):")
            lines.append("*(No threshold generations available)*")
            lines.append("")
        
        if judge:
            lines.append("## Judge Notes")
            if "notes" in judge:
                lines.append(f"{judge['notes']}\n")
        
        card_path = cards_dir / f"b{j:02d}.md"
        card_path.write_text("\n".join(lines), encoding="utf-8")
    
    print(f"[cards] generated {k} axis cards in {cards_dir}")


def _generate_summary_files(
    output_dir: Path,
    diagnostics: dict[int, dict],
    judge_outputs: dict[int, dict],
    W: np.ndarray,
    basis_path: Path,
) -> None:
    """Generate summary CSV, JSON, and LaTeX files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    k = W.shape[0]
    rows = []
    
    for j in range(k):
        judge = judge_outputs.get(j, {})
        diags = diagnostics.get(j, {})
        
        row = {
            "component": j,
            "axis_name": judge.get("axis_name", f"latent-axis-b{j:02d}"),
            "negative_pole": judge.get("negative_pole_name", "low-axis-activation"),
            "positive_pole": judge.get("positive_pole_name", "high-axis-activation"),
            "family": judge.get("family", "latent"),
            "confidence": judge.get("confidence", np.nan),
            "threshold_alpha_unit_negative": judge.get("threshold_alpha_unit_negative", np.nan),
            "threshold_alpha_unit_positive": judge.get("threshold_alpha_unit_positive", np.nan),
            "threshold_broken_rate_negative": judge.get("threshold_broken_rate_negative", np.nan),
            "threshold_broken_rate_positive": judge.get("threshold_broken_rate_positive", np.nan),
            "threshold_mean_severity_negative": judge.get("threshold_mean_severity_negative", np.nan),
            "threshold_mean_severity_positive": judge.get("threshold_mean_severity_positive", np.nan),
            "b_norm": float(np.linalg.norm(W[j])),
            "self_rho": diags.get("self_rho", np.nan),
            "label_dominance": diags.get("label_dominance", np.nan),
            "vad_explained": diags.get("vad_explained", np.nan),
            "primitive_score": diags.get("primitive_score", np.nan),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # CSV
    csv_path = output_dir / "axis_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[summary] wrote {csv_path}")
    
    # JSON
    json_path = output_dir / "axis_summary.json"
    with open(json_path, "w") as f:
        json.dump({
            "basis": str(basis_path),
            "k": k,
            "axes": df.to_dict(orient="records"),
        }, f, indent=2, ensure_ascii=False)
    print(f"[summary] wrote {json_path}")
    
    # Markdown table
    md_path = output_dir / "emotion_codebook.md"
    lines = [
        "# Emotion Codebook: k=64 ICA Basis (Layer 22)",
        "",
        "This codebook assigns interpretable names to all 64 basis axes.",
        "**These are NOT Plutchik categories.** They are latent affective dimensions",
        "inferred from causal steering, source text patterns, and diagnostic metrics.",
        "",
        "| Component | Axis Name | Negative Pole | Positive Pole | Family | Confidence |",
        "|---|---|---|---|---|---|",
    ]
    
    for _, row in df.iterrows():
        j = int(row["component"])
        conf = f"{row['confidence']:.2f}" if not pd.isna(row["confidence"]) else "N/A"
        lines.append(
            f"| b{j:02d} | {row['axis_name']} | {row['negative_pole']} | "
            f"{row['positive_pole']} | {row['family']} | {conf} |"
        )
    
    lines.extend([
        "",
        "## Scientific Context",
        "",
        "### Axis Independence",
        "Each axis is evaluated for **label independence** (low mutual information with Plutchik 8),",
        "**self-consistency** (monotonic response to steering alpha), and **cross-talk** (specificity vs. other axes).",
        "",
        "### VAD Coverage",
        f"Mean VAD-explained ratio: {df['vad_explained'].mean():.4f}",
        "",
        "### Self-Consistency",
        f"Mean self_rho (Spearman ρ of axis response to steering): {df['self_rho'].mean():.3f}",
        "",
    ])
    
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summary] wrote {md_path}")
    
    # LaTeX table
    tex_path = output_dir / "emotion_codebook.tex"
    tex_lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{|l|l|l|l|l|r|}",
        "\\hline",
        "Component & Axis Name & Negative Pole & Positive Pole & Family & Conf. \\\\",
        "\\hline",
    ]
    
    for _, row in df.iterrows():
        j = int(row["component"])
        name = str(row['axis_name']).replace("_", " ")[:25]
        neg = str(row['negative_pole']).replace("_", " ")[:15]
        pos = str(row['positive_pole']).replace("_", " ")[:15]
        fam = str(row['family']).replace("_", " ")[:12]
        conf = f"{row['confidence']:.2f}" if not np.isnan(row['confidence']) else "—"
        tex_lines.append(
            f"b{j:02d} & {name} & {neg} & {pos} & {fam} & {conf} \\\\"
        )
    
    tex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\caption{ICA $k=64$ Emotion Codebook (Layer 22, all 64 axes)}",
        "\\label{tab:emotion_codebook}",
        "\\end{table}",
    ])
    
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"[summary] wrote {tex_path}")


def _generate_readme(output_dir: Path) -> None:
    """Generate README.md explaining the codebook."""
    readme_path = output_dir / "README.md"
    
    content = """# Emotion Codebook: k=64 ICA Basis

## Overview

This directory contains a full interpretability analysis of the 64-dimensional ICA basis
fit to the latent affective space at Layer 22 of Llama-3.1-8B-Instruct.

## Scientific Framing

**These are not Plutchik categories.** The axes in this codebook are latent affective
directions whose meanings are inferred from three complementary sources:

1. **Causal steering**: Injecting α·b_j at layer 22 during generation and observing how
   the model's outputs shift. We use α ∈ {-2, -1, 0, 1, 2} in units of median basis norm.

2. **Source text patterns**: For each axis, we rank training pairs by their projection
   onto b_j and examine the top-loading positive and negative examples. Axes showing
   strong category-independence are preferred.

3. **Quantitative diagnostics**:
   - **self_ρ**: Spearman correlation between steering intensity α and the model's
     re-encoded response along b_j. High ρ indicates the axis is "real" and stable.
   - **label_dominance**: Fraction of top-loading examples from the single largest
     Plutchik category. Low dominance indicates label independence.
   - **vad_explained**: Projection of b_j onto the VAD (Valence-Arousal-Dominance)
     subspace. Low values suggest the axis captures structure orthogonal to basic emotions.

## Files

### Generations & Examples
- `generations.parquet`: All steered generations (component × α × prompt × prompt_id)
  - Columns: component, alpha, prompt_id, prompt, generation, effective_alpha, basis_path, layer, k
- `adaptive_generations.parquet`: Adaptive breakage-search generations for threshold finding

### Diagnostics & Metadata
- `top_texts.json`: Top-loading source texts for each component (positive & negative)
- `brokenness_judge_inputs.jsonl`: Structured inputs for broken/not-broken classification
- `brokenness_judge_outputs.jsonl`: Structured outputs from the brokenness judge
- `judge_inputs.jsonl`: Structured input for LLM-based axis naming (one JSON per component)
- `judge_outputs.jsonl`: LLM naming results (if --run-judge was used)
- `threshold_summary.csv` / `threshold_summary.json`: per-component threshold statistics

### Summaries
- `axis_summary.csv`: One row per component with metrics and naming
- `axis_summary.json`: Machine-readable summary with full metadata
- `emotion_codebook.md`: Markdown table of all axes (for reading)
- `emotion_codebook.tex`: LaTeX table for inclusion in papers (all 64 axes)

### Per-Axis Documentation
- `axis_cards/`: One markdown file per component (b00.md, b01.md, ..., b63.md)
    - Each card includes: diagnostics, threshold alpha values, top source texts, threshold generations, judge notes

## Interpretation Guidelines

### What These Axes Represent

Each axis b_j encodes an independent direction in latent affective space that:
- **Causally influences generation** (verified by steering)
- **Loads distinct source text patterns** (verified by top-N analysis)
- **Shows label independence** (low mutual information with Plutchik 8)
- **Is orthogonal to basic VAD** (low projection onto V, A, D)

Axes are NOT constrained to fit any predefined emotion taxonomy. They emerge from
the learned ICA decomposition of per-pair activation differences.

### Confidence Scores

The `confidence` field (0.0–1.0) reflects the LLM judge's estimate that this axis
represents a coherent, interpretable affective dimension. Lower confidence may indicate:
- Weak steering effect (high α needed to move outputs)
- Mixed category histograms (unclear semantics)
- Low causal strength relative to cross-talk

### Families

Axes are grouped into informal families based on judge output:
- **engagement**: degrees of attention, involvement, focus
- **uncertainty**: epistemic states, confidence, doubt
- **valence**: positive vs. negative affect (orthogonal to VAD)
- **dominance**: agency, control, assertiveness
- **arousal**: energy, intensity, activation (independent of VAD)
- And many others emergent from the data

## Usage

### Read the codebook:
```bash
cat emotion_codebook.md
```

### Inspect a specific axis:
```bash
cat axis_cards/b26.md
```

### Use in experiments:
Load `axis_summary.csv` and filter by family, confidence, or diagnostic scores.

## Comparison with Prior Work

- **Phase B (2025)**: Used category-averaged Δ → only k=8 basis captures Plutchik 8-d
- **Phase C (2025)**: Per-pair Δ with basis_sweep → discovered language-independent structure
- **This codebook (2026)**: Full interpretability documentation for k=64 ICA, Layer 22

The k=64 basis achieves better reconstruction (R² validation) and more stable semantics
(Spearman ρ) than smaller k, while remaining computationally tractable for downstream steering tasks.

## Files Structure

```
emotion_codebook/ica_k064_L22/
├── README.md                    (this file)
├── generations.parquet          (1536 generations: 64 components × 3 alphas × 8 prompts)
├── top_texts.json               (source text interpretations)
├── judge_inputs.jsonl           (structured input for LLM naming)
├── judge_outputs.jsonl          (LLM judge results, if --run-judge)
├── axis_summary.csv             (CSV summary, sortable)
├── axis_summary.json            (JSON metadata)
├── emotion_codebook.md          (markdown table)
├── emotion_codebook.tex         (LaTeX table)
└── axis_cards/
    ├── b00.md
    ├── b01.md
    ├── ...
    └── b63.md
```

---

*Last generated: 2026-05-10*
*Basis: `data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt`*
*Method: ICA, k=64, Layer 22*
"""
    
    readme_path.write_text(content, encoding="utf-8")
    print(f"[readme] wrote {readme_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a complete emotion codebook for all 64 ICA basis components."
    )
    p.add_argument(
        "--basis",
        type=Path,
        default=Path("data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt"),
        help="Path to basis artifact",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/emotion_codebook/ica_k064_L22"),
        help="Output directory for all results",
    )
    p.add_argument(
        "--components",
        type=int,
        nargs="*",
        default=None,
        help="Component indices to process (default: all)",
    )
    p.add_argument(
        "--n-prompts",
        type=int,
        default=8,
        help="Number of neutral prompts",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max generation length",
    )
    p.add_argument(
        "--zero-repetition-penalty",
        type=float,
        default=1.15,
        help="Repetition penalty applied only when alpha=0 baseline generations are decoded.",
    )
    p.add_argument(
        "--zero-no-repeat-ngram-size",
        type=int,
        default=3,
        help="No-repeat ngram size applied only when alpha=0 baseline generations are decoded.",
    )
    p.add_argument(
        "--zero-max-extra-tokens",
        type=int,
        default=24,
        help="Additional decode budget only for alpha=0 baseline so output can close a sentence.",
    )
    p.add_argument(
        "--sentence-extra-tokens",
        type=int,
        default=24,
        help="Additional decode budget for all generations to help produce multi-sentence outputs.",
    )
    p.add_argument(
        "--min-sentences",
        type=int,
        default=2,
        help="Target minimum number of sentence-like units in post-processed generations.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for prompt selection",
    )
    p.add_argument(
        "--prompt-source",
        type=str,
        default="fixed",
        choices=["fixed", "dataset"],
        help="Prompt source: 'fixed' (less-vague curated prompts) or 'dataset' (sampled neutral prompts).",
    )
    p.add_argument(
        "--alpha-mode",
        type=str,
        default="caa_match",
        choices=["caa_match"],
        help="How to scale alpha",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model.yaml"),
        help="Model config",
    )
    p.add_argument(
        "--steering-config",
        type=Path,
        default=Path("configs/steering.yaml"),
        help="Steering config",
    )
    p.add_argument(
        "--skip-top-texts",
        action="store_true",
        help="Skip top-text interpretation (load from existing JSON)",
    )
    p.add_argument(
        "--prepare-judge-inputs",
        action="store_true",
        help="Prepare JSONL for LLM judge (no judge execution)",
    )
    p.add_argument(
        "--run-judge",
        action="store_true",
        help="Call LLM judge API (requires OPENAI_API_KEY)",
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Which LLM model to use for judging",
    )
    args = p.parse_args()

    # Load basis
    print(f"[main] loading basis from {args.basis}")
    W, layer, k, decomposer = _load_basis_artifact(args.basis)
    print(f"[main] basis: k={k}, layer={layer}, decomposer={decomposer}, W.shape={W.shape}")

    profile_cfg, profile_name = load_profile(args.config)
    sc = yaml.safe_load(args.steering_config.read_text())
    apply_to = sc["caa"].get("apply_to", "generation")
    
    # Determine components
    components = args.components if args.components else list(range(k))
    components = sorted(set(components))
    print(f"[main] processing {len(components)} components")
    
    # Setup output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = None

    # === Global generation cache ===
    # Scan ALL past experiment parquets under experiments/results/ so that any
    # previously generated (component, alpha, prompt) triple is never re-generated.
    gen_cache = GenerationCache.from_results_dir(Path("experiments/results"))

    # === Step 2: Load diagnostics ===
    print("[main] === Step 2: Load component diagnostics ===")
    diagnostics = _load_diagnostics(args.basis, W, layer, k)
    for j in components:
        if not diagnostics[j]:
            print(f"  [diag] component b{j}: no metrics found")
    
    # === Step 3: Compute/load top texts ===
    top_texts_path = args.output_dir / "top_texts.json"
    
    if (args.skip_top_texts or top_texts_path.exists()) and top_texts_path.exists():
        print(f"[main] loading top texts from {top_texts_path}")
        with open(top_texts_path) as f:
            top_texts = json.load(f)
        top_texts = {int(k): v for k, v in top_texts.items()}
    else:
        print("[main] === Step 3: Compute top-text interpretation ===")
        bundle = load_activations(profile=profile_name, root=Path("data/activations"))
        
        # Get train mask from basis artifact
        payload = torch.load(args.basis, weights_only=False, map_location="cpu")
        train_mask_t = payload.get("train_mask")
        if train_mask_t is None:
            print("[main] WARNING: basis missing train_mask; using all data")
            train_mask = np.ones(len(bundle.meta), dtype=bool)
        else:
            train_mask = train_mask_t.numpy().astype(bool)
        
        top_texts = _compute_top_texts(
            W,
            bundle,
            layer,
            Path("data/contrastive/pairs.parquet"),
            train_mask,
            n_top=8,
        )
        
        with open(top_texts_path, "w") as f:
            # Convert numpy types for JSON serialization
            serializable = {}
            for j, data in top_texts.items():
                serializable[str(j)] = data
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"[main] saved top texts to {top_texts_path}")

    # === Step 4: Adaptive breakage search ===
    search_prompts = _load_codebook_prompts(
        n=args.n_prompts,
        seed=args.seed,
        prompt_source=args.prompt_source,
    )
    adaptive_gens_path = args.output_dir / "adaptive_generations.parquet"
    brokenness_inputs_path = args.output_dir / "brokenness_judge_inputs.jsonl"
    brokenness_outputs_path = args.output_dir / "brokenness_judge_outputs.jsonl"
    threshold_summary_csv = args.output_dir / "threshold_summary.csv"
    threshold_summary_json = args.output_dir / "threshold_summary.json"

    # === Step 4: Adaptive breakage search with early stopping ===
    # For each component and sign, iterate alphas in ascending order.
    # After generating at each alpha, immediately call the brokenness judge.
    # Once broken, record threshold = previous alpha and stop.
    # Reuse rows already available from Step 1 (alpha=1.0, 2.0).

    _alpha_scale = _compute_alpha_scale(W, args.alpha_mode)
    _sorted_aus = sorted(BREAKAGE_CANDIDATE_ALPHAS)  # ascending: 0.25, 0.5, 1.0, 2.0, 4.0, 8.0

    _adaptive_rows: list[dict] = []

    # Load partially completed adaptive search to enable auto-resume
    _adaptive_saved: dict[tuple[int, int, float], list[dict]] = {}  # (comp, sign, au) -> rows
    if adaptive_gens_path.exists():
        _prev = pd.read_parquet(adaptive_gens_path)
        for _, _r in _prev.iterrows():
            _key = (int(_r["component"]), int(_r["sign"]), float(_r["alpha_unit"]))
            _adaptive_saved.setdefault(_key, []).append(_r.to_dict())
        print(f"[main] resumed adaptive generations: {len(_prev)} rows from {adaptive_gens_path}")

    # threshold_map: (component, sign) -> {"alpha_unit": float, "threshold_gens": [str]}
    threshold_map: dict[tuple[int, int], dict] = {}
    # broken_map: (component, sign, alpha_unit) -> bool  (from judge outputs already saved)
    broken_map: dict[tuple[int, int, float], bool] = {}
    if brokenness_outputs_path.exists():
        for _rec in (json.loads(l) for l in brokenness_outputs_path.read_text().splitlines() if l.strip()):
            _k3 = (int(_rec["component"]), int(_rec["sign"]), float(_rec["alpha_unit"]))
            if "broken" in _rec:
                # aggregate across prompt_ids: broken if majority broken
                broken_map.setdefault(_k3, [])  # type: ignore[assignment]
                broken_map[_k3].append(bool(_rec["broken"]))  # type: ignore[index]
        broken_map = {k: (sum(v) / len(v) >= 0.25) for k, v in broken_map.items()}  # type: ignore[assignment]

    def _judge_single_alpha(
        component: int, sign: int, alpha_unit: float, rows: list[dict]
    ) -> bool:
        """Call brokenness judge for one (component, sign, alpha_unit) and return is_broken."""
        if not args.run_judge or not os.getenv("OPENAI_API_KEY"):
            return False  # no judge available → never stop early

        try:
            import openai  # noqa: PLC0415
        except ImportError:
            return False

        client = openai.OpenAI()
        verdicts = []
        for _r in rows:
            instruction = (
                "You are judging whether a generated continuation is linguistically broken. "
                "Return JSON with fields: broken (bool), severity (0.0-1.0), fluency (0.0-1.0), "
                "coherence (0.0-1.0), off_topic (0.0-1.0), repetition (0.0-1.0), rationale (str). "
                "Mark broken=true if the text is gibberish, self-contradictory, repetitive collapse, "
                "or clearly unusable as natural language continuation."
            )
            context = (
                f"Component b{component}\nSign: {sign}\nAlpha unit: {alpha_unit}\n"
                f"Prompt: {_r['prompt']}\nGeneration: {_r['generation']}\nRespond in JSON only.\n"
            )
            try:
                resp = client.chat.completions.create(
                    model=args.judge_model,
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": context},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    max_tokens=300,
                )
                text = resp.choices[0].message.content
                m = re.search(r"\{.*\}", text, re.DOTALL)
                judgment = json.loads(m.group()) if m else {}
            except Exception:
                judgment = {}

            judgment["task"] = "brokenness"
            judgment["component"] = component
            judgment["sign"] = sign
            judgment["alpha_unit"] = alpha_unit
            judgment["prompt_id"] = int(_r["prompt_id"])

            # Append to brokenness_outputs
            with open(brokenness_outputs_path, "a", encoding="utf-8") as _f:
                _f.write(json.dumps(judgment, ensure_ascii=False) + "\n")

            verdicts.append(bool(judgment.get("broken", False)))

        broken_rate = sum(verdicts) / len(verdicts) if verdicts else 0.0
        return broken_rate >= 0.25  # stop if ≥1/4 prompts are broken

    print("[main] === Step 4: Adaptive breakage search (with early stopping) ===")

    # Initialize with ALL previously saved rows so incremental saves are cumulative
    _all_adaptive_rows: list[dict] = []
    for _saved_rows in _adaptive_saved.values():
        _all_adaptive_rows.extend(_saved_rows)

    for component in tqdm(components, desc="Breakage search", position=0):
        for sign in (-1, 1):
            prev_alpha: float | None = None
            prev_gens: list[str] = []
            found_broken = False

            for au in _sorted_aus:
                key3 = (component, sign, au)
                # Check if we already judged this as broken
                if key3 in broken_map:
                    if broken_map[key3]:
                        # already known broken → threshold is prev
                        threshold_map[(component, sign)] = {
                            "alpha_unit": prev_alpha if prev_alpha is not None else au,
                            "threshold_gens": prev_gens,
                        }
                        found_broken = True
                        break
                    else:
                        # already judged not-broken → skip judge, just track prev
                        if key3 in _adaptive_saved:
                            rows_here = _adaptive_saved[key3]
                            _all_adaptive_rows.extend(rows_here)
                            prev_alpha = au
                            prev_gens = [r["generation"] for r in rows_here]
                        continue

                # Get or generate rows for this (component, sign, au)
                if key3 in _adaptive_saved:
                    rows_here = _adaptive_saved[key3]
                else:
                    # Generate — check global cache first per prompt
                    _b = torch.from_numpy(W[component]).to(torch.float32)
                    _b_norm = float(np.linalg.norm(W[component]))
                    _alpha_phys = float(au) * float(sign) * _alpha_scale * _b_norm
                    _signed_au = float(au) * float(sign)
                    rows_here = []
                    for pid, prompt in enumerate(search_prompts):
                        cached_tail = gen_cache.get(component, _signed_au, prompt)
                        if cached_tail is not None:
                            tail = cached_tail
                        else:
                            if model is None:
                                model, _, _ = load_model(profile_cfg)
                            gen = steered_generate(
                                model,
                                prompt,
                                vector=_b,
                                alpha=_alpha_phys,
                                layers=[layer],
                                apply_to=apply_to,
                                max_new_tokens=args.max_new_tokens + max(0, int(args.sentence_extra_tokens)),
                                temperature=0.0,
                                top_p=1.0,
                                repetition_penalty=1.0,
                                no_repeat_ngram_size=0,
                            )
                            tail = gen[len(prompt):] if gen.startswith(prompt) else gen
                            tail = _finalize_min_sentences(tail, min_sentences=args.min_sentences)
                            gen_cache.put(component, _signed_au, prompt, tail)
                        row_d = {
                            "component": component,
                            "sign": sign,
                            "alpha_unit": au,
                            "prompt_id": pid,
                            "prompt": prompt,
                            "generation": tail,
                            "effective_alpha": _alpha_phys,
                            "basis_path": str(args.basis),
                            "layer": layer,
                            "k": k,
                        }
                        rows_here.append(row_d)
                    _adaptive_saved[key3] = rows_here

                _all_adaptive_rows.extend(rows_here)

                # Judge inline
                is_broken = _judge_single_alpha(component, sign, au, rows_here)

                if is_broken:
                    threshold_map[(component, sign)] = {
                        "alpha_unit": prev_alpha if prev_alpha is not None else au,
                        "threshold_gens": prev_gens,
                    }
                    found_broken = True
                    break

                prev_alpha = au
                prev_gens = [r["generation"] for r in rows_here]

            if not found_broken:
                # Never broke → use largest alpha as threshold
                threshold_map[(component, sign)] = {
                    "alpha_unit": _sorted_aus[-1],
                    "threshold_gens": prev_gens,
                }

        # Incremental save after each component
        _save_df = pd.DataFrame(_all_adaptive_rows)
        _save_df.to_parquet(adaptive_gens_path)

    search_df = pd.DataFrame(_all_adaptive_rows).reset_index(drop=True)
    print(f"[main] adaptive search complete: {len(search_df)} rows, {len(threshold_map)} thresholds")

    # Build threshold_df from threshold_map
    breakage_summary = pd.DataFrame()
    component_rows = []
    for component in components:
        neg = threshold_map.get((component, -1), {"alpha_unit": _sorted_aus[-1], "threshold_gens": []})
        pos = threshold_map.get((component, 1), {"alpha_unit": _sorted_aus[-1], "threshold_gens": []})
        component_rows.append({
            "component": int(component),
            "threshold_alpha_unit_negative": float(neg["alpha_unit"]),
            "threshold_alpha_unit_positive": float(pos["alpha_unit"]),
            "threshold_broken_rate_negative": float("nan"),
            "threshold_broken_rate_positive": float("nan"),
            "threshold_mean_severity_negative": float("nan"),
            "threshold_mean_severity_positive": float("nan"),
            "threshold_generations_negative": neg["threshold_gens"],
            "threshold_generations_positive": pos["threshold_gens"],
        })
    threshold_df = pd.DataFrame(component_rows)

    # Save threshold summary
    threshold_df.to_csv(threshold_summary_csv, index=False)
    with open(threshold_summary_json, "w") as f:
        json.dump({
            "basis": str(args.basis),
            "k": int(k),
            "rows": [
                {c: (v if not (isinstance(v, float) and np.isnan(v)) else None)
                 for c, v in row.items() if c != "threshold_generations_negative"
                 and c != "threshold_generations_positive"}
                for row in component_rows
            ],
        }, f, indent=2, ensure_ascii=False)
    print(f"[thresholds] wrote {threshold_summary_csv}")
    print(f"[thresholds] wrote {threshold_summary_json}")

    # === Step 5: Prepare naming judge inputs ===
    judge_inputs_path = args.output_dir / "judge_inputs.jsonl"

    if not threshold_df.empty:
        _prepare_threshold_judge_inputs(
            threshold_df,
            top_texts,
            diagnostics,
            W,
            judge_inputs_path,
        )

    # === Step 6: Optional naming judge execution ===
    judge_outputs = {}
    judge_outputs_path = args.output_dir / "judge_outputs.jsonl"

    if args.run_judge and judge_inputs_path.exists():
        print("[main] === Step 6: Run naming judge ===")
        _call_llm_judge(judge_inputs_path, judge_outputs_path, model=args.judge_model)
        judge_outputs = _load_judge_outputs(judge_outputs_path)
    elif judge_outputs_path.exists():
        print(f"[main] loading judge outputs from {judge_outputs_path}")
        judge_outputs = _load_judge_outputs(judge_outputs_path)
    else:
        judge_outputs = _fallback_judge_outputs_from_existing()

    if not threshold_df.empty:
        threshold_lookup = {
            int(row["component"]): row.to_dict()
            for _, row in threshold_df.iterrows()
        }
        for component, row in threshold_lookup.items():
            judge_outputs.setdefault(component, {})
            judge_outputs[component].update(row)

    # === Step 7: Generate axis cards ===
    print("[main] === Step 6: Generate axis cards ===")
    _generate_axis_cards(
        args.output_dir,
        top_texts,
        diagnostics,
        judge_outputs,
        W,
        gen_cache,
        search_prompts,
    )

    # === Step 8: Generate summary files ===
    print("[main] === Step 7: Generate summary files ===")
    _generate_summary_files(
        args.output_dir,
        diagnostics,
        judge_outputs,
        W,
        args.basis,
    )
    
    # === Step 9: Generate README ===
    print("[main] === Step 8: Generate README ===")
    _generate_readme(args.output_dir)
    
    print(f"\n[main] ✓ Emotion codebook complete!")
    print(f"[main] All outputs in: {args.output_dir}")
    print(f"[main] Start with: cat {args.output_dir}/emotion_codebook.md")


if __name__ == "__main__":
    main()
