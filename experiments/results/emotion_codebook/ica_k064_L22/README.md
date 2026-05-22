# Emotion Codebook: k=64 ICA Basis

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
