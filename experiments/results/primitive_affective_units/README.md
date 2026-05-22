# Primitive Affective Units

This experiment identifies candidate **primitive affective units**: basis
directions that are label-independent, self-consistent, causally
steerable, and not reducible to named emotion categories (Plutchik) or
to the V/A/D affine subspace.

The script `experiments/eval_primitive_affective_units.py` consumes a
basis artifact (e.g. `data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt`)
and produces, per component `b_j`:

- behavioural metrics from an encode→steer→re-encode self-consistency
  test (`self_rho`, `sign_correct`, `causal_strength`, `cross_talk`,
  `specificity`),
- structural metrics reused from the sibling `*.metrics.json`
  (`mi`, `linear_sep_acc`, `category_top1_dominance` (label dominance),
  `vad_explained`),
- a transparent weighted PrimitiveScore,
- strong-α qualitative steering generations for the top-K candidates.

Outputs live under `<basis_stem>/` and include `primitive_scores.csv`,
`primitive_scores.json`, `selfcons_readouts.parquet`,
`strong_generations.parquet`, `top_candidates.md`, and `config.json`.
The `top_candidates.md` file leaves a `proposed_name` field blank for
human interpretation.

Scientifically, this operationalises the claim that primitive affective
units are *not* CAA category vectors themselves: they are discovered by
decomposing pairwise residual differences into latent basis directions
and selecting components that remain functional, label-independent, and
causally interpretable.
