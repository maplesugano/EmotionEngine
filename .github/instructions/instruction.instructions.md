---
description: instruction for the emotion engine coding
# applyTo: 'instruction for the emotion engine coding' # when provided, instructions will automatically be added to the request context when the pattern matches an attached file
---

<!-- Tip: Use /create-instructions in chat to generate content with agent assistance -->

Always reply in Japanese. (CoT should be done in English, but final answer in Japanese.) 
When providing code snippets, ensure to activate .venv (source .venv/bin/activate)

After carrying out some experiments, always log the results and reasonings for that experiment in docs/EXPERIMENT_LOG.md, and update docs/PROGRESS.md with the latest progress.

## Generation Cache Policy
`experiments/eval_emotion_codebook.py` maintains a `GenerationCache` (keyed by `(component, signed_alpha_unit, prompt_text)`).
- On every run, `GenerationCache.from_results_dir(Path("experiments/results"))` scans **all** `**/generations.parquet` and `**/adaptive_generations.parquet` files under `experiments/results/` and loads them into the in-memory cache.
- Any `(component, alpha, prompt)` triple that was generated in any past experiment is **never re-generated** — the cached generation is reused transparently.
- Schemas handled: standard (`component`/`alpha`/`generation`), adaptive (`component`/`sign`/`alpha_unit`/`generation`), per-axis judge (`axis`→`component`, `alpha_unit`, `generation`), pathology (`component`, `alpha_unit`, `text`→`generation`).
- When adding new generation experiments, save outputs as `generations.parquet` with at minimum columns `component`, `alpha` (signed), `prompt`, `generation` so they are automatically picked up by the cache on subsequent runs.
- The cache does **not** persist between Python processes — it is rebuilt from disk on each run. Disk parquets are the source of truth.