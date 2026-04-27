# EmotionEngine

Pre-verbal emotion extraction, manipulation, and re-synthesis via latent steering in LLMs.

## Setup

**Requirements**: Python ≥ 3.11, CUDA GPU (≥ 24 GB recommended — tested on RTX A5000).

```bash
# Install dependencies
pip install -e ".[dev]"
```

## Repository Layout

```
artifacts/          # LaTeX paper source
configs/            # model.yaml · steering.yaml · data.yaml
data/
  raw/              # downloaded source datasets
  contrastive/      # contrastive prompt pairs (JSONL)
  activations/      # residual-stream activations (safetensors)
  emotion_code/     # basis matrix, PAD mapping weights
experiments/        # evaluation scripts (shift accuracy, monotonicity, …)
notebooks/          # Jupyter notebooks (00_model_comparison, …)
src/
  data/             # dataset building
  activations/      # activation collection via transformer_lens hooks
  emotion_code/     # steering vector extraction, PCA/NMF basis, PAD mapping
  steering/         # hook-based inference-time steering
  api/              # FastAPI server (extract / steer / health)
tests/              # pytest smoke tests
ui/                 # React + Vite frontend (PAD cube + turntable UI)
```

## Model Decision (Phase 0)

| Role | Model | VRAM (bf16) |
|---|---|---|
| **Primary** | Qwen2.5-7B-Instruct | ~15.3 GB |
| Fast ablation | GPT-2 medium | ~0.7 GB |

See [notebooks/00_model_comparison.ipynb](notebooks/00_model_comparison.ipynb) for the full comparison.  
Switch active model in [configs/model.yaml](configs/model.yaml).

## Quick Start

```bash
# Verify GPU and hook smoke-test
python tests/test_hook_smoke.py

# Launch API server (after Phase C)
uvicorn src.api.server:app --reload --port 8000
```