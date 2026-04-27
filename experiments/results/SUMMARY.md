# Phase B-5 Evaluation Summary

| Metric | Value | Target | Pass |
|---|---|---|---|
| Shift accuracy (mean, α=+2) | 0.133 | >= 0.4 | ❌ |
| Monotonicity ρ (min over cats) | 0.157 | >= 0.7 | ❌ |
| Median max alpha_unit @ ratio≤2 | 2.500 | >= 1.0 | ✅ |
| VAD R² (min over V/A/D) | 0.158 | >= 0.5 | ❌ |

## Layer sweep

- best_layer: **16**
  - layer 13: val_acc=0.664
  - layer 16: val_acc=0.666
  - layer 19: val_acc=0.664
  - layer 22: val_acc=0.664

## Shift accuracy (per category)

| category | shift_acc | baseline | Δ | note |
|---|---|---|---|---|
| anger | 0.00 | 0.00 | +0.00 |  |
| anticipation | n/a | n/a | n/a | no classifier label |
| disgust | 0.00 | 0.00 | +0.00 |  |
| fear | 0.00 | 0.00 | +0.00 |  |
| joy | 0.40 | 0.30 | +0.10 |  |
| sadness | 0.30 | 0.00 | +0.30 |  |
| surprise | 0.10 | 0.10 | +0.00 |  |
| trust | n/a | n/a | n/a | no classifier label |

## Monotonicity (Spearman ρ per category)

| category | rho | p | n | note |
|---|---|---|---|---|
| anger | 0.28 | 0.047 | 50 |  |
| anticipation | n/a | n/a |  | no classifier label |
| disgust | 0.29 | 0.045 | 50 |  |
| fear | 0.23 | 0.11 | 50 |  |
| joy | 0.32 | 0.024 | 50 |  |
| sadness | 0.25 | 0.084 | 50 |  |
| surprise | 0.16 | 0.28 | 50 |  |
| trust | n/a | n/a |  | no classifier label |

## Perplexity guardrail

- anger: max α (≤×2.0 baseline) = **3.0** (baseline PPL=4.96)
- anticipation: max α (≤×2.0 baseline) = **2.0** (baseline PPL=4.96)
- disgust: max α (≤×2.0 baseline) = **5.0** (baseline PPL=4.96)
- fear: max α (≤×2.0 baseline) = **1.0** (baseline PPL=4.96)
- joy: max α (≤×2.0 baseline) = **5.0** (baseline PPL=4.96)
- sadness: max α (≤×2.0 baseline) = **1.0** (baseline PPL=4.96)
- surprise: max α (≤×2.0 baseline) = **4.0** (baseline PPL=4.96)
- trust: max α (≤×2.0 baseline) = **2.0** (baseline PPL=4.96)

## VAD regression (held-out)

- layer: 19
- V: R² = **0.561** ✅
- A: R² = **0.245** ❌
- D: R² = **0.158** ❌
