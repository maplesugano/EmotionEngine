# Per-axis best-matching word (judge other_label mode)

basis: `data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt` | layer=
alphas: pos=2.0, neg=-2.0, n_prompts=8

| axis | top@+α (count) | top@−α (count) | top@0 | top Plutchik@+α | mean other@+α |
|---:|---|---|---|---|---:|
| b0 | helplessness (1) | curiosity (1) | indecision (1) | anticipation | 0.44 |
| b1 | curiosity (2) | uncertainty (2) | confusion (1) | anticipation | 0.30 |
| b2 | self-care (1) | intellectual curiosity (1) | indecision (1) | anger | 0.20 |
| b3 | frustration (1) | uncertainty (3) | confusion (1) | anger | 0.33 |
| b4 | regret (1) | frustration (1) | indecision (1) | anticipation | 0.38 |
| b5 | anxiety (1) | uncertainty (2) | confusion (1) | anticipation | 0.34 |
| b6 | confusion (2) | curiosity (2) | confusion (1) | trust | 0.41 |
| b7 | commitment (1) | self-doubt (1) | confusion (1) | anticipation | 0.36 |
| b8 | frustration (2) | confusion (1) | confusion (1) | anger | 0.39 |
| b9 | uncertainty (1) | helplessness (1) | confusion (1) | anticipation | 0.36 |
| b10 | none (1) | determination (1) | confusion (1) | anticipation | 0.31 |
| b11 | uncertainty (1) | contentment (2) | indecision (1) | anger | 0.30 |
| b12 | determination (1) | curiosity (2) | indecision (1) | anticipation | 0.39 |
| b13 | frustration (2) | stress (1) | indecision (1) | anger | 0.45 |
| b14 | determination (1) | frustration (1) | indecision (1) | anticipation | 0.41 |
| b15 | indecision (1) | determination (1) | indecision (1) | anticipation | 0.40 |
| b16 | uncertainty (2) | skepticism (1) | indecision (1) | anticipation | 0.33 |
| b17 | frustration (1) | confusion (1) | indecision (1) | anger | 0.25 |
| b18 | uncertainty (1) | determination (1) | confusion (1) | anticipation | 0.17 |
| b19 | confusion (1) | self-doubt (1) | indecision (1) | anticipation | 0.25 |
| b20 | repetitive desire (2) | frustration (1) | indecision (1) | anticipation | 0.38 |
| b21 | motivation (1) | confusion (2) | indecision (1) | anticipation | 0.38 |
| b22 | unrequited affection (1) | confusion (1) | indecision (1) | anticipation | 0.39 |
| b23 | indecision (1) | frustration (1) | indecision (1) | anticipation | 0.35 |
| b24 | overwhelmed (1) | confusion (2) | indecision (1) | anticipation | 0.26 |
| b25 | excitement (1) | self-reflection (1) | indecision (1) | anticipation | 0.46 |
| b26 | determination (1) | curiosity (2) | indecision (1) | anticipation | 0.26 |
| b27 | self-doubt (1) | indecision (1) | indecision (1) | anger | 0.29 |
| b28 | frustration (2) | frustration (1) | confusion (1) | anticipation | 0.39 |
| b29 | indecision (1) | confusion (1) | confusion (1) | anticipation | 0.28 |
| b30 | self-doubt (1) | repetitive behavior (1) | confusion (1) | anticipation | 0.30 |
| b31 | frustration (1) | uncertainty (2) | indecision (1) | anticipation | 0.16 |
| b32 | enthusiasm (1) | anxiety about preparation (1) | indecision (1) | anger | 0.33 |
| b33 | self-doubt (1) | repetitive reassurance (1) | indecision (1) | anticipation | 0.46 |
| b34 | frustration (1) | confusion (1) | indecision (1) | anticipation | 0.41 |
| b35 | curiosity (2) | indecision (1) | indecision (1) | anticipation | 0.54 |
| b36 | paranoid uncertainty (1) | contentment (1) | indecision (1) | anticipation | 0.26 |
| b37 | self-doubt (1) | confusion (1) | confusion (1) | anticipation | 0.30 |
| b38 | paranoid hypervigilance (1) | frustration (1) | indecision (1) | anticipation | 0.38 |
| b39 | uncertainty (2) | frustration (1) | confusion (1) | anticipation | 0.46 |
| b40 | inner conflict (1) | eagerness (1) | indecision (1) | anticipation | 0.31 |
| b41 | indecision (1) | confusion (1) | indecision (1) | anticipation | 0.38 |
| b42 | frustration (1) | uncertainty (2) | confusion (1) | anticipation | 0.56 |
| b43 | cynicism (1) | confusion (2) | indecision (1) | anger | 0.38 |
| b44 | uncertainty (1) | confusion (1) | confusion (1) | anticipation | 0.25 |
| b45 | determination (1) | hunger (1) | indecision (1) | anticipation | 0.35 |
| b46 | fatigue (1) | apprehension (1) | confusion (1) | anticipation | 0.40 |
| b47 | frustration (2) | eagerness (1) | confusion (1) | anticipation | 0.31 |
| b48 | determination (1) | confusion (1) | confusion (1) | anticipation | 0.36 |
| b49 | academic ambition (1) | uncertainty (2) | indecision (1) | anticipation | 0.15 |
| b50 | enthusiasm (1) | uncertainty (2) | confusion (1) | anticipation | 0.24 |
| b51 | uncertainty (2) | frustration (2) | confusion (1) | anticipation | 0.56 |
| b52 | confusion (1) | encouragement (1) | confusion (1) | anticipation | 0.35 |
| b53 | frustration (1) | self-doubt (1) | indecision (1) | anger | 0.35 |
| b54 | frustration (1) | uncertainty (2) | indecision (1) | anticipation | 0.41 |
| b55 | curiosity (2) | self-doubt (1) | confusion (1) | anticipation | 0.39 |
| b56 | determination (1) | confusion (1) | indecision (1) | anticipation | 0.20 |
| b57 | uncertainty (1) | generosity (1) | confusion (1) | anticipation | 0.30 |
| b58 | existential uncertainty (1) | relaxed acceptance (1) | confusion (1) | anticipation | 0.41 |
| b59 | determination (1) | confusion (1) | confusion (1) | anticipation | 0.30 |
| b60 | confusion (1) | disappointment (1) | confusion (1) | anticipation | 0.54 |
| b61 | identity crisis (1) | excitement (1) | confusion (1) | anger | 0.22 |
| b62 | self-doubt (1) | frustration (1) | confusion (1) | anticipation | 0.34 |
| b63 | confusion (1) | disappointment (1) | indecision (1) | anticipation | 0.41 |