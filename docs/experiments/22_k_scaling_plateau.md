# 22. k スケーリングと retention 押し上げ（論文核心）

## 1. 実験の理由

[16_caa_basis_decomposition.md](16_caa_basis_decomposition.md) で k=8→16 で
R² が +0.10 上がり plateau 未到達と判明。L=22 で k∈{32, 64} を実走し、
**「named axes は basis の射影として再構成可能」**仮説を retention 1.0 に
近づけることが論文 Phase 1〜2 の核心目標。

## 2. 数式と定義

[16_caa_basis_decomposition.md](16_caa_basis_decomposition.md) と
[17_caa_basis_decomp_steering.md](17_caa_basis_decomp_steering.md) の指標を
$k \in \{16, 32, 64\}$ で再走。`caa_match` α 正規化 alpha_scale=0.18929。

スクリプト：
- Phase 1：[experiments/eval_caa_basis_decomposition.py](../../experiments/eval_caa_basis_decomposition.py)
- Phase 2：[experiments/eval_caa_basis_decomp_steering.py](../../experiments/eval_caa_basis_decomp_steering.py)

basis：[data/emotion_code/basis_sweep_L22/ica_k{032,064}_seed0.pt](../../data/emotion_code/basis_sweep_L22/)

## 3. 結果

### Phase 1（数値、L=22, ICA, OLS）

| k | median R² | median cos | per-cat 最低 R² |
|---:|---:|---:|---|
| 16 | 0.878 | 0.937 | joy 0.75 |
| 32 | 0.920 | 0.959 | joy 0.83 |
| **64** | **0.960** | **0.980** | joy 0.92 / surprise 0.93 |

→ R² 伸び幅 +0.042 → +0.040 と **線形ペース維持**。最小 R² も大幅改善し、
CAA 8 軸が 64 次元 basis の OLS でほぼ完全に張れる。

### Phase 2（行動、L=22 / k=64, n_prompts=32, 4608 gens, 約 19h）

| variant | mean_shift_acc | mean_delta | retention_vs_caa |
|---|---:|---:|---:|
| caa | 0.146 | +0.083 | 1.00 |
| **ols** | **0.135** | **+0.073** | **0.929** |
| lasso | 0.068 | +0.005 | 0.464 |
| nnls | 0.063 | 0.000 | 0.429 |
| vad | 0.063 | 0.000 | 0.429 |
| random | 0.042 | −0.021 | 0.286 |

k 推移：retention 0.71 (k=16/L19) → 0.79 (k=32/L22) → **0.93 (k=64/L22)**。

**論文上の主張「named axis = basis 線形射影」が retention=0.93 で実質的に閉じる**。

含意：
- k 単調増加は plateau 未到達だが、retention は逓減フェーズ。
- nnls/lasso/vad は k 増でも改善せず ⇒ CAA は basis の **両符号・密** な線形結合。
- random が baseline 以下に沈み、**方向の正当性** が retention を生むことが
  対照群側から担保。

成果物：
- [results/caa_basis_decomposition_L22/](../../experiments/results/caa_basis_decomposition_L22/)
- [results/caa_basis_decomp_steering_L22_k32/](../../experiments/results/caa_basis_decomp_steering_L22_k32/)
- [results/caa_basis_decomp_steering_L22_k64/](../../experiments/results/caa_basis_decomp_steering_L22_k64/)

## 4. 次の実験

→ [23_caa_pair_additivity.md](23_caa_pair_additivity.md) — 単一カテゴリの
線形再構成が出来た上で、**カテゴリ対の合成則** を測る加法性検定。
