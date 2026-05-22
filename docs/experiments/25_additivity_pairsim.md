# 25. 残差比 vs ペア類似度の相関

## 1. 実験の理由

[23_caa_pair_additivity.md](23_caa_pair_additivity.md) で発見された
readout 残差比 ≈ 0.8 が **ペアごとに変動する** ことを観察。
「basis 親和性が高いペアほど残差が大きい（同方向に押し合う高次項）」
という構造的説明を試み、二層構造を「直交ペアほど加法的」というクリーン
主張に書き換える。

## 2. 数式と定義

[22_k_scaling_plateau.md](22_k_scaling_plateau.md) の OLS 重み
$w_a, w_b \in \mathbb{R}^k$ から 3 種類のペア類似度：

- **`cos_w`**：$\cos(w_a, w_b)$（basis 重み空間）
- **`cos_recon`**：$\cos(\hat v_a, \hat v_b)$（再構成 CAA 軸）
- **`cos_caa_raw`**：$\cos(v_a, v_b)$（生 CAA）

各セルの残差比：
$$\mathrm{rr}(\alpha, \beta) = \frac{\|R(\alpha,\beta) - R(0,0) - r_a(\alpha) - r_b(\beta)\|_2}{\|r_a(\alpha) + r_b(\beta)\|_2}$$

**ペア統計**：median rr 等を pair-level（n=4）と cell-level（n=16）で
Pearson / Spearman 相関。

スクリプト：[experiments/eval_caa_additivity_pairsim.py](../../experiments/eval_caa_additivity_pairsim.py)

## 3. 結果（L=22 / k=64）

per-pair：

| pair | cos_w | cos_recon | median resid ratio |
|---|---:|---:|---:|
| **anger + fear** | **0.849** | 0.548 | **0.865** |
| disgust + joy | 0.123 | −0.086 | 0.825 |
| joy + sadness | 0.116 | 0.018 | 0.803 |
| joy + anger | 0.112 | 0.058 | 0.782 |

相関：
- pair-level (n=4)：cos_w → resid_ratio **Spearman r = +1.00**, Pearson +0.875
- cell-level (n=16)：cos_w → resid_ratio **Pearson r = +0.59 (p=0.016)**,
  Spearman r = +0.62 (p=0.011)
- cos_caa_raw / cos_recon もほぼ同方向（cell Pearson 0.52 / 0.51, p<0.05）

**含意**：
- **basis 親和性が高いペアほど残差が大きい**（同方向に押し合う高次項が大）。
- 逆相関ではない＝**「直交ペアほど加法的」という cleanly stated 主張**へ
  書き換え可能。
- 二層構造（[23_caa_pair_additivity.md](23_caa_pair_additivity.md)）は
  単なる artefact ではなく、ペアの幾何で説明される構造的現象。

成果物：[results/caa_additivity_pairsim_L22_k64/](../../experiments/results/caa_additivity_pairsim_L22_k64/)

## 4. 次の実験

→ [26_additivity_layer_k_robustness.md](26_additivity_layer_k_robustness.md) —
別構成（L=19, k=16）で同パターンが再現するかを確認する頑健性検証。
