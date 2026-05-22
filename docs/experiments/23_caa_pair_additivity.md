# 23. CAA カテゴリ対の加法性検定（二層構造の発見）

## 1. 実験の理由

[22_k_scaling_plateau.md](22_k_scaling_plateau.md) で「単独 CAA は basis 線形和
で再構成可能」を確立した。次の自然な問は **「カテゴリ対の joint steering も
線形和になるか」**。仮説「basis = 真の原子」の系として、joint = sum of
marginals が成り立つかを (i) readout 空間 (ii) shift_acc 空間の両方で測る。

## 2. 数式と定義

OLS 重み（$k=64, L=22$）から再構成した CAA 軸 $\hat v_a, \hat v_b$ を使って：
$$h^{(\ell)} \leftarrow h^{(\ell)} + \alpha \hat v_a + \beta \hat v_b$$

**Marginal 効果**：
- $r_a(\alpha) = R(\alpha, 0) - R(0, 0)$
- $r_b(\beta) = R(0, \beta) - R(0, 0)$

ここで $R(\alpha, \beta)$ は readout（再エンコード後の CAA 射影、または
shift_acc）。

**加法残差**：
$$\mathrm{resid}(\alpha, \beta) = R(\alpha,\beta) - R(0,0) - r_a(\alpha) - r_b(\beta)$$

**残差比**：$\|\mathrm{resid}\|_2 / \|r_a + r_b\|_2$。

スクリプト：[experiments/eval_caa_basis_additivity.py](../../experiments/eval_caa_basis_additivity.py)

設定：4 ペア (joy,sadness), (joy,anger), (anger,fear), (disgust,joy)
× α,β ∈ {−2, 0, +2} × n_prompts=8 × max_new_tokens=32 = **288 generations**
（約 35 min）。alpha_scale = 0.18929 (`caa_match`)。

## 3. 結果（off-diag、$\alpha,\beta\neq 0$ のみ）

| 指標 | median | mean |
|---|---:|---:|
| **readout 残差比** | **0.807** | 0.845 |
| **shift-acc 残差** $|r_a$| | **0.000** | 0.023 |
| shift-acc 残差 $|r_b|$ | 0.000 | 0.063 |

per-pair（α=β=+2）：

| pair | readout 残差比 | shift-acc 残差 a / b |
|---|---:|---:|
| joy + sadness | 0.81 | 0.00 / 0.00 |
| joy + anger | 0.82 | 0.00 / 0.00 |
| anger + fear | 0.91 | 0.00 / 0.00 |
| disgust + joy | 0.83 | 0.00 / 0.00 |

**二層構造の発見**：
- **readout レベルで加法性大崩れ**（残差が marginal 和の 80%）。
  basis 64 次元線形和では joint を説明しきれず、**高次相互作用が存在**。
- **shift-acc レベルでは完全加法**（中央値 0）。8 値分類器が割り当てる
  ラベルは「片方の単独軸と同じ」になり、判別境界は加法的に滑らか。
- 残差方向は [14_b8_qualitative.md](14_b8_qualitative.md) /
  [21_meta_emotion_cluster.md](21_meta_emotion_cluster.md) の **emergent
  meta-emotion** と整合：既存ラベルでは検出されない中間状態が残差表現には
  現れている。

**論文上の主張構造化**：
- Phase 1〜2（[22_k_scaling_plateau.md](22_k_scaling_plateau.md)）= **線形再構成**：retention 0.93
- Phase 3（本節以降）= **非線形合成**：readout 残差 0.81 が示す高次項が
  メタ感情の発生源候補

成果物：[results/caa_basis_additivity_L22_k64/](../../experiments/results/caa_basis_additivity_L22_k64/)

## 4. 次の実験

→ [24_additivity_metaemotion_proj.md](24_additivity_metaemotion_proj.md) —
残差ベクトルが [21_meta_emotion_cluster.md](21_meta_emotion_cluster.md) の
メタ感情方向に乗るかを直接射影で測る。
