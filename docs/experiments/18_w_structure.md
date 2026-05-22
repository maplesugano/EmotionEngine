# 18. 重み行列 W の構造解析（分散表現の直接証拠）

## 1. 実験の理由

[17_caa_basis_decomp_steering.md](17_caa_basis_decomp_steering.md) で
`OLS で再構成可能` は示した。ここでは重み行列
$W \in \mathbb{R}^{|\mathcal{C}|\times k}$ を **成分視点** で再分析し、
「joy 専用」「fear 専用」のような **片寄り成分が存在しない** ことを示す
（＝Plutchik は basis の独立片ではなく分散和）。同時に「どのカテゴリの
top-1 にもならない」**lexical-gap 候補成分**を抽出。

## 2. 数式と定義

成分 $j$ の各統計量（列 $W_{\cdot,j}$ について）：

**Participation ratio**：
$$\mathrm{PR}_j = \frac{(\sum_c |W_{c,j}|)^2}{\sum_c W_{c,j}^2}$$
低いほど特定カテゴリ寄り、最大値 $|\mathcal{C}|=8$ で完全一様。

**Sign balance**：$\mathrm{sgn\_bal}_j = \frac{\sum_c W_{c,j}}{\sum_c |W_{c,j}|} \in [-1, +1]$。

**Top-cat gap**：$\mathrm{gap}_j = |W_{c^*,j}| - |W_{c^{**},j}|$、$c^*$ は top-1 カテゴリ、$c^{**}$ は top-2。

**分類**：
- `cat_specific`：$\mathrm{PR}_j \le 2$
- `pan`：$\mathrm{PR}_j \ge 4$
- `lexical_gap`：$\mathrm{gap}_j \le 0.20$（top-1 が突出しない）

スクリプト：[experiments/eval_caa_basis_weight_structure.py](../../experiments/eval_caa_basis_weight_structure.py)

## 3. 結果（ICA k=16 seed=0, L=19, OLS）

**13 pan + 3 lexical_gap (b1, b11, b13) + 0 cat_specific**

→ いずれの basis 成分も「joy 専用」「fear 専用」のような片寄りを持たない。
**Plutchik 8 は basis 軸の独立片ではなく分散和**であることが直接示された。

3 つの lexical_gap 成分（b1, b11, b13）は、どのカテゴリの top-1 にもならず
**既存の Plutchik 名がない領域に意味を持つ候補**。これが
[19_lexical_gap_steering.md](19_lexical_gap_steering.md) の対象。

成果物：[results/caa_basis_weight_structure/ica_k016_seed0_L19_ols/](../../experiments/results/caa_basis_weight_structure/ica_k016_seed0_L19_ols/)
（`W_heatmap.png`, `W_heatmap_sorted_by_PR.png`,
`component_classification.csv`, `summary.json`）

## 4. 次の実験

→ [19_lexical_gap_steering.md](19_lexical_gap_steering.md) — gap 成分を
steer して「Plutchik 語彙の外」の生成挙動を観察する。
