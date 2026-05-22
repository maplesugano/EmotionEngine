# 08. VAD 線形写像 R²

## 1. 実験の理由

心理学では Plutchik と並んで **VAD 連続次元**（Valence / Arousal / Dominance）
が「感情の最小座標系」とされる。残差ストリームから VAD を線形に取り出せる
かを測ることで、(a) named axes の中での被覆率、(b) 後段で「VAD では足りない」
（[16_caa_basis_decomposition.md](16_caa_basis_decomposition.md)）という主張の
**baseline 数値**を作る。

## 2. 数式と定義

EmoBank（連続値 V/A/D アノテーション付き、$N$=10,061）を使い、
層 $\ell=19$ の last-token 残差 $h^{(\ell)}_i \in \mathbb{R}^d$ から
$y_i \in \mathbb{R}^3$ へ Ridge 回帰：
$$\min_{W,b} \sum_i \|y_i - (W h^{(\ell)}_i + b)\|_2^2 + \lambda\|W\|_F^2,\quad W\in\mathbb{R}^{3\times d}$$

各軸の決定係数：
$$R^2_{a} = 1 - \frac{\sum_i (y_{i,a} - \hat y_{i,a})^2}{\sum_i (y_{i,a} - \bar y_a)^2},\ a\in\{V,A,D\}$$

スクリプト：[experiments/eval_vad_r2.py](../../experiments/eval_vad_r2.py)
出力：[vad_r2.json](../../experiments/results/vad_r2.json)、
重み：[vad_mapping.pt](../../data/emotion_code/vad_mapping.pt)

## 3. 結果（layer 19）

- **V: 0.561** ✅
- A: 0.245 ❌
- D: 0.158 ❌
- mean: 0.321、min: 0.158（目標 ≥ 0.5 未達）

→ **Valence しか線形には取り出せない**。Arousal / Dominance は残差線形では
届かない。「VAD 3 軸で測ること自体が偏った投影」という Phase C 方針転換の
根拠であり、後段 [16_caa_basis_decomposition.md](16_caa_basis_decomposition.md)
で **VAD baseline R² = 0.017** vs basis R² = 0.81 の対比に直接使う。

## 4. Phase B-5 総括

| Metric | Value | Target | Pass |
|---|---:|---:|---|
| Shift accuracy mean (α=+2) | 0.133 | ≥ 0.4 | ❌ |
| Monotonicity ρ (min) | 0.157 | ≥ 0.7 | ❌ |
| Median max alpha_unit | 2.5 | ≥ 1.0 | ✅ |
| VAD R² (min) | 0.158 | ≥ 0.5 | ❌ |

4 指標中 1 つしか目標達成せず。失敗の最有力仮説：**入力（カテゴリ平均）と
評価（外部分類器・VAD）の双方が言語化に依存**しているため、モデル内在の
「真の感情軸」を測れていない。

## 5. 次の実験

→ [09_basis_sweep.md](09_basis_sweep.md) — per-pair Δ を入力にして
言語化前の基底を探索する Phase C へ。
