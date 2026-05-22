# 05. Shift accuracy 評価

## 1. 実験の理由

CAA ベクトルが「方向として正しい」かを、生成テキストの **外部分類器
ラベル変化率** で確認する。Phase B-5 で目標とする最も直接的な機能指標。

## 2. 数式と定義

中性プロンプト集合 $\mathcal{X}_0$、外部 Plutchik 分類器
$f: \text{text}\to\mathcal{C}$（j-hartmann/distilroberta）を用いて
$$\text{shift\_acc}_c(\alpha) = \frac{1}{|\mathcal{X}_0|}\sum_{x\in\mathcal{X}_0} \mathbb{1}\big[f\big(\text{gen}(x; \alpha v_c)\big) = c\big]$$

baseline は $\alpha = 0$。改善量 $\Delta_c = \text{shift\_acc}_c(\alpha) - \text{shift\_acc}_c(0)$。

スクリプト：[experiments/eval_shift_accuracy.py](../../experiments/eval_shift_accuracy.py)
出力：[shift_accuracy.csv](../../experiments/results/shift_accuracy.csv)

## 3. 結果（α=+2）

| category | shift_acc | baseline | Δ |
|---|---:|---:|---:|
| anger | 0.00 | 0.00 | +0.00 |
| disgust | 0.00 | 0.00 | +0.00 |
| fear | 0.00 | 0.00 | +0.00 |
| **joy** | **0.40** | 0.30 | **+0.10** |
| **sadness** | **0.30** | 0.00 | **+0.30** |
| surprise | 0.10 | 0.10 | +0.00 |
| anticipation / trust | n/a | n/a | n/a（外部分類器に該当ラベルなし）|

→ joy / sadness 以外、外部分類器ベースでは動かない。**「CAA 軸が弱い」のか
「分類器の感度が低い」のか切り分け不能**。Phase C-2 で純内在指標
（[13_self_consistency.md](13_self_consistency.md)）に転換する重要な反省点。

## 4. 次の実験

→ [06_monotonicity.md](06_monotonicity.md) — α を連続変化させたときの
強度の単調性を測る。
