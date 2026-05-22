# 06. Monotonicity（α と感情強度の単調性）

## 1. 実験の理由

shift_acc が「ラベルが変わるか否か」の二値判定なのに対し、こちらは
「α を増やすほど感情が **強く** なるか」を測る連続軸の検証。
ステアリングが scalable な制御になっているかの基本性質。

## 2. 数式と定義

各カテゴリ $c$ について $\alpha_1 < \alpha_2 < \cdots < \alpha_K$ を流し、
分類器スコア $s_c(\alpha) = f_c(\text{gen}(x; \alpha v_c))$（$f_c$ は
$c$ クラス確率）の **Spearman 順位相関**：
$$\rho_c = \mathrm{corr}_{\text{Spearman}}\big((\alpha_k)_k, (s_c(\alpha_k))_k\big)$$

各プロンプト $x$ で計算し平均。$\rho \ge 0.7$ が目標。

スクリプト：[experiments/eval_monotonicity.py](../../experiments/eval_monotonicity.py)
出力：[monotonicity.csv](../../experiments/results/monotonicity.csv)

## 3. 結果

| cat | ρ | p | n |
|---|---:|---:|---:|
| anger | 0.282 | 0.047 | 50 |
| disgust | 0.285 | 0.045 | 50 |
| fear | 0.228 | 0.111 | 50 |
| joy | 0.319 | 0.024 | 50 |
| sadness | 0.247 | 0.084 | 50 |
| surprise | 0.157 | 0.277 | 50 |

→ **弱いが正の相関**（min ρ = 0.157、目標 0.7 未達）。CAA は方向としては
機能するが「強度の単調性」は不十分。

## 4. 次の実験

→ [07_perplexity_guardrail.md](07_perplexity_guardrail.md) — α をどこまで
上げられるかの安全域を測る。
