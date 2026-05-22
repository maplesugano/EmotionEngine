# 07. Perplexity guardrail（安全域）

## 1. 実験の理由

α を上げれば感情は強まるが、生成が崩壊（無意味語列化）すると評価不能。
**「PPL が baseline の 2 倍を超えない最大 α」** を `alpha_unit` 単位で
測り、後段すべての実験で α 範囲の上限根拠を与える。

## 2. 数式と定義

参照モデル（Llama-3.1-8B-Instruct 自身）で生成テキスト $y$ の perplexity：
$$\mathrm{PPL}(y) = \exp\!\left(-\frac{1}{|y|}\sum_{t} \log p(y_t \mid y_{<t})\right)$$

baseline $\mathrm{PPL}_0 = \mathrm{PPL}\big(\text{gen}(x; 0)\big)$。
許容最大 $\alpha$：
$$\alpha^*_c = \max\big\{\alpha\;\big|\;\mathrm{PPL}\big(\text{gen}(x; \alpha v_c)\big) \le 2\,\mathrm{PPL}_0\big\}$$

スクリプト：[experiments/eval_perplexity.py](../../experiments/eval_perplexity.py)
出力：[perplexity_alpha.csv](../../experiments/results/perplexity_alpha.csv)

## 3. 結果

baseline PPL = 4.96。$\alpha^*_c$（PPL ≤ 2× baseline）：

| cat | max α |
|---|---:|
| anger | 3.0 |
| anticipation | 2.0 |
| disgust | 5.0 |
| fear | **1.0** |
| joy | 5.0 |
| sadness | **1.0** |
| surprise | 4.0 |
| trust | 2.0 |

α=5 まで上げると **fear / sadness / anticipation で PPL が 10⁴ オーダー**
まで爆発（生成崩壊）。joy / disgust / surprise は α=5 でも安定（PPL < 10²）。
**注入余地はカテゴリ依存**で、median max α ≈ 2.5。これが以後 α∈{−2, 0, +2}
グリッドの根拠。

## 4. 次の実験

→ [08_vad_r2.md](08_vad_r2.md) — Plutchik とは別軸（VAD）で残差の
線形可分性を測る。
