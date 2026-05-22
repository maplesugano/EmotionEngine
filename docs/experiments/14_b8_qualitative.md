# 14. b8 の質的解釈（steering 生成例）

## 1. 実験の理由

[10_label_independence_metrics.md](10_label_independence_metrics.md) の
top-text 解釈は「入力テキストとの相関」しか見えない。**軸の本当の意味は
steer された生成でしか分からない**。[13_self_consistency.md](13_self_consistency.md)
で最有力候補と判定された b8 を強い α で steer し、Plutchik では命名できない
次元の **最初の具体例** を作る。

## 2. 数式と定義

[07_perplexity_guardrail.md](07_perplexity_guardrail.md) の median max α ≈ 2.5
だと b8 の生成はベースラインとほぼ同じ。基底ノルム $\|b_8\|_2 \approx 0.4$
は CAA 中央値ノルム ≈4 より一桁小さいので、effective scale を：
$$v_8 = \frac{b_8}{\|b_8\|_2},\quad \alpha_{\text{eff}} \in \{-6, -3, 0, +3, +6\}$$
で注入（後段の `caa_match` α 正規化の原型）。

スクリプト：[experiments/eval_basis_qualitative.py](../../experiments/eval_basis_qualitative.py)
出力：[results/basis_qualitative_b8_strong.txt](../../experiments/results/basis_qualitative_b8_strong.txt)

## 3. 結果

- **負極 α=−6**：話者が相手に直接訴える修辞構造。
  例：`I'm not going to tell you again. ... You're not listening. You're not paying attention.`
  / `I'm here to learn. I'm here to see the world.`
- **正極 α=+6**：第三者視点・分析的フレーム。最強ではプロンプトを
  **読解問題に変換**：`What does the person want to do? A) ... B) ...`
- **非対称**：正極は α=+3 で飽和、負極は α=−6 まで escalation。
  負極＝「能動」、正極＝「deactivation / detachment」。

**軸の命名仮**：*addressivity（対人的訴求性）vs. analytical detachment*。

**方法論的教訓**：
- top-text による命名（「個人意見 vs 礼儀定型」）は入力相関に過ぎない。
- **因果的な軸の意味は steering 生成でしか出ない**。両方やって初めて軸の正体が見える。
- Plutchik 8 に対して「提領態度軸」は斜交：仲良くしたい怒りも、距離をとった
  喜びもありうる ⇒ **「Plutchik では表せない感情次元」の最初の具体例**。

## 4. 次の実験

→ [15_basis_additivity.md](15_basis_additivity.md) — 複数の基底軸を同時に
steer したとき、効果が線形和になるかを測る。
