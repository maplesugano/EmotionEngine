# 19. Lexical-gap steering（行動検証）

## 1. 実験の理由

[18_w_structure.md](18_w_structure.md) で抽出した lexical_gap 成分
（b1, b11, b13）が、**実際に Plutchik 8 とは違う方向の生成を引き起こすか**
を確認。対照群として pan 成分（b0, b5, b8）と 2 成分加算 combo を並べる。

## 2. 数式と定義

**ターゲット 9 種**：
- gap：$b_1, b_{11}, b_{13}$
- pan：$b_0, b_5, b_8$
- combo：$b_1 + b_{11}$, $b_1 + b_{13}$, $b_{11} + b_{13}$

**`caa_match` α 正規化**（重要実装）：basis 行ノルム $\|b_j\|_2 \approx 0.4$、
CAA 中央値ノルム ≈ 4 → そのまま α では無効果。
$$v = b_j \cdot \frac{\mathrm{median}_c \|v_c\|_2}{\|b_j\|_2},\quad h \leftarrow h + \alpha v$$
にリスケールし effective α を計算するモードを追加（このスケール合わせは
以後の必須前処理）。

**外部分類器**：j-hartmann/distilroberta で生成テキストを Plutchik 8 に分類。

スクリプト：[experiments/eval_lexical_gap_steering.py](../../experiments/eval_lexical_gap_steering.py)
規模：9 ターゲット × α∈{0, 1, 2} × 16 中性プロンプト = **432 generations**

## 3. 結果

外部分類器（Hartmann）では **gap 群と pan 群の Plutchik 分布が同一**：
**「Plutchik 語彙の外側」を既存分類器は捉えられない**ことが確認された
（→ [20_lexical_gap_judge.md](20_lexical_gap_judge.md) の動機）。

定性的には：
- **gap_b11**：自己分化・自己観察（"I should know more about myself..."）
- **combo_b1+b11**：過警戒的・偏執的内省（強い不確実性 + 防衛）
- **combo_b11+b13**：決断不能 + 内的葛藤

成果物：[results/lexical_gap_steering/](../../experiments/results/lexical_gap_steering/)
（`generations.parquet`, `classifier.parquet`, `summary_by_target.csv`,
`qualitative.md`）

## 4. 次の実験

→ [20_lexical_gap_judge.md](20_lexical_gap_judge.md) — Hartmann で取れない
差異を **LLM-as-judge + 自由記述ラベル** で実体化する。
