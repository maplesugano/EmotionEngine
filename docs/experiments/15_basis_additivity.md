# 15. basis 軸間の加法性検定（Phase C-2）

## 1. 実験の理由

仮説「感情 = 原子の線形結合」の最初の動作テスト。$\alpha b_i + \beta b_j$ の
同時注入による readout が、単独注入の周辺効果の単純和になるかを測る。
線形和なら「basis = 加法群」、崩れるなら高次相互作用の存在。

## 2. 数式と定義

注入応答 $r(\alpha, \beta) = \tilde h(\alpha b_i + \beta b_j) - \tilde h(0)$、
**加法予測**：
$$r_{\text{pred}}(\alpha, \beta) = r(\alpha, 0) + r(0, \beta)$$

**residual**：$\mathrm{resid}(\alpha,\beta) = r(\alpha,\beta) - r_{\text{pred}}(\alpha,\beta)$

**対象 2 軸の絶対誤差**（self-readout cosine ベース）：
$$\mathrm{err}_i = |\hat s_i^{\text{actual}} - \hat s_i^{\text{pred}}|$$

**全体 ratio**：$\|\mathrm{resid}\|_2 / \|r_{\text{pred}}\|_2$。

スクリプト：[experiments/eval_basis_additivity.py](../../experiments/eval_basis_additivity.py)

## 3. 結果（α∈{−1, 0, +1} と {−2, 0, +2}、4 オフ対角セル平均）

| ペア | err_i | err_j | k=16 全体 ratio (median) |
|---|---:|---:|---:|
| (b4, b7) | 0.019 | 0.026 | 0.76 |
| (b8, b4) | 0.021 | 0.015 | 0.81 |
| (b8, b7) | 0.026 | 0.015 | 0.53 |
| (b8, b11) | 0.028 | 0.030 | 0.94 |

readout 絶対値スケール 0.1–0.2 に対し err は **10–25%**。
**対象 2 軸上では加法性がきれいに成立**。

一方、k=16 全体 ratio は **0.5–0.9** と大きい
（他 14 成分への cross-talk が無視できない）。
b8 を含むペアで顕著で、b8 が emergent / 非線形に獲得された軸であることと整合。

成果物：
- α=±2: [results/basis_additivity/](../../experiments/results/basis_additivity/)
- α=±1: [results/basis_additivity_a1/](../../experiments/results/basis_additivity_a1/)

**結論**：
- ◯ **局所線形**（i, j 軸上）：仮説の局所支持。
- △ **大域 cross-talk**：他軸への漏れが残る。

主指標は今後 `err_i, err_j`、ratio は cross-talk 量の参考値とする。

## 4. 次の実験

→ [16_caa_basis_decomposition.md](16_caa_basis_decomposition.md) — CAA を
basis の線形結合として書き直せるかを数値検証する Phase C-3 へ。
