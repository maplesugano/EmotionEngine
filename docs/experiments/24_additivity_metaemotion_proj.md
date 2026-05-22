# 24. 残差 → メタ感情クラスタ射影

## 1. 実験の理由

[23_caa_pair_additivity.md](23_caa_pair_additivity.md) で発見された
readout 残差（ratio 0.81）が、ランダム noise なのか、それとも
[21_meta_emotion_cluster.md](21_meta_emotion_cluster.md) で同定したメタ感情
方向に乗るのかを直接検証する。乗るなら「Phase 3 の非線形成分の正体は
メタ感情」と主張できる。

## 2. 数式と定義

[23_caa_pair_additivity.md](23_caa_pair_additivity.md) の生成テキスト 288 件を
OpenAI `text-embedding-3-small` で埋め込み $e(\cdot) \in \mathbb{R}^{1536}$。
[21_meta_emotion_cluster.md](21_meta_emotion_cluster.md) の 7 クラスタ
centroid $\mu_k$ を再構築。

各 off-diag セル $(c_a, c_b, \alpha, \beta)$ で：
- $e_{\text{joint}} = e(\text{gen}(\alpha\hat v_a + \beta\hat v_b))$
- $e_{\text{marg}_a} = e(\text{gen}(\alpha\hat v_a))$、$e_{\text{marg}_b}$ 同様
- $e_{\text{base}} = e(\text{gen}(0))$

**埋め込み残差**：
$$e_{\text{resid}} = e_{\text{joint}} - e_{\text{marg}_a} - e_{\text{marg}_b} + e_{\text{base}}$$

**Gain**：
$$\mathrm{gain}_k = \cos(e_{\text{joint}}, \mu_k) - \max_{m\in\{a,b\}}\cos(e_{\text{marg}_m}, \mu_k)$$

**Null**：cell 内シャッフルで $\cos(e_{\text{resid}}, \mu_k)$ の p95 を生成。

スクリプト：[experiments/eval_caa_additivity_metaemotion.py](../../experiments/eval_caa_additivity_metaemotion.py)

## 3. 結果（L=22 / k=64）

| 指標 | 値 |
|---|---:|
| residual top-cos median | **0.068** |
| null p95（shuffle） | 0.044 |
| marginal top-cos median | 0.324 |
| 全 7 クラスタ median `joint − marg_max` | **−0.012 〜 −0.020（全て負）** |

最頻ヒット先（cell_top_cluster）：
`academic ambition / competitive determination` 9/16、
`ironic amusement` 4/16、`uncertainty / indecision` 3/16。

**含意**：
- 残差は **non-random**（null p95 の 1.5×）だが、
  **joint 生成は単独軸より各メタ感情に近づかない**。
- 今回 4 ペアは対立／反対方向が多く joint で打ち消し（cancellation）が
  起きている可能性 → negative result として明示。
- **§21 のメタ感情は CAA 8 軸対の合成ではなく、basis-native な高次
  相互作用に由来する**ことを定量で示す negative result。
- 論文上は「CAA-pair 残差 ≠ メタ感情方向」と書き、メタ感情の起源探索を
  basis-pair 側へ移す（→ [27_basis_native_additivity_metaemotion.md](27_basis_native_additivity_metaemotion.md)）。

成果物：[results/caa_additivity_metaemotion_L22_k64/](../../experiments/results/caa_additivity_metaemotion_L22_k64/)

## 4. 次の実験

→ [25_additivity_pairsim.md](25_additivity_pairsim.md) — 残差比の
**ペア類似度** との相関を測り、加法崩れを構造的に説明する。
