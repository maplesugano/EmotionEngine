# 12. 層間一貫性（Hungarian + |cos| マッチング）

## 1. 実験の理由

[04_layer_sweep.md](04_layer_sweep.md) で「層を変えても val_acc がほぼ同じ」
ことが分かっている。同じ (decomposer, k, seed) を複数層で学習し、成分が
**層を跨いで保存される** なら、その方向はモデルが同じ概念として扱う
**真の感情原子の最有力候補**。

## 2. 数式と定義

層 $\ell \in \{13, 16, 19, 22\}$ で同じ手続きで $B^{(\ell)} \in \mathbb{R}^{k\times d}$ を学習。
2 層 $a, b$ について Hungarian アルゴリズムでコスト
$C_{j,l} = -|\langle B^{(a)}_j, B^{(b)}_l\rangle|$ を最小化する割当 $\pi$ を求める。

**ペア平均**：$\overline{|\cos|}_{a,b} = \frac{1}{k}\sum_j |\langle B^{(a)}_j, B^{(b)}_{\pi(j)}\rangle|$

**成分別アンカー** ($\ell^* = 19$)：他 3 層へのマッチを取り、$|\cos|$ の平均が
高い成分が「層横断で生き残る」軸。

スクリプト：[experiments/eval_basis_layerconsistency.py](../../experiments/eval_basis_layerconsistency.py)
出力：[layer_pair_consistency.csv](../../data/emotion_code/basis_sweep/layer_pair_consistency.csv)、
[layer_component_consistency.csv](../../data/emotion_code/basis_sweep/layer_component_consistency.csv)

## 3. 結果（ICA k=16）

ペア平均 |cos|：

| L_a | L_b | mean | min | median |
|---|---|---|---|---|
| 13 | 16 | 0.51 | 0.17 | 0.57 |
| 13 | 19 | 0.45 | 0.19 | 0.44 |
| 13 | 22 | 0.35 | 0.21 | 0.36 |
| 16 | 19 | 0.63 | 0.34 | 0.71 |
| 16 | 22 | 0.52 | 0.31 | 0.58 |
| **19** | **22** | **0.73** | 0.42 | 0.76 |

L19 アンカーの成分別平均 |cos|（注目成分のみ）：

| L19 j | mean abs_cos | candidate |
|---|---|---|
| **b7** | **0.70** | ○ 断片化軸 |
| **b4** | **0.64** | ○ |
| **b8** | 0.47 (下位 2/16) | ◎ 低層では見えていない |

サニティチェック：L16 で b8 の Hungarian マッチ先（b5, |cos|=0.34）を
そのまま [13_self_consistency.md](13_self_consistency.md) に通すと
ρ=0.17, self_−2=0.000 → **マッチだけでは機能的保存ではない**。

→ **b8 は L19〜22 の中後段で emergent に定着した軸**。「全層で生き残る原子」
ではないが、中後段では明確に確立した軸。

## 4. 次の実験

→ [13_self_consistency.md](13_self_consistency.md) — 軸が「自分が出した変化を
自分で読める」（hero metric）かを測る。
