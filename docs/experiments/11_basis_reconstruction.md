# 11. Held-out 再構成 R²

## 1. 実験の理由

[09_basis_sweep.md](09_basis_sweep.md) で k∈{8, 16} を試したが、どこまで
$k$ を上げると **内在次元** に到達するかは未確認。train で学んだ基底を
val Δ に当てはめた R² が plateau に達する $k$ が「真の内在次元」の下限。

## 2. 数式と定義

train で得た $B \in \mathbb{R}^{k\times d}$ に対し、val サンプル $\Delta^{\text{val}}_i$
を最小二乗で表現：
$$\hat s_i = \arg\min_s \|\Delta^{\text{val}}_i - s B\|_2^2 = \Delta^{\text{val}}_i B^\top (BB^\top)^{-1}$$

再構成 $\hat\Delta_i = \hat s_i B$。

**R²**：
$$R^2_{\text{val}} = 1 - \frac{\sum_i \|\Delta^{\text{val}}_i - \hat\Delta_i\|_2^2}{\sum_i \|\Delta^{\text{val}}_i - \bar\Delta^{\text{val}}\|_2^2}$$

スクリプト：[experiments/eval_basis_reconstruction.py](../../experiments/eval_basis_reconstruction.py)
出力：[reconstruction.csv](../../data/emotion_code/basis_sweep/reconstruction.csv)

## 3. 結果（layer 19）

| decomposer | k | r2_train | r2_val |
|---|---:|---:|---:|
| ica | 8 | 0.254 | 0.244 |
| **ica** | **16** | **0.339** | **0.324** |
| nmf | 8 | 0.236 | 0.224 |
| nmf | 16 | 0.314 | 0.300 |
| pca | 8 | 0.254 | 0.244 |
| pca | 16 | 0.339 | 0.324 |

→ k=8→16 で R² が +0.08 程度伸び、**plateau 未到達**。
PCA と ICA は理論通り同 R²（同じ線形空間）、NMF は非負制約のぶん劣る。
内在次元は 16 を超える ⇒ 後段 [22_k_scaling_plateau.md](22_k_scaling_plateau.md)
で k∈{32, 64} を実走することになる。

## 4. 次の実験

→ [12_layer_consistency.md](12_layer_consistency.md) — 層を跨いで保存される
成分があるか（軸の物理的実体性）を測る。
