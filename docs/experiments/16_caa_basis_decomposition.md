# 16. CAA → basis 線形分解（数値検証）

## 1. 実験の理由

Phase B-5 の CAA カテゴリベクトル $v_{c,\ell}$ と Phase C の基底 $\{b_k\}$ は
**同じ層・同じ残差空間** $\mathbb{R}^{d}$ に住み、生成元データ
（pos/neg ペア $\Delta$）を共有している。したがって CAA は basis の線形結合
として書き直せるはず：
$$v_{\text{joy},\ell} \approx \sum_{k=1}^{K} w_k\, b_k$$
これが成り立てば「Plutchik 8 は独立軸ではなく basis の混合表現」という
根本仮説の **数値証拠** になる。VAD（3 軸）との比較で「named axes では
捉えきれない構造を basis が掴んでいる」ことも判定する。

## 2. 数式と定義

CAA 行列 $V \in \mathbb{R}^{|\mathcal{C}|\times d}$、基底行列 $B\in\mathbb{R}^{k\times d}$。
カテゴリ $c$ ごとに $w_c \in \mathbb{R}^k$ を：

- **OLS**：$\min_w \|v_c - w^\top B\|_2^2$（`np.linalg.lstsq`）
- **NNLS**：上式に $w \ge 0$ 制約（`scipy.optimize.nnls`、NMF basis と整合）
- **LASSO**：$\min_w \|v_c - w^\top B\|_2^2 + \lambda \|w\|_1$、$\lambda$ は
  $\alpha_{\text{rel}} \in \{10^{-3}, 10^{-2}, 10^{-1}\}$ の 3 点

評価指標：
- $R^2_c = 1 - \|v_c - \hat v_c\|_2^2 / \|v_c - \bar v\|_2^2$
- $\cos_c = \langle v_c, \hat v_c\rangle / (\|v_c\|\|\hat v_c\|)$
- $\|w_c\|_0$（疎度）

**VAD baseline**：[08_vad_r2.md](08_vad_r2.md) の $W^{\text{VAD}}\in\mathbb{R}^{3\times d}$ を
basis 同様に扱い、$\min_w \|v_c - w^\top W^{\text{VAD}}\|^2$ を解いて R²。

スクリプト：[experiments/eval_caa_basis_decomposition.py](../../experiments/eval_caa_basis_decomposition.py)

## 3. 結果

**Headline（中央値、全 artifact / 全カテゴリ / 全層）**

| 手法 | median R² | median cos | median ‖w‖₀ |
|---|---:|---:|---:|
| **OLS（basis）** | **0.813** | **0.902** | 16 |
| NNLS（basis） | 0.458 | 0.676 | 8 |
| LASSO（basis 集約） | 0.153 | 0.569 | 1 |
| **VAD baseline** | **0.017** | — | 3 |

→ **basis（k=16, OLS）は VAD の約 48 倍の説明力**。仮説強支持。

**ベスト構成（OLS）**

| decomposer | k | layer | median R² | median cos |
|---|---:|---:|---:|---:|
| ICA | 16 | 22 | **0.878** | 0.937 |
| ICA | 16 | 19 | 0.864 | 0.929 |
| PCA | 16 | 19 | 0.864 | 0.929 |
| ICA | 16 | 13/16 | 0.85 | 0.92 |
| NMF | 16 | 19 | 0.830 | 0.911 |

含意：
- k=8 → k=16 で R² +0.10。後段 [22_k_scaling_plateau.md](22_k_scaling_plateau.md) で
  k=32, 64 を試す動機。
- decomposer 間差は小さい（ICA ≈ PCA > NMF）。「数値はほぼ等価、解釈性で選べ」。
- **NNLS で R² が半減** ⇒ CAA は basis の **両符号** の重ね合わせ（純加算ではない）。
- LASSO α=1e-1 で ‖w‖₀≈1 に潰れ R²=0.15 ⇒ 1 成分では CAA を表せない。

成果物：
- [results/caa_basis_decomposition/decomposition.csv](../../experiments/results/caa_basis_decomposition/decomposition.csv) (640 行)
- [vad_baseline.csv](../../experiments/results/caa_basis_decomposition/vad_baseline.csv)
- `weights/{sweep}__{artifact}.pt`

## 4. 次の実験

→ [17_caa_basis_decomp_steering.md](17_caa_basis_decomp_steering.md) — 数値で
書き直せたなら、**実際にモデルを動かす力（steer）も保たれるか** を測る。
