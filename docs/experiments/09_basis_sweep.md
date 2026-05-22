# 09. 多分解器・多 k スイープ（per-pair Δ 入力）

## 1. 実験の理由

Phase B-5 の基底（[src/emotion_code/basis.py](../../src/emotion_code/basis.py)）は
入力が **カテゴリ平均 Δ**（8 行）なので、原理的に Plutchik 8 部分空間
しか張れない。仮説「言語化粒度より細かい潜在基底」を検証するには、
**カテゴリラベルを忘れた per-pair Δ 行列**を入力にし、複数の分解器
（NMF / PCA / ICA / sparse dictionary）と複数の $k$ で系統スイープする
必要がある。

## 2. 数式と定義

入力データ行列：
$$X^{(\ell)} \in \mathbb{R}^{N\times d},\quad X^{(\ell)}_i = \Delta^{(\ell)}_i$$
（$N=3200$ 訓練ペア、$d=4096$、層 $\ell=19$）。

各分解器は共通シグネチャで $X \approx S\, B$ に分解、
$B \in \mathbb{R}^{k\times d}$ が **基底**、$S\in\mathbb{R}^{N\times k}$ が
**ローディング**：

| 分解器 | 制約・目的関数 |
|---|---|
| **PCA** | $\min \|X - SB\|_F^2$ s.t. $BB^\top = I$（直交） |
| **NMF** | $\min \|X - SB\|_F^2$ s.t. $S, B \ge 0$ |
| **ICA** | $S$ の列が独立成分（FastICA: kurtosis 最大化） |
| **Dict learning** | $\min \|X - SB\|_F^2 + \lambda\|S\|_1$ |

実装：[src/emotion_code/decompose.py](../../src/emotion_code/decompose.py)
（純関数 API） / [src/emotion_code/basis_sweep.py](../../src/emotion_code/basis_sweep.py)
（CLI スイーパ）。

各 artifact に **`train_mask`** を同梱 → 後続の metrics / interpret が
同じ Δ 分割を再現できる。

## 3. 結果（layer 19、k∈{8, 16}、{NMF, PCA, ICA}、2 seeds）

[sweep.summary.json](../../data/emotion_code/basis_sweep/sweep.summary.json)：

| decomposer | k | seed | 収束情報 |
|---|---:|---:|---|
| nmf | 8 | 0/1 | n_iter=469, converged ✅, recon_err=759.65 |
| nmf | 16 | 0/1 | n_iter=1500, converged ❌, recon_err=734.46 |
| pca | 8 | 0/1 | explained_variance ≈ 0.256 |
| pca | 16 | 0/1 | explained_variance ≈ 0.340 |
| ica | 8 | 0/1 | n_iter ≤ 29, converged ✅ |
| ica | 16 | 0/1 | n_iter 51/78, converged ✅ |

成果物：[data/emotion_code/basis_sweep/](../../data/emotion_code/basis_sweep/)
配下の `{decomposer}_k{NNN}_seed{N}.pt`。

## 4. 次の実験

→ [10_label_independence_metrics.md](10_label_independence_metrics.md) — 得た
基底が Plutchik / VAD ラベルから独立しているかを定量化する。
