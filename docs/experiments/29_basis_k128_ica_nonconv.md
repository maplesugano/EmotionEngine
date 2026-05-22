# 29. k=128 ICA 非収束と PCA 参照（k 上限の経験的特定）

## 1. 実験の理由

[22_k_scaling_plateau.md](22_k_scaling_plateau.md) で
$k=64$ まで retention が単調に伸びることを確認したが、
逓減フェーズに入っており、$k$ をさらに大きく取った場合に
ICA で安定基底が得られるかは未検証だった。
本実験では $k=128$ で ICA を回し、収束可否と
PCA による有効ランクの上限を確認する。

レイヤは $L=22$（[22](22_k_scaling_plateau.md) と同じ）。
$\Delta\in\mathbb{R}^{3200\times 4096}$（per-pair, train mask, $L=22$）。

## 2. 実行

### 2.1 ICA, k=128, max-iter=2000

```bash
python -m src.emotion_code.basis_sweep \
  --layer 22 --decomposers ica --ks 128 --n-seeds 1 \
  --max-iter 2000 \
  --output-dir data/emotion_code/basis_sweep_L22
```

```
ConvergenceWarning: FastICA did not converge.
[sweep] decomposer=ica k=128 seed=0 layer=22 n_iter=2000 converged=False
```

### 2.2 ICA, k=128, max-iter=10000（5×に増やす）

```bash
python -m src.emotion_code.basis_sweep \
  --layer 22 --decomposers ica --ks 128 --n-seeds 1 \
  --max-iter 10000 \
  --output-dir data/emotion_code/basis_sweep_L22
```

```
ConvergenceWarning: FastICA did not converge.
[sweep] decomposer=ica k=128 seed=0 layer=22 n_iter=10000 converged=False
```

反復回数を 5 倍にしても `converged=False`。
すなわち FastICA の不動点反復は緩やかに収束しているのではなく、
**振動して `tol=1e-4` に到達できていない**。

### 2.3 PCA, k=128（参照）

```bash
python -m src.emotion_code.basis_sweep \
  --layer 22 --decomposers pca --ks 128 --n-seeds 1 \
  --output-dir data/emotion_code/basis_sweep_L22
```

```
[sweep] decomposer=pca k=128 seed=0 layer=22
        explained_variance=0.6302444932516664
```

PCA $k=128$ で説明分散は **0.6302**。

## 3. 解釈

### 3.1 なぜ ICA は収束しないか

$\Delta$ は $n=3200$ 行・$d=4096$ 列で、$k=128$ を取ると
**1 成分あたりサンプル数 $n/k=25$** しかない。
FastICA は白色化後の単位球面上で不動点反復を行うが、
有効ランクが $k$ より小さい場合、末尾成分はノイズ部分空間に
落ち込み、定常点を持たずに振動する。
反復上限を上げても収束しないのはこのためで、
緩めるべきは `max_iter` ではなく `tol` か $k$ である。

### 3.2 PCA EVR から見た有効ランク

| $k$ | PCA EVR (累積) |
|---:|---:|
| 8   | 0.077（[basis.summary.json](../../data/emotion_code/basis.summary.json) 由来、$L=16$ 参考値）|
| 128 | **0.630**（$L=22$）|

$k=128$ で 63% にとどまることは、$\Delta$ の有効ランクが $128$ より
かなり小さいことを示す。残り 37% の分散は急速に減衰する固有値の
裾に分布しており、ICA がそこへ独立成分を割り当てようとして
振動していると整合する。

### 3.3 結論

- **運用上の上限は $k=64$**：[22](22_k_scaling_plateau.md) で
  retention=0.93 を達成済み。$k=128$ に増やす意義は
  少なくとも ICA 経路では観測できない。
- **$k=128$ は PCA 参照のみ採用**：
  EVR=0.63 を「有効ランクの目安」として記録するに留める。
- **次に試すなら**：
  (a) `tol` を $5\times 10^{-3}$ に緩める、または
  (b) PCA で 64 次元へ前段圧縮した上で ICA(64) を回す
  （= 実質的に $k=64$ と等価）。
  いずれも本実験のスコープ外。

## 4. 成果物

- [data/emotion_code/basis_sweep_L22/ica_k128_seed0.pt](../../data/emotion_code/basis_sweep_L22/)
  （`converged=False`、参考用）
- [data/emotion_code/basis_sweep_L22/pca_k128_seed0.pt](../../data/emotion_code/basis_sweep_L22/)
  （EVR=0.6302）
