# 10. ラベル独立性 + Stability メトリクス

## 1. 実験の理由

[09_basis_sweep.md](09_basis_sweep.md) で得た基底 $\{b_j\}$ が、
**(a) 既存ラベル（Plutchik 8 / VAD）から独立**で、かつ **(b) seed を変えても
再現される安定構造**であることを定量化する。仮説「言語化粒度より細かい
潜在基底」の最初の支持証拠を作る。

## 2. 数式と定義

各 Δ サンプル $i$ のローディング $s_{i,j} = (S)_{i,j}$、Plutchik ラベル
$y_i \in \mathcal{C}$、VAD 連続値 $\tilde y_i \in \mathbb{R}^3$。

**MI（mutual information）**：
$\mathrm{MI}(s_{\cdot,j}, y) = H(y) - H(y\mid s_{\cdot,j})$、$s$ は分位ビン化。

**Linear separability acc**：5-fold CV で
$\mathrm{LogReg}(s_{\cdot,j}) \to y$ の accuracy。chance = $1/8 = 0.125$。

**Top-N category dominance**：
$\mathrm{dom}_j = \max_c \frac{|\{i\in \text{top-N}_j : y_i = c\}|}{N}$。
top-N は $|s_{i,j}|$ 上位サンプル。**低いほどラベル独立**。

**VAD explained ratio**：
$\mathrm{vad\_expl}_j = R^2(s_{\cdot,j} \to \tilde y)$（線形回帰）。

**Cosine silhouette of $H$**：基底ベクトル行 $B_{j,\cdot}$ をカテゴリ
クラスタとした場合の silhouette（cosine 距離）。

**Hungarian-cosine stability**：seed $a, b$ の基底 $B^{(a)}, B^{(b)}$ について
コスト行列 $-|\langle B^{(a)}_j, B^{(b)}_l\rangle|$ で Hungarian 割当を解き、
マッチした $|\cos|$ の平均：
$$\mathrm{stab}(B^{(a)}, B^{(b)}) = \frac{1}{k}\sum_j |\langle B^{(a)}_j, B^{(b)}_{\pi(j)}\rangle|$$

実装：[src/emotion_code/basis_metrics.py](../../src/emotion_code/basis_metrics.py)
解釈：[src/emotion_code/basis_interpret.py](../../src/emotion_code/basis_interpret.py)

## 3. 結果

[metrics.summary.csv](../../data/emotion_code/basis_sweep/metrics.summary.csv) /
[stability.summary.csv](../../data/emotion_code/basis_sweep/stability.summary.csv)：

| 指標 | NMF | PCA | ICA |
|---|---|---|---|
| sep_mean (chance 0.125) | ≈ 0.16 | ≈ 0.15 | ≈ 0.15 |
| dom_mean (top-N) | ≈ 0.59 | ≈ 0.50 | ≈ 0.50 |
| vad_explained mean | ≈ 0.005 | ≈ 0.006 | ≈ 0.006 |
| silhouette_cosine_H (k=16) | −0.044 | −0.052 | −0.026 |

Stability：

| decomposer | k | mean pairwise stability |
|---|---:|---:|
| nmf | 8/16 | 1.0000 / 1.0000 |
| pca | 8/16 | 1.0000 / 1.0000 |
| ica | 8 | 1.0000 |
| **ica** | **16** | **0.9543** |

**4 つの重要発見**：
1. どの decomposer でも大半の成分が **Plutchik 8 から強く外れる**。
2. **VAD で説明できる成分はほぼゼロ**（< 1%）→ 仮説を強く支持。
3. **Plutchik 8 silhouette が負** → そもそも残差空間で Plutchik は離散塊にならない。
4. **ICA k=16** がラベル独立性 × stability の両立で最有力候補。

ICA k=16 component 0 例（[ica_k016_seed0.interpret.ica.json](../../data/emotion_code/basis_sweep/ica_k016_seed0.interpret.ica.json)）：
- カテゴリヒスト：`{surprise:5, joy:4, disgust:1}` → 単一感情ではない
- top+：`"Unbelievable! I could never do that."` / `"You have no idea how happy I am for you."`
- top−：`"He brings with him some talented people..."` / `"They're totally unreliable."`
→ 「強い感情を伴う発話 vs. 平板な発話」を横断する**強度軸的な何か**で、
言語化された 8 カテゴリのいずれにも一意対応しない。

## 4. 次の実験

→ [11_basis_reconstruction.md](11_basis_reconstruction.md) — 内在次元の
下限を held-out R² で測る Phase C-2 へ。
