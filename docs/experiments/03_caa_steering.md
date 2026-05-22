# 03. CAA ベクトルとステアリング機構

## 1. 実験の理由

「残差に方向ベクトルを足して感情を操作できる」という Activation Steering の
最も素朴な実装（CAA = Category Average Activation）を、本プロジェクトの
contrastive ペア集合で再現し、以後すべての評価のベースラインを作る。
複雑な分解（NMF/PCA/ICA）に進む前に、**この最小構成で何が動き何が動かないか**
を切り分ける。

## 2. 数式と定義

**CAA ベクトル**（[src/emotion_code/caa.py](../../src/emotion_code/caa.py)）：
$$v_{c,\ell} = \frac{1}{|\mathcal{P}_c|}\sum_{i\in\mathcal{P}_c} h^{(\ell)}(x^+_i) - \frac{1}{|\mathcal{P}_c|}\sum_{i\in\mathcal{P}_c} h^{(\ell)}(x^-_i)$$

カテゴリ $c\in\mathcal{C}$、層 $\ell\in\mathcal{L}$、$\mathcal{P}_c$ は
カテゴリ $c$ のペア集合。形は `[8, 4, d]`。

**ステアリング注入**（pre-forward hook、[src/steering/hook.py](../../src/steering/hook.py)）：
$$h^{(\ell)} \leftarrow h^{(\ell)} + \alpha\, v_{c,\ell}$$

**alpha_unit 正規化**：CAA はカテゴリ間でノルムが 1.5–2× 違う
（[02_activation_collection.md](02_activation_collection.md)）ので、
$\hat v_{c,\ell} = v_{c,\ell} / \|v_{c,\ell}\|_2$ を 1.0 単位として
$\alpha \in \mathbb{R}$ を渡す。これで $\alpha = +2$ が「自然 L2 の 2 倍」を
意味する。

## 3. 結果

- 出力：[data/emotion_code/caa.pt](../../data/emotion_code/caa.pt)
- サマリ：[data/emotion_code/caa.summary.json](../../data/emotion_code/caa.summary.json)
- ステアリング生成：[src/steering/generate.py](../../src/steering/generate.py)
- ステアリング機構自体は壊れていない（後段 [05_shift_accuracy.md](05_shift_accuracy.md)
  で joy/sadness で正の効果を確認）。

## 4. 次の実験

→ [04_layer_sweep.md](04_layer_sweep.md) — どの層で steer すべきかを定量化する。
