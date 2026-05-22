# 02. 残差ストリーム活性化の収集

## 1. 実験の理由

CAA / 基底 / VAD いずれも入力は同じ「残差ストリーム上の last-token ベクトル」。
複数層を一度に保存しておけば後段の **層スイープ** や **層間一貫性検証**
（[12_layer_consistency.md](12_layer_consistency.md)）が再収集なしで回せる。
GPU 時間が支配的なので「収集は 1 回、評価は何度でも」体制を作る。

## 2. 数式と定義

**残差ストリーム**：Transformer 層 $\ell$ の入力 $h^{(\ell)} \in \mathbb{R}^{T\times d}$。
hook は **pre-forward** で $h^{(\ell)}$ を読む（Attention/MLP 適用前）。

**last-token 抽出**：プロンプト最終トークン位置 $t = T-1$ の
$h^{(\ell)}_{T-1} \in \mathbb{R}^d$ を保存。

**Contrastive Δ**：ペア $(x^+_i, x^-_i)$ について
$$\Delta_i^{(\ell)} = h^{(\ell)}_{T-1}(x^+_i) - h^{(\ell)}_{T-1}(x^-_i) \in \mathbb{R}^d$$

これが Phase B-5 (CAA) と Phase C (基底) の共通入力。

## 3. 結果

- モデル：`meta-llama/Llama-3.1-8B-Instruct`、$d = 4096$
- フック層：$\mathcal{L} = \{13, 16, 19, 22\}$（中後段の残差）
- token 位置：last
- 出力：[data/activations/llama/](../../data/activations/llama/) に
  `shard_NNNNN_{pos,neg}.safetensors`（4000 ペア × 4 層）

カテゴリ別ベクトル L2 ノルム平均（[caa.summary.json](../../data/emotion_code/caa.summary.json)）：

| cat | L13 | L16 | L19 | L22 |
|---|---:|---:|---:|---:|
| anger | 1.95 | 2.70 | 3.62 | 4.78 |
| anticipation | 2.38 | 3.40 | 4.51 | 6.22 |
| disgust | 2.11 | 2.86 | 3.97 | 5.34 |
| fear | 2.11 | 2.96 | 3.93 | 5.23 |
| joy | 1.59 | 2.13 | 2.85 | 3.62 |
| sadness | 2.40 | 3.32 | 4.42 | 5.90 |
| surprise | 1.63 | 2.23 | 3.11 | 4.06 |
| trust | 2.04 | 2.89 | 3.95 | 5.44 |

→ 層が深いほどノルム単調増加、カテゴリ間 1.5–2× ばらつき。
**「α スケーリングだけでは強度を揃えられない」**ことが判明し、
ステアリング側で `alpha_unit` 正規化を入れる根拠になった。

## 4. 次の実験

→ [03_caa_steering.md](03_caa_steering.md) — 収集した Δ から CAA ベクトルを
作り、ステアリング機構を確立する。
