# 13. Encode-Steer-Re-encode 自己整合性（hero metric）

## 1. 実験の理由

[05_shift_accuracy.md](05_shift_accuracy.md) は外部分類器に依存し、
[08_vad_r2.md](08_vad_r2.md) は VAD ラベルに依存する。**ラベルも分類器も
使わず、モデル内部だけで「これは本当に軸か」を問う** ための最厳格指標。
「自分が出した変化を自分で読める ＝ 軸として閉じている」ことの直接証拠。

## 2. 数式と定義

基底 $b_j$、係数 $\alpha$、プロンプト集合 $\mathcal{X}_0$ について：

1. **注入**：$h^{(\ell)} \leftarrow h^{(\ell)} + \alpha b_j$ で生成 $y(\alpha; j)$。
2. **再エンコード**：$y(\alpha; j)$ を同層 $\ell$ に通し、last-token 残差
	$\tilde h(\alpha; j) \in \mathbb{R}^d$ を取り出す。
3. **全基底へ射影**：$\hat s_l(\alpha; j) = \langle \tilde h(\alpha; j) - \tilde h(0; j),\, b_l\rangle / \|b_l\|^2$。

**Self-monotonicity**：
$$\rho_j = \mathrm{Spearman}\big((\alpha_k)_k,\ (\hat s_j(\alpha_k; j))_k\big),\ \text{平均は }\mathcal{X}_0$$

**Specificity at extreme**：
$$\mathrm{spec}_j(\alpha) = \hat s_j(\alpha; j) - \frac{1}{k-1}\sum_{l\neq j} |\hat s_l(\alpha; j)|$$

**self_delta_cosine**：
$\mathrm{self\_\Delta}_j(\alpha) = \cos\big(\tilde h(\alpha; j) - \tilde h(0; j),\ b_j\big)$

スクリプト：[experiments/eval_basis_selfconsistency.py](../../experiments/eval_basis_selfconsistency.py)

## 3. 結果

### 3.1 CAA pseudo-basis sanity（α∈{−2, 0, +2}, n_prompts=4）

[caa__overall.json](../../experiments/results/basis_selfconsistency/caa__overall.json)：
monotonicity_mean = **0.125**, specificity = **−0.053**。

| component | self+2 | self−2 | ρ_mean |
|---|---:|---:|---:|
| 4 (joy) | +0.040 | −0.005 | 0.125 |
| 5 (sadness) | −0.016 | −0.022 | 0.125 |
| 0 (anger) | +0.011 | −0.048 | 0.125 |

→ **CAA は self-consistency をほぼ満たさない**。shift_acc は通っていた
（[05_shift_accuracy.md](05_shift_accuracy.md), joy 0.40 / sadness 0.30）が、
self-consistency は **生成テキストを再エンコードした残差が steer 方向に再度
乗ること** を要求するため、圧倒的に厳しい。

### 3.2 ICA k=16 全成分（α∈{−2,0,+2}, n_prompts=3）

[ica_k016_seed0__overall.json](../../experiments/results/basis_selfconsistency_full/ica_k016_seed0__overall.json)：
monotonicity_mean = **0.146**（CAA より高い）。

注目成分：

| j | self+2 | self−2 | ρ_self | label_dom | MI | 判定 |
|---:|---:|---:|---:|---:|---:|---|
| **b8** | +0.009 | −0.038 | **0.91** | 0.25 | 0.14 | ★ 最有力 |
| **b7** | +0.021 | −0.024 | 0.79 | 0.375 | 0.21 | ★ |
| **b4** | +0.004 | −0.010 | 0.38 | 0.375 | 0.16 | ○ |
| b11 | +0.003 | −0.002 | 0.91 | **1.00** | 0.21 | ✗ label-leak |
| b5 | 0.0 | +0.047 | 0.866 | 0.5 | — | △ 単側のみ |

### 3.3 ラベル独立 × 機能性クロス

| 区分 | 件数 |
|---|---:|
| functional（α=±2 で符号正） | 6 / 16 |
| label-independent（dom ≤ 0.375） | 6 / 16 |
| **両立する候補感情原子** | **3 / 16** (b4, b7, b8) |

→ **ICA b8 は ρ=0.91 / dominance=0.25** で、CAA のどの軸より明確に
self-consistent。「ラベル独立な感情原子」最有力候補。

## 4. 次の実験

→ [14_b8_qualitative.md](14_b8_qualitative.md) — b8 を強い α で steer して
**軸の意味を因果的に読む**。
