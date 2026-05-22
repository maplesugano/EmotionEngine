# 17. 再構成 CAA で steering（行動検証）

## 1. 実験の理由

[16_caa_basis_decomposition.md](16_caa_basis_decomposition.md) で得た
$\hat v_c = \sum_k w_k b_k$ を CAA の代わりに steer 注入し、shift_acc の
保持率を測る。**「数値で書き直せた」≠「同じく動かせる」** の溝を埋める検証。

## 2. 数式と定義

各カテゴリ $c$、variant $\in \{\text{caa}, \text{ols}, \text{nnls}, \text{lasso}, \text{vad}, \text{random}\}$
について steer ベクトル $u_c$ を：

| variant | $u_c$ |
|---|---|
| caa | $v_c$（生 CAA） |
| ols / nnls / lasso | $\hat v_c = \sum_k w_k^{(\text{variant})} b_k$ |
| vad | $\hat v_c = \sum_a w_a^{(\text{vad})} W^{\text{VAD}}_a$ |
| random | $\|w^{\text{ols}}\|_2$ に揃えたガウス重みで再構成 |

steer 注入し [05_shift_accuracy.md](05_shift_accuracy.md) と同じ shift_acc を測る。

**Retention**：
$$\mathrm{ret}^{(\text{variant})} = \frac{\overline{\mathrm{shift\_acc}}^{(\text{variant})}}{\overline{\mathrm{shift\_acc}}^{(\text{caa})}}$$

スクリプト：[experiments/eval_caa_basis_decomp_steering.py](../../experiments/eval_caa_basis_decomp_steering.py)
（増分 parquet 保存・`--resume` 対応）

## 3. 結果（ICA k=16, L=19, n_prompts=32, α∈{−2,0,2}, 4608 gens, 16h53m）

| variant | mean_shift_acc | mean_baseline | mean_delta | retention_vs_caa |
|---|---:|---:|---:|---:|
| caa | 0.161 | 0.063 | +0.099 | 1.00 |
| **ols** | **0.115** | 0.063 | **+0.052** | **0.71** |
| random | 0.089 | 0.063 | +0.026 | 0.55 |
| nnls | 0.083 | 0.063 | +0.021 | 0.52 |
| lasso | 0.068 | 0.063 | +0.005 | 0.42 |
| vad | 0.057 | 0.063 | −0.005 | 0.35 |

→ **OLS 再構成は CAA の steer 力を 71% 保持**。仮説「CAA = basis の線形結合」
が **数値・行動の両面** で支持。
- VAD は baseline 以下、NNLS/LASSO は random 並み。
- Phase 1 R²=0.86 → Phase 2 retention=0.71 が直接対応。

成果物：[results/caa_basis_decomp_steering/](../../experiments/results/caa_basis_decomp_steering/)

## 4. 次の実験

→ [18_w_structure.md](18_w_structure.md) — OLS 重み行列 $W \in \mathbb{R}^{8\times 16}$
の **構造** を分析し、「Plutchik = 分散表現」を直接示す。
