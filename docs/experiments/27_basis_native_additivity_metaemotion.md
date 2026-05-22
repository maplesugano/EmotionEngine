# 27. basis-native 加法性 + メタ感情射影

## 1. 実験の理由

[24_additivity_metaemotion_proj.md](24_additivity_metaemotion_proj.md) で
「CAA-pair 残差 ≠ メタ感情方向」が示された。次の仮説は
**「メタ感情は basis-native なペア合成（$\alpha b_i + \beta b_j$）で induce
される」**。lexical-gap 軸対（b1, b11, b13）と pan 軸対（b8, b4, b7）で
直接検証する。同時に [15_basis_additivity.md](15_basis_additivity.md) を
メタ感情射影付きで再実走する位置付け。

## 2. 数式と定義

[23_caa_pair_additivity.md](23_caa_pair_additivity.md) の加法性指標 +
[24_additivity_metaemotion_proj.md](24_additivity_metaemotion_proj.md) の
gain / null を **basis 軸対** $b_i, b_j$ に適用。`caa_match` 正規化必須。

スクリプト：[experiments/eval_basis_additivity_metaemotion.py](../../experiments/eval_basis_additivity_metaemotion.py)

設定：5 ペア（(b1,b11), (b1,b13), (b11,b13), (b8,b4), (b8,b7)）
× 3×3 α × 8 prompts = **360 gens**

## 3. 結果

### 3.1 ICA k=16 / L=19

**(a) readout 加法性**（off-diag セル平均）：

| ペア | err_i | err_j | resid_ratio |
|---|---:|---:|---:|
| (b1, b11) | 0.064 | 0.077 | 0.897 |
| (b1, b13) | 0.080 | 0.052 | 0.855 |
| (b11, b13) | 0.064 | 0.057 | 0.799 |
| (b8, b4) | 0.066 | 0.059 | 0.839 |
| (b8, b7) | 0.061 | 0.086 | 0.782 |
| **median** | **0.069** | **0.061** | **0.817** |

→ CAA-pair（[15_basis_additivity.md](15_basis_additivity.md), err≈0.015–0.030）の
**2–4 倍大きい err** ⇒ basis 軸対は CAA 対より局所線形性が弱い。

**(b) per-cluster gain（`joint − max(marg)` median）**：

| cluster | median gain |
|---|---:|
| frustration / despair | −0.013 |
| academic ambition / competitive | −0.019 |
| ironic amusement | −0.019 |
| self-doubt / encouragement | −0.020 |
| enthusiasm / curiosity | −0.021 |
| romantic idealism | −0.022 |
| **uncertainty / indecision** | **−0.026** |

**全 7 クラスタで gain 負**。joint は単独軸の最大 cos を上回らない。
[24_additivity_metaemotion_proj.md](24_additivity_metaemotion_proj.md) と
**同じ negative result が basis-pair でも再現**。

per-cell 残差は構造化（最大 |cos_resid| ≈ 0.21、`+2,+2` セルで複数ペアが
*academic ambition* に共通正残差 = strong-α アトラクタ）。

### 3.2 ICA k=64 / L=22 で再走（容量仮説の切り分け）

「k=16 容量不足が原因」を切り分けるため champion basis で再走。

| 構成 | err_i | err_j | resid_ratio |
|---|---:|---:|---:|
| L19 / k=16 | 0.069 | 0.061 | 0.817 |
| **L22 / k=64** | **0.049** | 0.063 | **0.803** |

per-cluster gain は依然 **全 7 クラスタ負**、ただし 5/7 で gain 縮小
（uncertainty 半減）。**容量を上げると marginal がメタ感情を吸収しきり、
joint で初めて出る方向は新たには見えない**。

### 3.3 結論

- 二層構造は basis-pair でも同パターン再現、ただし err は 2–4 倍悪い
  ⇒ basis 軸はより小さい局所性しか持たない。
- メタ感情射影は **L19/k16 と L22/k64 の両方で全 7 クラスタ joint < marg_max**。
  **「メタ感情 = basis 2 軸合成」仮説は反証寄り**。
- 仮説 (i)「k=16 容量不足が原因」は否定。
- **方針転換**：仮説 (iii)「単独 basis 軸 × strong α の周辺効果がメタ感情の
  正体」を主軸に据える。[14_b8_qualitative.md](14_b8_qualitative.md) で
  observed したような *single-axis × strong-α* で十分再現できる、という
  方向で検証を組み直す。

成果物：
- [results/basis_additivity_metaemotion_L19_k16/](../../experiments/results/basis_additivity_metaemotion_L19_k16/)
- [results/basis_additivity_metaemotion_L22_k64/](../../experiments/results/basis_additivity_metaemotion_L22_k64/)

## 4. 次の実験（未実走、提案）

→ **single-axis × strong-α でメタ感情を直接 induce**：
[14_b8_qualitative.md](14_b8_qualitative.md) の枠組みを judge + クラスタ
射影パイプラインに通し、**per-axis × per-cluster gain matrix** を作る。
basis_qualitative の出力 parquet を `eval_caa_additivity_metaemotion.py`
互換スキーマで保存できる薄い adapter を書き、α=0 を baseline として
`α≠0 − α=0` cosine gain を取る形式が最小実装。
