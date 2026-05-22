# 26. 二層構造の構成不変性（layer / k 頑健性）

## 1. 実験の理由

[23_caa_pair_additivity.md](23_caa_pair_additivity.md) で発見された
**「readout 大崩れ × shift 完全加法」の二層構造** が、(L=22, k=64) という
特定構成に依存する artefact なのか、構成不変な性質なのかを切り分ける。
別構成（L=19, k=16、§17 の比較対象）で同条件で再走。

## 2. 数式と定義

[23_caa_pair_additivity.md](23_caa_pair_additivity.md) と同じ加法性指標を、
ICA k=16, L=19 の OLS 重みで再構成した $\hat v_a, \hat v_b$ に適用。
alpha_scale = 0.25384 (`caa_match`)、288 generations、約 43 min。

スクリプト：[experiments/eval_caa_basis_additivity.py](../../experiments/eval_caa_basis_additivity.py)
（同一、weights/basis のみ差し替え）

## 3. 結果

| 指標 | L22 / k=64 | **L19 / k=16** |
|---|---:|---:|
| readout 残差比 median | 0.807 | **0.788** |
| readout 残差比 mean | 0.845 | 0.990 |
| shift-acc 残差 median | 0.000 | **0.000** |

→ **二層構造は構成不変**（layer / k を変えても再現） ⇒ artefact ではない。

ペア類似度相関（[25_additivity_pairsim.md](25_additivity_pairsim.md) と同手順）：
L19/k16 では Pearson 非有意・Spearman r=0.80 (n=4)。disgust+joy の mean
resid ratio が 1.64 と外れ値、k=16 で basis が荒く `cos_w` シグナル弱い、
が原因。**方向は同じ**だが、解像度は k=64 の方がきれい。

成果物：[results/caa_basis_additivity_L19_k16/](../../experiments/results/caa_basis_additivity_L19_k16/)

**Phase 3 残課題の現状整理**：
- (a) layer/k 頑健性 ← **本実験で確認**
- (b) ペア類似度との相関 ← [25_additivity_pairsim.md](25_additivity_pairsim.md) で確認
- (c) 残差 → メタ感情射影 ← [24_additivity_metaemotion_proj.md](24_additivity_metaemotion_proj.md)
  で **negative result**：CAA-pair の残差は既存メタ感情カタログには向かない
- ⇒ 次は **basis-native なペア合成** がメタ感情を induce するかの直接検証へ。

## 4. 次の実験

→ [27_basis_native_additivity_metaemotion.md](27_basis_native_additivity_metaemotion.md) —
CAA-pair ではなく **basis 軸対** で同じ加法性 + メタ感情射影を測る。
