# 04. 層スイープ評価

## 1. 実験の理由

CAA ベクトルは層ごとに独立に作れる（[03_caa_steering.md](03_caa_steering.md)）。
どの層に注入すべきかを決めるとともに、「層間で表現がどれだけ似ているか」
の最初の手掛かりを得る。後段の **層間一貫性検証**
（[12_layer_consistency.md](12_layer_consistency.md)）の動機付け。

## 2. 数式と定義

**Validation accuracy**：held-out ペア $(x^+, x^-)$ について、$x^+$ と
$x^-$ の last-token 残差を CAA 軸 $v_{c,\ell}$ に射影した内積符号で
極性を分類した accuracy：
$$\text{val\_acc}_\ell = \frac{1}{N}\sum_i \mathbb{1}\big[\langle h^{(\ell)}(x^+_i),v_{c,\ell}\rangle > \langle h^{(\ell)}(x^-_i),v_{c,\ell}\rangle\big]$$

スクリプト：[experiments/eval_layer_sweep.py](../../experiments/eval_layer_sweep.py)

## 3. 結果

[layer_sweep.json](../../experiments/results/layer_sweep.json)：

| layer | val_acc |
|---:|---:|
| 13 | 0.66375 |
| **16** | **0.665625** |
| 19 | 0.664375 |
| 22 | 0.66375 |

→ 層を変えても精度はほぼ一定（差 < 0.002）。
**「残差表現が層をまたいで似ている」**ことを示唆 → Phase C-2 の
層間一貫性 [12_layer_consistency.md](12_layer_consistency.md) で深掘り。

## 4. 次の実験

→ [05_shift_accuracy.md](05_shift_accuracy.md) — 層を固定して、実際に
ステアリングで生成テキストの感情ラベルが変わるかを測る。
