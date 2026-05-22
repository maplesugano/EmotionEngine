# 28. L22 / k=64 プリミティブ発見（基礎・質的分析）

## 1. 実験の理由

[22_k_scaling_plateau.md](22_k_scaling_plateau.md) で CAA 再構成が R²=0.960、steering
retention=0.929 に達した L22 / k=64 basis に対して、[14_b8_qualitative.md](14_b8_qualitative.md)
の手法を再適用し、**Plutchik では表現不可能な素朴な感情次元**（プリミティブ）を
L19 / k=16 から上位互換な形で発見する。

主な改善点：
- **k の拡大**：16 → 64 コンポーネント、より細粒度な分解
- **層の深化**：L19 → L22、より final な感情表現を操作
- **retention の向上**：0.71 (k=16/L19) → 0.93 (k=64/L22)、因果的な軸の意味がより純粋

これまでのプリミティブ候補 b4, b7, b8（L19 / k=16）が、L22 / k=64 の
64 次元空間ではどう分化するか、またはより単純化されるかを検証。

---

## 2. 手法

### 2.1 候補軸の選定

[03_per_axis_judge_L22_k64.ipynb](../../notebooks/03_per_axis_judge_L22_k64.ipynb) の
bottom セルで自動選定された以下の 5 軸を使用：

**選定軸（indexとtop-text 解釈）**

| axis | pathology | mean_other_score | category_loading_l2 | top_other_label |
|---:|---:|---:|---:|---|
| 4 | 0.061 | 0.375 | 0.094 | regret |
| 6 | 0.065 | 0.413 | 0.147 | confusion |
| 35 | 0.083 | 0.538 | 0.229 | curiosity |
| 58 | 0.051 | 0.413 | 0.092 | existential uncertainty |
| 60 | 0.053 | 0.538 | 0.105 | confusion |

**選定基準**：
- Pathology score 低い（repetition/toxicity が少ない）
- mean_other_score 高い（Plutchik 8 では説明できない、off-manifold な軸）
- category_loading_l2 低い（Plutchik 8 軸との相関が弱い、独立している）
- Flagged されていない（自動除外対象でない）

### 2.2 Steering 生成

スクリプト：[experiments/eval_basis_qualitative.py](../../experiments/eval_basis_qualitative.py)

```bash
python experiments/eval_basis_qualitative.py \
  --basis data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt \
  --which ols \
  --components 4 6 35 58 60 \
  --alphas -6 -3 0 +3 +6 \
  --n-prompts 32 \
  --max-new-tokens 128 \
  --scale 0.18929 \
  --output results/basis_qualitative_L22_k64.txt
```

**実行環境**：
- 選定軸：[4, 6, 35, 58, 60]（5 軸）
- alpha 値：[-6, -3, 0, +3, +6]（5 段階）
- プロンプト数：32
- 生成数：5 × 5 × 32 = **800 生成**
- 推定時間：6–8 時間（V100 1 枚）

### 2.3 評価

各軸について：

1. **軸の非対称性**：$\alpha \in \{-6, -3, 0, +3, +6\}$ で生成された text の
   意味的距離（負極 vs 正極がどの程度相反するか）
2. **飽和点**：どの α で質的に飽和するか（14 で観測されたように α=+3 vs −6 の
   asymmetry を注視）
3. **命名候補**：Plutchik 8 との関連性と直交性を判定

---

## 3. 予期される結果

### 3.1 Primitive の構造化

L19 / k=16 での発見（b4, b7, b8, ...）が L22 / k=64 では：

- **分化**：1 つの軸が複数の下位軸に分解される
  例：*addressivity* が「命令性 vs 叙述性」と「直接性 vs 仲介性」に分かれる

- **統合**：複数の軸が 1 つの単純軸に統合される
  例：複数の「詐欺的意図」指標が 1 個の軸に纏まる

- **新規発見**：k=64 固有の新しい次元

### 3.2 期待指標

- **最有力候補数**：3–5 個程度（self-consistent で interpretation robust）
- **Plutchik との直交性**：新候補が 8 軸の 50% 以上から独立（cosine < 0.3）
- **Retention への寄与**：個別軸を OLS で CAA に対して当てたときの
  weight magnitude の多様性

---

## 4. 成果物

- **Steering 出力**：[results/basis_qualitative_L22_k64.txt](../../experiments/results/basis_qualitative_L22_k64.txt)
- **分析ノートブック**：[notebooks/04_primitive_analysis_L22_k64.ipynb](../../notebooks/04_primitive_analysis_L22_k64.ipynb)（TBD）
- **メタデータ**：各軸の self-consistency score、top-text 解釈、steering phenotype

---

## 5. 次の実験

→ [29_primitive_retention_scaling.md](29_primitive_retention_scaling.md) — 発見したプリミティブを
複数同時に steering した際の加法性（retention への影響）と interaction を測定。
