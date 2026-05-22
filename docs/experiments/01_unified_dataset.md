# 01. 統合感情データセット構築

## 1. 実験の理由

LLM 残差ストリーム上で「感情ベクトル」を抽出するには **(a) 感情カテゴリの
分布が幅広いコーパス**、**(b) 共通スキーマ**、**(c) 品質フィルタ済みの
contrastive ペア**、の 3 点が必要。既存単一データセットは語彙バイアスが強く、
Plutchik 8 軸を均等に被覆しないので、複数源を統合し正規化したコーパスを
自前で構築する。

## 2. 数式と定義

**Plutchik 8 軸**：$\mathcal{C} = \{\text{joy, sadness, anger, fear, trust, disgust, anticipation, surprise}\}$。

**反対関係**（[src/data/schema.py](../../src/data/schema.py) `PLUTCHIK_OPPOSITE`）：
$$\text{joy}\leftrightarrow\text{sadness},\ \text{anger}\leftrightarrow\text{fear},\ \text{trust}\leftrightarrow\text{disgust},\ \text{anticipation}\leftrightarrow\text{surprise}$$

**Annotator agreement**：複数アノテータがあるコーパスで、最頻ラベル割合
$\mathrm{agree}(x) = \max_c \frac{1}{N}\sum_i \mathbb{1}[y_i = c]$。
$\mathrm{agree}(x) \ge 0.5$ を kept、未満を bypass。

**Contrastive pair**：同一カテゴリ $c$ について
$\text{pair} = (x^{+}_{c}, x^{-}_{c})$、$x^-$ は $c$ の opposite category サンプル。
3 戦略で生成：
- *mined*：同コーパスから opposite category を確率的に抽出
- *llm_swap*：本文を保ち感情極性のみ LLM で反転
- *template*：テンプレートに語彙を流し込む

## 3. 結果

ソース別取り込み（[data/unified/stats.json](../../data/unified/stats.json)）：

| source | rows |
|---|---:|
| go_emotions | 53,206 |
| semeval2018 | 10,689 |
| daily_dialog | 102,979 |
| isear | 7,102 |
| emobank | 10,061 |
| **total** | **184,037** |

フィルタ後（`min_agreement=0.5`）：input 184,037 → kept 113,582 / bypass 25,553。

カテゴリ別 kept：

| anger | anticipation | disgust | fear | joy | neutral | sadness | surprise | trust |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,885 | 5,882 | 1,114 | 3,146 | 12,510 | 70,831 | 3,379 | 2,225 | 9,610 |

Contrastive ペア（[data/contrastive/pairs.stats.json](../../data/contrastive/pairs.stats.json)）：
**8 × 500 = 4000 pairs**（mined 60% / llm_swap 30% / template 10%）。

## 4. 次の実験

→ [02_activation_collection.md](02_activation_collection.md) — 構築したペアを
Llama-3.1-8B-Instruct にかけて残差ストリームを収集する。
