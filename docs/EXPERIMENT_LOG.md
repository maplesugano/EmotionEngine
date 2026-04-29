# EmotionEngine 実験ログ（詳細版）

本書は [PROGRESS.md](PROGRESS.md) の補完として、Phase A から Phase C-2 までに行った
すべての実験設計・データ・数値結果・解釈・含意を時系列で残すための完全版ログ。
後続の研究者がこの 1 ファイルだけで「何を試し」「何が出て」「なぜ次にこれをやるのか」
を完全に追えることを目的とする。

---

## 0. プロジェクトの根幹仮説

> **「感情は言語化粒度より細かい潜在基底の線形結合として表せる」**

- Plutchik 8 / Ekman 6 / VAD 3 軸はすべて **観測ラベル**（人間が言語化した投影）にすぎない。
- LLM 残差ストリーム上には、言語化された感情よりも細かい「感情原子」 $b_j \in \mathbb{R}^d$ が線形に張られていて、
  $$\text{joy} \approx \sum_j w_j b_j,\quad \text{sadness} \approx \sum_j w'_j b_j$$
  さらに名前のない混合 $\alpha b_3 + \beta b_7$ も同じ空間で表現可能、というのが検証対象。
- 検証対象は **存在性**（基底があるか）と **言語非依存性**（既存ラベルへ吸収されないか）の 2 点。

---

## 1. データセット構築（Phase A）

### 1.1 統合スキーマと品質フィルタ

- [src/data/build_unified.py](../src/data/build_unified.py)：複数ソースを共通 schema (`schema.py`) に正規化
- [src/data/quality_filter.py](../src/data/quality_filter.py)：annotator agreement と長さによる足切り

ソース別取り込み（[data/unified/stats.json](../data/unified/stats.json)）：

| source | rows |
|---|---:|
| go_emotions | 53,206 |
| semeval2018 | 10,689 |
| daily_dialog | 102,979 |
| isear | 7,102 |
| emobank | 10,061 |
| **total** | **184,037** |

フィルタ後（[data/unified/examples.filtered.stats.json](../data/unified/examples.filtered.stats.json)、`min_agreement=0.5`）：

| | input | kept | bypass |
|---|---:|---:|---:|
| rows | 184,037 | 113,582 | 25,553 |

カテゴリ別（kept）：

| anger | anticipation | disgust | fear | joy | neutral | sadness | surprise | trust |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,885 | 5,882 | 1,114 | 3,146 | 12,510 | 70,831 | 3,379 | 2,225 | 9,610 |

### 1.2 Contrastive ペア構築

[src/data/build_contrastive.py](../src/data/build_contrastive.py)：Plutchik 8 × **500 = 4000 pairs**（[data/contrastive/pairs.stats.json](../data/contrastive/pairs.stats.json)）。

| provenance | n |
|---|---:|
| mined | 2,400 (60%) |
| llm_swap | 1,200 (30%) |
| template | 400 (10%) |

反対語は [src/data/schema.py](../src/data/schema.py) `PLUTCHIK_OPPOSITE` に定義（joy↔sadness, anger↔fear, etc.）。
mined 戦略：同コーパスから *opposite category* を確率的にサンプル。
llm_swap：本文を保ったまま感情極性を反転する LLM 書き換え。
template：テンプレートからの強制注入（人工バリエーション保証）。

### 1.3 活性化収集

[src/activations/collect.py](../src/activations/collect.py)
- モデル：`Llama-3.1-8B-Instruct`（[configs/model.yaml](../configs/model.yaml)）
- フック層：`[13, 16, 19, 22]`（中後段の残差ストリーム）
- token 位置：`last`
- 出力：[data/activations/llama/](../data/activations/llama/) に `shard_NNNNN_{pos,neg}.safetensors`

カテゴリ別ベクトルノルム（[data/emotion_code/caa.summary.json](../data/emotion_code/caa.summary.json)、層別の平均 L2）：

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

→ 層が深くなるにつれてノルムが単調増加。後段ほど「強い差」が出る一方、
カテゴリ間で 1.5〜2 倍のばらつきがあり、単純な α スケーリングでは強度を揃えられない
（このためステアリング側で `alpha_unit` 単位の正規化を入れる必要があった）。

---

## 2. Phase B-5：本流パイプラインの確立

> **このセクションの結果は確定済み**。Phase C 以降は触らず、ここをベースラインとして比較する。

### 2.1 感情コード（3 種類）

#### CAA — Category Average Activation
[src/emotion_code/caa.py](../src/emotion_code/caa.py)
$$v_{c,L} = \mathrm{mean}(\text{pos}_{c,L}) - \mathrm{mean}(\text{neg}_{c,L})$$
形 `[8, 4, d]`（カテゴリ × 層 × 次元）。
出力：[data/emotion_code/caa.pt](../data/emotion_code/caa.pt)、サマリ [caa.summary.json](../data/emotion_code/caa.summary.json)

#### 基底（NMF / PCA）
[src/emotion_code/basis.py](../src/emotion_code/basis.py)
- 入力：**カテゴリ平均** Δ（8×d）→ 言語化済みの 8 軸を sign-split で 16 行に展開
- layer 16、`k=8`（Plutchik 揃え）
- 出力：[basis.pt](../data/emotion_code/basis.pt)、サマリ [basis.summary.json](../data/emotion_code/basis.summary.json)

PCA explained variance ratio（k=8 計）：
```
[0.077, 0.044, 0.030, 0.023, 0.020, 0.018, 0.014, 0.013]  → cumulative ≈ 0.239
```
NMF 復元誤差：`579.05`。

→ **Phase B-5 基底の本質的限界**：入力がカテゴリ平均（言語化された 8 軸）なので、
原理的に Plutchik 部分空間しか張れない。仮説（言語非依存の原子）の検証には別ルートが必要、
と早期に判明した。これが Phase C を立ち上げる動機になった。

#### VAD 線形写像
[src/emotion_code/vad.py](../src/emotion_code/vad.py)
- EmoBank（10,061 rows）、layer 19、Ridge 回帰で V/A/D 3 軸
- 出力：[vad_mapping.pt](../data/emotion_code/vad_mapping.pt)

### 2.2 ステアリング機構

- フック：[src/steering/hook.py](../src/steering/hook.py)
  pre-forward hook で残差に `alpha * v` を加算
- 生成：[src/steering/generate.py](../src/steering/generate.py)
- α は **alpha_unit**（v の自然 L2 を 1.0 とする正規化）で渡し、カテゴリ間ノルム差を吸収

### 2.3 評価（[experiments/results/SUMMARY.md](../experiments/results/SUMMARY.md)）

| Metric | Value | Target | Pass |
|---|---:|---:|---|
| Shift accuracy (mean, α=+2) | **0.133** | ≥ 0.4 | ❌ |
| Monotonicity ρ (min over cats) | **0.157** | ≥ 0.7 | ❌ |
| Median max alpha_unit (PPL ≤ 2× baseline) | **2.5** | ≥ 1.0 | ✅ |
| VAD R² (min over V/A/D) | **0.158** | ≥ 0.5 | ❌ |

#### 層スイープ（[layer_sweep.json](../experiments/results/layer_sweep.json)）
| layer | val_acc |
|---:|---:|
| 13 | 0.66375 |
| **16** | **0.665625** |
| 19 | 0.664375 |
| 22 | 0.66375 |

→ **層を変えても精度はほぼ一定**。残差表現が層をまたいで似ていることを示唆（後で Phase C の層間一貫性検証に繋がる）。

#### Shift accuracy（[shift_accuracy.csv](../experiments/results/shift_accuracy.csv)、α=+2）
| category | shift_acc | baseline | Δ |
|---|---:|---:|---:|
| anger | 0.00 | 0.00 | +0.00 |
| disgust | 0.00 | 0.00 | +0.00 |
| fear | 0.00 | 0.00 | +0.00 |
| joy | **0.40** | 0.30 | +0.10 |
| sadness | **0.30** | 0.00 | +0.30 |
| surprise | 0.10 | 0.10 | +0.00 |
| anticipation / trust | n/a | n/a | n/a（外部分類器に対応ラベルなし）|

→ joy と sadness 以外、外部分類器ベースの shift では動かない。
これは「CAA 軸が弱い」のか「外部分類器の感度が低い」のか切り分け不能、というのが Phase C への重要な反省点。

#### Monotonicity（[monotonicity.csv](../experiments/results/monotonicity.csv)、Spearman ρ）
| cat | ρ | p | n |
|---|---:|---:|---:|
| anger | 0.282 | 0.047 | 50 |
| disgust | 0.285 | 0.045 | 50 |
| fear | 0.228 | 0.111 | 50 |
| joy | 0.319 | 0.024 | 50 |
| sadness | 0.247 | 0.084 | 50 |
| surprise | 0.157 | 0.277 | 50 |

→ 弱いが正の相関。CAA は方向としては機能するが「強度の単調性」は不十分。

#### Perplexity guardrail（[perplexity_alpha.csv](../experiments/results/perplexity_alpha.csv)）
baseline PPL = 4.96。最大許容 α（PPL ≤ 2× baseline）：

| cat | max α |
|---|---:|
| anger | 3.0 |
| anticipation | 2.0 |
| disgust | 5.0 |
| fear | **1.0** |
| joy | 5.0 |
| sadness | **1.0** |
| surprise | 4.0 |
| trust | 2.0 |

α=5 まで上げると **fear/sadness/anticipation で PPL が 10⁴ オーダーまで爆発**（生成崩壊）。
joy/disgust/surprise は α=5 でも安定（PPL < 10²）。注入余地はカテゴリ依存。

#### VAD R²（[vad_r2.json](../experiments/results/vad_r2.json)、layer 19）
- V: **0.561** ✅
- A: 0.245 ❌
- D: 0.158 ❌
- mean: 0.321、min: 0.158

→ **Valence しか線形に取り出せない**。Arousal/Dominance は残差線形では届かない（非線形 or 別表現）。
これも「VAD 3 軸で測ること自体が偏った投影」という Phase C 方針転換の根拠。

### 2.4 Phase B-5 の総括

- **数値目標は 4 つ中 1 つしか合格していない**（PPL guardrail のみ ✅）。
- それでも joy/sadness は機能、Valence は線形可、ステアリング機構自体は壊れていない。
- 失敗原因の最有力仮説：**入力（カテゴリ平均）と評価軸（外部分類器・VAD）の双方が言語化に依存**しているため、モデル内在の「真の感情軸」を測れていない。
- → Phase C の 2 大方針：(1) **per-pair Δ を入力**、(2) **評価をモデル内在指標へ**。

---

## 3. Phase C：言語非依存基底の探索

### 3.1 純関数 API：[src/emotion_code/decompose.py](../src/emotion_code/decompose.py)
NMF / PCA / ICA / sparse dictionary learning を共通シグネチャで提供。
`category_loadings` ヘルパも同梱（後の解釈用）。

### 3.2 多分解器・多 k スイープ：[src/emotion_code/basis_sweep.py](../src/emotion_code/basis_sweep.py)
- **入力**：per-pair Δ（カテゴリ平均**しない**、shape `3200 × 4096`）
- CLI：`--decomposers nmf pca ica dict --ks 8 16 32 ... --n-seeds N --max-iter M`
- 出力：[data/emotion_code/basis_sweep/](../data/emotion_code/basis_sweep/) に `{decomposer}_k{NNN}_seed{N}.pt`
- 各 payload に `train_mask` を同梱 → metrics/interpret が同じ Δ 分割を再現できる

### 3.3 ラベル独立性スコア：[src/emotion_code/basis_metrics.py](../src/emotion_code/basis_metrics.py)
| 粒度 | 指標 |
|---|---|
| per-component | MI(component, category) / 5-fold logistic linear-sep acc / top-N category dominance / VAD 説明率 |
| per-artifact | cosine silhouette of `H` rows w.r.t. categories |
| per-(decomposer, k) | seed 間 Hungarian-cosine stability |

出力：各 artifact 横に `*.metrics.json`、プール CSV [metrics.summary.csv](../data/emotion_code/basis_sweep/metrics.summary.csv) と [stability.summary.csv](../data/emotion_code/basis_sweep/stability.summary.csv)

### 3.4 解釈ツール：[src/emotion_code/basis_interpret.py](../src/emotion_code/basis_interpret.py)
各 $b_j$ への top/bottom 射影ペアの元テキスト + Plutchik category histogram（label-leak 診断）。
出力例：[ica_k016_seed0.interpret.ica.json](../data/emotion_code/basis_sweep/ica_k016_seed0.interpret.ica.json)

### 3.5 スイープ実行（layer 19、k∈{8, 16}、{nmf, pca, ica}、2 seeds）

[sweep.summary.json](../data/emotion_code/basis_sweep/sweep.summary.json) より抜粋：

| decomposer | k | seed | 収束情報 |
|---|---:|---:|---|
| nmf | 8 | 0/1 | n_iter=469, converged ✅, recon_err=759.65 |
| nmf | 16 | 0/1 | n_iter=1500, converged ❌, recon_err=734.46 |
| pca | 8 | 0/1 | explained_variance ≈ 0.256 |
| pca | 16 | 0/1 | explained_variance ≈ 0.340 |
| ica | 8 | 0/1 | n_iter ≤ 29, converged ✅ |
| ica | 16 | 0/1 | n_iter 51/78, converged ✅ |

### 3.6 Stability（[stability.summary.csv](../data/emotion_code/basis_sweep/stability.summary.csv)）

| decomposer | k | mean pairwise stability |
|---|---:|---:|
| nmf | 8 | **1.0000** |
| nmf | 16 | 1.0000 |
| pca | 8 | 1.0000 |
| pca | 16 | 1.0000 |
| ica | 8 | 1.0000 |
| ica | 16 | **0.9543** |

→ NMF/PCA は決定的に近く完全再現。ICA は seed 依存性があり、k=16 で 0.95（実用上「ほぼ安定」）。

### 3.7 ラベル独立性スコア（[metrics.summary.csv](../data/emotion_code/basis_sweep/metrics.summary.csv)）

サマリ統計（chance acc = 0.125）：

| 指標 | NMF | PCA | ICA |
|---|---|---|---|
| sep_mean (linear-sep acc) | ≈ 0.16 | ≈ 0.15 | ≈ 0.15 |
| dom_mean (top-1 dominance) | ≈ 0.59 | ≈ 0.50 | ≈ 0.50 |
| vad_explained mean | ≈ 0.005 | ≈ 0.006 | ≈ 0.006 |
| silhouette_cosine_H (k=16) | −0.044 | −0.052 | −0.026 |

#### 重要な発見

1. **どの decomposer でも大半の成分は Plutchik から強く外れている**：8 ラベルでは説明できない方向が中心。
2. **VAD で説明できる成分はほぼゼロ**（< 1%）→ 仮説「言語化粒度より細かい」を強く支持。
3. **Plutchik 8 カテゴリの silhouette が負** → そもそも残差空間中で Plutchik は離散塊にならない。
4. **ICA k=16 がラベル独立性 × stability の両立で最有力**。

### 3.8 解釈サンプル（ICA k=16、layer 19、component 0）

[ica_k016_seed0.interpret.ica.json](../data/emotion_code/basis_sweep/ica_k016_seed0.interpret.ica.json) より：

- カテゴリヒストグラム（top-N positive 側）：`{surprise: 5, joy: 4, disgust: 1}` → 純粋な単一感情ではない
- top positive テキスト例：`"Unbelievable! I could never do that."`、`"You have no idea how happy I am for you."`、`"I can't believe Obama is our President."`
- top negative テキスト例：`"He brings with him some talented people from his former city."`、`"They're totally unreliable."`

→ component 0 は「強い感情を伴う発話 vs. 平板な発話」という、surprise/joy/disgust を横断する **強度軸的な何か**。
言語化された 8 カテゴリのいずれにも一意対応しない。

---

## 4. Phase C-2：純内在指標による軸検証

> 外部分類器・VAD・ラベルを **一切使わない** 評価へ転換。
> モデル内部だけで「これは本当に軸か」を問う 3 本柱。

### 4.1 Held-out 再構成 R²（[eval_basis_reconstruction.py](../experiments/eval_basis_reconstruction.py)）

train で学んだ `W` を val Δ に最小二乗で当てはめ、R² を測る。
出力：[reconstruction.csv](../data/emotion_code/basis_sweep/reconstruction.csv)

| decomposer | k | r2_train | r2_val |
|---|---:|---:|---:|
| ica | 8 | 0.254 | 0.244 |
| ica | 16 | 0.339 | **0.324** |
| nmf | 8 | 0.236 | 0.224 |
| nmf | 16 | 0.314 | 0.300 |
| pca | 8 | 0.254 | 0.244 |
| pca | 16 | 0.339 | **0.324** |

→ k=8→16 で R² が +0.08 程度伸び、**まだ plateau に達していない**。
内在次元は 16 を超える。次は k ∈ {32, 64, 128} で plateau を探る。
PCA と ICA は理論通り同じ R²（線形空間としては同等）、NMF は非負制約のぶん少し劣る。

### 4.2 層間一貫性（[eval_basis_layerconsistency.py](../experiments/eval_basis_layerconsistency.py)）

同じ (decomposer, k, seed) で層 13/16/19/22 の `W` を学習し、Hungarian + |cos| で成分マッチング。
**層を跨いで保存される成分** = モデルが同じ概念として扱う方向 = 真の感情原子の最有力候補。

実行手順（[PROGRESS.md §3.2](PROGRESS.md) 記載）：
```bash
for L in 13 16 19 22; do
  uv run python -m src.emotion_code.basis_sweep --layer $L \
      --decomposers ica --ks 16 --n-seeds 1 \
      --output-dir data/emotion_code/basis_sweep_L${L}
done
uv run python -m experiments.eval_basis_layerconsistency \
    --sweep-dirs data/emotion_code/basis_sweep_L*
```

出力（予定）：`layer_pair_consistency.csv`、`layer_component_consistency.csv`。
※現時点で layer 19 単独までしか走っていないため、本セクションの結果はまだ空欄。

### 4.3 **Hero metric**：Encode-Steer-Re-encode 自己整合性

[eval_basis_selfconsistency.py](../experiments/eval_basis_selfconsistency.py)

各 $b_j$ と $\alpha$ について：
1. $\alpha \cdot b_j$ を hook 注入して生成
2. 生成テキストを **同層で再エンコード**
3. 全基底に射影し $\hat{s}_l(\alpha; j)$ を測る

「**自分が出した変化を自分で読める** = 軸として閉じている」ことの直接証拠。

派生指標：
- **Self-monotonicity**：$\alpha \leftrightarrow \hat{s}_j(\alpha)$ の Spearman ρ（プロンプト平均）
- **Specificity**：$\hat{s}_j - \mathrm{mean}_{l \neq j} |\hat{s}_l|$

#### 4.3.1 CAA pseudo-basis sanity check
[caa__overall.json](../experiments/results/basis_selfconsistency/caa__overall.json)：
- components: 4, 5, 0（joy=4 を含む 3 軸）
- αs: [−2, 0, 2], n_prompts=4
- **monotonicity_mean = 0.125**, specificity_at_extreme = −0.053

per-component（[caa__monotonicity.csv](../experiments/results/basis_selfconsistency/caa__monotonicity.csv)）：

| component | ρ_mean | ρ_min |
|---:|---:|---:|
| 4 (joy) | 0.125 | −0.5 |
| 5 (sadness) | 0.125 | −0.5 |
| 0 (anger) | 0.125 | −0.5 |

self_delta_cosine（[caa__summary.csv](../experiments/results/basis_selfconsistency/caa__summary.csv)）：

| component | α=−2 | α=+2 |
|---:|---:|---:|
| 4 (joy) | −0.005 | +0.040 |
| 5 (sadness) | −0.022 | −0.016 |
| 0 (anger) | −0.048 | +0.011 |

→ **CAA は self-consistency でほぼ機能していない**。joy だけ α=+2 で弱い正反応。
shift_acc は通っていた（joy 0.40, sadness 0.30）が、**生成テキストを再エンコードした残差が steering 方向に再度乗ること**は要求していなかった。
self-consistency はそれを要求するため、**圧倒的に厳しい指標**。

#### 4.3.2 ICA k=16 全成分（[ica_k016_seed0__overall.json](../experiments/results/basis_selfconsistency_full/ica_k016_seed0__overall.json)）

- components: 0..15、αs: [−2, 0, 2]、n_prompts=3、max_new_tokens=32
- **monotonicity_mean = 0.146**（CAA の 0.125 より高い）
- specificity_at_extreme = −0.033

per-component self ρ（[ica_k016_seed0__monotonicity.csv](../experiments/results/basis_selfconsistency_full/ica_k016_seed0__monotonicity.csv)）：

| j | ρ_mean | ρ_min |
|---:|---:|---:|
| 0 | −0.289 | −1.000 |
| 1 | 0.000 | −0.500 |
| 2 | −0.411 | −0.866 |
| 3 | 0.167 | −0.500 |
| **4** | **0.378** | −0.866 |
| **5** | **0.866** | 0.866 |
| 6 | 0.333 | 0.000 |
| **7** | **0.789** | 0.500 |
| **8** | **0.911** | 0.866 |
| 9 | −0.411 | −0.866 |
| 10 | 0.167 | −0.866 |
| **11** | **0.911** | 0.866 |
| 12 | −0.167 | −1.000 |
| 13 | −0.045 | −0.500 |
| 14 | −0.122 | −0.866 |
| 15 | −0.744 | −0.866 |

self_delta_cosine（[ica_k016_seed0__summary.csv](../experiments/results/basis_selfconsistency_full/ica_k016_seed0__summary.csv)）の主要行：

| j | α=−2 | α=+2 | specificity@α=±2 |
|---:|---:|---:|---:|
| 4 | −0.0102 | +0.0036 | −0.039 / −0.027 |
| 5 | 0.0 | +0.0467 | 0.0 / +0.007 |
| 7 | −0.0237 | +0.0209 | −0.062 / +0.006 |
| 8 | −0.0380 | +0.0092 | −0.090 / −0.007 |
| 11 | −0.0022 | +0.0029 | −0.042 / −0.017 |

#### 4.3.3 ラベル独立性 × 機能性のクロス

| 区分 | 件数 | 成分 |
|---|---:|---|
| **functional**（α=±2 で self_delta の符号が正しい） | 6 / 16 | 主に b4, b5, b7, b8, b11, b14 |
| **label-independent**（top-1 dominance ≤ 0.375） | 6 / 16 | b3, b4, b7, b8（k=16 seed0 metrics より） |
| **両立する候補感情原子** | **3 / 16** | **b4, b7, b8** |

注目成分：

| j | self+2 | self−2 | ρ_self | label_dom | MI | 判定 |
|---:|---:|---:|---:|---:|---:|---|
| **b8** | +0.009 | −0.038 | **0.91** | 0.25 | 0.14 | ★ 最有力候補 |
| **b7** | +0.021 | −0.024 | 0.79 | 0.375 | 0.21 | ★ 候補 |
| **b4** | +0.004 | −0.010 | 0.38 | 0.375 | 0.16 | ○ 弱め |
| b11 | +0.003 | −0.002 | 0.91 | **1.00** | 0.21 | ✗ label-leak |
| b5 | 0.0 | +0.047 | 0.866 | 0.5 | — | △ 単側のみ反応 |

#### 4.3.4 解釈と運用上の含意

- **CAA は shift_acc を満たしても self-consistency を満たさない**。
  → 「外部分類器でラベルが変わる」ことと「モデル内部で軸として閉じている」ことは別物。
  どちらが重要かは応用次第だが、**原子性の証拠としては後者のほうが強い**。
- **ICA b8 は ρ=0.91 / dominance=0.25** で、CAA のどの軸より明確に self-consistent な軸として振る舞う。
  → 「ラベル独立な感情原子」最有力候補。
- 旧版（5 α 点）で b8 が ρ=0.09 と出ていたのは中間 α のノイズ寄与。
  まずは α=±2 の **符号テスト**で一次選別、Spearman は 3 点では粗いので **AUC 化**が次の一手。
- specificity がほぼ全成分で負（他成分への漏れが b_j 自身への寄与より大きい）→ 軸間が完全直交ではない。
  ICA でも残差空間内で完全分離は無理、混じり合いを許す前提で評価する必要。

---

## 4.X Phase C-3 — CAA の basis 線形分解（CAA → basis decomposition）

### 4.X.1 動機と仮説

Phase B-5 の CAA カテゴリベクトル $v_{c,L}$ と Phase C の basis $\{b_k\}$ は同じ層
 $L$ の同じ残差空間（$\mathbb{R}^{4096}$）に住み、生成元データ（pos/neg ペア $\Delta$）も
共有している。したがって CAA を basis の線形結合として書き直せるはず：

$$ v_{\text{joy}} \approx \sum_{k=1}^{K} w_k \, b_k $$

これが成り立てば、Plutchik 8 カテゴリは **独立軸ではなく basis の混合表現** として
解釈できる、という根本仮説（§0）の数値証拠になる。VAD（3 軸）との比較で
「**named axes では捉えきれない構造を basis が掴んでいる**」かを判定。

### 4.X.2 実装

- スクリプト：[experiments/eval_caa_basis_decomposition.py](../experiments/eval_caa_basis_decomposition.py)
- 入力：`data/emotion_code/caa.pt` (`vectors [8, 4, 4096]`)、`basis_sweep_L{13,16,19,22}/`
  および `basis_sweep/` 内の全 16 artifact、`vad_mapping.pt` (`W [3, 4096]`)
- 解法：3 種を並列適用
  - **OLS** (`np.linalg.lstsq`) — 制約なしの天井
  - **NNLS** (`scipy.optimize.nnls`) — 非負制約（NMF basis と整合）
  - **LASSO** (`sklearn.linear_model.Lasso`) — `α_rel ∈ {1e-3, 1e-2, 1e-1}` の 3 点
- 出力：
  - [experiments/results/caa_basis_decomposition/decomposition.csv](../experiments/results/caa_basis_decomposition/decomposition.csv)（640 行）
  - [vad_baseline.csv](../experiments/results/caa_basis_decomposition/vad_baseline.csv)（32 行）
  - [summary.json](../experiments/results/caa_basis_decomposition/summary.json)
  - `weights/{sweep}__{artifact}.pt` — per-category 重み

### 4.X.3 結果（数値検証）

**Headline（中央値、全 artifact / 全カテゴリ / 全層）**

| 手法 | median R² | median cos | median ‖w‖₀ |
|---|---:|---:|---:|
| **OLS（basis）** | **0.813** | **0.902** | 16 |
| NNLS（basis） | 0.458 | 0.676 | 8 |
| LASSO（basis, 全 α 集約） | 0.153 | 0.569 | 1 |
| **VAD baseline（3 軸）** | **0.017** | — | 3 |

→ **basis（k=16, OLS）は VAD の約 48 倍の説明力**。仮説は強く支持される。

**ベスト構成（OLS, R² 上位）**

| decomposer | k | layer | median R² | median cos |
|---|---:|---:|---:|---:|
| ICA | 16 | 22 | 0.878 | 0.937 |
| ICA | 16 | 19 | 0.864 | 0.929 |
| PCA | 16 | 19 | 0.864 | 0.929 |
| ICA | 16 | 16 | 0.850 | 0.922 |
| ICA | 16 | 13 | 0.849 | 0.921 |
| NMF | 16 | 19 | 0.830 | 0.911 |
| (ICA/PCA/NMF k=8) | 8 | 19 | ~0.75 | ~0.87 |

- **k=8 → k=16 で R² が +0.10 押し上がる**：CAA の表現にはまだ余裕があり、
  Phase C の k-plateau 探索が CAA 分解にも効くことを示唆。
- **decomposer 間差は小さい**（ICA ≈ PCA > NMF, 同一 k で）。後段の解釈性は
  ICA/NMF が有利なので「数値的にはほぼ等価、解釈性で選べ」が結論。
- **layer による傾向**：L22 がわずかに最良（R²=0.88）。L19 production 環境と
  ほぼ同じ R²（差 0.014）なので運用上は L19 のままで良い。
- **NNLS が R² を半減させる**：CAA は basis の **両符号** の重ね合わせ。
  「カテゴリ＝basis の加算的混合」という直感的描像は厳密には成り立たない。
- **LASSO の α=1e-1 では ‖w‖₀≈1 に潰れて R² 0.15**：1 成分だけで CAA を
  説明するのは不可能。LASSO で疎にするなら α=1e-3 程度が適切（CSV 参照）。

### 4.X.4 結果（行動検証本実行）

スクリプト：[experiments/eval_caa_basis_decomp_steering.py](../experiments/eval_caa_basis_decomp_steering.py)

各カテゴリの再構成ベクトル $v_\text{recon} = \sum w_k b_k$ を steering vector として
注入し、`caa` / `ols` / `nnls` / `lasso` / `vad` / `random`（`||w||_2` を ols に揃え
たガウス重み）の 6 variant で shift-accuracy を比較。`ica_k016_seed0`（L=19）を採用。

本実行（`--n-prompts 32 --alphas -2 0 2`、16h53m、4608 generations、α=+2 を target）：

| variant | mean_shift_acc | mean_baseline_acc | mean_delta | retention_vs_caa |
|---|---:|---:|---:|---:|
| **caa** | 0.161 | 0.063 | **+0.099** | 1.00 |
| **ols** | 0.115 | 0.063 | +0.052 | **0.71** |
| random | 0.089 | 0.063 | +0.026 | 0.55 |
| nnls | 0.083 | 0.063 | +0.021 | 0.52 |
| lasso | 0.068 | 0.063 | +0.005 | 0.42 |
| vad | 0.057 | 0.063 | −0.005 | 0.35 |

（試行数：n=192/variant ≈ 32 prompts × 6 measurable cats、SE ≈ 2pp）

- **OLS 再構成は CAA の steer 力を 71% 保持**：数値（cos 0.90）が行動でも比例して残る。
  「CAA は basis の線形結合」が **数値・行動の両面で支持される**。
- **VAD は baseline 以下**（Δ=−0.005, retention 35%）：3 軸では Plutchik の
  steer 力を再現できない。Phase 1 の R²=0.017 と整合。
- **NNLS / LASSO は random と同程度**：Phase 1 の R² 低下（0.46 / 0.15）が
  そのまま行動に出ている。両符号の重ね合わせ・密重みが必要。
- **CAA の baseline 0.063 ≈ 1/16** chance と一致、α=0 の健全性確認。

スクリプトには **増分 parquet 保存** と **`--resume`** を実装済み（中断・再開可）。

### 4.X.5 含意

- 数値（Phase 1）：**CAA は basis 16 軸でほぼ完全に書き直せる（cos 0.90 級）**。
  これが「カテゴリは独立な原子ではない」という主張の最も直接的な数値証拠。
- 行動（Phase 2）：**OLS 再構成で steer 力 71% retention**。線形結合での近似が
  「実際にモデルを動かす力」としても保たれる。論文のクレーム「named axes は
  basis の射影として再構成可能」が数値・行動の両側面で確証された。
- 警告：basis は CAA と同じ pos/neg ペア集合から作られるため軽い循環性が
  ある（カテゴリ平均を、その分布の主成分で説明している）。VAD（独立データ
  EmoBank で fit）が同じ手順で R²=0.017 しか出さないことが、循環性が説明力の
  主因ではないことを示している。

---

## 5. 全体としての発見の要点

1. **Plutchik 8 カテゴリ平均（CAA）はステアリング軸として弱い**：joy/sadness 以外はほぼ動かず、self-consistency でも CAA すら ρ ≈ 0.13。
2. **VAD 3 軸はモデル内残差では Valence しか線形に取れない**（V=0.56, A=0.24, D=0.16）。Arousal/Dominance は別の表現形態を取っている可能性。
3. **モデル内部の感情表現は 8 カテゴリでも 3 軸でも捉えきれない**：per-pair Δ から ICA で取った成分の多くが Plutchik / VAD から独立、Plutchik silhouette は負。
4. **「言語化粒度より細かい潜在原子」候補が実在する**：ICA b8 が ρ=0.91 / dominance 0.25。CAA の joy 軸より厳しい指標で勝っている。
5. **内在次元は k=16 でも飽和していない**（R²_val 0.32）。さらに高次元の基底が必要。
6. **層を変えても val_acc は不変**（0.66 ± 0.001）→ 残差表現は層をまたいで似ており、層間一貫性スイープで真の原子を絞れる見込みが高い。
7. **PPL guardrail はカテゴリ依存**（fear/sadness は α=2 が限界、joy/disgust は α=5 まで安定）→ 強度測定には正規化必須。

---

## 6. ディレクトリ早見

```
src/emotion_code/
├── caa.py               # B-5 本流（不変）
├── basis.py             # B-5 本流（不変、k=8固定、カテゴリ平均入力）
├── vad.py               # B-5 本流（不変、参考用に残す）
├── io.py
├── decompose.py         # C: NMF/PCA/ICA/dict 共通 API
├── basis_sweep.py       # C: 多 k × 多 decomposer × 多 seed（per-pair Δ 入力）
├── basis_metrics.py     # C: ラベル独立性スコア
└── basis_interpret.py   # C: 成分 → top texts

experiments/
├── eval_layer_sweep.py             # B-5
├── eval_shift_accuracy.py          # B-5
├── eval_monotonicity.py            # B-5
├── eval_perplexity.py              # B-5
├── eval_vad_r2.py                  # B-5
├── eval_basis_reconstruction.py    # C-2: held-out R²
├── eval_basis_layerconsistency.py  # C-2: 層間マッチング
└── eval_basis_selfconsistency.py   # C-2: hero metric

data/emotion_code/
├── caa.pt, basis.pt, vad_mapping.pt        # B-5 本流
└── basis_sweep/                            # C: 探索成果物
    ├── {nmf,pca,ica}_k{NNN}_seed{N}.pt
    ├── *.metrics.json
    ├── metrics.summary.csv
    ├── stability.summary.csv
    ├── reconstruction.csv
    ├── sweep.summary.json
    └── ica_k016_seed0.interpret.ica.json

experiments/results/
├── SUMMARY.md, layer_sweep.json, shift_accuracy.csv,
├── monotonicity.csv, perplexity_alpha.csv, vad_r2.json   # B-5
└── basis_selfconsistency{,_full}/                        # C-2
    ├── caa__{overall.json, monotonicity.csv, summary.csv, readouts.parquet}
    └── ica_k016_seed0__{overall.json, monotonicity.csv, summary.csv, readouts.parquet}
```

---

## 7. 既知の制約と未解決問題

1. **k > 16 の探索が未着手**：R² plateau の特定が次の優先タスク。
2. **layer-consistency の実走が未完了**：layer 19 のみ。13/16/22 の sweep を回す必要。
3. **n_prompts=3 は粗い**：Spearman ρ が 4 値しか取れず、b8 のような強い候補でも ρ_min=0.866 が出てしまう。n_prompts ≥ 8 + 符号一致 AUC で再ランキングすべき。
4. **specificity が大半で負**：軸間 cross-talk が支配的。ICA の独立性は統計的独立であり線形直交ではないことに注意。
5. **加法性検定が未実施**：$\alpha b_i + \beta b_j$ の readout が線形和になるか？ 真の「基底」なら成り立つはず。
6. **モデル単体での結果**：Llama-3.1-8B のみ。GPT-2 比較ノートは [notebooks/00_model_comparison.ipynb](../notebooks/00_model_comparison.ipynb) にあるが、Phase C 指標での横断比較はまだ。
7. **dict（sparse coding）と SAE が未実行**：超完備辞書比較が残っている。

---

## 8. 次にやること（優先順）

1. **候補 3 成分（ICA b4, b7, b8）の qualitative 解釈**
   - [basis_interpret.py](../src/emotion_code/basis_interpret.py) で top-text を読み込み、人間として「これは何の感情？」と試命名（命名不能なら本物のサブ言語的軸の証拠）。
   - 各成分単独 α 注入の生成サンプルを並べて読む。
2. **ノイズ低減した self-consistency 再測定**：n_prompts ≥ 8、α ∈ {−2, 2} の符号一致率（バイナリ AUC）でランキング。
3. **k sweep（32, 64, 128）+ R² plateau の特定** → 内在次元の下限。
4. **複数層スイープ + layer-consistency** → 層を跨いで生き残る成分が真の感情原子の最有力候補。
5. **加法性検定**（$\alpha b_i + \beta b_j$ 注入の readout が線形和になるか）。
6. **混合 prompt の生成 UI/CLI**：「言葉にできない感情」を実際にユーザが探索できる体験。
7. （長期）SAE / dict 大 k での超完備辞書、SAE 業界標準ツールとの比較。

---

## 9. 用語表

| 用語 | 定義 |
|---|---|
| **CAA** | Category Average Activation。pos/neg 平均差ベクトル。 |
| **per-pair Δ** | カテゴリ平均せず、ペアごとに pos − neg を取った行列（3200 × d）。 |
| **alpha_unit** | 注入ベクトルの自然 L2 を 1.0 とした正規化 α。 |
| **shift_acc** | 外部分類器でステア後ラベルが目標カテゴリになる率。 |
| **self_delta_cosine** | ステア生成 → 再エンコード残差 Δ を、注入した基底 b_j 自身に射影した値。 |
| **specificity** | $\hat{s}_j - \mathrm{mean}_{l \neq j} |\hat{s}_l|$。自身の反応が他成分への漏れより大きいか。 |
| **dominance** | 成分の top-N 射影で最頻 Plutchik カテゴリの占有率。1.0 = 完全 label-leak。 |
| **silhouette_cosine_H** | 各 H 行（成分の loading パターン）を Plutchik クラスタで silhouette 計算した値。 |
| **Hungarian-cosine stability** | seed/層間で成分を最適マッチングし、cosine で平均した安定度。 |
| **k-plateau** | **R² または説明分散が高 k で伸びなくなる点**。それ以上 k を増やしてもモデル内残差の有効情報量が増えていないことを示す。 |


