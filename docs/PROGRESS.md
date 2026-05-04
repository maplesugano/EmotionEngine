# EmotionEngine — Phase B 以降の進捗まとめ

> 後進が「いま何が動いていて、なぜそれを作ったのか」を追えるようにするための要約です。
> Phase A〜B-5 の確定済みパイプラインは無傷で残し、Phase C 以降は **追加モジュールのみ** で実験を進めています。

---

## 0. プロジェクトの根幹仮説

**「感情は言語化粒度より細かい潜在基底の線形結合として表せる」**

- 既存の心理学カテゴリ（Plutchik 8、Ekman 6、VAD 3 軸）は *観測ラベル* に過ぎない。
- LLM の残差ストリームには、これら言語化された感情よりも細かい「感情原子」`b_j` が線形に張られていて、`joy ≈ Σ w_j · b_j`、`sadness ≈ Σ w'_j · b_j`、そして名前のない混合 `α·b_3 + β·b_7` も同じ空間で表現可能 — というのが検証対象。

---

## 1. Phase B-5（確定済み・本流パイプライン）

> 触らない。すべての追加実験はこの結果をベースラインとして比較する。

### データ
- [src/data/build_unified.py](../src/data/build_unified.py)：ISEAR / EmoBank / DailyDialog 等を共通スキーマに正規化
- [src/data/quality_filter.py](../src/data/quality_filter.py)：品質フィルタ
- [src/data/build_contrastive.py](../src/data/build_contrastive.py)：Plutchik 8 カテゴリ × 500 ペア = 4000 contrastive pairs
  - 構成：mined 60% / llm_swap 30% / template 10%
  - 反対語は [src/data/schema.py](../src/data/schema.py) の `PLUTCHIK_OPPOSITE`

### 活性化収集
- [src/activations/collect.py](../src/activations/collect.py)：Llama-3.1-8B-Instruct、hook layers `[13, 16, 19, 22]`、last-token 残差を pos/neg shard 別に safetensors で保存（[data/activations/llama/](../data/activations/llama/)）

### 感情コード（本流）
- **CAA**：[src/emotion_code/caa.py](../src/emotion_code/caa.py)
  カテゴリ平均差 `v_{c,L} = mean(pos) - mean(neg)`、形 `[8, 4]` ファミリの prototype
  → [data/emotion_code/caa.pt](../data/emotion_code/caa.pt)
- **基底（NMF/PCA）**：[src/emotion_code/basis.py](../src/emotion_code/basis.py)
  layer 16 で sign-split NMF と PCA、k=8（Plutchik 揃え）
  → [data/emotion_code/basis.pt](../data/emotion_code/basis.pt)
- **VAD 線形写像**：[src/emotion_code/vad.py](../src/emotion_code/vad.py)
  EmoBank で layer 19 → V/A/D Ridge 回帰
  → [data/emotion_code/vad_mapping.pt](../data/emotion_code/vad_mapping.pt)
- **ステアリング**：[src/steering/hook.py](../src/steering/hook.py)、[src/steering/generate.py](../src/steering/generate.py)

### 評価
| 指標 | スクリプト | 結果ファイル |
|---|---|---|
| 層スイープ | [experiments/eval_layer_sweep.py](../experiments/eval_layer_sweep.py) | [layer_sweep.json](../experiments/results/layer_sweep.json) |
| Shift accuracy | [experiments/eval_shift_accuracy.py](../experiments/eval_shift_accuracy.py) | [shift_accuracy.csv](../experiments/results/shift_accuracy.csv) |
| Monotonicity | [experiments/eval_monotonicity.py](../experiments/eval_monotonicity.py) | [monotonicity.csv](../experiments/results/monotonicity.csv) |
| Perplexity guard | [experiments/eval_perplexity.py](../experiments/eval_perplexity.py) | [perplexity_alpha.csv](../experiments/results/perplexity_alpha.csv) |
| VAD R² | [experiments/eval_vad_r2.py](../experiments/eval_vad_r2.py) | [vad_r2.json](../experiments/results/vad_r2.json) |

ノートブックでまとめ：[notebooks/01_phaseB_results.ipynb](../notebooks/01_phaseB_results.ipynb)

---

## 2. Phase C — 言語非依存な感情基底の探索

Phase B-5 の基底は「Plutchik 8 のカテゴリ平均」を入力にしているため、原理的に Plutchik 部分空間しか張れない。
仮説検証には **(a) ペア単位 Δ を入力**、**(b) k と decomposer を sweep**、**(c) ラベル/VADに依存しない指標**、の 3 点を足す必要がある。

### 2.1 純関数 API
[src/emotion_code/decompose.py](../src/emotion_code/decompose.py) — NMF / PCA / ICA / sparse dictionary learning を共通インターフェースで提供。`category_loadings` ヘルパも同梱。

### 2.2 多分解器・多 k スイープ
[src/emotion_code/basis_sweep.py](../src/emotion_code/basis_sweep.py)
- 入力：per-pair Δ（カテゴリ平均しない、3200 × 4096）
- `--decomposers nmf pca ica dict --ks 8 16 32 ... --n-seeds N --max-iter M`
- 出力：`{decomposer}_k{k:03d}_seed{seed}.pt` を [data/emotion_code/basis_sweep/](../data/emotion_code/basis_sweep/) に
- 各 payload に `train_mask` を同梱（後続 metrics/interpret が同じ Δ を再現するため）

### 2.3 ラベル独立性スコア
[src/emotion_code/basis_metrics.py](../src/emotion_code/basis_metrics.py)
- per-component：`MI(component, category)` / 線形分離 acc（5-fold logistic on s_{·,j}）/ top-N category dominance / VAD 説明率（参考）
- per-artifact：cosine silhouette of `H` rows w.r.t. categories
- per-(decomposer, k)：seed 間 Hungarian-cosine stability
- 出力：各 artifact 横に `*.metrics.json`、プール CSV [metrics.summary.csv](../data/emotion_code/basis_sweep/metrics.summary.csv) と [stability.summary.csv](../data/emotion_code/basis_sweep/stability.summary.csv)

### 2.4 解釈ツール
[src/emotion_code/basis_interpret.py](../src/emotion_code/basis_interpret.py)
- 各 `b_j` への top/bottom 射影ペアの元テキスト + Plutchik category histogram（label-leak 診断）

### 2.5 最初の発見（layer 19、k∈{8,16}、NMF/PCA/ICA、2 seeds）

```
chance acc = 0.125
NMF/PCA/ICA いずれも sep_mean ≈ 0.15–0.16, dom_mean ≈ 0.50–0.59
vad_explained ≈ 0.006   （全成分平均）
silhouette  ≈ −0.05    （カテゴリは塊にならない）
ICA k=16 stability = 0.95
```

- **どの decomposer も大半の成分は Plutchik から強く外れている**：8 ラベルでは説明できない方向が中心。
- **ICA k=16 がラベル独立性と stability の両立で最有力**。トップ「言語非依存候補」は ICA 成分 13 / 2 / 8 と PCA 成分 4 / 7。
- **VAD で説明できる成分はほぼゼロ**（< 1%）。仮説の「言語化粒度より細かい」を強く支持。
- 8 カテゴリの silhouette が負 → そもそも Plutchik は離散塊ではない。

### 2.6 方針転換：VAD と外部分類器を捨てる

VAD 軸は 3 つの **名前付き** 軸であり、それで測ること自体が「語彙化された感情座標」への偏った投影。
以後、Phase C の評価は **モデル内部だけ** で完結する純内在指標に切り替える（外部分類器も使わない）。

---

## 3. Phase C-2 — 純内在的な軸検証

3 つの評価を追加。すべて **ラベル不要・VAD不要・外部分類器不要**。

### 3.1 Held-out 再構成 R²（軽量、即座に回せる）
[experiments/eval_basis_reconstruction.py](../experiments/eval_basis_reconstruction.py)
- train で学習した `W` を val Δ に最小二乗で当てはめ、R² を測る
- k を sweep して plateau を見つければ「内在次元の下限」が分かる
- 出力：[data/emotion_code/basis_sweep/reconstruction.csv](../data/emotion_code/basis_sweep/reconstruction.csv)

初回結果（layer 19）：

| decomposer | k=8 R²_val | k=16 R²_val |
|---|---|---|
| NMF | 0.224 | 0.300 |
| PCA / ICA | 0.244 | 0.324 |

→ k=16 でもまだ伸び代あり。次は k∈{32, 64, 128} で plateau を探る。

### 3.2 層間一貫性（実体性の物理的証拠）
[experiments/eval_basis_layerconsistency.py](../experiments/eval_basis_layerconsistency.py)
- 同じ (decomposer, k, seed) で層 13/16/19/22 の `W` を学習
- Hungarian + |cosine| で成分マッチング
- **層を跨いで保存される成分** = モデルが同じ概念として扱う方向
- 出力：[layer_pair_consistency.csv](../data/emotion_code/basis_sweep/layer_pair_consistency.csv)、[layer_component_consistency.csv](../data/emotion_code/basis_sweep/layer_component_consistency.csv)

実行手順：

```bash
for L in 13 16 19 22; do
  uv run python -m src.emotion_code.basis_sweep --layer $L \
      --decomposers ica --ks 16 --n-seeds 1 \
      --output-dir data/emotion_code/basis_sweep_L${L}
done
uv run python -m experiments.eval_basis_layerconsistency \
    --sweep-dirs data/emotion_code/basis_sweep_L*
```

### 3.3 Encode-Steer-Re-encode 自己整合性（**hero metric**）
[experiments/eval_basis_selfconsistency.py](../experiments/eval_basis_selfconsistency.py)

各 `b_j` と α について：
1. `α · b_j` を hook 注入して生成
2. 生成テキストを同層で再エンコード
3. 全基底に射影し `ŝ_l(α; j)` を測る

**「自分が出した変化を自分で読める」=軸として閉じている** ことの直接証拠。
2 つの派生指標：

- **Self-monotonicity**：`α ↔ ŝ_j(α)` の Spearman ρ（プロンプト平均）
- **Specificity**：`ŝ_j − mean_{l≠j} |ŝ_l|`

実行例：

```bash
# CAA を pseudo-basis として渡す sanity check
uv run python -m experiments.eval_basis_selfconsistency \
    --basis data/emotion_code/caa.pt --caa-layer 16 \
    --components 4 5 0 --alphas -2 0 2 --n-prompts 4

# ICA k=16 全成分
uv run python -m experiments.eval_basis_selfconsistency \
    --basis data/emotion_code/basis_sweep/ica_k016_seed0.pt \
    --alphas -2 0 2 --n-prompts 3 --max-new-tokens 32 \
    --output-dir experiments/results/basis_selfconsistency_full
```

出力（[experiments/results/basis_selfconsistency/](../experiments/results/basis_selfconsistency/) と `..._full/`）：
- `*__readouts.parquet` 全行
- `*__summary.csv` per (component, α)
- `*__monotonicity.csv` per component の self ρ
- `*__overall.json` 1 ライン要約

#### Phase C-2 初回結果（layer 19、ICA k=16）

ラベル独立性（`metrics.summary.csv`）と機能性（self-consistency）を交差させた結果：

| 区分 | 件数 |
|---|---|
| **functional**（α=±2 で self_delta の符号が正しい） | 6 / 16 |
| **label-independent**（top-N category dominance ≤ 0.375） | 6 / 16 |
| **両方を満たす（候補感情原子）** | **3 / 16**（成分 4, 7, 8）|

注目成分：

| j | self+2 | self−2 | ρ_self | label_dom | MI |
|---|---|---|---|---|---|
| **b8** | +0.009 | −0.038 | 0.91 | 0.25 | 0.14 |
| **b7** | +0.021 | −0.024 | 0.79 | 0.375 | 0.21 |
| **b4** | +0.004 | −0.010 | 0.38 | 0.375 | 0.16 |
| b11 | +0.003 | −0.002 | 0.91 | 1.00 | 0.21 |（label-leak）

所見：
- **CAA sanity check** では joy α=+2 だけが弱い正反応（ρ=0.13）。**self-consistency は shift_acc よりはるかに厳しい**（生成内容の感情ラベルではなく、生成テキストを再エンコードした残差が steering 方向に再度乗ることを要求）。
- それでも ICA b8 は ρ=0.91、dominance=0.25 で **CAA より明確に self-consistent な軸**として振る舞う。「ラベル独立な感情原子」候補として最有力。
- 旧版（5 α 点）で b8 が ρ=0.09 と出ていたのは中間 α のノイズ寄与。**まずは α=±2 の符号テストで一次選別する** のが正しい運用。

#### Phase C-2 追記 1：層横断一貫性（ICA k=16 を L∈{13,16,19,22} で再 sweep）

[experiments/eval_basis_layerconsistency.py](../experiments/eval_basis_layerconsistency.py) を実走し、Hungarian マッチングの |cosine| で層間マッチ。

ペア平均 |cos|（[layer_pair_consistency.csv](../data/emotion_code/basis_sweep/layer_pair_consistency.csv)）：

| L_a | L_b | mean | min | median |
|---|---|---|---|---|
| 13 | 16 | 0.51 | 0.17 | 0.57 |
| 13 | 19 | 0.45 | 0.19 | 0.44 |
| 13 | 22 | 0.35 | 0.21 | 0.36 |
| 16 | 19 | 0.63 | 0.34 | 0.71 |
| 16 | 22 | 0.52 | 0.31 | 0.58 |
| 19 | 22 | **0.73** | 0.42 | 0.76 |

L19 をアンカーにした三層マッチの per-component 平均：

| L19 j | mean abs_cos | candidate |
|---|---|---|
| **b7** | **0.70** | ○ 断片化軸 |
| **b4** | **0.64** | ○ |
| **b8** | **0.47**（下位 2/16） | ◎ 低層では見えていない |

サニティチェックとして L16 の Hungarian マッチ先 (b5, |cos|=0.34) をそのまま self-consistency にかけると ρ=0.17, self_−2=0.000。**マッチだけでは機能的に保存されていない**ことを確認。

→ **b8 は L19〜22 の中後段で emergent な軸**。「全層で生き残る原子」ではないが、**中後段では明確に定着した軸**。

#### Phase C-2 追記 2：b8 の質的解釈（steering 生成例）

[experiments/eval_basis_qualitative.py](../experiments/eval_basis_qualitative.py) で α∈{−6,−3,0,+3,+6} ステアリングした生成例を [results/basis_qualitative_b8_strong.txt](../experiments/results/basis_qualitative_b8_strong.txt) に保存。

軽い α=±2 だと生成はほぼベースラインと同一。`scale = 1/median||b||` にさらに 3–6 を掛けると軸の意味が見える：

- **負極 (−6)**：話者が相手に直接訴える修辞構造。`I'm not going to tell you again. ... You're not listening. You're not paying attention.` / `I'm here to learn. I'm here to see the world.`
- **正極 (+6)**：第三者視点・分析的フレーム。最強ではプロンプトを **読解問題に変換** (`What does the person want to do? A) ... B) ...`)。
- **非対称性**：正極は α=+3 で飽和、負極は α=−6 まで escalation。負極が「能動」、正極が「deactivation/detachment」。

**軸の命名仮**：*addressivity（対人的訴求性）vs analytical detachment*。

方法論上の重要な教訓：
- `basis_interpret.py` の top-text による命名（「個人意見 vs 礼儀定型」）は **入力テキストの相関**。
- **因果的な軸の意味は steering 生成でしか分からない**。両方をやって初めて「軸の正体」が見える。
- Plutchik 8 に対して「提領態度軸」は斜交している：仲良くしたい怒りも、距離をとった喜びもありうる。これが**「Plutchik では表せない感情次元」の最初の具体例**。

#### Phase C-2 追記 3：加法性検定（[experiments/eval_basis_additivity.py](../experiments/eval_basis_additivity.py)）

`α·b_i + β·b_j` 同時注入の readout が、単独注入の周辺効果の和になるか：

    predicted(α, β) = r(α, 0) + r(0, β) − r(0, 0)
    resid = actual(α, β) − predicted

ペア (b8,b4), (b8,b7), (b4,b7), (b8,b11) を α∈{−2,0,2} と α∈{−1,0,1} の 2 グリッドで実走。

**self-readout（対象 2 軸の cosine）の絶対誤差**（α=±1、4 オフ対角セル平均）：

| ペア | err_i | err_j | k=16 全体 ratio (median) |
|---|---|---|---|
| (b4, b7) | 0.019 | 0.026 | 0.76 |
| (b8, b4) | 0.021 | 0.015 | 0.81 |
| (b8, b7) | 0.026 | 0.015 | 0.53 |
| (b8, b11) | 0.028 | 0.030 | 0.94 |

readout の絶対値スケール（0.1〜0.2）に対して **err は 10–25%**。**対象 2 軸の上では加法性がきれいに成立**。

一方、**k=16 全体ベクトルでの residual ratio は 0.5〜0.9** と大きい。これは「他の 14 成分への cross-talk が無視できない」ことを意味する。ratio metric の分母（marginal effect の大きさ）が α=±1 では小さくなり S/N が悪化するため、**ratio は飽和域 (α=±2) より α=±1 の方が悪く見える**点に注意。

所見：
- ◯ **局所的（i, j 軸上）には線形和**として振る舞う → 「感情 = 原子の線形結合」仮説の局所的支持。
- △ **大域的には他軸への漏れあり**。特に b8 を含むペアで顕著で、b8 が emergent / 非線形に獲得された軸であることと整合。
- 評価指標の選択：今後は `err_i, err_j`（対象軸の絶対誤差）を主指標にし、`k 全体 ratio` は cross-talk 量の参考値とする。

ファイル：
- 結果 (α=±2): [results/basis_additivity/ica_k016_seed0__additivity{,_summary}.{parquet,csv}](../experiments/results/basis_additivity/)
- 結果 (α=±1): [results/basis_additivity_a1/ica_k016_seed0__additivity{,_summary}.{parquet,csv}](../experiments/results/basis_additivity_a1/)

---

## 3.X Phase C-3 — CAA → basis 線形分解

仮説：Plutchik カテゴリ平均 $v_c$ は、同じ層の basis $\{b_k\}$ で書き直せる
（$v_c \approx \sum_k w_k b_k$）。これが成り立てば「カテゴリ＝独立軸」ではなく
「basis の混合」と解釈できる。

実装：
- Phase 1（数値）：[experiments/eval_caa_basis_decomposition.py](../experiments/eval_caa_basis_decomposition.py)
  全 16 basis artifact × 8 cat × {OLS, NNLS, LASSO} で $w$ を fit、R²/cos/疎度を比較。
  VAD（$W \in \mathbb{R}^{3 \times 4096}$）を同じ手順で baseline 化。
- Phase 2（行動）：[experiments/eval_caa_basis_decomp_steering.py](../experiments/eval_caa_basis_decomp_steering.py)
  再構成ベクトル $v_\text{recon} = \sum w_k b_k$ を steering して shift-accuracy で
  CAA / 各 fit / VAD / random と比較。`--resume`、増分 parquet 保存対応。

数値結果（中央値）：

| 手法 | median R² | median cos |
|---|---:|---:|
| OLS（basis k=16） | **0.813** | **0.902** |
| NNLS（basis k=16） | 0.458 | 0.676 |
| LASSO（basis） | 0.153 | 0.569 |
| VAD baseline | 0.017 | — |

ベスト：ICA k=16 L=22 で R²=0.878 / cos=0.937。L19 production と差は 0.014 のみ。
**basis は VAD の約 48 倍の説明力**、仮説は数値的に強く支持される。
NNLS で R² が半減 → CAA は basis の **両符号** の重ね合わせ（純加算ではない）。

行動結果（本実行、ICA k=16 L=19、n_prompts=32、4608 gens、16h53m）：

| variant | mean_shift_acc | mean_delta | retention_vs_caa |
|---|---:|---:|---:|
| caa | 0.161 | +0.099 | 1.00 |
| **ols** | **0.115** | **+0.052** | **0.71** |
| random | 0.089 | +0.026 | 0.55 |
| nnls | 0.083 | +0.021 | 0.52 |
| lasso | 0.068 | +0.005 | 0.42 |
| vad | 0.057 | −0.005 | 0.35 |

→ **OLS 再構成は CAA の steer 力を 71% 保持**。仮説「CAA = basis の線形結合」は
数値・行動の両面で支持。VAD は baseline 以下、NNLS/LASSO は random 並み。

成果物：
- [experiments/results/caa_basis_decomposition/](../experiments/results/caa_basis_decomposition/)（`decomposition.csv`, `vad_baseline.csv`, `summary.json`, `weights/`）
- [experiments/results/caa_basis_decomp_steering/](../experiments/results/caa_basis_decomp_steering/)（`generations.parquet`, `generations_classified.parquet`, `shift_by_variant.csv`, `summary_by_variant.csv`, `summary.json`）

### 3.X.1 重み行列 W の構造（B：分散表現の直接証拠）

スクリプト：[experiments/eval_caa_basis_weight_structure.py](../experiments/eval_caa_basis_weight_structure.py)

8 カテゴリ × 16 basis 成分の OLS 重み行列を成分視点で再分析（`participation_ratio`,
`sign_balance`, `top_cat_gap` で分類）。

結果：**13 pan + 3 lexical_gap (b1, b11, b13) + 0 cat_specific**
→ 「joy 専用」「fear 専用」のような片寄り成分は **存在しない**。Plutchik 8 が
basis 軸の **独立片** ではなく **分散和** であることが直接示された。

成果物：[experiments/results/caa_basis_weight_structure/ica_k016_seed0_L19_ols/](../experiments/results/caa_basis_weight_structure/ica_k016_seed0_L19_ols/)

### 3.X.2 Lexical-gap steering（C：行動検証）

スクリプト：[experiments/eval_lexical_gap_steering.py](../experiments/eval_lexical_gap_steering.py)

3 つの gap 成分 + 3 つの pan 成分 + 3 つの 2 成分加算 combo の計 9 ターゲット ×
α∈{0,1,2} × 16 中性プロンプト = **432 generations**。

実装上の重要点：basis 行ノルム ≈0.4、CAA 中央値ノルム ≈4 → そのまま α では無効果。
`--alpha-mode caa_match` で v を `median(||CAA||)` にリスケールしてから effective α
を計算するモードを追加（このスケール合わせは以後の必須前処理）。

外部分類器（Hartmann/distilroberta）では gap 群と pan 群の Plutchik 分布が同一で、
**「Plutchik 語彙の外側」を既存分類器は捉えられない**ことが確認された。
定性的には gap_b11 = 自己観察、combo_b1+b11 = 過警戒的内省、combo_b11+b13 = 決断不能。

成果物：[experiments/results/lexical_gap_steering/](../experiments/results/lexical_gap_steering/)

### 3.X.3 LLM-as-judge とメタ感情クラスタ（E：「言葉にない感情」を実体化）

スクリプト：

- [experiments/eval_lexical_gap_judge.py](../experiments/eval_lexical_gap_judge.py)
  GPT-4o-mini で Plutchik 8 を 0–1 採点 + **自由記述ラベル（1–4 単語）と強度** を
  構造化 JSON で取得（temperature=0、`--resume`、16 行 checkpoint）。
- [experiments/eval_lexical_gap_cluster.py](../experiments/eval_lexical_gap_cluster.py)
  自由記述ラベルを `text-embedding-3-small` で埋め込み、
  cosine + agglomerative で 7 メタ感情クラスタへ分類。

定量結果（α=+2、judge）：

| target | other_score | frac_other_dom |
|---|---:|---:|
| **combo_b11+b13** | **0.55** | **0.75** |
| **gap_b11** | 0.52 | 0.63 |
| pan_b8 | 0.41 | 0.56 |

**メタ感情クラスタ（109 ラベル / 61 unique → 7 群）**：

| cluster | name | n |
|---:|---|---:|
| 2 | uncertainty / indecision | 43 |
| 6 | enthusiasm / curiosity | 28 |
| 0 | self-doubt / encouragement | 25 |
| 1 | frustration / despair | 9 |
| 4 | academic ambition / competitive determination | 2 |
| 3 | romantic idealism | 1 |
| 5 | ironic amusement | 1 |

→ 約 6 割（68/109）が「不確実性 / 自己疑念 / 優柔不断」群。

成果物：[experiments/results/lexical_gap_judge/](../experiments/results/lexical_gap_judge/)
（`judgments.parquet`, `summary_by_target.csv`, `other_labels.csv`,
`cluster_assignment.csv`, `cluster_summary.csv`,
`cluster_scatter.png`, `cluster_per_target.png`）

### 3.X.4 仮説の精緻化

当初「言葉にない感情」と呼んでいたものは、**Plutchik より細かい現象学的
グラニュラリティ — とくにメタ認知的状態（uncertainty, self-doubt,
indecision, ironic amusement, …）** であることが judge + クラスタで明らかになった。
これらは英語の単一語では命名困難だが、judge LLM が複数語句で記述できる
程度には言語化可能な「中間状態」である。

### 3.X.5 retention 押し上げ（A：k=32 / L=22）

Phase 1 ベスト構成（ICA k=16 L=22, R²=0.878）に **k=32** を追加して再走。

- basis：[data/emotion_code/basis_sweep_L22/ica_k032_seed0.pt](../data/emotion_code/basis_sweep_L22/ica_k032_seed0.pt)
- Phase 1（[results/caa_basis_decomposition_L22/](../experiments/results/caa_basis_decomposition_L22/)）
  → **R²(OLS) = 0.920**（k=16 L=22: 0.878、k=16 L=19: 0.864 → +0.06）
- Phase 2（[results/caa_basis_decomp_steering_L22_k32/](../experiments/results/caa_basis_decomp_steering_L22_k32/)、19h、4608 gens）

| variant | mean_shift_acc | retention_vs_caa |
|---|---:|---:|
| caa | 0.146 | 1.00 |
| **ols** | **0.115** | **0.786** |
| nnls | 0.083 | 0.571 |
| vad | 0.063 | 0.429 |
| lasso | 0.052 | 0.357 |
| random | 0.036 | 0.250 |

→ **OLS retention 0.786**（L=19 k=16 比 +7.2pp）。Phase 1 R² 改善が行動でも
比例して伸びた。**k=32 でもまだ R² プラトーに達していない可能性** があり、
次は k=64 で plateau 探索が候補。

### 3.X.6 k=64 / L=22（plateau 探索）

§3.X.5 の k∈{8,16,32} の単調増加（R² 0.75→0.88→0.92）が k=64 で頭打ちに
なるかを確認するため、同 L=22 で k=64 の Phase 1+2 を実走。

- basis：[data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt](../data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt)
- Phase 1（[results/caa_basis_decomposition_L22/](../experiments/results/caa_basis_decomposition_L22/)）

| k | median R²(OLS) | median cos |
|---:|---:|---:|
| 16 | 0.878 | 0.937 |
| 32 | 0.920 | 0.959 |
| **64** | **0.960** | **0.980** |

per-cat の最低も joy=0.92 / surprise=0.93 まで上がり、CAA の 8 軸全てが
basis 64 次元の OLS でほぼ完全再構成可能になった。R² の伸び幅は
+0.04 → +0.04 と **線形ペースを維持**（plateau には未到達）。

- Phase 2（[results/caa_basis_decomp_steering_L22_k64/](../experiments/results/caa_basis_decomp_steering_L22_k64/)、4608 gens、約 19h）

| variant | mean_shift_acc | mean_delta | retention_vs_caa |
|---|---:|---:|---:|
| caa | 0.146 | +0.083 | 1.00 |
| **ols** | **0.135** | **+0.073** | **0.929** |
| lasso | 0.068 | +0.005 | 0.464 |
| nnls | 0.063 | 0.000 | 0.429 |
| vad | 0.063 | 0.000 | 0.429 |
| random | 0.042 | −0.021 | 0.286 |

→ **OLS retention 0.929**（k=32 比 +14pp、L=19 k=16 比 +22pp）。Phase 1 R² の
+0.04 が Phase 2 retention の +0.14 を生んだ。**「named axis (CAA) は basis の
線形射影として実質的に再構成可能」**という主張が、数値（R²=0.96）と行動
（retention=0.93）の両面で強く支持された。

含意・残課題：
- k=8→64 で R² が 0.75 → 0.96 まで単調増加。**plateau 未確認** だが、
  R² の限界 1.0 が近く、効用差は逓減フェーズに入りつつある可能性。
- nnls/lasso は k を上げても改善せず（むしろ random と同水準）。
  CAA は basis の **両符号スパースでない密な線形結合** として表現される。
- **論文の Phase 1〜2 セクション（CAA ≈ Σ w_k b_k）はこの結果で実質完了**。
  以降は (D) 加法性検定で「basis 同士の合成則」を確立し、論文の理論核心
  （basis = 真の原子）を閉じる段階。

### 3.X.7 加法性検定（D：CAA カテゴリ対 × OLS 再構成 basis、L=22 / k=64）

仮説「basis = 真の原子」の系として、**joint steering = sum of marginals** が
成り立つかを CAA カテゴリ対で検証。
[experiments/eval_caa_basis_additivity.py](../experiments/eval_caa_basis_additivity.py) を新設し、
§3.X.6 で得た OLS 重み（ICA k=64, L=22）から再構成した CAA 軸
$\hat v_a, \hat v_b$ について、$\alpha\hat v_a + \beta\hat v_b$ で steering して
生成・再エンコード・CAA 射影を測定。

- pairs：(joy, sadness), (joy, anger), (anger, fear), (disgust, joy)
- α, β ∈ {−2, 0, +2}、n_prompts=8、max_new_tokens=32
- 出力：[results/caa_basis_additivity_L22_k64/](../experiments/results/caa_basis_additivity_L22_k64/)
  （`generations.parquet`、`readouts.parquet`、`additivity_readout.csv`、
  `additivity_shift.csv`、`summary.json`）

結果（off-diagonal、$\alpha,\beta\neq 0$ のみ）：

| 指標 | 値 |
|---|---:|
| **readout 残差比** median \|resid\|/\|marg\| | **0.807** |
| readout 残差比 mean | 0.845 |
| **shift-acc 残差** median \|resid_to_a\| | **0.000** |
| shift-acc 残差 median \|resid_to_b\| | 0.000 |

含意：
- **readout 空間では加法性が大きく崩れる**（残差が marginal 和の 80% に達する）。
  joint steering の残差ストリーム表現は「単独軸の単純な線形和」にはなって
  おらず、basis 間に **明確な高次相互作用** が存在する。
- **行動（shift-acc）レベルでは残差ほぼゼロ**。joint で生成しても 8 カテゴリ
  分類器が割り当てるラベルは「片方の単独軸と同じ」になり、加算で新カテゴリ
  に飛ぶ訳ではない（=判別境界は加法的に滑らか）。
- 両者の食い違いは §3.X.2 の **emergent meta-emotion**（combo_b1+b11 →
  「過警戒的内省」など、Plutchik では命名できない混合状態）と整合：
  既存ラベルでは検出されない「中間状態」が残差ストリームには出ている。
- 結論として **basis は厳密な加法群ではない** が、**離散ラベル空間に射影
  すれば加法的に振る舞う** という二層構造が定量化された。論文では
  「線形再構成（Phase 1〜2）」と「非線形合成（Phase 3）」を分けて主張する
  土台ができた。

### 3.X.8 二層構造の追検証（D の発展：c → b → a）

§3.X.7 で出した「readout 加法崩れ × shift 加法成立」の二層構造を、
3 軸（メタ感情射影 / ペア類似度との相関 / 別構成での再現）で追検証。

**(c) 残差ベクトル → メタ感情クラスタ射影**
[experiments/eval_caa_additivity_metaemotion.py](../experiments/eval_caa_additivity_metaemotion.py) を新設。
生成テキストを `text-embedding-3-small` で埋め込み、§3.X.3 の 7 メタ感情
クラスタ centroid（`results/lexical_gap_judge/cluster_summary.csv`）への
コサインで「joint − marg_max」gain を測定。
出力：[results/caa_additivity_metaemotion_L22_k64/](../experiments/results/caa_additivity_metaemotion_L22_k64/)

| 指標 | 値 |
|---|---:|
| residual top cos median | **0.068** |
| null p95（shuffle）| 0.044 |
| marginal top cos median | 0.324 |
| 全 7 クラスタで median gain `joint − marg_max` | **−0.012 〜 −0.020（全て負）** |

→ 残差は **non-random**（null p95 の 1.5×）だが、**joint 生成は単独軸より
メタ感情クラスタに近づかない**。最頻ヒット先は "academic ambition /
competitive determination" (9/16)、"ironic amusement" (4/16)、
"uncertainty / indecision" (3/16)。

**解釈**：今回の 4 ペア（joy×sadness, joy×anger, anger×fear, disgust×joy）は
対立 / 反対方向の組み合わせが多く、joint で **打ち消し** が起きている可能性が
高い。§3.X.3 のメタ感情は単独 basis 軸（b1, b11 など）の steering で出た
現象であり、CAA 8 カテゴリ対の合成からは直接は再現されない。
論文上は「CAA-pair 残差 ≠ メタ感情方向」を **明示的な negative result** として
書き、メタ感情の発生源は CAA の組合せではなく **basis-native な高次相互作用**
である、と分離して主張するのが妥当。

**(b) per-pair 残差比 vs ペア類似度の相関**
[experiments/eval_caa_additivity_pairsim.py](../experiments/eval_caa_additivity_pairsim.py) を新設。
各ペアの cos 類似度（basis 重み `cos_w`、再構成軸 `cos_recon`、生 CAA `cos_caa_raw`）と
residual ratio の相関を測定。
出力：[results/caa_additivity_pairsim_L22_k64/](../experiments/results/caa_additivity_pairsim_L22_k64/)

| pair | cos_w | median resid ratio |
|---|---:|---:|
| anger + fear  | **0.849** | **0.865** |
| disgust + joy | 0.123 | 0.825 |
| joy + sadness | 0.116 | 0.803 |
| joy + anger   | 0.112 | 0.782 |

相関：
- pair-level（n=4）: Spearman cos_w r = **+1.00**
- cell-level（n=16）: Pearson cos_w r = **+0.59 (p=0.016)**, Spearman r = +0.62 (p=0.011)

→ **basis 重みが似ているペアほど残差が大きい**（同方向に押し合う高次項が
大きい）。逆相関ではない＝「直交ペアほど加法的」というクリーンな主張が出た。

**(a) k=32 / L=19 での再現**
別構成（ICA k=16, L=19、§3.X.5 比較対象）で同じ加法性検定を実走。
出力：[results/caa_basis_additivity_L19_k16/](../experiments/results/caa_basis_additivity_L19_k16/)

| 指標 | L22/k64 | **L19/k16** |
|---|---:|---:|
| readout 残差比 median | 0.807 | **0.788** |
| shift-acc 残差 median | 0.000 | **0.000** |

→ 二層構造（**readout で大崩れ ≈ 0.8、shift で完全加法 ≈ 0**）は **層 / k を
変えても再現**。pair-sim 相関は L19/k16 では弱化（Pearson 非有意、Spearman
r=0.80 (n=4) で方向は同じ）。これは disgust+joy の mean resid ratio が 1.64 と
外れ値であること、k=16 で basis 軸が荒く `cos_w` シグナルが弱いことに起因。

**結論（D の発展全体）**：
- 二層構造は構成不変（layer/k）で確認 ⇒ artefact ではない。
- 残差は random でも noise でもなく、ペアの basis 親和性で説明できる
  「同方向に押し合う高次項」である。
- ただし残差方向は既知メタ感情カタログ（§3.X.3）には射影できない ⇒
  **「Phase 3 の非線形成分」の正体は CAA 軸ペアの合成ではない**。
  メタ感情を意図的に誘発するには basis-native な合成（b_i + b_j）が要る、
  という仮説が次の検証対象になる。

### 3.X.9 basis-native 加法性検定 + メタ感情射影（D-c の続き、ICA k=16 / L=19）

§3.X.8(c) の続報。`eval_basis_additivity_metaemotion.py` で
basis 軸対 `α b_i + β b_j` を caa_match スケールで joint steering し、
生成テキストを §3.X.3 の 7 メタ感情クラスタへ射影。
ペア＝ §3.X.2 の `combo_b1+b11` 系 (b1,b11), (b1,b13), (b11,b13) と
pan 軸対照 (b8,b4), (b8,b7) の計 5 ペア × 3×3 α × 8 prompts = **360 gens**。

成果物：[results/basis_additivity_metaemotion_L19_k16/](../experiments/results/basis_additivity_metaemotion_L19_k16/)
（`generations.parquet`, `readouts.parquet`, `additivity_readout.csv`,
 `additivity_readout_detail.parquet`, `summary.json`,
 `metaemotion/{embeddings.npz, centroids.npz, cell_residual_cos.csv,
 cell_top_cluster.csv, per_text_cos.csv, per_text_summary.csv,
 per_cluster_summary.csv, summary.json}`）

**(a) readout 加法性**（off-diag セル平均）：

| ペア | err_i | err_j | resid_ratio |
|---|---:|---:|---:|
| (b1, b11) | 0.064 | 0.077 | 0.897 |
| (b1, b13) | 0.080 | 0.052 | 0.855 |
| (b11, b13) | 0.064 | 0.057 | 0.799 |
| (b8, b4) | 0.066 | 0.059 | 0.839 |
| (b8, b7) | 0.061 | 0.086 | 0.782 |
| **median** | **0.069** | **0.061** | **0.817** |

→ **CAA-pair (§3.X.7、err≈0.015–0.030) より 2–4 倍大きい err**。
basis 軸対は CAA 対よりも局所線形性が弱い。
全体 |resid|/|marg| ≈ 0.82 は CAA-pair (0.81) と同水準。

**(b) メタ感情射影**（per-cluster median `joint − max(marg)` cosine）：

| cluster | median gain |
|---|---:|
| frustration / despair | −0.013 |
| academic ambition / competitive determination | −0.019 |
| ironic amusement | −0.019 |
| self-doubt / encouragement | −0.020 |
| enthusiasm / curiosity | −0.021 |
| romantic idealism | −0.022 |
| **uncertainty / indecision** | **−0.026** |

**全 7 クラスタで gain が負**。joint は単独軸の最大 cos を上回らない。
**§3.X.8(c) と同じ negative result が basis 軸対でも再現** された。

ただし per-cell 残差は構造化されている（null p95 ≈ 0.05 に対し）：

| pair | α, β | cluster | cos_resid |
|---|---|---|---:|
| (b1, b11) | −2,+2 | romantic idealism | **−0.21** |
| (b1, b13) | −2,+2 | academic ambition / competitive | +0.17 |
| (b8, b7) | +2,+2 | academic ambition / competitive | +0.16 |
| **(b8, b4)** | **+2,+2** | **uncertainty / indecision** | **+0.16** |
| (b8, b4) | −2,+2 | uncertainty / indecision | +0.12 |
| (b1, b11) | +2,+2 | romantic idealism | −0.14 |
| (b1, b11) | +2,+2 | self-doubt / encouragement | −0.13 |

唯一の正の per-cluster gain は **(b8, b4) → frustration/despair (+0.005)**、
有意ではない（n=32）。`+2,+2` セルでは複数のペアが
"academic ambition / competitive determination" 方向に共通して正残差を出し、
強い steering 下のアトラクタ的な cross-talk と読める。

**(c) §3.X.2 の "過警戒的内省" は再現せず**：(b1,b11) の `+2,+2` セルで
"uncertainty / indecision" cos = 0.185 だが、b1 単独 (0.196) と b11 単独
(0.192) が既に飽和しており joint は両者を下回る。
3.X.2 で観察された質感は **b11 単独軸の周辺効果** で説明でき、
b1 との合成は新しい方向を加えていない。

サンプル生成（(b1,b11)）：

| α, β | 生成（先頭一例） |
|---|---|
| −2,+2 | "I mean, I'm not saying that you're not a good friend, but sometimes you can be a bit... distant." |
| +2,+2 | "I've got to get out of here. I'm like, totally sick of this place." |
| −2,−2 | "I have a new friend, a young woman named Rachel, who is a member of the local community choir." |
| +2,−2 | "I've been waiting for ages for you to get it. I've been trying to get you to see it for ages." |

→ いずれも Plutchik 隣接の表層感情（不満 / 退屈 / 描写 / 焦燥）。
"meta-emotion" と呼べる中間状態は外部から確認できない。

**結論（D-c の続き）**：
- §3.X.7–8 の二層構造（readout 大崩れ × shift 完全加法）は **basis 軸対でも
  同パターンで再現**、ただし err_i/err_j は CAA 対より 2–4 倍悪い
  → basis 軸は「より小さい局所性しか持たない」。
- メタ感情クラスタへの射影は **全 7 クラスタで joint < marg_max**。
  §3.X.8(c) の "CAA-pair 残差は既存メタ感情に向かない" は **basis-pair でも
  そのまま成立**。**「Plutchik より細かい現象学」を 2 軸合成で induce する
  最小構成は、ICA k=16 / L=19 では存在しない**。
- ただし per-cell には null を超える signed 残差（最大 |cos| ≈ 0.21）があり、
  ランダムノイズではない。残差方向は **強 steering 下で複数ペア共通して
  "academic ambition / competitive determination" に向かう** という
  attractor-like な構造が見えており、これは **k=16 basis の表現容量限界**
  によるアーティファクトの可能性が高い。
- 仮説：メタ感情 induce には (i) より細かい basis（k=64 / L=22）、
  または (ii) 3 軸以上の合成、または (iii) そもそも単独 basis 軸（b11 など）
  の周辺効果が本質、のいずれか。次の優先は **同 sweep を k=64/L=22 で再走** で
  basis 容量側の説明を切り分けること。

### 3.X.10 basis-native 加法性 + メタ感情射影（k=64 / L=22 で再走）

§3.X.9 で立てた仮説 (i)「k=16 の容量限界が原因」を切り分けるため、§3.X.6 の
champion basis（ICA k=64 / L=22, R²=0.96, retention=0.93）で同 5 ペア × 3×3 α ×
8 prompts = 360 gens を再走。**注意：ICA は (L,k) ごとに成分が並び替わるため
b1/b11/b13 等のインデックスは L19/k16 とは別軸**（layer-consistency map で対応
付けしていないので "literal index" 比較）。

成果物：[results/basis_additivity_metaemotion_L22_k64/](../experiments/results/basis_additivity_metaemotion_L22_k64/)

**(a) readout 加法性**：

| 構成 | err_i | err_j | resid_ratio |
|---|---:|---:|---:|
| L19 / k=16 (§3.X.9) | 0.069 | 0.061 | 0.817 |
| **L22 / k=64**     | **0.049** | **0.063** | **0.803** |

→ err_i が 30% 改善、resid_ratio はほぼ同水準。**basis 容量を 4× にしても
joint と marginal の差は 0.8 のまま**。「readout 大崩れ」は構成不変。

**(b) メタ感情 per-cluster gain（`joint − max(marg)` 中央値）**：

| cluster | L19/k16 gain | **L22/k64 gain** |
|---|---:|---:|
| frustration / despair | −0.013 | **−0.014** |
| ironic amusement | −0.019 | **−0.010** |
| self-doubt / encouragement | −0.020 | **−0.013** |
| uncertainty / indecision | **−0.026** | **−0.014** |
| enthusiasm / curiosity | −0.021 | **−0.015** |
| academic ambition / competitive | −0.019 | **−0.024** |
| romantic idealism | −0.022 | **−0.027** |

→ **依然として全 7 クラスタで gain 負**。ただし 5/7 で gain 縮小（uncertainty
は半減）、(b1,b13) は uncertainty/indecision で **gain = −0.003**（実質ほぼ
0）まで来た。**容量を増やすと "marginal がメタ感情を吸収しきる" 効果が強まる**
方向であり、joint で初めて出るメタ感情成分が新たに見えるわけではない。

**(c) per-cell |cos_resid|**：null p95 ≈ 0.05 に対し最大 0.17。L19/k16 の
0.21 より小さく、cluster 分布も *academic ambition* (8) / *uncertainty* (7) /
*self-doubt* (4) と分散。**強 steering 下のアトラクタ** という解釈は維持される
が、k 増加で symptom は軽減。

**結論（§3.X.10）**：
- 「k=16 の容量不足が negative result の原因」**仮説 (i) は否定**。
  k=64 でも全クラスタ gain 負・joint < marg_max は不変。
- err_i / per-cluster gain magnitude は **k 増加で漸近的に縮小** しており、
  「basis 数を上げるほど marginal の表現力が伸びて joint が新規方向を出さなく
  なる」という単調傾向。これは "メタ感情は basis 軸対の合成で生まれる" 仮説
  そのものへの反証寄り。
- **方針転換**：仮説 (iii)「単独 basis 軸（b11 など）の周辺効果がメタ感情の
  正体」を主軸に据える。§3.X.2 で観察された "過警戒的内省" 等の現象学は
  *single-axis × strong α* で十分再現できる、という方向で検証を組み直す。
  3 軸以上の合成 (ii) は副次優先。

src/emotion_code/
├── caa.py               # Phase B-5 本流（不変）
├── basis.py             # Phase B-5 本流（不変、k=8固定）
├── vad.py               # Phase B-5 本流（不変、参考用に残す）
├── io.py
├── decompose.py         # Phase C: NMF/PCA/ICA/dict 共通 API
├── basis_sweep.py       # Phase C: 多 k × 多 decomposer × 多 seed
├── basis_metrics.py     # Phase C: ラベル独立性スコア
└── basis_interpret.py   # Phase C: 成分 → top texts

experiments/
├── eval_layer_sweep.py             # Phase B-5
├── eval_shift_accuracy.py          # Phase B-5
├── eval_monotonicity.py            # Phase B-5
├── eval_perplexity.py              # Phase B-5
├── eval_vad_r2.py                  # Phase B-5
├── eval_basis_reconstruction.py    # Phase C-2: held-out R²
├── eval_basis_layerconsistency.py  # Phase C-2: 層間マッチング
├── eval_basis_selfconsistency.py   # Phase C-2: hero metric
├── eval_basis_qualitative.py       # Phase C-2: steering 生成例
├── eval_basis_additivity.py        # Phase C-2: basis 成分間の加法性
├── eval_caa_basis_decomposition.py # Phase C-3: CAA を basis で再構成（Phase 1）
├── eval_caa_basis_decomp_steering.py # Phase C-3: 再構成 CAA で steering（Phase 2）
├── eval_caa_basis_additivity.py    # Phase C-3: 再構成 CAA カテゴリ対の加法性（D）
├── eval_caa_additivity_metaemotion.py # Phase C-3: 残差 → メタ感情クラスタ射影（D-c）
├── eval_caa_additivity_pairsim.py  # Phase C-3: 残差比 vs ペア類似度（D-b）
└── eval_basis_additivity_metaemotion.py # Phase C-3: basis 軸対の加法性 + メタ感情射影（D-c 続き）

data/emotion_code/
├── caa.pt, basis.pt, vad_mapping.pt        # B-5 本流
├── basis_sweep/                            # C: 探索成果物 (L19)
│   ├── {nmf,pca,ica,dict}_k{NNN}_seed{N}.pt
│   ├── *.metrics.json
│   ├── metrics.summary.csv
│   ├── stability.summary.csv
│   ├── reconstruction.csv
│   ├── layer_pair_consistency.csv
│   └── layer_component_consistency.csv
└── basis_sweep_L{13,16,22}/                # C: 他層 ICA k=16
```

---

## 5. 次にやること（優先順）

> **完了済み**：(A) k=32/L=22 retention 押し上げ（§3.X.5、retention 0.786）→
> **k=64/L=22 で retention 0.929 に到達**（§3.X.6）、(B) W 構造解析、
> (C) lexical-gap steering、(D) **加法性検定**（§3.X.7、readout 残差 0.81 /
> shift-acc 残差 0.00 → 二層構造を定量化）→ **(D) 追検証 c/b/a 完了**
> （§3.X.8、layer/k 不変で再現、cos_w ↔ 残差 Spearman +1.0、ただし残差は
> 既存メタ感情カタログには射影できず）、(E) LLM judge + メタ感情
> クラスタ、(F) **basis-native 加法性 + メタ感情射影（§3.X.9 ICA k=16/L=19、
> §3.X.10 ICA k=64/L=22 — どちらも全 7 クラスタで joint < marg_max、容量を
> 上げると gain 縮小傾向 ⇒「メタ感情 = basis 2 軸合成」仮説は反証寄り）**。
> 残タスクは下記。

1. **single-axis × strong-α でメタ感情を直接 induce（§3.X.10 結論の主軸）**
   §3.X.9–10 で 2 軸合成は negative。次は **単独 basis 軸 × α∈{±3, ±6}** で
   §3.X.3 の 7 メタ感情クラスタを induce できる軸を網羅。
   既存 [eval_basis_qualitative.py](../experiments/eval_basis_qualitative.py)
   は §3.X.2 で b8 の addressivity 軸を発見済み。これを judge + クラスタ
   射影パイプラインに通して **per-axis × per-cluster gain matrix** を作る。
   実装案：`eval_basis_qualitative.py` の出力 parquet を
   `eval_caa_additivity_metaemotion.py` 互換スキーマで保存できるよう薄い
   adapter を書き、α=0 を baseline として `joint − marg_max` の代わりに
   `α≠0 − α=0` cosine gain を取る。

2. **3 軸以上の合成（副次優先）**
   §3.X.10 で「2 軸では出ない」が確定したので、(b1, b11, b13) 等の 3 軸
   simultaneous steering を試す価値はある。`eval_basis_additivity_metaemotion.py`
   を 3 軸対応に拡張する場合、グリッドサイズが 3³=27 セル × ペア数で
   容易に膨張するため α ∈ {0, ±2} の 3 値固定推奨。

3. **k=128 で R² plateau の最終確認（任意）**
   k=8→64 で R² が 0.75→0.96 と単調増加。k=128 が頭打ちになるかで「内在
   次元の上限」を論文クレームできる。retention 側は既に 0.93 まで来ており
   投資対効果は逓減フェーズ。

4. **メタ感情合成の人手評価 / UI**
   §3.X.3 で同定された 7 メタ感情クラスタ（とくに *uncertainty/indecision*,
   *self-doubt/encouragement*）を **目的的に induce** する steering UI を作り、
   人手で「Plutchik で言えない」率を確認。§3.X.10 の結果から、UI は
   *single-axis × α* スライダーで十分（2 軸合成 UI は不要）。

5. **judge の信頼性確認**
   GPT-4o-mini judge を別 LLM（Claude / Llama-3-70B）で再走 + Krippendorff α
   で agreement を測る。`other_score` が judge bias でないことを保証。

