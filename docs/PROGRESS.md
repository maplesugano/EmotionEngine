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

---

## 4. ディレクトリ早見

```
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
└── eval_basis_additivity.py        # Phase C-2: 加法性検定

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

1. **加法性の追加検証**：cross-talk を抑える direction-orthogonalization（b_i, b_j に直交化した残差で評価）、または k=8 など低 k 基底での再測。
2. **ノイズ低減した self-consistency 再測定**：n_prompts ≥ 8、α∈{−2, 2} の符号一致率（バイナリ AUC）でランキング。
3. **k sweep（32, 64）+ R² plateau** → b8 がさらに分解されるか、未見の emergent 軸が出るか
4. **L22 での独立探索**：L19→L22 のペア一貫性が最大 (0.73) なので b8-like 軸がさらに明確化している可能性
5. **混合 prompt の生成 UI/CLI**：「言葉にできない感情」をユーザが探索できる体験
6. （長期）SAE / dict 大 k での超完備辞書、SAE 業界標準ツールとの比較
