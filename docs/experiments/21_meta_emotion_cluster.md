# 21. メタ感情クラスタリング

## 1. 実験の理由

[20_lexical_gap_judge.md](20_lexical_gap_judge.md) で得た「Plutchik 外」
自由記述ラベル群を **意味的に集約** することで、basis が分離している
「言葉にない感情」のカタログ化を行う。論文上、「メタ感情の発生源」を
議論するための共通参照点。

## 2. 数式と定義

各自由記述 $\ell_i$ を OpenAI `text-embedding-3-small` で
$e_i \in \mathbb{R}^{1536}$ にエンコード。Cosine 距離行列
$D_{ij} = 1 - \langle e_i, e_j\rangle / (\|e_i\|\|e_j\|)$。

**Agglomerative clustering**（average linkage）で 7 クラスタに分割。
クラスタ centroid：$\mu_k = \frac{1}{|C_k|}\sum_{i\in C_k} e_i$。

スクリプト：[experiments/eval_lexical_gap_cluster.py](../../experiments/eval_lexical_gap_cluster.py)

## 3. 結果（α=+2、109 ラベル / 61 unique → 7 群）

| cluster | name | n_assignments |
|---:|---|---:|
| 2 | **uncertainty / indecision** | **43** |
| 6 | enthusiasm / curiosity | 28 |
| 0 | **self-doubt / encouragement** | **25** |
| 1 | frustration / despair | 9 |
| 4 | academic ambition / competitive determination | 2 |
| 3 | romantic idealism | 1 |
| 5 | ironic amusement | 1 |

→ **約 6 割（68/109）が「不確実性 / 自己疑念 / 優柔不断」群**。
b1, b11, b13 が掴んでいるのは Plutchik 8 より一段上の **メタ認知的状態**
（uncertainty, self-doubt, indecision, …）。これらは英語の単一語では
命名困難だが、judge LLM が複数語句で記述できる程度には言語化可能な
「中間状態」。

**仮説の精緻化**：当初「言葉にない感情」と呼んでいたものは、より正確には
**Plutchik より細かい現象学的グラニュラリティ — とくにメタ認知的状態**である。

成果物：[results/lexical_gap_judge/](../../experiments/results/lexical_gap_judge/)
（`cluster_assignment.csv`, `cluster_summary.csv`,
`cluster_scatter.png`, `cluster_per_target.png`）

## 4. 次の実験

→ [22_k_scaling_plateau.md](22_k_scaling_plateau.md) — basis の表現力 $k$ を
上げて R² と retention の plateau を探る。論文 Phase 1〜2 の核心結果。
