# 20. LLM-as-judge による「Plutchik 外」採点

## 1. 実験の理由

[19_lexical_gap_steering.md](19_lexical_gap_steering.md) で外部分類器が
gap 群と pan 群を弁別できなかった。これが (a) gap 成分が無効か (b) 既存
分類器の語彙が狭いかを切り分けるため、**LLM judge に Plutchik 8 を採点
させた上で、Plutchik で表しきれないものに自由記述ラベルを付けさせる**。

## 2. 数式と定義

GPT-4o-mini に temperature=0 で構造化 JSON 出力：
```json
{
	"plutchik": {"joy": 0.0..1.0, ..., "trust": 0.0..1.0},
	"other": {"label": "1-4 word phrase", "score": 0.0..1.0}
}
```

**指標**：
- $\mathrm{plutchik\_max}(y) = \max_c p_c(y)$
- $\mathrm{other\_score}(y) = s_{\text{other}}(y)$
- $\mathrm{frac\_other\_dom}(\text{target}) = \frac{1}{N}\sum_y \mathbb{1}[s_{\text{other}}(y) > \max_c p_c(y)]$

スクリプト：[experiments/eval_lexical_gap_judge.py](../../experiments/eval_lexical_gap_judge.py)
（`--resume`、16 行 checkpoint 対応）

## 3. 結果（α=+2 集計）

| target | plutchik_max | other_score | frac_other_dom |
|---|---:|---:|---:|
| **combo_b11+b13** | 0.49 | **0.55** | **0.75** |
| **gap_b11** | 0.51 | 0.52 | 0.63 |
| pan_b8 | 0.40 | 0.41 | 0.56 |
| その他 | ≤ 0.50 | 0.30–0.45 | 0.40–0.55 |

→ gap 群は pan 群より **「Plutchik 8 軸より自由記述の方が支配的」な比率が
一貫して高い**。Hartmann では取れなかった差異が顕在化。
**(b) 既存分類器の語彙が狭い** が答えだと判明。

成果物：[results/lexical_gap_judge/](../../experiments/results/lexical_gap_judge/)
（`judgments.parquet`, `summary_by_target.csv`, `other_labels.csv`）

## 4. 次の実験

→ [21_meta_emotion_cluster.md](21_meta_emotion_cluster.md) — 自由記述ラベル
を埋め込みクラスタ化し、「メタ感情カタログ」を構築する。
