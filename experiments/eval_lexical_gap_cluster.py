"""Cluster GPT-4o-mini's other_labels into meta-emotion clusters.

Pipeline:
  1. Load judgments.parquet, take rows where other_label != "" at α=+max
  2. Embed each unique other_label with OpenAI text-embedding-3-small
  3. Agglomerative clustering (cosine distance, average linkage)
  4. Heuristic cluster-name = top-2 most-frequent labels in the cluster
  5. Plot:
     a) UMAP/PCA 2-D scatter colored by cluster
     b) Stacked bar: per-target distribution over clusters
  6. Save cluster_assignment.csv, cluster_summary.csv, figures.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


def _embed(client, model: str, texts: list[str]) -> np.ndarray:
    out = []
    BATCH = 128
    for s in range(0, len(texts), BATCH):
        chunk = texts[s:s + BATCH]
        resp = client.embeddings.create(model=model, input=chunk)
        out.extend([d.embedding for d in resp.data])
    return np.array(out, dtype=np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judgments", type=Path,
                   default=Path("experiments/results/lexical_gap_judge/judgments.parquet"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/results/lexical_gap_judge"))
    p.add_argument("--n-clusters", type=int, default=7)
    p.add_argument("--embed-model", type=str, default="text-embedding-3-small")
    p.add_argument("--alpha", type=float, default=2.0,
                   help="Use rows where alpha_unit == this")
    args = p.parse_args()

    load_dotenv()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    judg = pd.read_parquet(args.judgments)
    sub = judg[(judg["alpha_unit"] == args.alpha)
               & (judg["other_label"].astype(str).str.strip() != "")].copy()
    sub["other_label"] = sub["other_label"].str.strip().str.lower()
    print(f"[cluster] {len(sub)} non-empty other_labels at alpha={args.alpha}")

    uniq_labels = sorted(sub["other_label"].unique())
    print(f"[cluster] {len(uniq_labels)} unique labels; embedding...")
    embs = _embed(client, args.embed_model, uniq_labels)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    # Agglomerative clustering on cosine distance
    n_clusters = min(args.n_clusters, len(uniq_labels))
    clu = AgglomerativeClustering(n_clusters=n_clusters,
                                  metric="cosine", linkage="average")
    labels_idx = clu.fit_predict(embs)

    label_to_cid = dict(zip(uniq_labels, labels_idx))

    # Cluster name = top-2 most-frequent labels in cluster (weighted by freq in sub)
    freq = sub["other_label"].value_counts().to_dict()
    cluster_members: dict[int, list[tuple[str, int]]] = {}
    for lab, cid in label_to_cid.items():
        cluster_members.setdefault(cid, []).append((lab, freq.get(lab, 0)))
    cluster_name: dict[int, str] = {}
    for cid, members in cluster_members.items():
        members.sort(key=lambda x: -x[1])
        top = [m[0] for m in members[:2]]
        cluster_name[cid] = " / ".join(top)

    # Assignment table
    sub["cluster_id"] = sub["other_label"].map(label_to_cid)
    sub["cluster_name"] = sub["cluster_id"].map(cluster_name)
    sub[["target", "alpha_unit", "prompt_id", "other_label",
         "other_score", "cluster_id", "cluster_name"]].to_csv(
        args.output_dir / "cluster_assignment.csv", index=False)

    # Summary: per-cluster info
    rows_summary = []
    for cid in sorted(cluster_members):
        members = cluster_members[cid]
        n_sub = int((sub["cluster_id"] == cid).sum())
        rows_summary.append({
            "cluster_id": cid,
            "cluster_name": cluster_name[cid],
            "n_unique_labels": len(members),
            "n_total_assignments": n_sub,
            "top_labels": ", ".join(f"{l}({c})" for l, c in members[:6]),
        })
    summary_df = pd.DataFrame(rows_summary).sort_values("n_total_assignments",
                                                       ascending=False)
    summary_df.to_csv(args.output_dir / "cluster_summary.csv", index=False)
    print("[cluster] === cluster summary ===")
    print(summary_df[["cluster_id", "cluster_name", "n_unique_labels",
                      "n_total_assignments"]].to_string(index=False))

    # 2-D PCA scatter
    pca = PCA(n_components=2)
    pts = pca.fit_transform(embs)
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.get_cmap("tab10")
    for cid in sorted(cluster_members):
        m = labels_idx == cid
        ax.scatter(pts[m, 0], pts[m, 1], s=60, alpha=0.7,
                   color=cmap(cid % 10), label=f"{cid}: {cluster_name[cid]}")
    # annotate top labels
    for cid in sorted(cluster_members):
        members = cluster_members[cid]
        for lab, _ in members[:3]:
            i = uniq_labels.index(lab)
            ax.annotate(lab, (pts[i, 0], pts[i, 1]), fontsize=7, alpha=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"other_labels @ α=+{args.alpha} clustered into "
                 f"{n_clusters} meta-emotion groups")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "cluster_scatter.png", dpi=150)
    plt.close(fig)

    # Per-target stacked bar over clusters
    pivot = (sub.groupby(["target", "cluster_id"]).size()
             .unstack(fill_value=0))
    # column order = same as cluster_summary order
    col_order = sorted(pivot.columns.tolist())
    pivot = pivot[col_order]
    # normalize to fraction
    frac = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = np.zeros(len(frac))
    for cid in col_order:
        vals = frac[cid].values
        ax.bar(frac.index, vals, bottom=bottoms,
               color=cmap(cid % 10),
               label=f"{cid}: {cluster_name[cid]}")
        bottoms += vals
    ax.set_ylabel("fraction of other_labels")
    ax.set_title(f"Per-target distribution over meta-emotion clusters (α=+{args.alpha})")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(args.output_dir / "cluster_per_target.png", dpi=150)
    plt.close(fig)

    print(f"[cluster] wrote {args.output_dir}/cluster_*.{{csv,png}}")


if __name__ == "__main__":
    main()
