"""Project additivity residuals onto meta-emotion cluster centroids.

Background
----------
``eval_caa_basis_additivity.py`` showed that joint steering with
``α v_A + β v_B`` deviates from the linear sum of marginals in *readout*
space (median |resid|/|marg| ≈ 0.81) but is essentially additive in
*shift-acc* space.  This is consistent with a two-layer story:

    joint = (linear sum of marginals)  +  (nonlinear residual)

where the residual is invisible to the 8-way Plutchik classifier yet large
in the basis readout.  The §3.X.3 / §4.X.8 LLM-judge analysis already
identified 7 "meta-emotion" clusters (uncertainty/indecision,
self-doubt/encouragement, enthusiasm/curiosity, …) that are *not*
expressible by Plutchik but recur across single-component steering.

This script asks: **does the additivity residual point in the direction
of those meta-emotion clusters?**  If yes, the nonlinear interaction term
*is* the meta-emotion content — closing the Phase 3 story.

Pipeline
--------
1. Load joint generations from
   ``experiments/results/caa_basis_additivity_<tag>/generations.parquet``.
2. Re-embed every generation with the same OpenAI model used for the
   meta-emotion clusters (``text-embedding-3-small`` by default), L2 normalise.
3. Re-embed the unique judge labels in
   ``experiments/results/lexical_gap_judge/cluster_assignment.csv`` and
   average within each ``cluster_id`` to recover normalised centroids
   ``c_k ∈ R^D``.
4. For every (cat_a, cat_b, α, β) cell, average the per-prompt embeddings
   to obtain ``e(α, β)``.  In each off-diagonal cell compute
       resid_emb = e(α,β) − e(α,0) − e(0,β) + e(0,0)
       marg_emb  = e(α,0) + e(0,β) − e(0,0)              (linear part)
       full_emb  = e(α,β) − e(0,0)                        (full effect)
   plus per-generation cosines for a non-subtraction alternative test.
5. Score each vector against every cluster centroid with cosine.

Outputs (in ``--output-dir``):
- ``embeddings.parquet``      raw L2-normalised embeddings per row
- ``centroids.npz``           cluster centroids and ids/names
- ``cell_residual_cos.csv``   per (cell, cluster) cosines for resid/marg/full
- ``per_text_cos.csv``        per generation × cluster cosines
- ``per_text_summary.csv``    joint vs marginal closeness summary per cluster
- ``summary.json``            top clusters per residual + medians

Usage
-----
    uv run python -m experiments.eval_caa_additivity_metaemotion \
        --additivity-dir experiments/results/caa_basis_additivity_L22_k64 \
        --cluster-dir    experiments/results/lexical_gap_judge \
        --output-dir     experiments/results/caa_additivity_metaemotion_L22_k64
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


def _embed(client, model: str, texts: list[str]) -> np.ndarray:
    out: list[list[float]] = []
    BATCH = 128
    for s in range(0, len(texts), BATCH):
        chunk = texts[s:s + BATCH]
        # OpenAI embedding API rejects empty strings — substitute a single space.
        chunk = [t if t.strip() else " " for t in chunk]
        resp = client.embeddings.create(model=model, input=chunk)
        out.extend([d.embedding for d in resp.data])
    arr = np.asarray(out, dtype=np.float32)
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
    return arr


def _build_centroids(client, model: str, cluster_csv: Path) -> dict:
    df = pd.read_csv(cluster_csv)
    df["other_label"] = df["other_label"].astype(str).str.strip().str.lower()
    df = df[df["other_label"] != ""]
    uniq = (df.drop_duplicates("other_label")
              .sort_values("cluster_id")
              .reset_index(drop=True))
    print(f"[meta] re-embedding {len(uniq)} unique judge labels ...")
    embs = _embed(client, model, uniq["other_label"].tolist())  # already normalised

    centroids: list[np.ndarray] = []
    cids: list[int] = []
    names: list[str] = []
    for cid, sub in uniq.groupby("cluster_id"):
        idx = sub.index.to_numpy()
        c = embs[idx].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        centroids.append(c)
        cids.append(int(cid))
        names.append(str(sub["cluster_name"].iloc[0]))
    C = np.stack(centroids, axis=0)  # [K, D]
    return {"C": C, "cluster_ids": cids, "cluster_names": names,
            "label_embs": embs, "label_table": uniq}


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine between rows of a [N,D] (normalised) and b [K,D] (normalised)."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_n @ b_n.T  # [N, K]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--additivity-dir", type=Path, required=True,
                   help="Directory containing generations.parquet from "
                        "eval_caa_basis_additivity.py")
    p.add_argument("--cluster-dir", type=Path,
                   default=Path("experiments/results/lexical_gap_judge"),
                   help="Directory with cluster_assignment.csv from "
                        "eval_lexical_gap_cluster.py")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--embed-model", type=str, default="text-embedding-3-small")
    p.add_argument("--n-random", type=int, default=64,
                   help="Number of random unit-vector baselines for null cosine.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # ---------- Generations ----------
    gen_path = args.additivity_dir / "generations.parquet"
    df = pd.read_parquet(gen_path)
    df = df.sort_values(["cat_a", "cat_b", "alpha", "beta", "prompt_id"]
                        ).reset_index(drop=True)
    texts = df["generation"].astype(str).tolist()
    print(f"[meta] embedding {len(texts)} generations ...")
    G = _embed(client, args.embed_model, texts)  # [N, D]
    df["emb_idx"] = np.arange(len(df))

    # Persist embeddings (compact npz, plus cell index in parquet)
    np.savez_compressed(args.output_dir / "embeddings.npz", G=G)
    df[["cat_a", "cat_b", "alpha", "beta", "prompt_id", "emb_idx"]].to_parquet(
        args.output_dir / "embeddings.parquet", index=False)

    # ---------- Cluster centroids ----------
    bundle = _build_centroids(client, args.embed_model,
                              args.cluster_dir / "cluster_assignment.csv")
    C = bundle["C"]                     # [K, D]
    cids = bundle["cluster_ids"]
    names = bundle["cluster_names"]
    K = len(cids)
    np.savez_compressed(args.output_dir / "centroids.npz",
                        C=C, cluster_ids=np.asarray(cids),
                        cluster_names=np.asarray(names))

    # ---------- Per-text cosines ----------
    G_to_C = _cos(G, C)  # [N, K]
    per_text_rows: list[dict] = []
    for i, row in df.iterrows():
        for kj, cid in enumerate(cids):
            per_text_rows.append({
                "cat_a": row["cat_a"], "cat_b": row["cat_b"],
                "alpha": float(row["alpha"]), "beta": float(row["beta"]),
                "prompt_id": int(row["prompt_id"]),
                "cluster_id": cid, "cluster_name": names[kj],
                "cos": float(G_to_C[i, kj]),
            })
    pt = pd.DataFrame(per_text_rows)
    pt.to_csv(args.output_dir / "per_text_cos.csv", index=False)

    # ---------- Per-cell mean embeddings + residual ----------
    cell_emb: dict[tuple[str, str, float, float], np.ndarray] = {}
    for (cat_a, cat_b, a, b), sub in df.groupby(
        ["cat_a", "cat_b", "alpha", "beta"], sort=False,
    ):
        idx = sub["emb_idx"].to_numpy()
        cell_emb[(cat_a, cat_b, float(a), float(b))] = G[idx].mean(axis=0)

    pairs = sorted({(r["cat_a"], r["cat_b"]) for _, r in df.iterrows()})
    alphas = sorted(set(float(a) for a in df["alpha"].unique()))

    rng = np.random.default_rng(args.seed)
    R = rng.standard_normal((args.n_random, C.shape[1])).astype(np.float32)

    cell_rows: list[dict] = []
    summary_top: list[dict] = []
    for (cat_a, cat_b) in pairs:
        e00 = cell_emb[(cat_a, cat_b, 0.0, 0.0)]
        for a in alphas:
            for b in alphas:
                if a == 0.0 and b == 0.0:
                    continue
                if a == 0.0 or b == 0.0:
                    # only measure off-diagonal residuals; marginal cells are
                    # used as building blocks.
                    continue
                e_joint = cell_emb[(cat_a, cat_b, a, b)]
                e_a = cell_emb[(cat_a, cat_b, a, 0.0)]
                e_b = cell_emb[(cat_a, cat_b, 0.0, b)]
                resid = e_joint - e_a - e_b + e00
                marg = e_a + e_b - e00
                full = e_joint - e00

                # null distribution: cosine of random unit vectors with each centroid
                null_cos = _cos(R, C)                                  # [n_random, K]
                null_p95 = np.quantile(np.abs(null_cos), 0.95, axis=0)  # [K]

                vecs = {"resid": resid, "marg": marg, "full": full}
                cosines = {name: _cos(v[None, :], C)[0] for name, v in vecs.items()}
                norms = {name: float(np.linalg.norm(v)) for name, v in vecs.items()}

                for kj, cid in enumerate(cids):
                    cell_rows.append({
                        "cat_a": cat_a, "cat_b": cat_b,
                        "alpha": a, "beta": b,
                        "cluster_id": cid, "cluster_name": names[kj],
                        "cos_resid": float(cosines["resid"][kj]),
                        "cos_marg":  float(cosines["marg"][kj]),
                        "cos_full":  float(cosines["full"][kj]),
                        "null_p95":  float(null_p95[kj]),
                        "norm_resid": norms["resid"],
                        "norm_marg":  norms["marg"],
                        "norm_full":  norms["full"],
                    })

                # top cluster per vector type
                for name in ("resid", "marg", "full"):
                    kj_top = int(np.argmax(cosines[name]))
                    summary_top.append({
                        "cat_a": cat_a, "cat_b": cat_b,
                        "alpha": a, "beta": b,
                        "vector": name,
                        "top_cluster_id": cids[kj_top],
                        "top_cluster_name": names[kj_top],
                        "top_cos": float(cosines[name][kj_top]),
                        "null_p95": float(null_p95[kj_top]),
                    })

    cell_df = pd.DataFrame(cell_rows)
    cell_df.to_csv(args.output_dir / "cell_residual_cos.csv", index=False)
    top_df = pd.DataFrame(summary_top)
    top_df.to_csv(args.output_dir / "cell_top_cluster.csv", index=False)

    # ---------- Per-text summary: does joint beat marginals? ----------
    # For each generation, the "joint cosine" is cos(joint_text, centroid_k).
    # The control is the average cos of the two single-axis generations
    # produced for the same (cat_a, cat_b, prompt_id). Aggregate per cluster.
    ptab = pt.copy()
    ptab["cell"] = list(zip(ptab.cat_a, ptab.cat_b, ptab.alpha, ptab.beta,
                            ptab.prompt_id, ptab.cluster_id))
    pivot = (ptab.set_index(["cat_a", "cat_b", "prompt_id", "cluster_id",
                             "cluster_name", "alpha", "beta"])["cos"]
             .unstack(level=["alpha", "beta"]))

    rows_pt: list[dict] = []
    for (cat_a, cat_b, prompt_id, cid, cname), s in pivot.iterrows():
        if (0.0, 0.0) not in s.index:
            continue
        base = s[(0.0, 0.0)]
        for a in alphas:
            for b in alphas:
                if a == 0.0 or b == 0.0:
                    continue
                if (a, b) not in s.index or (a, 0.0) not in s.index \
                   or (0.0, b) not in s.index:
                    continue
                joint = s[(a, b)]
                marg_a = s[(a, 0.0)]
                marg_b = s[(0.0, b)]
                rows_pt.append({
                    "cat_a": cat_a, "cat_b": cat_b,
                    "prompt_id": int(prompt_id),
                    "cluster_id": int(cid), "cluster_name": cname,
                    "alpha": float(a), "beta": float(b),
                    "cos_joint":   float(joint),
                    "cos_marg_a":  float(marg_a),
                    "cos_marg_b":  float(marg_b),
                    "cos_baseline": float(base),
                    "joint_minus_marg_max": float(joint - max(marg_a, marg_b)),
                    "joint_minus_marg_avg": float(joint - 0.5*(marg_a + marg_b)),
                })
    pt_df = pd.DataFrame(rows_pt)
    pt_df.to_csv(args.output_dir / "per_text_summary.csv", index=False)

    # Aggregate: per cluster, median (joint - max(marg_a, marg_b)) across all cells.
    if len(pt_df):
        agg = (pt_df.groupby(["cluster_id", "cluster_name"])
                     .agg(median_joint=("cos_joint", "median"),
                          median_marg_max=("cos_marg_a",
                                           lambda s: float(np.median(
                                               np.maximum(s.values,
                                                          pt_df.loc[s.index, "cos_marg_b"].values)))),
                          median_resid_gain_to_max=("joint_minus_marg_max", "median"),
                          median_resid_gain_to_avg=("joint_minus_marg_avg", "median"),
                          n=("cos_joint", "size"))
                     .reset_index())
        agg = agg.sort_values("median_resid_gain_to_max", ascending=False)
        agg.to_csv(args.output_dir / "per_cluster_summary.csv", index=False)
    else:
        agg = pd.DataFrame()

    # ---------- Top-line summary ----------
    summary = {
        "additivity_dir": str(args.additivity_dir),
        "cluster_dir": str(args.cluster_dir),
        "embed_model": args.embed_model,
        "n_generations": int(len(df)),
        "n_clusters": K,
        "cluster_ids": cids,
        "cluster_names": names,
        "n_pairs": len(pairs),
        "n_offdiag_cells": int(len(cell_df) // K) if K else 0,
        "median_top_cos": {
            "resid": float(top_df[top_df.vector == "resid"]["top_cos"].median()),
            "marg":  float(top_df[top_df.vector == "marg"]["top_cos"].median()),
            "full":  float(top_df[top_df.vector == "full"]["top_cos"].median()),
        },
        "median_null_p95": float(top_df["null_p95"].median()),
        "top_resid_clusters": (top_df[top_df.vector == "resid"]
                               .groupby("top_cluster_name").size()
                               .sort_values(ascending=False).to_dict()),
    }
    if not agg.empty:
        summary["per_cluster_top_gain"] = agg.head(3).to_dict(orient="records")

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---------- Console print ----------
    print("\n[meta] === residual top-cluster distribution ===")
    if not top_df.empty:
        rd = top_df[top_df.vector == "resid"]
        print(rd.groupby("top_cluster_name").size()
                .sort_values(ascending=False).to_string())
        print(f"\n[meta] median top cos (resid) = "
              f"{summary['median_top_cos']['resid']:.3f}, "
              f"(marg) = {summary['median_top_cos']['marg']:.3f}, "
              f"(full) = {summary['median_top_cos']['full']:.3f}, "
              f"null p95 ≈ {summary['median_null_p95']:.3f}")
    if not agg.empty:
        print("\n[meta] === per-cluster median (joint - max(marg)) cosine ===")
        print(agg.to_string(index=False))
    print(f"\n[meta] wrote {args.output_dir}/")


if __name__ == "__main__":
    main()
