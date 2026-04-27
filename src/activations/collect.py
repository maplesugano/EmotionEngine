"""Collect last-token residual-stream activations for contrastive pairs.

For each pair (positive, negative) we run the active model from
configs/model.yaml, record `resid_post` at every layer in `hook_layers`, and
write the last-token vector to a safetensors shard. We emit two parallel
shards per chunk (`pos.safetensors`, `neg.safetensors`), each containing one
tensor per layer with shape `[shard_size, d_model]`, plus an `index.parquet`
with row-aligned metadata (`pair_id`, `category`, `provenance`).

Usage
-----
    uv run python -m src.activations.collect \
        --pairs   data/contrastive/pairs.parquet \
        --config  configs/model.yaml \
        --output  data/activations \
        --shard-size 512
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from safetensors.torch import save_file
from tqdm.auto import tqdm

from src.activations._runtime import collect_batch as _collect_batch
from src.activations._runtime import load_model as _load_model


def _flush_shard(
    out_dir: Path,
    shard_idx: int,
    side: str,                       # "pos" or "neg"
    per_layer: dict[int, list[torch.Tensor]],
) -> None:
    payload = {
        f"layer_{l}": torch.cat(per_layer[l], dim=0).contiguous()
        for l in per_layer
    }
    path = out_dir / f"shard_{shard_idx:05d}_{side}.safetensors"
    save_file(payload, str(path))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path,
                   default=Path("data/contrastive/pairs.parquet"))
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--output", type=Path, default=Path("data/activations"))
    p.add_argument("--shard-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0,
                   help="Optional cap on number of pairs (smoke test).")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    profile = cfg["profiles"][cfg["active"]]
    hook_layers: list[int] = list(profile["hook_layers"])
    print(f"[act] active profile = {cfg['active']}, layers = {hook_layers}")

    pairs = pd.read_parquet(args.pairs)
    if args.limit:
        pairs = pairs.head(args.limit).reset_index(drop=True)
    print(f"[act] {len(pairs)} pairs")

    out_dir = args.output / cfg["active"]
    out_dir.mkdir(parents=True, exist_ok=True)

    model, device, _ = _load_model(profile)

    # Per-shard buffers
    pos_buf: dict[int, list[torch.Tensor]] = {l: [] for l in hook_layers}
    neg_buf: dict[int, list[torch.Tensor]] = {l: [] for l in hook_layers}
    shard_meta: list[dict] = []
    shard_idx = 0
    in_shard = 0

    bs = args.batch_size
    for start in tqdm(range(0, len(pairs), bs),
                      desc="[act] batches", unit="batch"):
        chunk = pairs.iloc[start : start + bs]
        pos_texts = chunk["pos_text"].astype(str).tolist()
        neg_texts = chunk["neg_text"].astype(str).tolist()

        pos_acts = _collect_batch(model, pos_texts, hook_layers, device)
        neg_acts = _collect_batch(model, neg_texts, hook_layers, device)
        for l in hook_layers:
            pos_buf[l].append(pos_acts[l])
            neg_buf[l].append(neg_acts[l])

        for j, (_, row) in enumerate(chunk.iterrows()):
            shard_meta.append({
                "shard": shard_idx,
                "row": in_shard + j,
                "pair_id": int(row["pair_id"]),
                "category": row["category"],
                "provenance": row["provenance"],
            })
        in_shard += len(chunk)

        if in_shard >= args.shard_size or start + bs >= len(pairs):
            _flush_shard(out_dir, shard_idx, "pos", pos_buf)
            _flush_shard(out_dir, shard_idx, "neg", neg_buf)
            print(f"[act] flushed shard {shard_idx} ({in_shard} rows)")
            shard_idx += 1
            in_shard = 0
            pos_buf = {l: [] for l in hook_layers}
            neg_buf = {l: [] for l in hook_layers}

    meta_df = pd.DataFrame.from_records(shard_meta)
    meta_path = out_dir / "index.parquet"
    meta_df.to_parquet(meta_path, index=False)

    manifest = {
        "model_profile": cfg["active"],
        "model_name": profile["name"],
        "hook_layers": hook_layers,
        "hook_position": profile.get("hook_position", "last_token"),
        "n_pairs": int(len(pairs)),
        "n_shards": shard_idx,
        "shard_size": args.shard_size,
        "d_model": int(model.cfg.d_model),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[act] wrote {shard_idx} shards + index → {out_dir}")


if __name__ == "__main__":
    main()
