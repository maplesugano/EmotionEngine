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

import numpy as np
import pandas as pd
import torch
import yaml
from safetensors.torch import save_file
from tqdm.auto import tqdm


def _load_model(profile: dict):
    """Load via transformer_lens.

    For large (>=7B) models we first load the HF weights directly onto the
    GPU with the requested dtype, then hand that instance to TL. This avoids
    HF's default behaviour of materialising an fp32 CPU copy first, which on
    a 32 GB / 0-swap host triggers an OOM kill before the GPU is ever used.
    """
    from transformer_lens import HookedTransformer

    name = profile["name"]
    dtype_str = profile.get("dtype", "float32")
    dtype = {"float32": torch.float32,
             "bfloat16": torch.bfloat16,
             "float16": torch.float16}[dtype_str]

    family = profile.get("family", "")
    is_large = family in {"llama"} or "8b" in name.lower() or "7b" in name.lower()

    if is_large and not torch.cuda.is_available():
        raise RuntimeError(
            f"Model '{name}' requires CUDA; torch.cuda.is_available()=False. "
            "Refusing to load on CPU (would OOM the host)."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[act] loading {name} dtype={dtype_str} device={device} large={is_large}")

    if is_large:
        # Stream weights straight to GPU; no fp32 CPU copy.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        hf_model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": device},
        )
        model = HookedTransformer.from_pretrained_no_processing(
            name,
            hf_model=hf_model,
            tokenizer=tok,
            device=device,
            dtype=dtype,
        )
        del hf_model
    else:
        model = HookedTransformer.from_pretrained_no_processing(
            name, device=device, dtype=dtype
        )

    model.eval()
    if device == "cuda":
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"[act] cuda mem after load: free={free/1e9:.2f} GB / total={total/1e9:.2f} GB")
    return model, device, dtype


@torch.inference_mode()
def _collect_batch(
    model,
    texts: list[str],
    hook_layers: list[int],
    device: str,
) -> dict[int, torch.Tensor]:
    """Return {layer: tensor[B, d_model]} of last-token resid_post."""
    captured: dict[int, torch.Tensor] = {}

    # transformer_lens tokenizes with left-padding via `prepend_bos=True`
    # by default; we right-pad and read the last *real* token per row.
    tok_lists = [model.to_tokens(t, prepend_bos=True)[0] for t in texts]
    max_len = max(t.shape[0] for t in tok_lists)
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id
    padded = torch.full((len(tok_lists), max_len), pad_id,
                        dtype=tok_lists[0].dtype, device=device)
    real_len = torch.empty(len(tok_lists), dtype=torch.long)
    for i, t in enumerate(tok_lists):
        padded[i, : t.shape[0]] = t.to(device)
        real_len[i] = t.shape[0]

    def _make_hook(layer_idx: int):
        def _hook(act, hook):
            idx = (real_len - 1).to(act.device)
            gather = act[torch.arange(act.shape[0], device=act.device), idx, :]
            captured[layer_idx] = gather.detach().to("cpu", torch.float32)
        return _hook

    fwd_hooks = [
        (f"blocks.{l}.hook_resid_post", _make_hook(l)) for l in hook_layers
    ]
    with model.hooks(fwd_hooks=fwd_hooks):
        _ = model(padded)

    return captured


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
