"""Shared runtime helpers used by collection and downstream activation passes.

Extracted from ``src.activations.collect`` so other modules (e.g.
``src.emotion_code.vad``) can reuse the same model-loading and last-token
extraction logic without import cycles.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml


def load_profile(config_path: Path | str = Path("configs/model.yaml"),
                 profile: str | None = None) -> tuple[dict, str]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    name = profile or cfg["active"]
    return cfg["profiles"][name], name


def load_model(profile: dict):
    """Load via transformer_lens, streaming weights to GPU for >=7B models."""
    from transformer_lens import HookedTransformer

    name = profile["name"]
    dtype_str = profile.get("dtype", "float32")
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_str]

    family = profile.get("family", "")
    is_large = family in {"llama"} or "8b" in name.lower() or "7b" in name.lower()
    if is_large and not torch.cuda.is_available():
        raise RuntimeError(
            f"Model '{name}' requires CUDA; refusing to load on CPU."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[runtime] loading {name} dtype={dtype_str} device={device} large={is_large}")

    if is_large:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        hf_model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dtype, low_cpu_mem_usage=True, device_map={"": device},
        )
        model = HookedTransformer.from_pretrained_no_processing(
            name, hf_model=hf_model, tokenizer=tok, device=device, dtype=dtype,
        )
        del hf_model
    else:
        model = HookedTransformer.from_pretrained_no_processing(
            name, device=device, dtype=dtype,
        )

    model.eval()
    if device == "cuda":
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"[runtime] cuda mem after load: free={free/1e9:.2f} GB / total={total/1e9:.2f} GB")
    return model, device, dtype


@torch.inference_mode()
def collect_batch(
    model,
    texts: list[str],
    hook_layers: list[int],
    device: str,
) -> dict[int, torch.Tensor]:
    """Return ``{layer: tensor[B, d_model]}`` of last-token resid_post (fp32, CPU)."""
    captured: dict[int, torch.Tensor] = {}

    tok_lists = [model.to_tokens(t, prepend_bos=True)[0] for t in texts]
    max_len = max(t.shape[0] for t in tok_lists)
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id
    padded = torch.full(
        (len(tok_lists), max_len), pad_id, dtype=tok_lists[0].dtype, device=device
    )
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
