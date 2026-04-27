"""
tests/test_hook_smoke.py
------------------------
Smoke test: load GPT-2 medium via TransformerLens and extract
residual-stream activations from the configured hook layers.

Run:
    python tests/test_hook_smoke.py
    # or via pytest:
    pytest tests/test_hook_smoke.py -v
"""

import sys
import pathlib

# Allow running from repo root or tests/ directly
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import yaml
import torch
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_config(profile: str = "gpt2") -> dict:
    cfg_path = pathlib.Path(__file__).parent.parent / "configs" / "model.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["profiles"][profile]


def get_residual_streams_tl(model_name: str, hook_layers: list[int], text: str):
    """Extract last-token residual-stream vectors using TransformerLens."""
    import transformer_lens

    model = transformer_lens.HookedTransformer.from_pretrained(
        model_name, center_writing_weights=False
    )
    model.eval()

    cache: dict[str, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(value, hook):  # noqa: ANN001
            # value shape: (batch, seq_len, d_model)
            cache[f"layer_{layer_idx}"] = value[:, -1, :].detach().cpu()
        return hook_fn

    hooks = [
        (f"blocks.{layer}.hook_resid_post", make_hook(layer))
        for layer in hook_layers
    ]

    tokens = model.to_tokens(text)
    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=hooks)

    return cache


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGPT2HookSmoke:
    cfg = load_model_config("gpt2")

    def test_transformer_lens_import(self):
        import transformer_lens  # noqa: F401

    def test_residual_stream_extraction(self):
        cache = get_residual_streams_tl(
            model_name=self.cfg["name"],
            hook_layers=self.cfg["hook_layers"],
            text="The emotion of joy fills my heart.",
        )

        # One entry per hook layer
        assert set(cache.keys()) == {
            f"layer_{l}" for l in self.cfg["hook_layers"]
        }, f"Expected keys {self.cfg['hook_layers']}, got {list(cache.keys())}"

        # Each activation should be shape (1, d_model=1024 for GPT-2 medium)
        for key, tensor in cache.items():
            assert tensor.ndim == 2, f"{key}: expected 2D tensor, got {tensor.shape}"
            assert tensor.shape[0] == 1, f"{key}: batch dim should be 1"
            assert tensor.shape[1] == 1024, (
                f"{key}: expected d_model=1024 for GPT-2 medium, got {tensor.shape[1]}"
            )
        print(f"\n✓ Extracted residual streams from layers {self.cfg['hook_layers']}")
        for k, v in cache.items():
            print(f"  {k}: shape={tuple(v.shape)}, "
                  f"mean={v.mean().item():.4f}, std={v.std().item():.4f}")

    def test_hook_idempotent(self):
        """Running twice should yield identical results (no side effects from hooks)."""
        text = "A sense of calm and quiet."
        cache1 = get_residual_streams_tl(
            self.cfg["name"], self.cfg["hook_layers"], text
        )
        cache2 = get_residual_streams_tl(
            self.cfg["name"], self.cfg["hook_layers"], text
        )
        for key in cache1:
            assert torch.allclose(cache1[key], cache2[key]), (
                f"Hook not idempotent for {key}"
            )
        print("\n✓ Hook is idempotent (two forward passes yield identical results)")


# ---------------------------------------------------------------------------
# Entry point (also runnable as a plain script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t = TestGPT2HookSmoke()
    print("=== test_transformer_lens_import ===")
    t.test_transformer_lens_import()
    print("PASS")

    print("\n=== test_residual_stream_extraction ===")
    t.test_residual_stream_extraction()
    print("PASS")

    print("\n=== test_hook_idempotent ===")
    t.test_hook_idempotent()
    print("PASS")

    print("\n✓ All smoke tests passed.")
