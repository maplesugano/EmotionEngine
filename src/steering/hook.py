"""Inference-time steering hooks for residual-stream injection.

A steering vector ``v ∈ R^{d_model}`` is added to ``hook_resid_post`` at one
or more layers. Two application modes:

* ``apply_to="all"``         — add to every position (prompt + generation).
* ``apply_to="generation"``  — add only to positions strictly after the
                                 prompt length captured at first call.

The hook keeps no global state across requests; build a fresh hook per call.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def make_caa_hook(
    vector: torch.Tensor,
    alpha: float,
    apply_to: str = "generation",
    prompt_len: int | None = None,
):
    """Return a ``transformer_lens`` forward hook that adds ``alpha*vector``.

    Parameters
    ----------
    vector : Tensor[d_model]
    alpha : float
    apply_to : "generation" | "all"
    prompt_len : int or None
        Length of the original prompt (number of tokens including BOS).
        Required for ``apply_to="generation"``. Tokens at positions
        ``>= prompt_len`` receive the perturbation. During autoregressive
        decoding TL feeds either the full sequence or single-token KV
        increments; both cases are handled.
    """
    if apply_to not in {"generation", "all"}:
        raise ValueError(f"apply_to must be 'generation' or 'all', got {apply_to!r}")
    if apply_to == "generation" and prompt_len is None:
        raise ValueError("prompt_len is required when apply_to='generation'")

    v = vector.detach().to(dtype=torch.float32)

    def _hook(act, hook):  # noqa: ANN001
        # act: [batch, seq, d_model] or [batch, 1, d_model] during decode
        delta = (alpha * v).to(device=act.device, dtype=act.dtype)
        if apply_to == "all":
            act = act + delta
            return act
        # apply_to == "generation"
        seq = act.shape[1]
        if seq == 1:
            # single new token in cached decoding; no way to know absolute
            # position from `act` alone, so trust prompt_len: every single-
            # token forward after the initial one is a generated token.
            # The first call (prompt) has seq == prompt_len > 1.
            act = act + delta
            return act
        # Full-sequence forward: only positions >= prompt_len are generation.
        if seq <= prompt_len:
            return act
        act = act.clone()
        act[:, prompt_len:, :] = act[:, prompt_len:, :] + delta
        return act

    return _hook


@contextmanager
def steering_hooks(
    model,
    vector: torch.Tensor,
    alpha: float,
    layers: list[int],
    apply_to: str = "generation",
    prompt_len: int | None = None,
) -> Iterator[None]:
    """Context manager that installs the CAA hook at every layer in ``layers``."""
    hooks = [
        (
            f"blocks.{l}.hook_resid_post",
            make_caa_hook(vector, alpha, apply_to=apply_to, prompt_len=prompt_len),
        )
        for l in layers
    ]
    with model.hooks(fwd_hooks=hooks):
        yield
