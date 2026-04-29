"""Generation utilities with steering hooks.

We do *not* use ``HookedTransformer.generate`` because, on bf16 Llama-3.1
loaded via the HF route, its KV-cache decode path triggers a dtype mismatch
in attention (``q @ k`` Float vs BFloat16). Instead we run a simple greedy
decode by re-doing a full-sequence forward each step — slower per token,
but identical to the path that ``src.activations.collect`` already exercises
successfully.
"""

from __future__ import annotations

import torch

from src.steering.hook import steering_hooks


@torch.inference_mode()
def steered_generate(
    model,
    prompt: str,
    vector: torch.Tensor,
    alpha: float,
    layers: list[int],
    apply_to: str = "generation",
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> str:
    """Greedy (or top-p) decode with optional CAA steering injected at ``layers``.

    Extra knobs to escape greedy self-loops:
      - ``repetition_penalty`` (>1.0) divides logits of already-emitted tokens.
      - ``no_repeat_ngram_size`` (>0) forbids repeating any n-gram of that size.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)  # [1, S]
    prompt_len = int(tokens.shape[1])
    device = next(model.parameters()).device
    tokens = tokens.to(device)

    eos_id = model.tokenizer.eos_token_id

    def _banned_by_ngram() -> list[int]:
        n = no_repeat_ngram_size
        if n <= 0 or tokens.shape[1] < n:
            return []
        seq = tokens[0].tolist()
        prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
        banned: set[int] = set()
        for i in range(len(seq) - n + 1):
            if tuple(seq[i : i + n - 1]) == prefix:
                banned.add(seq[i + n - 1])
        return list(banned)

    def _step():
        logits = model(tokens)              # [1, S, V]
        next_logits = logits[0, -1].float() # [V]
        if repetition_penalty != 1.0:
            seen = torch.unique(tokens[0])
            pos = next_logits[seen] > 0
            scaled = next_logits[seen].clone()
            scaled[pos] = scaled[pos] / repetition_penalty
            scaled[~pos] = scaled[~pos] * repetition_penalty
            next_logits[seen] = scaled
        banned = _banned_by_ngram()
        if banned:
            next_logits[banned] = float("-inf")
        if temperature <= 0:
            return int(next_logits.argmax())
        probs = torch.softmax(next_logits / max(temperature, 1e-6), dim=-1)
        if 0.0 < top_p < 1.0:
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_p, dim=-1)
            mask = cum > top_p
            mask[0] = False
            sorted_p[mask] = 0.0
            sorted_p = sorted_p / sorted_p.sum()
            choice = int(torch.multinomial(sorted_p, 1).item())
            return int(sorted_i[choice].item())
        return int(torch.multinomial(probs, 1).item())

    if alpha == 0.0:
        for _ in range(max_new_tokens):
            nxt = _step()
            tokens = torch.cat([tokens, torch.tensor([[nxt]], device=device, dtype=tokens.dtype)], dim=1)
            if eos_id is not None and nxt == eos_id:
                break
    else:
        with steering_hooks(
            model, vector=vector, alpha=alpha, layers=layers,
            apply_to=apply_to, prompt_len=prompt_len,
        ):
            for _ in range(max_new_tokens):
                nxt = _step()
                tokens = torch.cat(
                    [tokens, torch.tensor([[nxt]], device=device, dtype=tokens.dtype)], dim=1,
                )
                if eos_id is not None and nxt == eos_id:
                    break

    new_tokens = tokens[0, prompt_len:]
    return model.tokenizer.decode(new_tokens, skip_special_tokens=True)

