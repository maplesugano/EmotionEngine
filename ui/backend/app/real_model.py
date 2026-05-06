"""Real-model backend for the Emotion DJ Booth.

Loads Llama-3.1-8B-Instruct (or whatever ``configs/model.yaml`` has active,
provided d_model matches the basis) plus the 64-D ICA basis from
``data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt`` and exposes the same
function surface that ``emotion_engine.py`` previously consumed from the
mock implementation.

Pipeline
--------
* ``text_to_basis(text)``
    1. Run a single forward pass over ``text`` with a hook on layer 22 to
       capture the last-token residual ``h ∈ R^{4096}``.
    2. Project into the 64-D ICA basis: ``z = W @ h``  (rows of ``W`` are the
       64 ICA components in 4096-D residual space).
* ``basis_to_macro(z)``     — sigmoid of category loadings · z (8 Plutchik).
* ``basis_to_projection(z)`` — first two PCs of the 8 category loadings,
  applied to ``z``, squashed with tanh.
* ``rewrite_text(source, target_basis, …)``
    1. Recompute source basis ``z_src`` from ``source`` (cached by hash).
    2. Build a steering vector in 4096-D: ``Δh = (z_target − z_src) · W``.
    3. Generate via ``src.steering.generate.steered_generate`` with the
       chat-templated prompt, injecting ``Δh`` at layer 22.
* ``PRESET_VECTORS``       — 6 named meta-emotion directions in 64-D space,
  derived from CAA category directions.

Module-level globals are populated lazily on first use to keep import time
cheap (and so the FastAPI worker doesn't block uvicorn startup if model
loading fails partway through).
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import yaml

# ── Make the repo importable when running uvicorn from ui/backend/ ──────────
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.activations._runtime import collect_batch, load_model, load_profile  # noqa: E402
from src.steering.generate import steered_generate  # noqa: E402

from .models import BASIS_DIM, BASIS_LABELS, MACRO_EMOTIONS  # noqa: E402
from .utils import seed_from_text  # noqa: E402


# ── Configuration ──────────────────────────────────────────────────────────

BASIS_PATH = (
    _REPO_ROOT / "data" / "emotion_code" / "basis_sweep_L22" / "ica_k064_seed0.pt"
)
BASIS_EXCLUDE_PATH = BASIS_PATH.with_suffix(".exclude.json")
MODEL_CFG = _REPO_ROOT / "configs" / "model.yaml"
STEER_CFG = _REPO_ROOT / "configs" / "steering.yaml"

# Layer at which the 64-D basis was learned, and where we both *read* the
# residual (for analyze) and *inject* the steering delta (for rewrite).
BASIS_LAYER = 22


# ── Lazy-init singleton ────────────────────────────────────────────────────

_lock = threading.Lock()
_state: dict = {}


def _ensure_loaded() -> dict:
    """Load model + basis on first use. Subsequent calls return cached state."""
    if _state.get("ready"):
        return _state
    with _lock:
        if _state.get("ready"):
            return _state

        print(f"[real_model] loading basis from {BASIS_PATH}")
        bundle = torch.load(BASIS_PATH, map_location="cpu", weights_only=False)
        if bundle["k"] != BASIS_DIM:
            raise RuntimeError(
                f"basis k={bundle['k']} does not match BASIS_DIM={BASIS_DIM}"
            )
        if bundle["layer"] != BASIS_LAYER:
            raise RuntimeError(
                f"basis layer={bundle['layer']} does not match "
                f"BASIS_LAYER={BASIS_LAYER}"
            )
        # W: [64, 4096] basis vectors (one row per ICA component).
        W = bundle["ica"]["W"].to(torch.float32).contiguous()
        # category_loadings: [8, 64] cosine of each Plutchik category onto basis.
        cat_load = bundle["ica"]["category_loadings"].to(torch.float32)
        cat_names: list[str] = list(bundle["categories"])

        # Reorder category_loadings rows to match MACRO_EMOTIONS order so
        # downstream code can index by Plutchik name without translation.
        cat_idx = [cat_names.index(name) for name in MACRO_EMOTIONS]
        W_macro = cat_load[cat_idx, :].numpy()                 # [8, 64]
        W_macro_pinv = np.linalg.pinv(W_macro)                 # [64, 8]

        # 2-D projection axes = first two PCs of category loadings in basis
        # space. Stable, interpretable map of where Plutchik categories sit
        # in the 64-D code.
        cl_centered = W_macro - W_macro.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(cl_centered, full_matrices=False)
        W_proj = Vt[:2].copy()                                  # [2, 64]
        # SVD sign is arbitrary — pin signs deterministically so the +x and
        # +y directions always point toward the basis component with the
        # single largest contribution (positive). Without this the axis
        # labels could flip on every server restart.
        for row in range(2):
            j = int(np.argmax(np.abs(W_proj[row])))
            if W_proj[row, j] < 0:
                W_proj[row] *= -1.0
        W_proj_pinv = np.linalg.pinv(W_proj)                   # [64, 2]

        # ── Verbal names for each of the 64 basis components ─────────────
        # Each component i gets a phrase. Prefer a hand-curated /
        # judge-derived label from BASIS_LABELS (Plutchik-free); fall back
        # to a category_loadings-derived phrase only when no label exists.
        basis_phrases: list[str] = []
        for i in range(BASIS_DIM):
            curated = BASIS_LABELS.get(i)
            if curated:
                basis_phrases.append(curated)
                continue
            col = W_macro[:, i]                                 # [8]
            pos_j = int(np.argmax(col))
            neg_j = int(np.argmin(col))
            pos_w = float(col[pos_j])
            neg_w = float(col[neg_j])
            pos_name = MACRO_EMOTIONS[pos_j]
            neg_name = MACRO_EMOTIONS[neg_j]
            if max(pos_w, -neg_w) < 0.04:
                phrase = "neutral component"
            elif pos_w > 0.04 and -neg_w > 0.04 and -neg_w > 0.4 * pos_w:
                phrase = f"{pos_name}↑ {neg_name}↓"
            elif pos_w >= -neg_w:
                phrase = f"{pos_name}-leaning"
            else:
                phrase = f"anti-{neg_name}"
            basis_phrases.append(phrase)

        # ── Axis labels: top-K basis components per direction ────────────
        K_AXIS = 3

        def _axis_entries(weights: np.ndarray, descending: bool) -> list[dict]:
            order = np.argsort(weights)
            picks = order[::-1] if descending else order
            out: list[dict] = []
            for idx in picks[:K_AXIS]:
                idx_int = int(idx)
                out.append(
                    {
                        "index": idx_int,
                        "weight": float(weights[idx_int]),
                        "phrase": basis_phrases[idx_int],
                    }
                )
            return out

        axis_labels = {
            "pos_x": _axis_entries(W_proj[0], descending=True),
            "neg_x": _axis_entries(W_proj[0], descending=False),
            "pos_y": _axis_entries(W_proj[1], descending=True),
            "neg_y": _axis_entries(W_proj[1], descending=False),
        }

        # Preset directions in 64-D basis space, built from category loadings
        # so each preset sits on actual CAA-meaningful axes.
        def _norm(v: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v) + 1e-8
            return v / n

        joy = W_macro[MACRO_EMOTIONS.index("joy")]
        trust = W_macro[MACRO_EMOTIONS.index("trust")]
        fear = W_macro[MACRO_EMOTIONS.index("fear")]
        sadness = W_macro[MACRO_EMOTIONS.index("sadness")]
        anger = W_macro[MACRO_EMOTIONS.index("anger")]
        anticipation = W_macro[MACRO_EMOTIONS.index("anticipation")]
        surprise = W_macro[MACRO_EMOTIONS.index("surprise")]

        preset_vectors: Dict[str, np.ndarray] = {
            "uncertainty":           _norm(fear + surprise - trust),
            "self_doubt":            _norm(fear + sadness - trust),
            "analytical_detachment": _norm(-(joy + sadness + anger)),
            "addressivity":          _norm(anticipation + surprise),
            "warmth":                _norm(joy + trust),
            "urgency":               _norm(anger + anticipation),
        }
        # Note: preset vectors get the pathology mask applied AFTER the
        # exclude list is loaded below, so they too cannot push along
        # flagged axes.

        # Steering-gain calibration. We want one "unit" of basis edit to
        # produce a 4096-D injection comparable in magnitude to one CAA
        # std-norm unit. Empirically Llama-3.1-8B residuals at L22 have
        # ||h|| in O(50–150); adjust ``reference_h_norm`` to taste.
        median_W_norm = float(np.median(np.linalg.norm(W.numpy(), axis=1)))
        reference_h_norm = 80.0
        steer_gain = reference_h_norm / max(median_W_norm, 1e-6)

        sc = yaml.safe_load(STEER_CFG.read_text())
        apply_to = sc["caa"].get("apply_to", "generation")

        # ── Pathological-axis exclude list ───────────────────────────────
        # Produced by ``experiments/eval_basis_pathology.py``; sidecar JSON
        # next to the basis artifact lists axis indices that drive the
        # model into repetition loops or dirty-language emissions. We zero
        # those components in every *delta* (preset, macro slider, latent
        # drag, rewrite) so user edits never push along them. The source
        # ``z`` from ``text_to_basis`` is left untouched so analyze stays
        # faithful. Set EMOTION_BASIS_EXCLUDE_DISABLE=1 to bypass.
        excluded: list[int] = []
        if os.environ.get("EMOTION_BASIS_EXCLUDE_DISABLE") not in ("1", "true", "TRUE"):
            if BASIS_EXCLUDE_PATH.exists():
                try:
                    payload = json.loads(BASIS_EXCLUDE_PATH.read_text())
                    excluded = sorted({
                        int(i) for i in payload.get("exclude", [])
                        if 0 <= int(i) < BASIS_DIM
                    })
                    print(
                        f"[real_model] basis exclude list: {len(excluded)}/{BASIS_DIM} "
                        f"axes from {BASIS_EXCLUDE_PATH.name} -> {excluded}"
                    )
                except Exception as exc:
                    print(
                        f"[real_model] WARN: failed to parse {BASIS_EXCLUDE_PATH.name}: "
                        f"{exc}; no axes excluded."
                    )
            else:
                print(
                    f"[real_model] no exclude sidecar at {BASIS_EXCLUDE_PATH.name}; "
                    "all 64 axes active."
                )
        excluded_set = set(excluded)
        # Mask: 0 on excluded axes, 1 elsewhere. Multiply against any 64-D
        # delta to silence those components.
        delta_mask = np.ones(BASIS_DIM, dtype=np.float32)
        if excluded:
            delta_mask[np.asarray(excluded, dtype=np.int64)] = 0.0
            # Apply mask + re-normalise preset directions in place so
            # ``apply_preset`` cannot move along pathological axes.
            for name, vec in list(preset_vectors.items()):
                masked = vec * delta_mask
                n = float(np.linalg.norm(masked))
                preset_vectors[name] = masked / n if n > 1e-8 else masked

        # Load model.
        profile, prof_name = load_profile(MODEL_CFG)
        if profile.get("family") != "llama":
            raise RuntimeError(
                f"basis_sweep_L22/ica_k064_seed0.pt was built from llama "
                f"activations; configs/model.yaml has active='{prof_name}' "
                f"({profile.get('family')!r}). Set active: llama and retry."
            )
        print(f"[real_model] loading model profile={prof_name}")
        model, device, _dtype = load_model(profile)

        _state.update(
            ready=True,
            model=model,
            device=device,
            W=W,                          # torch [64, 4096] cpu fp32
            W_np=W.numpy(),
            W_macro=W_macro,              # [8, 64]
            W_macro_pinv=W_macro_pinv,    # [64, 8]
            W_proj=W_proj,                # [2, 64]
            W_proj_pinv=W_proj_pinv,      # [64, 2]
            preset_vectors=preset_vectors,
            basis_phrases=basis_phrases,
            axis_labels=axis_labels,
            steer_gain=steer_gain,
            apply_to=apply_to,
            source_basis_cache={},        # text-hash -> np.ndarray[64]
            excluded_components=excluded,
            excluded_set=excluded_set,
            delta_mask=delta_mask,
        )
        # Make presets visible through the module-level proxy.
        PRESET_VECTORS.update(preset_vectors)  # type: ignore[attr-defined]
        print(
            f"[real_model] ready. layer={BASIS_LAYER} "
            f"steer_gain={steer_gain:.3f} median_W_norm={median_W_norm:.4f}"
        )
        return _state


# ── Internal helpers ───────────────────────────────────────────────────────


def _get_residual(text: str) -> np.ndarray:
    """Return last-token residual at BASIS_LAYER as fp32 numpy [4096]."""
    s = _ensure_loaded()
    captured = collect_batch(
        s["model"], [text], hook_layers=[BASIS_LAYER], device=s["device"]
    )
    return captured[BASIS_LAYER][0].numpy().astype(np.float32)


def _basis_from_residual(h: np.ndarray) -> np.ndarray:
    s = _ensure_loaded()
    return s["W_np"] @ h  # [64]


def _cache_source(text: str, basis: np.ndarray) -> None:
    s = _ensure_loaded()
    cache: dict = s["source_basis_cache"]
    cache[seed_from_text(text)] = basis
    if len(cache) > 32:
        # FIFO eviction
        for old_key in list(cache.keys())[:-32]:
            cache.pop(old_key, None)


def _get_or_compute_source_basis(text: str) -> np.ndarray:
    s = _ensure_loaded()
    key = seed_from_text(text)
    cached = s["source_basis_cache"].get(key)
    if cached is not None:
        return cached
    h = _get_residual(text)
    z = _basis_from_residual(h)
    _cache_source(text, z)
    return z


def _chat_prompt(text: str) -> str:
    s = _ensure_loaded()
    tok = s["model"].tokenizer
    messages = [
        {
            "role": "user",
            "content": (
                "Rewrite the following text. Keep the meaning roughly the "
                "same; only vary the phrasing.\n\n"
                f"{text}\n\nRewritten:"
            ),
        }
    ]
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"Rewrite the following text:\n\n{text}\n\nRewritten:"


# ── Public surface (matches the previous mock_model API) ───────────────────

# Populated by _ensure_loaded(); emotion_engine indexes this by preset name.
PRESET_VECTORS: Dict[str, np.ndarray] = {}


def get_basis_phrases() -> list[str]:
    """Return the verbal label for each of the 64 basis components."""
    return list(_ensure_loaded()["basis_phrases"])


def get_axis_labels() -> dict:
    """Return top-K basis contributors per latent-map axis direction."""
    return _ensure_loaded()["axis_labels"]


def get_excluded_components() -> list[int]:
    """Return basis component indices flagged as pathological (repetition/toxicity).

    Used by /api/meta so the frontend can hide them in the palette and by
    the rewrite path to zero them in the steering delta.
    """
    return list(_ensure_loaded()["excluded_components"])


def text_to_basis(text: str) -> np.ndarray:
    """Real text → 64-D basis vector via Llama L22 residual + ICA projection."""
    h = _get_residual(text or " ")
    z = _basis_from_residual(h)
    _cache_source(text, z)
    return z


def basis_to_macro(basis: np.ndarray) -> Dict[str, float]:
    """64-D basis → 8 Plutchik macro-emotion scores."""
    s = _ensure_loaded()
    logits = s["W_macro"] @ basis                           # [8]
    probs = 1.0 / (1.0 + np.exp(-logits * 0.5))             # mild temperature
    return {name: float(p) for name, p in zip(MACRO_EMOTIONS, probs)}


def macro_to_basis_delta(
    current_basis: np.ndarray,
    target_macro: Dict[str, float],
    blend: float = 0.5,
) -> np.ndarray:
    """Pseudo-inverse update for macro-slider edits."""
    s = _ensure_loaded()
    W_macro: np.ndarray = s["W_macro"]
    W_pinv: np.ndarray = s["W_macro_pinv"]

    current_logits = W_macro @ current_basis
    current_probs = 1.0 / (1.0 + np.exp(-current_logits * 0.5))
    eps = 1e-4
    target_probs = np.array(
        [
            max(eps, min(1.0 - eps, target_macro.get(n, current_probs[i])))
            for i, n in enumerate(MACRO_EMOTIONS)
        ]
    )
    # invert sigmoid(0.5 * x) → x = 2 * logit(p)
    target_logits = 2.0 * np.log(target_probs / (1.0 - target_probs))
    delta = W_pinv @ (target_logits - current_logits)
    delta = delta * s["delta_mask"]  # zero pathological axes
    return current_basis + blend * delta


def basis_to_projection(basis: np.ndarray) -> Tuple[float, float]:
    """64-D basis → 2-D map coordinates in (-1, 1)."""
    s = _ensure_loaded()
    p = np.tanh((s["W_proj"] @ basis) / 4.0)
    return float(p[0]), float(p[1])


def projection_to_basis_delta(
    current_basis: np.ndarray,
    target_x: float,
    target_y: float,
    blend: float = 0.6,
) -> np.ndarray:
    """Pseudo-inverse update for latent-map drag."""
    s = _ensure_loaded()
    target = np.clip(np.array([target_x, target_y], dtype=np.float64), -0.98, 0.98)
    target_pre = 4.0 * np.arctanh(target)
    current_pre = s["W_proj"] @ current_basis
    delta = s["W_proj_pinv"] @ (target_pre - current_pre)
    delta = delta * s["delta_mask"]  # zero pathological axes
    return current_basis + blend * delta


def rewrite_text(
    source: str,
    basis: np.ndarray,
    strength: float = 1.0,
    mode: str = "balanced",
) -> str:
    """Real activation-steered rewrite.

    Δh = (target_basis − source_basis) · W   ∈ R^{4096}
    is injected at layer ``BASIS_LAYER`` while greedy-decoding the
    chat-templated rewrite prompt.
    """
    if not source.strip():
        return source

    s = _ensure_loaded()
    target = np.asarray(basis, dtype=np.float32)
    src_basis = _get_or_compute_source_basis(source)
    delta_z = target - src_basis  # [64]
    delta_z = delta_z * s["delta_mask"]  # silence pathological axes

    W: torch.Tensor = s["W"]                                       # [64, 4096]
    delta_h = torch.from_numpy(delta_z.astype(np.float32)) @ W     # [4096]

    mode_scale = {"subtle": 0.5, "balanced": 1.0, "strong": 1.6}.get(mode, 1.0)
    alpha = float(strength) * mode_scale * float(s["steer_gain"])

    has_steer = bool(np.linalg.norm(delta_z) > 1e-4) and alpha != 0.0
    prompt = _chat_prompt(source)

    out = steered_generate(
        s["model"],
        prompt=prompt,
        vector=delta_h,
        alpha=alpha if has_steer else 0.0,
        layers=[BASIS_LAYER],
        apply_to=s["apply_to"],
        max_new_tokens=120,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
    )
    return out.strip()
