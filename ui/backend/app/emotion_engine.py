"""Public engine surface used by the FastAPI routes.

The route handlers in ``main.py`` should never reach into ``mock_model``
directly — they only call functions defined here. To swap in a real model
later, replace the imports below (or implement these functions on top of
``src.steering.generate.steered_generate`` etc.) and the API stays stable.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from . import real_model as _backend
from .models import BASIS_DIM, label_for
from .schemas import (
    DiffSegment,
    MacroEmotions,
    Projection,
    TopBasisComponent,
)
from .utils import diff_words


# ── Conversions between schema types and numpy ─────────────────────────────


def _vec_from_list(v: List[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    if arr.shape != (BASIS_DIM,):
        raise ValueError(f"basis vector must have length {BASIS_DIM}")
    return arr


def _macro_from_schema(m: MacroEmotions) -> Dict[str, float]:
    return m.model_dump()


def _macro_to_schema(m: Dict[str, float]) -> MacroEmotions:
    return MacroEmotions(**m)


def _projection_from_xy(xy: Tuple[float, float]) -> Projection:
    return Projection(x=xy[0], y=xy[1])


def _top_components(basis: np.ndarray, k: int = 10) -> List[TopBasisComponent]:
    abs_w = np.abs(basis)
    idxs = np.argsort(abs_w)[::-1][:k]
    phrases = _backend.get_basis_phrases()
    out: List[TopBasisComponent] = []
    for i in idxs:
        i_int = int(i)
        # Prefer hand-written qualitative label if we have one, else fall
        # back to the data-derived verbal phrase.
        label = label_for(i_int)
        if label == "latent component":
            label = phrases[i_int]
        out.append(
            TopBasisComponent(
                index=i_int,
                weight=float(basis[i_int]),
                label=label,
            )
        )
    return out


# ── Public API ─────────────────────────────────────────────────────────────


def analyze(text: str) -> dict:
    """Estimate latent emotion code for a piece of text."""
    basis = _backend.text_to_basis(text)
    macro = _backend.basis_to_macro(basis)
    proj = _backend.basis_to_projection(basis)

    return {
        "source_text": text,
        "basis_vector": basis.tolist(),
        "macro_emotions": _macro_to_schema(macro),
        "projection": _projection_from_xy(proj),
        "top_basis_components": _top_components(basis),
    }


def rewrite(
    source_text: str,
    basis_vector: List[float],
    macro_emotions: MacroEmotions,
    strength: float = 1.0,
    mode: str = "balanced",
) -> dict:
    """Generate a rewritten version conditioned on the latent vector."""
    basis = _vec_from_list(basis_vector)
    rewritten = _backend.rewrite_text(
        source_text, basis, strength=strength, mode=mode,
    )
    new_macro = _backend.basis_to_macro(basis)
    proj = _backend.basis_to_projection(basis)
    diff = [
        DiffSegment(type=t, text=tx)
        for (t, tx) in diff_words(source_text, rewritten)
    ]
    return {
        "rewritten_text": rewritten,
        "basis_vector": basis.tolist(),
        "macro_emotions": _macro_to_schema(new_macro),
        "projection": _projection_from_xy(proj),
        "diff": diff,
    }


def apply_preset(
    current_basis_vector: List[float],
    preset: str,
    strength: float = 1.0,
) -> dict:
    """Mix the current basis vector with the chosen preset direction."""
    basis = _vec_from_list(current_basis_vector)
    direction = _backend.PRESET_VECTORS[preset]
    # Preset vectors are unit-norm in 64-D space, while real ICA basis
    # coefficients are typically O(1–3). Scale so one ``strength`` unit
    # of preset is comparable to the typical basis magnitude.
    preset_scale = 6.0
    new_basis = basis + preset_scale * float(strength) * direction
    macro = _backend.basis_to_macro(new_basis)
    proj = _backend.basis_to_projection(new_basis)
    return {
        "basis_vector": new_basis.tolist(),
        "macro_emotions": _macro_to_schema(macro),
        "projection": _projection_from_xy(proj),
        "top_basis_components": _top_components(new_basis),
    }


def update_from_macro(
    current_basis_vector: List[float],
    target_macro: MacroEmotions,
    blend: float = 0.5,
) -> dict:
    """Pseudo-inverse update when a user moves a macro slider."""
    basis = _vec_from_list(current_basis_vector)
    new_basis = _backend.macro_to_basis_delta(
        basis, _macro_from_schema(target_macro), blend=blend,
    )
    macro = _backend.basis_to_macro(new_basis)
    proj = _backend.basis_to_projection(new_basis)
    return {
        "basis_vector": new_basis.tolist(),
        "macro_emotions": _macro_to_schema(macro),
        "projection": _projection_from_xy(proj),
        "top_basis_components": _top_components(new_basis),
    }


def update_from_projection(
    current_basis_vector: List[float], x: float, y: float,
) -> dict:
    """Pseudo-inverse update when a user drags the latent map."""
    basis = _vec_from_list(current_basis_vector)
    new_basis = _backend.projection_to_basis_delta(basis, x, y)
    macro = _backend.basis_to_macro(new_basis)
    proj = _backend.basis_to_projection(new_basis)
    return {
        "basis_vector": new_basis.tolist(),
        "macro_emotions": _macro_to_schema(macro),
        "projection": _projection_from_xy(proj),
        "top_basis_components": _top_components(new_basis),
    }
