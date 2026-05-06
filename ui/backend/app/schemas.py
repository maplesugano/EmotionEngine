"""Pydantic schemas for the Emotion DJ Booth prototype API."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, conlist


MacroEmotion = Literal[
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
]

PresetName = Literal[
    "uncertainty",
    "self_doubt",
    "analytical_detachment",
    "addressivity",
    "warmth",
    "urgency",
]

Mode = Literal["subtle", "balanced", "strong"]

# 64-dim basis vector (loose bounds — we clamp on the server side too).
BasisVector = conlist(float, min_length=64, max_length=64)


class Projection(BaseModel):
    x: float
    y: float


class TopBasisComponent(BaseModel):
    index: int
    weight: float
    label: str


class MacroEmotions(BaseModel):
    joy: float = 0.0
    trust: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    sadness: float = 0.0
    disgust: float = 0.0
    anger: float = 0.0
    anticipation: float = 0.0


# ── Request models ─────────────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)


class RewriteRequest(BaseModel):
    source_text: str
    basis_vector: BasisVector  # type: ignore[valid-type]
    macro_emotions: MacroEmotions
    strength: float = 1.0
    mode: Mode = "balanced"


class PresetRequest(BaseModel):
    current_basis_vector: BasisVector  # type: ignore[valid-type]
    preset: PresetName
    strength: float = 1.0


class MacroSliderRequest(BaseModel):
    """Optional helper: ask backend to push a basis vector toward a target
    macro-emotion profile (used when sliders move)."""
    current_basis_vector: BasisVector  # type: ignore[valid-type]
    target_macro_emotions: MacroEmotions
    blend: float = 0.5  # 0 = keep current, 1 = full target


# ── Response models ────────────────────────────────────────────────────────


class AnalyzeResponse(BaseModel):
    source_text: str
    basis_vector: List[float]
    macro_emotions: MacroEmotions
    projection: Projection
    top_basis_components: List[TopBasisComponent]


class DiffSegment(BaseModel):
    type: Literal["same", "removed", "added"]
    text: str


class RewriteResponse(BaseModel):
    rewritten_text: str
    basis_vector: List[float]
    macro_emotions: MacroEmotions
    projection: Projection
    diff: List[DiffSegment]


class PresetResponse(BaseModel):
    basis_vector: List[float]
    macro_emotions: MacroEmotions
    projection: Projection
    top_basis_components: List[TopBasisComponent]


class AxisLabelEntry(BaseModel):
    index: int
    weight: float
    phrase: str


class AxisLabels(BaseModel):
    pos_x: List[AxisLabelEntry]
    neg_x: List[AxisLabelEntry]
    pos_y: List[AxisLabelEntry]
    neg_y: List[AxisLabelEntry]


class MetaResponse(BaseModel):
    basis_dim: int
    macro_emotions: List[str]
    presets: List[str]
    basis_labels: Dict[int, str]
    basis_phrases: List[str]
    axis_labels: AxisLabels
    excluded_components: List[int] = Field(
        default_factory=list,
        description=(
            "Basis component indices flagged as pathological (repetition / "
            "dirty language) by experiments/eval_basis_pathology.py. The "
            "backend zeroes these in every steering delta; the frontend "
            "should hide or grey them out in the palette."
        ),
    )
