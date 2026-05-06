"""FastAPI entrypoint for the Emotion DJ Booth prototype.

Run from the repo root::

    cd ui/dj_booth/backend
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn app.main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import emotion_engine
from . import real_model as _backend
from .models import BASIS_DIM, BASIS_LABELS, MACRO_EMOTIONS, PRESETS
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AxisLabels,
    MacroSliderRequest,
    MetaResponse,
    PresetRequest,
    PresetResponse,
    RewriteRequest,
    RewriteResponse,
)


app = FastAPI(
    title="Emotion DJ Booth",
    version="0.1.0",
    description=(
        "Prototype API exposing a 64-dimensional latent emotion code. "
        "All model logic is mocked but the surface mirrors what a real "
        "activation-steered LLM would expose."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/meta", response_model=MetaResponse)
def get_meta() -> MetaResponse:
    # Triggers model load on first call so we can return data-derived labels.
    phrases = _backend.get_basis_phrases()
    axis_labels = _backend.get_axis_labels()
    excluded = _backend.get_excluded_components()
    return MetaResponse(
        basis_dim=BASIS_DIM,
        macro_emotions=MACRO_EMOTIONS,
        presets=PRESETS,
        basis_labels={int(k): v for k, v in BASIS_LABELS.items()},
        basis_phrases=phrases,
        axis_labels=AxisLabels(**axis_labels),
        excluded_components=excluded,
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    return AnalyzeResponse(**emotion_engine.analyze(req.text))


@app.post("/api/rewrite", response_model=RewriteResponse)
def rewrite(req: RewriteRequest) -> RewriteResponse:
    return RewriteResponse(
        **emotion_engine.rewrite(
            source_text=req.source_text,
            basis_vector=list(req.basis_vector),
            macro_emotions=req.macro_emotions,
            strength=req.strength,
            mode=req.mode,
        )
    )


@app.post("/api/preset", response_model=PresetResponse)
def preset(req: PresetRequest) -> PresetResponse:
    if req.preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"unknown preset {req.preset!r}")
    return PresetResponse(
        **emotion_engine.apply_preset(
            current_basis_vector=list(req.current_basis_vector),
            preset=req.preset,
            strength=req.strength,
        )
    )


@app.post("/api/update_from_macro", response_model=PresetResponse)
def update_from_macro(req: MacroSliderRequest) -> PresetResponse:
    """Optional helper for macro-slider edits — see emotion_engine."""
    return PresetResponse(
        **emotion_engine.update_from_macro(
            current_basis_vector=list(req.current_basis_vector),
            target_macro=req.target_macro_emotions,
            blend=req.blend,
        )
    )


from pydantic import BaseModel
from .schemas import BasisVector


class ProjectionEditRequest(BaseModel):
    current_basis_vector: BasisVector  # type: ignore[valid-type]
    x: float
    y: float


@app.post("/api/update_from_projection", response_model=PresetResponse)
def update_from_projection(req: ProjectionEditRequest) -> PresetResponse:
    return PresetResponse(
        **emotion_engine.update_from_projection(
            current_basis_vector=list(req.current_basis_vector),
            x=req.x,
            y=req.y,
        )
    )


@app.get("/")
def root() -> dict:
    return {"name": "Emotion DJ Booth", "ok": True}
