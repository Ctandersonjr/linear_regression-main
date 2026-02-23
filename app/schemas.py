from __future__ import annotations

from pydantic import BaseModel, Field


class Metrics(BaseModel):
    r2: float = Field(..., description="R-squared score on the holdout test split")
    mse: float = Field(..., description="Mean squared error on the holdout test split")


class TopImprover(BaseModel):
    player: str
    pts: float
    predicted_next_pts: float
    predicted_improvement: float


class ImprovementResponse(BaseModel):
    season: int
    samples: int
    metrics: Metrics
    top_improvers: list[TopImprover]
