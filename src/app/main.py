from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from app.model import build_training_data, train_and_rank
from app.nba_client import BallDontLieClient, NBAApiError
from app.schemas import ImprovementResponse, Metrics, TopImprover

app = FastAPI(title="NBA Player Improvement Predictor", version="1.0.0")
client = BallDontLieClient()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_model=ImprovementResponse)
def predict_improvement(
    season: int = Query(2022, ge=1980, le=2100),
    player_count: int = Query(200, ge=50, le=400),
    top_n: int = Query(10, ge=1, le=25),
) -> ImprovementResponse:
    try:
        dataset = build_training_data(client, season=season, player_count=player_count)
        result = train_and_rank(dataset, season=season, top_n=top_n)
        return ImprovementResponse(
            season=result.season,
            samples=result.samples,
            metrics=Metrics(r2=result.r2, mse=result.mse),
            top_improvers=[TopImprover(**item) for item in result.top_improvers], # type: ignore
        )
    except (NBAApiError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
