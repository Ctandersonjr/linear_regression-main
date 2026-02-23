from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.nba_client import BallDontLieClient, NBAApiError, Player


FEATURES = ["pts", "ast", "reb", "min"]


@dataclass
class ModelResult:
    season: int
    r2: float
    mse: float
    samples: int
    top_improvers: list[dict[str, float | str]]



def _as_minutes(value: str | float | int | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (float, int)):
        return float(value)
    if ":" in value:
        minute_str, *_ = value.split(":", 1)
        return float(minute_str)
    return float(value)


def _to_frame(averages: list[dict], players_by_id: dict[int, Player], season: int) -> pd.DataFrame:
    rows = []
    for item in averages:
        player_id = item["player_id"]
        if player_id not in players_by_id:
            continue
        rows.append(
            {
                "player_id": player_id,
                "player": players_by_id[player_id].full_name,
                "season": season,
                "pts": float(item.get("pts", 0.0)),
                "ast": float(item.get("ast", 0.0)),
                "reb": float(item.get("reb", 0.0)),
                "min": _as_minutes(item.get("min")),
            }
        )
    return pd.DataFrame(rows)


def build_training_data(client: BallDontLieClient, season: int, player_count: int) -> pd.DataFrame:
    players = client.list_players(max_players=player_count)
    players_by_id = {p.id: p for p in players}
    ids = tuple(players_by_id.keys())

    current = _to_frame(client.season_averages(season, ids), players_by_id, season)
    nxt = _to_frame(client.season_averages(season + 1, ids), players_by_id, season + 1)
    if current.empty or nxt.empty:
        raise NBAApiError("NBA API did not return enough season data to train a model.")

    merged = current.merge(
        nxt[["player_id", "pts"]].rename(columns={"pts": "next_pts"}),
        on="player_id",
        how="inner",
    )
    return merged.dropna(subset=FEATURES + ["next_pts"])


def train_and_rank(df: pd.DataFrame, season: int, top_n: int = 10) -> ModelResult:
    if len(df) < 25:
        raise ValueError("Insufficient data after preprocessing; need at least 25 rows.")

    x = df[FEATURES]
    y = df["next_pts"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)
    pred_test = model.predict(x_test)

    full_pred = model.predict(x)
    scored = df.assign(predicted_next_pts=full_pred)
    scored["predicted_improvement"] = scored["predicted_next_pts"] - scored["pts"]

    top_improvers = (
        scored.sort_values("predicted_improvement", ascending=False)
        .head(top_n)[["player", "pts", "predicted_next_pts", "predicted_improvement"]]
        .to_dict(orient="records")
    )

    return ModelResult(
        season=season,
        r2=float(r2_score(y_test, pred_test)),
        mse=float(mean_squared_error(y_test, pred_test)),
        samples=int(len(df)),
        top_improvers=top_improvers,
    )
