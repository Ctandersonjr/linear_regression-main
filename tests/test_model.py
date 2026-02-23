import pandas as pd

from app.model import _as_minutes, train_and_rank


def test_as_minutes_parsing() -> None:
    assert _as_minutes("34:22") == 34.0
    assert _as_minutes("18") == 18.0
    assert _as_minutes(12) == 12.0
    assert _as_minutes(None) == 0.0


def test_train_and_rank_returns_scores() -> None:
    rows = []
    for i in range(60):
        rows.append(
            {
                "player": f"Player {i}",
                "pts": float(10 + i % 8),
                "ast": float(2 + i % 4),
                "reb": float(3 + i % 5),
                "min": float(20 + i % 10),
                "next_pts": float(11 + i % 8),
            }
        )

    df = pd.DataFrame(rows)
    result = train_and_rank(df, season=2022, top_n=5)

    assert result.samples == 60
    assert len(result.top_improvers) == 5
    assert isinstance(result.r2, float)
    assert isinstance(result.mse, float)
