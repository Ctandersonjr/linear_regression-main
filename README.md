# linear_regression

Production-ready API service that predicts which NBA players are likely to improve their points-per-game next season.

## What changed

- Replaced notebook-only workflow with a deployable FastAPI app.
- Replaced local CSV dependency with live NBA data from the **balldontlie API**.
- Added model training pipeline, error handling, health endpoint, and unit tests.

## API endpoints

- `GET /health` -> service health check.
- `GET /predict-improvement?season=2022&player_count=200&top_n=10` -> trains a linear regression model from API data and returns top projected improvers.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Optional environment variables:

- `BALLDONTLIE_API_KEY` (if your API plan requires auth)
- `BALLDONTLIE_BASE_URL` (override API base URL)

## Test

```bash
pytest -q
```
