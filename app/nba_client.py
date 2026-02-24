from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import requests


class NBAApiError(RuntimeError):
    """Raised when the NBA stats API cannot be reached or returns invalid data."""


@dataclass(frozen=True)
class Player:
    id: int
    first_name: str
    last_name: str

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


class BallDontLieClient:
    """Small HTTP client for the public balldontlie API."""

    def __init__(self, base_url: str | None = None, timeout: int = 15) -> None:
        self.base_url = (base_url or os.getenv("BALLDONTLIE_BASE_URL") or "https://api.balldontlie.io/v1").rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

        api_key = os.getenv("BALLDONTLIE_API_KEY")
        if api_key:
            self.session.headers["Authorization"] = api_key

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise NBAApiError(f"Failed to call NBA API endpoint {path}: {exc}") from exc

    def list_players(self, max_players: int = 200) -> list[Player]:
        players: list[Player] = []
        cursor = 0

        while len(players) < max_players:
            payload = self._get("/players", {"per_page": 100, "cursor": cursor})
            for raw_player in payload.get("data", []):
                players.append(
                    Player(
                        id=raw_player["id"],
                        first_name=raw_player["first_name"],
                        last_name=raw_player["last_name"],
                    )
                )
                if len(players) >= max_players:
                    break

            meta = payload.get("meta", {})
            next_cursor = meta.get("next_cursor")
            if not next_cursor:
                break
            cursor = int(next_cursor)

        if not players:
            raise NBAApiError("NBA API returned zero players.")

        return players

    @lru_cache(maxsize=32)
    def season_averages(self, season: int, player_ids: tuple[int, ...]) -> list[dict[str, Any]]:
        payload = self._get(
            "/season_averages",
            {
                "season": season,
                "player_ids[]": list(player_ids),
            },
        )
        return payload.get("data", [])