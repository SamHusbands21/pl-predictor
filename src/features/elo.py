"""
Standard Elo rating system updated from actual match results.

Ratings are computed in chronological order and stored as pre-match values —
i.e. the rating recorded for a match is the rating BEFORE that match was
played. This is essential to prevent data leakage.

Usage:
    from src.features.elo import EloSystem
    elo = EloSystem(k=20, home_advantage=75, initial_rating=1500)
    ratings_df = elo.fit_transform(matches_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# Expected outcome formula: E = 1 / (1 + 10^((Rb - Ra) / 400))
# We model match result as: Home=1, Draw=0.5, Away=0


@dataclass
class EloSystem:
    k: float = 20.0
    home_advantage: float = 75.0
    initial_rating: float = 1500.0
    ratings: dict[str, float] = field(default_factory=dict)

    def _get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)

    def _expected(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _update(self, team: str, delta: float) -> None:
        self.ratings[team] = self._get_rating(team) + delta

    def process_match(
        self,
        home_team: str,
        away_team: str,
        result: str,  # "H", "D", or "A"
    ) -> tuple[float, float]:
        """
        Update ratings from one match result.

        Returns (pre_match_home_elo, pre_match_away_elo) — the ratings
        before this match was factored in.
        """
        pre_home = self._get_rating(home_team)
        pre_away = self._get_rating(away_team)

        # Apply home advantage to effective home rating for expectation calc
        eff_home = pre_home + self.home_advantage
        e_home = self._expected(eff_home, pre_away)
        e_away = 1.0 - e_home

        score_home = {"H": 1.0, "D": 0.5, "A": 0.0}[result]
        score_away = 1.0 - score_home

        self._update(home_team, self.k * (score_home - e_home))
        self._update(away_team, self.k * (score_away - e_away))

        return pre_home, pre_away

    def fit_transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Process all matches chronologically and return a DataFrame of
        pre-match Elo ratings alongside the original match data.

        Expects columns: Date, HomeTeam, AwayTeam, FTR
        """
        matches = matches.sort_values("Date").reset_index(drop=True)
        self.ratings = {}

        pre_home_elos, pre_away_elos = [], []

        for _, row in matches.iterrows():
            home_elo, away_elo = self.process_match(
                row["HomeTeam"], row["AwayTeam"], row["FTR"]
            )
            pre_home_elos.append(home_elo)
            pre_away_elos.append(away_elo)

        result = matches.copy()
        result["elo_home"] = pre_home_elos
        result["elo_away"] = pre_away_elos
        result["elo_diff"] = result["elo_home"] - result["elo_away"]
        return result

    def get_current_ratings(self) -> dict[str, float]:
        """Return the current (post all-processed-matches) Elo ratings."""
        return dict(self.ratings)
