"""
xG-based Elo rating system.

Instead of using the binary/ternary match result (W/D/L), the Elo update
uses the xG share as the "score":

    score_home = xg_home / (xg_home + xg_away)

This produces a continuous update that is far less noisy than actual
results and better reflects the underlying performance of each team.

Pre-match values are stored — no leakage.

Usage:
    from src.features.xg_elo import XgEloSystem
    xg_elo = XgEloSystem(k=20, home_advantage=75)
    df = xg_elo.fit_transform(merged_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class XgEloSystem:
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
        xg_home: float,
        xg_away: float,
    ) -> tuple[float, float]:
        """
        Update ratings from one match's xG values.

        Returns (pre_match_home_xg_elo, pre_match_away_xg_elo).
        """
        pre_home = self._get_rating(home_team)
        pre_away = self._get_rating(away_team)

        total_xg = xg_home + xg_away
        if total_xg <= 0:
            # No xG info — treat as a draw
            score_home = 0.5
        else:
            score_home = xg_home / total_xg

        score_away = 1.0 - score_home

        eff_home = pre_home + self.home_advantage
        e_home = self._expected(eff_home, pre_away)
        e_away = 1.0 - e_home

        self._update(home_team, self.k * (score_home - e_home))
        self._update(away_team, self.k * (score_away - e_away))

        return pre_home, pre_away

    def fit_transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Process all matches chronologically and return pre-match xG Elo
        ratings alongside the original match data.

        Expects columns: Date (or date), HomeTeam (or home_team),
        AwayTeam (or away_team), xg_home, xg_away
        """
        df = matches.copy()

        # Normalise column names to allow either football-data or understat style
        col_map = {}
        if "Date" in df.columns and "date" not in df.columns:
            col_map["Date"] = "date"
        if "HomeTeam" in df.columns and "home_team" not in df.columns:
            col_map["HomeTeam"] = "home_team"
        if "AwayTeam" in df.columns and "away_team" not in df.columns:
            col_map["AwayTeam"] = "away_team"
        if col_map:
            df = df.rename(columns=col_map)

        df = df.sort_values("date").reset_index(drop=True)
        self.ratings = {}

        pre_home_elos, pre_away_elos = [], []

        for _, row in df.iterrows():
            home_elo, away_elo = self.process_match(
                row["home_team"],
                row["away_team"],
                float(row.get("xg_home", 0)),
                float(row.get("xg_away", 0)),
            )
            pre_home_elos.append(home_elo)
            pre_away_elos.append(away_elo)

        df["xg_elo_home"] = pre_home_elos
        df["xg_elo_away"] = pre_away_elos
        df["xg_elo_diff"] = df["xg_elo_home"] - df["xg_elo_away"]
        return df

    def get_current_ratings(self) -> dict[str, float]:
        return dict(self.ratings)
