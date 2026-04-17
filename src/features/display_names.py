"""
Human-readable names for model features.

Shared between `src/models/evaluate.py` (global SHAP plot) and
`src/pipeline/live.py` (per-fixture SHAP contributions in recommendations.json).
"""

from __future__ import annotations

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "elo_home":          "Home Elo rating",
    "elo_away":          "Away Elo rating",
    "elo_diff":          "Elo difference (Home − Away)",
    "xg_elo_home":       "Home xG-Elo rating",
    "xg_elo_away":       "Away xG-Elo rating",
    "xg_elo_diff":       "xG-Elo difference (Home − Away)",
    "home_ppg_5":        "Home pts/game (last 5)",
    "home_ppg_10":       "Home pts/game (last 10)",
    "away_ppg_5":        "Away pts/game (last 5)",
    "away_ppg_10":       "Away pts/game (last 10)",
    "home_xgf_5":        "Home xG for (last 5)",
    "home_xga_5":        "Home xG against (last 5)",
    "away_xgf_5":        "Away xG for (last 5)",
    "away_xga_5":        "Away xG against (last 5)",
    "home_gf_5":         "Home goals for (last 5)",
    "home_ga_5":         "Home goals against (last 5)",
    "away_gf_5":         "Away goals for (last 5)",
    "away_ga_5":         "Away goals against (last 5)",
    "home_days_rest":    "Home days since last match",
    "away_days_rest":    "Away days since last match",
    "h2h_home_win_rate": "H2H home win rate (last 5 meetings)",
    "home_advantage":    "Home advantage",
}


def display_name(feature: str) -> str:
    """Return the human-readable name for a feature, falling back to the raw key."""
    return FEATURE_DISPLAY_NAMES.get(feature, feature)
