"""
Daily live pipeline entrypoint.

Steps:
  1. Download latest historical results from football-data.co.uk
  2. Download/refresh understat xG data (current season always re-fetched via ScraperAPI)
  3. Rebuild ELO / xG-ELO ratings up to today
  4. Load trained models
  5. Fetch upcoming EPL fixtures + current odds from Betfair Exchange
  6. Generate predictions and identify value bets
  7. Write output/recommendations.json

This script is called by GitHub Actions daily at 08:00 UTC.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on path when run as a script
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from src.collect.betfair import get_upcoming_epl_fixtures
from src.collect.football_data import download_all as download_fd
from src.collect.understat import download_all as download_us
from src.features.elo import EloSystem
from src.features.xg_elo import XgEloSystem
from src.features.engineer import MODEL_FEATURES, _rolling_team_stats, _h2h_win_rate
from src.models.train import load_models, _ensemble_proba

logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "output"
EV_THRESHOLD = 1.05
MAX_KELLY = 0.25


def _get_current_elo_ratings(
    hist_df: pd.DataFrame,
    xg_df: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit Elo and xG-Elo on all historical data and return current ratings."""
    elo = EloSystem(k=20, home_advantage=75)
    elo.fit_transform(hist_df)

    # Merge xG for xG-Elo
    merged = hist_df.rename(columns={
        "HomeTeam": "home_team", "AwayTeam": "away_team", "Date": "date"
    })
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    xg_clean = xg_df[["date", "home_team", "away_team", "xg_home", "xg_away"]].copy()
    xg_clean["date"] = pd.to_datetime(xg_clean["date"]).dt.normalize()
    merged = merged.merge(xg_clean, on=["date", "home_team", "away_team"], how="left")
    merged["xg_home"] = merged["xg_home"].fillna(0.0)
    merged["xg_away"] = merged["xg_away"].fillna(0.0)

    xg_elo = XgEloSystem(k=20, home_advantage=75)
    xg_elo.fit_transform(merged)

    return elo.get_current_ratings(), xg_elo.get_current_ratings()


def _build_fixture_features(
    fixture: dict,
    elo_ratings: dict[str, float],
    xg_elo_ratings: dict[str, float],
    hist_df: pd.DataFrame,
    xg_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Build model features for a single upcoming fixture.
    Rolling stats use strictly historical data (no future leakage).
    """
    home, away = fixture["home"], fixture["away"]
    default_elo = 1500.0

    elo_home = elo_ratings.get(home, default_elo)
    elo_away = elo_ratings.get(away, default_elo)
    xg_elo_home = xg_elo_ratings.get(home, default_elo)
    xg_elo_away = xg_elo_ratings.get(away, default_elo)

    # Rolling stats — reuse engineer._rolling_team_stats on historical data
    # We need the most recent rolling values for each team
    hist = hist_df.rename(columns={
        "HomeTeam": "home_team", "AwayTeam": "away_team", "Date": "date"
    }).copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
    xg_clean = xg_df[["date", "home_team", "away_team", "xg_home", "xg_away"]].copy()
    xg_clean["date"] = pd.to_datetime(xg_clean["date"]).dt.normalize()
    hist = hist.merge(xg_clean, on=["date", "home_team", "away_team"], how="left")
    hist["xg_home"] = hist.get("xg_home", pd.Series(0.0)).fillna(0.0)
    hist["xg_away"] = hist.get("xg_away", pd.Series(0.0)).fillna(0.0)

    hist_with_stats = _rolling_team_stats(hist)

    def last_stat(team_col: str, team: str, stat: str) -> float:
        rows = hist_with_stats[hist_with_stats[team_col] == team]
        if rows.empty:
            return 0.0
        return float(rows.iloc[-1].get(stat, 0.0) or 0.0)

    def days_rest(team_col: str, team: str) -> float:
        rows = hist_with_stats[hist_with_stats[team_col] == team]
        if rows.empty:
            return 7.0
        last_date = rows.iloc[-1]["date"]
        fix_date = pd.to_datetime(fixture["date"]).tz_localize(None)
        return min(float((fix_date - last_date).days), 30.0)

    # H2H win rate from recent meetings (in historical data only)
    h2h_mask = (
        ((hist["home_team"] == home) & (hist["away_team"] == away))
        | ((hist["home_team"] == away) & (hist["away_team"] == home))
    )
    h2h_matches = hist[h2h_mask].tail(5)
    if len(h2h_matches) == 0:
        h2h_rate = 0.5
    else:
        wins = 0.0
        for _, m in h2h_matches.iterrows():
            if m["home_team"] == home:
                wins += 1 if m["FTR"] == "H" else (0.5 if m["FTR"] == "D" else 0)
            else:
                wins += 1 if m["FTR"] == "A" else (0.5 if m["FTR"] == "D" else 0)
        h2h_rate = wins / len(h2h_matches)

    return {
        "elo_home": elo_home,
        "elo_away": elo_away,
        "elo_diff": elo_home - elo_away,
        "xg_elo_home": xg_elo_home,
        "xg_elo_away": xg_elo_away,
        "xg_elo_diff": xg_elo_home - xg_elo_away,
        "home_ppg_5": last_stat("home_team", home, "home_ppg_5"),
        "home_ppg_10": last_stat("home_team", home, "home_ppg_10"),
        "away_ppg_5": last_stat("away_team", away, "away_ppg_5"),
        "away_ppg_10": last_stat("away_team", away, "away_ppg_10"),
        "home_xgf_5": last_stat("home_team", home, "home_xgf_5"),
        "home_xga_5": last_stat("home_team", home, "home_xga_5"),
        "away_xgf_5": last_stat("away_team", away, "away_xgf_5"),
        "away_xga_5": last_stat("away_team", away, "away_xga_5"),
        "home_gf_5": last_stat("home_team", home, "home_gf_5"),
        "home_ga_5": last_stat("home_team", home, "home_ga_5"),
        "away_gf_5": last_stat("away_team", away, "away_gf_5"),
        "away_ga_5": last_stat("away_team", away, "away_ga_5"),
        "home_days_rest": days_rest("home_team", home),
        "away_days_rest": days_rest("away_team", away),
        "h2h_home_win_rate": h2h_rate,
        "home_advantage": 1,
    }


def _value_bets_for_fixture(
    model_probs: dict,
    betfair_odds: dict,
) -> list[dict]:
    """Return list of value bets for a fixture."""
    outcomes = ["home", "draw", "away"]
    bets = []
    for outcome in outcomes:
        p = model_probs.get(outcome, 0.0)
        o = betfair_odds.get(outcome)
        if o is None or o <= 1.0 or p <= 0:
            continue
        ev = p * o
        if ev < EV_THRESHOLD:
            continue
        kelly = min(MAX_KELLY, (p * o - 1) / (o - 1))
        bets.append({
            "outcome": outcome,
            "ev": round(ev, 4),
            "kelly_fraction": round(kelly, 4),
        })
    return sorted(bets, key=lambda x: x["ev"], reverse=True)


def run_pipeline(days_ahead: int = 7) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading historical data...")
    hist_df = download_fd()  # downloads any missing seasons, uses cache otherwise

    logger.info("Fetching understat xG data (current season refreshed)...")
    try:
        xg_df = download_us(force_current=True)
    except Exception as exc:
        logger.warning(f"Understat xG fetch failed ({exc}); xG-Elo will use zeros.")
        xg_df = pd.DataFrame(columns=["date", "home_team", "away_team", "xg_home", "xg_away"])

    logger.info("Building current Elo ratings...")
    elo_ratings, xg_elo_ratings = _get_current_elo_ratings(hist_df, xg_df)

    logger.info("Loading trained models...")
    xgb_model, rf_model = load_models()

    logger.info("Fetching upcoming fixtures from Betfair...")
    fixtures = get_upcoming_epl_fixtures(days_ahead=days_ahead)

    if not fixtures:
        logger.warning("No upcoming fixtures found. Writing empty recommendations.")
        output = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "fixtures": [],
        }
        with open(OUTPUT_DIR / "recommendations.json", "w") as f:
            json.dump(output, f, indent=2)
        return

    logger.info(f"Generating predictions for {len(fixtures)} fixtures...")
    output_fixtures = []

    for fixture in fixtures:
        try:
            features = _build_fixture_features(
                fixture, elo_ratings, xg_elo_ratings, hist_df, xg_df
            )
            X = np.array([[features[col] for col in MODEL_FEATURES]])
            proba = _ensemble_proba(xgb_model, rf_model, X)[0]

            model_probs = {
                "home": round(float(proba[0]), 4),
                "draw": round(float(proba[1]), 4),
                "away": round(float(proba[2]), 4),
            }

            betfair_odds = fixture["betfair_odds"]
            value_bets = _value_bets_for_fixture(model_probs, betfair_odds)

            date_val = fixture["date"]
            date_str = date_val.isoformat() if hasattr(date_val, "isoformat") else str(date_val)

            output_fixtures.append({
                "home": fixture["home"],
                "away": fixture["away"],
                "date": date_str,
                "model_probs": model_probs,
                "betfair_odds": {
                    k: round(v, 2) if v else None
                    for k, v in betfair_odds.items()
                },
                "value_bets": value_bets,
            })

            logger.info(
                f"  {fixture['home']} vs {fixture['away']}: "
                f"H={model_probs['home']} D={model_probs['draw']} A={model_probs['away']} "
                f"| {len(value_bets)} value bet(s)"
            )

        except Exception as exc:
            logger.warning(f"  Skipping {fixture.get('home')} vs {fixture.get('away')}: {exc}")
            continue

    output = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "fixtures": sorted(output_fixtures, key=lambda x: x["date"]),
    }

    out_path = OUTPUT_DIR / "recommendations.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nRecommendations written to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_pipeline()
