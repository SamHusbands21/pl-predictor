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

from src.collect.betfair import get_upcoming_epl_fixtures, get_upcoming_fixtures_fotmob
from src.collect.football_data import download_all as download_fd
from src.collect.understat import download_all as download_us
from src.features.display_names import FEATURE_DISPLAY_NAMES
from src.features.elo import EloSystem
from src.features.xg_elo import XgEloSystem
from src.features.engineer import MODEL_FEATURES, _rolling_team_stats, _h2h_win_rate
from src.features.team_names import canonical_team_name
from src.models.train import load_models, _ensemble_proba

logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "output"
EV_THRESHOLD = 1.25          # only flag bets with strong positive expected value
ALLOWED_OUTCOMES = {"home", "away"}  # draw excluded — model's draw recall is near-zero
MAX_KELLY = 0.25

OUTCOME_NAMES = ["home", "draw", "away"]
SHAP_TOP_N = 10
SCHEMA_VERSION = 2


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


def _prepare_hist_with_xg(
    hist_df: pd.DataFrame,
    xg_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the historical match table renamed to the engineer.py convention
    (home_team / away_team / date) and left-joined to understat xG.

    Extracted from the per-fixture path so the merge only runs once per pipeline
    run rather than once per fixture.
    """
    hist = hist_df.rename(columns={
        "HomeTeam": "home_team", "AwayTeam": "away_team", "Date": "date"
    }).copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
    xg_clean = xg_df[["date", "home_team", "away_team", "xg_home", "xg_away"]].copy()
    xg_clean["date"] = pd.to_datetime(xg_clean["date"]).dt.normalize()
    hist = hist.merge(xg_clean, on=["date", "home_team", "away_team"], how="left")
    hist["xg_home"] = hist.get("xg_home", pd.Series(0.0)).fillna(0.0)
    hist["xg_away"] = hist.get("xg_away", pd.Series(0.0)).fillna(0.0)
    return hist


def _last_n_results(hist: pd.DataFrame, team: str, n: int = 5) -> list[str]:
    """
    Return the most recent n results for a team as W/D/L codes (newest first),
    taken from both home and away appearances in `hist`.
    """
    home_rows = hist.loc[hist["home_team"] == team, ["date", "FTR"]].copy()
    home_rows["code"] = home_rows["FTR"].map({"H": "W", "D": "D", "A": "L"})
    away_rows = hist.loc[hist["away_team"] == team, ["date", "FTR"]].copy()
    away_rows["code"] = away_rows["FTR"].map({"H": "L", "D": "D", "A": "W"})
    combined = pd.concat(
        [home_rows[["date", "code"]], away_rows[["date", "code"]]],
        ignore_index=True,
    ).dropna(subset=["code"])
    combined = combined.sort_values("date", ascending=False)
    return combined["code"].head(n).tolist()


def _build_fixture_features(
    fixture: dict,
    elo_ratings: dict[str, float],
    xg_elo_ratings: dict[str, float],
    hist: pd.DataFrame,
    hist_with_stats: pd.DataFrame,
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


def _build_fixture_context(
    home: str,
    away: str,
    features: dict[str, float],
    hist: pd.DataFrame,
) -> dict:
    """
    Assemble the human-readable explanation payload (form + match context) shown
    under an expanded fixture tile on the website.
    """
    def team_block(prefix: str, team: str) -> dict:
        return {
            "elo":    round(float(features[f"elo_{prefix}"]), 1),
            "xg_elo": round(float(features[f"xg_elo_{prefix}"]), 1),
            "ppg_5":  round(float(features[f"{prefix}_ppg_5"]), 2),
            "ppg_10": round(float(features[f"{prefix}_ppg_10"]), 2),
            "xgf_5":  round(float(features[f"{prefix}_xgf_5"]), 2),
            "xga_5":  round(float(features[f"{prefix}_xga_5"]), 2),
            "gf_5":   round(float(features[f"{prefix}_gf_5"]), 2),
            "ga_5":   round(float(features[f"{prefix}_ga_5"]), 2),
            "last5":  _last_n_results(hist, team, 5),
        }

    return {
        "form": {
            "home": team_block("home", home),
            "away": team_block("away", away),
        },
        "match_context": {
            "rest_days": {
                "home": int(round(float(features["home_days_rest"]))),
                "away": int(round(float(features["away_days_rest"]))),
            },
            "h2h_home_winrate": round(float(features["h2h_home_win_rate"]), 3),
        },
    }


def _per_class_shap(explainer, X_row: np.ndarray, cls: int) -> tuple[np.ndarray, float]:
    """
    Return (shap_vector_for_class, base_value_for_class) for a single feature row.

    Handles both the legacy list-of-arrays format and the newer 3-D ndarray format
    returned by shap.TreeExplainer.shap_values on a multiclass XGBoost model.
    """
    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):
        vec = np.asarray(sv[cls])[0]
    else:
        sv = np.asarray(sv)
        vec = sv[0, :, cls] if sv.ndim == 3 else sv[0]

    ev = explainer.expected_value
    if hasattr(ev, "__len__"):
        base = float(np.asarray(ev).ravel()[cls])
    else:
        base = float(ev)
    return vec, base


def _shap_contributions(
    explainer,
    X_row: np.ndarray,
    features: dict[str, float],
    cls: int,
    top_n: int = SHAP_TOP_N,
) -> tuple[float, list[dict]]:
    """
    Compute top-N feature contributions for the given class from the XGBoost
    explainer, ordered by absolute SHAP value. Returns (base_value, contributions).
    """
    shap_vec, base = _per_class_shap(explainer, X_row, cls)
    pairs = []
    for i, name in enumerate(MODEL_FEATURES):
        pairs.append((name, float(features[name]), float(shap_vec[i])))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    contributions = [
        {
            "feature":    name,
            "human_name": FEATURE_DISPLAY_NAMES.get(name, name),
            "value":      round(value, 4),
            "shap":       round(shap, 4),
        }
        for name, value, shap in pairs[:top_n]
    ]
    return base, contributions


def _value_bets_for_fixture(
    model_probs: dict,
    betfair_odds: dict,
) -> list[dict]:
    """Return list of value bets for a fixture, filtered to ALLOWED_OUTCOMES."""
    bets = []
    for outcome in ["home", "draw", "away"]:
        if outcome not in ALLOWED_OUTCOMES:
            continue
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


def run_pipeline(days_ahead: int = 30) -> None:
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

    logger.info("Preparing rolling team stats (once for all fixtures)...")
    hist = _prepare_hist_with_xg(hist_df, xg_df)
    hist_with_stats = _rolling_team_stats(hist)

    logger.info("Loading trained models...")
    xgb_model, rf_model = load_models()

    logger.info("Building SHAP explainer for XGBoost component...")
    try:
        import shap
        explainer = shap.TreeExplainer(xgb_model)
    except Exception as exc:
        logger.warning(f"SHAP explainer unavailable ({exc}); explanations will be omitted.")
        explainer = None

    logger.info("Fetching upcoming fixtures from Betfair...")
    try:
        fixtures = get_upcoming_epl_fixtures(days_ahead=days_ahead)
    except Exception as exc:
        logger.warning(
            f"Betfair API unavailable ({exc}); "
            "falling back to Fotmob for fixtures (odds will be unavailable)."
        )
        fixtures = get_upcoming_fixtures_fotmob(days_ahead=days_ahead)

    if not fixtures:
        logger.warning("No upcoming fixtures found. Writing empty recommendations.")
        output = {
            "schema_version": SCHEMA_VERSION,
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
            # Defensive: even if a collector forgets to canonicalise, make sure
            # fixture["home"]/["away"] match historical naming before lookups.
            fixture = {
                **fixture,
                "home": canonical_team_name(fixture.get("home")),
                "away": canonical_team_name(fixture.get("away")),
            }
            features = _build_fixture_features(
                fixture, elo_ratings, xg_elo_ratings, hist, hist_with_stats
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

            record = {
                "home": fixture["home"],
                "away": fixture["away"],
                "date": date_str,
                "model_probs": model_probs,
                "betfair_odds": {
                    k: round(v, 2) if v else None
                    for k, v in betfair_odds.items()
                },
                "value_bets": value_bets,
            }

            # Per-fixture explanation payload (form + match context + SHAP).
            # Wrapped so any failure here doesn't block the core prediction from
            # reaching the website — we just skip the explanation for that tile.
            try:
                context = _build_fixture_context(
                    fixture["home"], fixture["away"], features, hist
                )
                record.update(context)

                if explainer is not None:
                    predicted_idx = int(np.argmax(
                        [model_probs["home"], model_probs["draw"], model_probs["away"]]
                    ))
                    base_value, contributions = _shap_contributions(
                        explainer, X, features, predicted_idx
                    )
                    record["explanation"] = {
                        "method": "shap_xgb",
                        "predicted_outcome": OUTCOME_NAMES[predicted_idx],
                        "base_value": round(base_value, 4),
                        "contributions": contributions,
                    }
            except Exception as exc:
                logger.warning(
                    f"  Explanation step failed for "
                    f"{fixture['home']} vs {fixture['away']}: {exc}"
                )

            output_fixtures.append(record)

            logger.info(
                f"  {fixture['home']} vs {fixture['away']}: "
                f"H={model_probs['home']} D={model_probs['draw']} A={model_probs['away']} "
                f"| {len(value_bets)} value bet(s)"
            )

        except Exception as exc:
            logger.warning(f"  Skipping {fixture.get('home')} vs {fixture.get('away')}: {exc}")
            continue

    output = {
        "schema_version": SCHEMA_VERSION,
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
