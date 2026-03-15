"""
Full feature engineering pipeline.

Takes the merged football-data + understat xG DataFrame, computes Elo / xG-Elo
ratings, then builds all match-level features. All features are strictly
pre-match (no leakage).

Outputs a parquet file at data/processed/features.parquet with columns:

  --- identifiers ---
  date, home_team, away_team, season

  --- Elo ---
  elo_home, elo_away, elo_diff
  xg_elo_home, xg_elo_away, xg_elo_diff

  --- form (rolling, team-level) ---
  home_ppg_5, home_ppg_10       # points per game last 5/10 matches
  away_ppg_5, away_ppg_10
  home_xgf_5, home_xga_5        # rolling xG for/against last 5 matches
  away_xgf_5, away_xga_5
  home_gf_5, home_ga_5          # rolling goals for/against last 5
  away_gf_5, away_ga_5

  --- rest ---
  home_days_rest, away_days_rest

  --- head-to-head ---
  h2h_home_win_rate             # home team win rate in last 5 H2H meetings

  --- flags ---
  home_advantage                # always 1 (explicit feature for tree models)

  --- target ---
  target                        # 0=Home win, 1=Draw, 2=Away win

  --- odds (kept for evaluation only, NOT fed to model) ---
  odds_home, odds_draw, odds_away
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.elo import EloSystem
from src.features.xg_elo import XgEloSystem

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


def _merge_xg(fd_df: pd.DataFrame, us_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join understat xG onto football-data matches.

    Merges on (date, home_team, away_team). football-data uses HomeTeam /
    AwayTeam / Date; understat uses home_team / away_team / date.
    Unmatched rows get xg_home=0, xg_away=0 (handled gracefully by xG-Elo).
    """
    fd = fd_df.rename(columns={
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "Date": "date",
    })
    fd["date"] = pd.to_datetime(fd["date"]).dt.normalize()

    us = us_df[["date", "home_team", "away_team", "xg_home", "xg_away"]].copy()
    us["date"] = pd.to_datetime(us["date"]).dt.normalize()

    merged = fd.merge(us, on=["date", "home_team", "away_team"], how="left")
    merged["xg_home"] = merged["xg_home"].fillna(0.0)
    merged["xg_away"] = merged["xg_away"].fillna(0.0)
    return merged.sort_values("date").reset_index(drop=True)


def _encode_target(ftr_series: pd.Series) -> pd.Series:
    """H → 0, D → 1, A → 2"""
    return ftr_series.map({"H": 0, "D": 1, "A": 2})


def _best_odds(row: pd.Series) -> tuple[float | None, float | None, float | None]:
    """Pick best available odds: Betfair Exchange first, then Bet365 fallback."""
    def pick(bf, b365):
        return bf if pd.notna(bf) and bf > 1.0 else (b365 if pd.notna(b365) and b365 > 1.0 else None)
    return (
        pick(row.get("BFEH"), row.get("B365H")),
        pick(row.get("BFED"), row.get("B365D")),
        pick(row.get("BFEA"), row.get("B365A")),
    )


def _rolling_team_stats(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute rolling per-team features using only past matches (shift by 1).
    Adds columns for home and away teams separately.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Build a long-form table: one row per team per match
    home_rows = df[["date", "home_team", "FTR", "xg_home", "xg_away",
                    "FTHG", "FTAG"]].copy()
    home_rows.columns = ["date", "team", "result", "xgf", "xga", "gf", "ga"]
    home_rows["points"] = home_rows["result"].map({"H": 3.0, "D": 1.0, "A": 0.0})
    home_rows["is_home"] = True

    away_rows = df[["date", "away_team", "FTR", "xg_away", "xg_home",
                    "FTAG", "FTHG"]].copy()
    away_rows.columns = ["date", "team", "result", "xgf", "xga", "gf", "ga"]
    away_rows["points"] = away_rows["result"].map({"H": 0.0, "D": 1.0, "A": 3.0})
    away_rows["is_home"] = False

    long = pd.concat([home_rows, away_rows], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    def rolling_shifted(series: pd.Series, window: int) -> pd.Series:
        """Shift by 1 before rolling so we never include the current match."""
        return series.shift(1).rolling(window, min_periods=1).mean()

    team_stats = {}
    for team, grp in long.groupby("team"):
        grp = grp.sort_values("date")
        team_stats[team] = pd.DataFrame({
            "date": grp["date"].values,
            "ppg_5": rolling_shifted(grp["points"], 5).values,
            "ppg_10": rolling_shifted(grp["points"], 10).values,
            "xgf_5": rolling_shifted(grp["xgf"], 5).values,
            "xga_5": rolling_shifted(grp["xga"], 5).values,
            "gf_5": rolling_shifted(grp["gf"], 5).values,
            "ga_5": rolling_shifted(grp["ga"], 5).values,
            "last_date": grp["date"].shift(1).values,
        })

    # Map back to home/away
    def get_stat(team: str, date: pd.Timestamp, col: str) -> float:
        ts = team_stats.get(team)
        if ts is None:
            return np.nan
        row = ts[ts["date"] == date]
        return float(row[col].iloc[0]) if len(row) > 0 else np.nan

    def get_last_date(team: str, date: pd.Timestamp):
        ts = team_stats.get(team)
        if ts is None:
            return pd.NaT
        row = ts[ts["date"] == date]
        if len(row) == 0:
            return pd.NaT
        val = row["last_date"].iloc[0]
        return pd.NaT if pd.isna(val) else val

    stat_cols = ["ppg_5", "ppg_10", "xgf_5", "xga_5", "gf_5", "ga_5"]
    for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
        for col in stat_cols:
            df[f"{prefix}_{col}"] = [
                get_stat(row[team_col], row["date"], col)
                for _, row in df.iterrows()
            ]
        df[f"{prefix}_last_date"] = [
            get_last_date(row[team_col], row["date"])
            for _, row in df.iterrows()
        ]

    df["home_days_rest"] = (
        df["date"] - pd.to_datetime(df["home_last_date"])
    ).dt.days.clip(upper=30)
    df["away_days_rest"] = (
        df["date"] - pd.to_datetime(df["away_last_date"])
    ).dt.days.clip(upper=30)

    df = df.drop(columns=["home_last_date", "away_last_date"])
    return df


def _h2h_win_rate(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """
    Home team's win rate in the last `n` head-to-head meetings (strictly
    before the current match).
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    win_rates = []

    for i, row in df.iterrows():
        home, away, date = row["home_team"], row["away_team"], row["date"]
        # Matches between these two teams (in either direction) before this match
        mask = (
            (df.index < i)
            & (
                ((df["home_team"] == home) & (df["away_team"] == away))
                | ((df["home_team"] == away) & (df["away_team"] == home))
            )
        )
        past = df[mask].tail(n)
        if len(past) == 0:
            win_rates.append(0.5)  # no history → neutral
            continue

        wins = 0
        for _, m in past.iterrows():
            if m["home_team"] == home:
                wins += 1 if m["FTR"] == "H" else (0.5 if m["FTR"] == "D" else 0)
            else:
                wins += 1 if m["FTR"] == "A" else (0.5 if m["FTR"] == "D" else 0)
        win_rates.append(wins / len(past))

    return pd.Series(win_rates, index=df.index)


def build_features(
    fd_df: pd.DataFrame,
    us_df: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Main entry point. Merges data sources, computes all features, returns
    a clean feature DataFrame.
    """
    logger.info("Merging football-data and understat xG...")
    df = _merge_xg(fd_df, us_df)

    logger.info("Computing Elo ratings...")
    elo = EloSystem(k=20, home_advantage=75)
    df = elo.fit_transform(df.rename(columns={
        "home_team": "HomeTeam", "away_team": "AwayTeam", "date": "Date"
    }))
    df = df.rename(columns={"HomeTeam": "home_team", "AwayTeam": "away_team", "Date": "date"})

    logger.info("Computing xG-Elo ratings...")
    xg_elo = XgEloSystem(k=20, home_advantage=75)
    df = xg_elo.fit_transform(df)

    logger.info("Computing rolling team stats...")
    df = _rolling_team_stats(df)

    logger.info("Computing H2H win rates...")
    df["h2h_home_win_rate"] = _h2h_win_rate(df)

    # Odds
    df[["odds_home", "odds_draw", "odds_away"]] = df.apply(
        lambda r: pd.Series(_best_odds(r)), axis=1
    )

    df["home_advantage"] = 1
    df["target"] = _encode_target(df["FTR"])

    # Final feature set
    feature_cols = [
        "date", "home_team", "away_team", "season",
        "elo_home", "elo_away", "elo_diff",
        "xg_elo_home", "xg_elo_away", "xg_elo_diff",
        "home_ppg_5", "home_ppg_10",
        "away_ppg_5", "away_ppg_10",
        "home_xgf_5", "home_xga_5",
        "away_xgf_5", "away_xga_5",
        "home_gf_5", "home_ga_5",
        "away_gf_5", "away_ga_5",
        "home_days_rest", "away_days_rest",
        "h2h_home_win_rate",
        "home_advantage",
        "target",
        "odds_home", "odds_draw", "odds_away",
    ]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns (will be filled with NaN): {missing}")
        for c in missing:
            df[c] = np.nan

    out = df[feature_cols].copy()
    out = out.dropna(subset=["target"])

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        path = PROCESSED_DIR / "features.parquet"
        out.to_parquet(path, index=False)
        logger.info(f"Saved features to {path} ({len(out)} rows)")

    return out


MODEL_FEATURES = [
    "elo_home", "elo_away", "elo_diff",
    "xg_elo_home", "xg_elo_away", "xg_elo_diff",
    "home_ppg_5", "home_ppg_10",
    "away_ppg_5", "away_ppg_10",
    "home_xgf_5", "home_xga_5",
    "away_xgf_5", "away_xga_5",
    "home_gf_5", "home_ga_5",
    "away_gf_5", "away_ga_5",
    "home_days_rest", "away_days_rest",
    "h2h_home_win_rate",
    "home_advantage",
]


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from src.collect.football_data import load_cached as load_fd
    from src.collect.understat import download_all as download_us

    try:
        fd_df = load_fd()
        us_df = download_us()
    except Exception as e:
        print(f"Error: {e}")
        print("Run src/collect/football_data.py first; set SCRAPERAPI_KEY for understat on CI.")
        sys.exit(1)

    features = build_features(fd_df, us_df)
    print(features.head())
    print(f"\nShape: {features.shape}")
    print(f"\nTarget distribution:\n{features['target'].value_counts()}")
