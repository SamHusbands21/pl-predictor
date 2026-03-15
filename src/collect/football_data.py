"""
Download historical Premier League match data from football-data.co.uk.

Each season CSV includes results, shots, and Betfair exchange closing odds
(columns BFEH, BFED, BFEA for home/draw/away).

Seasons available from 1993/94 onward; we collect from 2014/15 to align
with understat xG coverage.
"""

import io
import time
import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "football_data"
BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Season codes: football-data.co.uk uses YYZZ format e.g. 1415 for 2014/15
SEASONS = [
    "1415", "1516", "1617", "1718", "1819",
    "1920", "2021", "2122", "2223", "2324", "2425",
]

COLS_KEEP = [
    "Div", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",          # full-time goals + result
    "HS", "AS",                      # shots
    "HST", "AST",                    # shots on target
    "BFEH", "BFED", "BFEA",         # Betfair exchange odds (closing)
    "B365H", "B365D", "B365A",      # Bet365 as fallback
]


def _season_url(season_code: str) -> str:
    return f"{BASE_URL}/{season_code}/E0.csv"


def _download_season(season_code: str, retries: int = 3) -> pd.DataFrame:
    url = _season_url(season_code)
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text), encoding="latin-1")
            # keep only columns that exist in this season's CSV
            cols = [c for c in COLS_KEEP if c in df.columns]
            df = df[cols].copy()
            # parse date — football-data uses DD/MM/YY or DD/MM/YYYY
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
            year_start = int("20" + season_code[:2])
            df["season"] = f"{year_start}/{year_start + 1}"
            logger.info(f"  Downloaded {season_code}: {len(df)} matches")
            return df
        except Exception as exc:
            logger.warning(f"  Attempt {attempt + 1} failed for {season_code}: {exc}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download season {season_code} after {retries} retries")


def download_all(seasons: list[str] = SEASONS, force: bool = False) -> pd.DataFrame:
    """Download all seasons, cache to CSV, return combined DataFrame."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    for season_code in seasons:
        cache_path = RAW_DIR / f"E0_{season_code}.csv"
        if cache_path.exists() and not force:
            logger.info(f"  Using cached {cache_path.name}")
            df = pd.read_csv(cache_path, parse_dates=["Date"])
        else:
            logger.info(f"  Downloading season {season_code}...")
            df = _download_season(season_code)
            df.to_csv(cache_path, index=False)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)
    logger.info(f"Total matches loaded: {len(combined)}")
    return combined


def load_cached() -> pd.DataFrame:
    """Load all cached season CSVs without re-downloading."""
    frames = []
    for path in sorted(RAW_DIR.glob("E0_*.csv")):
        frames.append(pd.read_csv(path, parse_dates=["Date"]))
    if not frames:
        raise FileNotFoundError(
            f"No cached data found in {RAW_DIR}. Run download_all() first."
        )
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("Date").reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = download_all()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Seasons: {df['season'].unique()}")
    print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
