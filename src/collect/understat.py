"""
Scrape xG per match from understat.com for Premier League seasons.

Uses the `understat` Python package (async). Saves one JSON file per season
and returns a merged DataFrame with columns:
  date, home_team, away_team, xg_home, xg_away, season

Understat coverage starts at 2014/15.
"""

import asyncio
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "understat"

# Understat seasons are identified by the start year as a string
START_YEARS = list(range(2014, 2025))  # 2014 = 2014/15, ..., 2024 = 2024/25

# Mapping from understat team names to football-data.co.uk names for merging
TEAM_NAME_MAP = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Tottenham": "Tottenham",
    "Wolverhampton Wanderers": "Wolves",
    "Leicester City": "Leicester",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    "Leeds United": "Leeds",
    "Brighton and Hove Albion": "Brighton",
    "Aston Villa": "Aston Villa",
    "Brentford": "Brentford",
    "Fulham": "Fulham",
    "Bournemouth": "Bournemouth",
    "Luton": "Luton",
    "Burnley": "Burnley",
    "Everton": "Everton",
    "Chelsea": "Chelsea",
    "Arsenal": "Arsenal",
    "Liverpool": "Liverpool",
    "Southampton": "Southampton",
    "Crystal Palace": "Crystal Palace",
    "Watford": "Watford",
    "Norwich": "Norwich",
    "Ipswich": "Ipswich",
}


async def _fetch_season(year: int) -> list[dict]:
    """Fetch all EPL matches for the season starting in `year`."""
    try:
        import aiohttp
        from understat import Understat
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            matches = await understat.get_league_results("EPL", year)
        return matches
    except Exception as exc:
        logger.error(f"  Failed to fetch understat season {year}: {exc}")
        return []


def _parse_matches(matches: list[dict], year: int) -> pd.DataFrame:
    rows = []
    for m in matches:
        try:
            rows.append({
                "date": pd.to_datetime(m["datetime"]).normalize(),
                "home_team": TEAM_NAME_MAP.get(m["h"]["title"], m["h"]["title"]),
                "away_team": TEAM_NAME_MAP.get(m["a"]["title"], m["a"]["title"]),
                "xg_home": float(m["xG"]["h"]),
                "xg_away": float(m["xG"]["a"]),
                "goals_home": int(m["goals"]["h"]),
                "goals_away": int(m["goals"]["a"]),
                "season": f"{year}/{year + 1}",
            })
        except (KeyError, TypeError, ValueError):
            continue
    return pd.DataFrame(rows)


def download_all(years: list[int] = START_YEARS, force: bool = False) -> pd.DataFrame:
    """Download all seasons from understat, cache as JSON, return DataFrame."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    for year in years:
        cache_path = RAW_DIR / f"EPL_{year}.json"
        if cache_path.exists() and not force:
            logger.info(f"  Using cached {cache_path.name}")
            with open(cache_path) as f:
                matches = json.load(f)
        else:
            logger.info(f"  Fetching understat {year}/{year + 1}...")
            matches = asyncio.run(_fetch_season(year))
            if matches:
                with open(cache_path, "w") as f:
                    json.dump(matches, f)

        if matches:
            df = _parse_matches(matches, year)
            logger.info(f"  {year}/{year + 1}: {len(df)} matches")
            frames.append(df)

    if not frames:
        raise RuntimeError("No understat data collected.")

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("date").reset_index(drop=True)


def load_cached() -> pd.DataFrame:
    """Load all cached understat JSON files."""
    frames = []
    for path in sorted(RAW_DIR.glob("EPL_*.json")):
        year = int(path.stem.split("_")[1])
        with open(path) as f:
            matches = json.load(f)
        if matches:
            frames.append(_parse_matches(matches, year))
    if not frames:
        raise FileNotFoundError(
            f"No cached understat data in {RAW_DIR}. Run download_all() first."
        )
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = download_all()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
