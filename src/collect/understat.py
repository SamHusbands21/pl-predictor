"""
Scrape xG per match from understat.com for Premier League seasons.

Scrapes directly via requests + BeautifulSoup — no understat package required.
Understat embeds match data as JSON inside <script> tags on each league page.

Coverage starts at 2014/15.
"""

import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "understat"
START_YEARS = list(range(2014, 2025))  # 2014 = 2014/15, ..., 2024 = 2024/25

TEAM_NAME_MAP = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "Leicester City": "Leicester",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Nottingham Forest": "Nott'm Forest",
    "Brighton and Hove Albion": "Brighton",
    "Leeds United": "Leeds",
}


def _fetch_season(year: int, retries: int = 3) -> list[dict]:
    url = f"https://understat.com/league/EPL/{year}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "html.parser")
            # Understat embeds data as: var datesData = JSON.parse('...');
            scripts = soup.find_all("script")
            for script in scripts:
                if not script.string or "datesData" not in script.string:
                    continue
                # Understat embeds: var datesData = JSON.parse('...')
                # The inner string uses \' for quotes and \uXXXX for unicode
                match = re.search(
                    r'datesData\s*=\s*JSON\.parse\(\'(.*?)\'\)',
                    script.string,
                    re.DOTALL,
                )
                if match:
                    raw = match.group(1)
                    # Fix escaped single quotes then let json.loads handle \uXXXX
                    raw = raw.replace("\\'", "'")
                    return json.loads(raw)
            logger.warning(f"  No datesData found for {year}")
            return []
        except Exception as exc:
            logger.warning(f"  Attempt {attempt + 1} failed for {year}: {exc}")
            time.sleep(2 ** attempt)
    return []


def _parse_matches(matches: list[dict], year: int) -> pd.DataFrame:
    rows = []
    for m in matches:
        try:
            if m.get("isResult") is False:
                continue
            home = TEAM_NAME_MAP.get(m["h"]["title"], m["h"]["title"])
            away = TEAM_NAME_MAP.get(m["a"]["title"], m["a"]["title"])
            xg_h = float(m["xG"]["h"])
            xg_a = float(m["xG"]["a"])
            rows.append({
                "date": pd.to_datetime(m["datetime"]).normalize(),
                "home_team": home,
                "away_team": away,
                "xg_home": xg_h,
                "xg_away": xg_a,
                "goals_home": int(m["goals"]["h"]),
                "goals_away": int(m["goals"]["a"]),
                "season": f"{year}/{year + 1}",
            })
        except (KeyError, TypeError, ValueError):
            continue
    return pd.DataFrame(rows)


def download_all(years: list[int] = START_YEARS, force: bool = False) -> pd.DataFrame:
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
            matches = _fetch_season(year)
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
