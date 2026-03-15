"""
Scrape xG per match from understat.com for Premier League seasons.

Scrapes directly via requests + BeautifulSoup.
Understat embeds match data as JSON inside <script> tags on each league page.

IP blocking note
----------------
GitHub Actions runners use Azure cloud IPs that understat.com blocks.
Set the SCRAPERAPI_KEY environment variable (from scraperapi.com, free tier
gives 5,000 credits/month — we use ~20/month) to route requests through
ScraperAPI's residential proxies, which bypass the block.

Without SCRAPERAPI_KEY the scraper makes direct requests (works from any
non-blocked IP, e.g. a local dev machine).

Coverage starts at 2014/15; current season always re-fetched.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "understat"

TEAM_NAME_MAP = {
    "Manchester United":       "Man United",
    "Manchester City":         "Man City",
    "Newcastle United":        "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "Leicester City":          "Leicester",
    "West Bromwich Albion":    "West Brom",
    "West Ham United":         "West Ham",
    "Nottingham Forest":       "Nott'm Forest",
    "Brighton and Hove Albion":"Brighton",
    "Leeds United":            "Leeds",
    "Sheffield United":        "Sheffield United",
    "Luton Town":              "Luton",
    "Ipswich Town":            "Ipswich",
    "Swansea City":            "Swansea",
    "Stoke City":              "Stoke",
    "Hull City":               "Hull",
    "Cardiff City":            "Cardiff",
    "Norwich City":            "Norwich",
}


def _current_year() -> int:
    """Return the start year of the currently active PL season."""
    today = date.today()
    # PL runs Aug–May; from July onward we're in the new season
    return today.year if today.month >= 7 else today.year - 1


def _all_years() -> list[int]:
    """Return start years from 2014/15 through the current season."""
    return list(range(2014, _current_year() + 1))


def _scraperapi_url(target_url: str) -> str:
    """
    Wrap a URL with ScraperAPI if SCRAPERAPI_KEY is set.
    ScraperAPI routes requests through residential proxies that bypass
    understat's Azure IP block.
    """
    api_key = os.environ.get("SCRAPERAPI_KEY", "").strip()
    if api_key:
        return f"https://api.scraperapi.com?api_key={api_key}&url={quote(target_url)}"
    return target_url


def _fetch_season(year: int, retries: int = 4) -> list[dict]:
    """Fetch raw match JSON for one understat season (year = start year)."""
    target = f"https://understat.com/league/EPL/{year}"
    url = _scraperapi_url(target)

    for attempt in range(retries):
        try:
            resp = requests.get(
                url,
                timeout=60,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "html.parser")
            for script in soup.find_all("script"):
                if not script.string or "datesData" not in script.string:
                    continue
                m = re.search(
                    r'datesData\s*=\s*JSON\.parse\(\'(.*?)\'\)',
                    script.string,
                    re.DOTALL,
                )
                if m:
                    raw = m.group(1).replace("\\'", "'")
                    return json.loads(raw)
            logger.warning(f"  No datesData found for {year} (page may be a bot-detection page)")
            return []
        except Exception as exc:
            logger.warning(f"  Attempt {attempt + 1}/{retries} failed for {year}: {exc}")
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
            rows.append({
                "date":       pd.to_datetime(m["datetime"]).normalize(),
                "home_team":  home,
                "away_team":  away,
                "xg_home":    float(m["xG"]["h"]),
                "xg_away":    float(m["xG"]["a"]),
                "goals_home": int(m["goals"]["h"]),
                "goals_away": int(m["goals"]["a"]),
                "season":     f"{year}/{year + 1}",
            })
        except (KeyError, TypeError, ValueError):
            continue
    return pd.DataFrame(rows)


def download_all(years: list[int] | None = None, force_current: bool = True) -> pd.DataFrame:
    """
    Download understat xG for all seasons.

    Historical seasons are read from cache unless the cache file is missing.
    The current season is always re-fetched to pick up new matches.

    Parameters
    ----------
    years : list[int] | None
        Start years to collect. Defaults to all seasons from 2014 to present.
    force_current : bool
        If True (default), always re-fetch the current season even if cached.
    """
    if years is None:
        years = _all_years()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    current = _current_year()
    frames: list[pd.DataFrame] = []

    for year in years:
        cache_path = RAW_DIR / f"EPL_{year}.json"
        is_current = (year == current)
        use_cache = cache_path.exists() and not (force_current and is_current)

        if use_cache:
            logger.info(f"  Using cached {cache_path.name}")
            with open(cache_path) as f:
                matches = json.load(f)
        else:
            logger.info(f"  Fetching understat {year}/{year + 1}...")
            if is_current and not os.environ.get("SCRAPERAPI_KEY"):
                logger.warning(
                    "  SCRAPERAPI_KEY not set — direct request may fail on CI. "
                    "Sign up free at scraperapi.com and add the key as a GitHub Secret."
                )
            matches = _fetch_season(year)
            if matches:
                with open(cache_path, "w") as f:
                    json.dump(matches, f)

        if matches:
            df = _parse_matches(matches, year)
            logger.info(f"    {year}/{year + 1}: {len(df)} matches")
            frames.append(df)
        else:
            logger.warning(f"    {year}/{year + 1}: no data collected")

    if not frames:
        raise RuntimeError(
            "No understat xG data available. "
            "If on CI, set SCRAPERAPI_KEY (free at scraperapi.com). "
            "Locally, make sure understat.com is accessible from your IP."
        )

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("date").reset_index(drop=True)


def load_cached() -> pd.DataFrame:
    """Load understat data from local cache (no network requests)."""
    frames: list[pd.DataFrame] = []
    for path in sorted(RAW_DIR.glob("EPL_*.json")):
        year = int(path.stem.split("_")[1])
        with open(path) as f:
            matches = json.load(f)
        if matches:
            frames.append(_parse_matches(matches, year))
    if not frames:
        raise FileNotFoundError(
            f"No cached understat data in {RAW_DIR}. "
            "Run download_all() first (requires SCRAPERAPI_KEY on CI)."
        )
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = download_all()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nSeason distribution:\n{df['season'].value_counts().sort_index()}")
