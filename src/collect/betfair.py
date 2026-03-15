"""
Betfair Exchange API client.

Fetches upcoming Premier League match odds using betfairlightweight.
Requires credentials in .env:
  BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY
  BETFAIR_CERT_PATH, BETFAIR_KEY_PATH

How to obtain API access:
  1. Log into betfair.com
  2. Account → My Account → API Developer Programme
  3. Create a Delayed Data application key (free)
  4. Download SSL certs: client-2048.crt and client-2048.key
  5. Place certs in project root `certs/` directory
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Betfair event type ID for soccer
SOCCER_EVENT_TYPE_ID = "1"
# Betfair competition ID for Premier League
EPL_COMPETITION_ID = "10932509"

# Market names we want odds for
MATCH_ODDS_MARKET = "MATCH_ODDS"


def _get_client():
    """Authenticate and return a betfairlightweight trading client."""
    import betfairlightweight as bfl

    username = os.environ["BETFAIR_USERNAME"]
    password = os.environ["BETFAIR_PASSWORD"]
    app_key = os.environ["BETFAIR_APP_KEY"]
    # betfairlightweight expects a directory path and scans it for .crt/.key files
    certs_dir = str(Path(__file__).parents[2] / "certs")

    client = bfl.APIClient(
        username=username,
        password=password,
        app_key=app_key,
        certs=certs_dir,
    )
    client.login()
    logger.info("Betfair login successful")
    return client


def get_upcoming_epl_fixtures(days_ahead: int = 7) -> list[dict]:
    """
    Return upcoming EPL fixtures with best back odds from the Exchange.

    Returns a list of dicts:
      {
        "home": str,
        "away": str,
        "date": datetime (UTC),
        "betfair_odds": {"home": float, "draw": float, "away": float},
        "market_id": str,
      }
    """
    import betfairlightweight.filters as bfl_filters

    trading = _get_client()

    now = datetime.now(timezone.utc)
    from_time = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    to_time = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # List EPL match-odds markets in the next `days_ahead` days
    market_filter = bfl_filters.market_filter(
        event_type_ids=[SOCCER_EVENT_TYPE_ID],
        competition_ids=[EPL_COMPETITION_ID],
        market_countries=["GB"],
        market_type_codes=[MATCH_ODDS_MARKET],
        market_start_time=bfl_filters.time_range(from_=from_time, to=to_time),
    )

    catalogues = trading.betting.list_market_catalogue(
        filter=market_filter,
        market_projection=["RUNNER_DESCRIPTION", "EVENT", "MARKET_START_TIME"],
        max_results=50,
    )

    if not catalogues:
        logger.info("No upcoming EPL markets found.")
        return []

    market_ids = [c.market_id for c in catalogues]

    # Fetch best available back prices
    price_proj = bfl_filters.price_projection(price_data=["EX_BEST_OFFERS"])
    market_books = trading.betting.list_market_book(
        market_ids=market_ids,
        price_projection=price_proj,
    )

    id_to_book = {b.market_id: b for b in market_books}
    fixtures = []

    for cat in catalogues:
        book = id_to_book.get(cat.market_id)
        if book is None:
            continue

        runners = {r.runner_name: r for r in cat.runners}
        odds = {}
        for runner in book.runners:
            name = next(
                (k for k, v in runners.items() if v.selection_id == runner.selection_id),
                None,
            )
            if name is None:
                continue
            best_back = (
                runner.ex.available_to_back[0].price
                if runner.ex.available_to_back
                else None
            )
            if best_back:
                odds[name] = best_back

        # Runners are typically: Home team, Draw, Away team
        runner_names = list(runners.keys())
        if len(runner_names) < 3 or len(odds) < 3:
            continue

        home_name, draw_name, away_name = runner_names[0], "The Draw", runner_names[1]
        if "Draw" in runner_names:
            draw_name = "The Draw" if "The Draw" in runner_names else "Draw"
            non_draw = [r for r in runner_names if "Draw" not in r]
            home_name, away_name = non_draw[0], non_draw[1]

        try:
            fixture = {
                "home": home_name,
                "away": away_name,
                "date": cat.market_start_time,
                "betfair_odds": {
                    "home": odds.get(home_name),
                    "draw": odds.get(draw_name),
                    "away": odds.get(away_name),
                },
                "market_id": cat.market_id,
            }
            fixtures.append(fixture)
            logger.info(
                f"  {home_name} vs {away_name} | "
                f"H:{odds.get(home_name)} D:{odds.get(draw_name)} A:{odds.get(away_name)}"
            )
        except Exception as exc:
            logger.warning(f"  Skipping market {cat.market_id}: {exc}")
            continue

    trading.logout()
    return fixtures


def get_upcoming_fixtures_fotmob(days_ahead: int = 7) -> list[dict]:
    """
    Fallback fixture source using the Fotmob public API.

    Returns upcoming EPL fixtures in the same format as get_upcoming_epl_fixtures()
    but without odds (betfair_odds values are None).  Used when Betfair login
    fails due to geographic restriction from CI runners.

    Fotmob league ID 47 = Premier League.
    """
    import requests

    FOTMOB_TEAM_MAP: dict[str, str] = {
        "Man Utd":       "Man United",
        "Sheffield Utd": "Sheffield United",
    }

    try:
        resp = requests.get(
            "https://www.fotmob.com/api/leagues",
            params={"id": 47, "ccode3": "GBR"},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=20,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.error(f"Fotmob API request failed: {exc}")
        return []

    data = resp.json()
    all_matches = data.get("fixtures", {}).get("allMatches", [])

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=days_ahead)
    fixtures = []

    for m in all_matches:
        status = m.get("status", {})
        if status.get("finished") or status.get("cancelled"):
            continue
        utc_time = status.get("utcTime")
        if not utc_time:
            continue
        try:
            match_time = datetime.fromisoformat(utc_time.replace("Z", "+00:00"))
        except ValueError:
            continue
        if match_time <= now or match_time > cutoff:
            continue

        home_short = m.get("home", {}).get("shortName", "")
        away_short = m.get("away", {}).get("shortName", "")
        home = FOTMOB_TEAM_MAP.get(home_short, home_short)
        away = FOTMOB_TEAM_MAP.get(away_short, away_short)
        if not home or not away:
            continue

        fixtures.append({
            "home": home,
            "away": away,
            "date": match_time,
            "betfair_odds": {"home": None, "draw": None, "away": None},
        })
        logger.info(f"  [Fotmob] {home} vs {away} on {match_time.date()}")

    logger.info(f"Fotmob: {len(fixtures)} upcoming fixtures in next {days_ahead} days")
    return fixtures


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fixtures = get_upcoming_epl_fixtures(days_ahead=7)
    for f in fixtures:
        print(f)
