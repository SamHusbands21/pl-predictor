"""
Quick smoke test for Betfair API connection.

Run this after setting up your .env and placing SSL certs in certs/:
    python -m src.collect.betfair_test
"""

import logging
from src.collect.betfair import get_upcoming_epl_fixtures

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Testing Betfair API connection...")
    try:
        fixtures = get_upcoming_epl_fixtures(days_ahead=14)
        if fixtures:
            logger.info(f"\nFound {len(fixtures)} upcoming EPL fixtures:")
            for f in fixtures:
                date_str = f["date"].strftime("%d %b %Y %H:%M") if hasattr(f["date"], "strftime") else str(f["date"])
                odds = f["betfair_odds"]
                logger.info(
                    f"  {f['home']} vs {f['away']} | {date_str} | "
                    f"H:{odds['home']} D:{odds['draw']} A:{odds['away']}"
                )
        else:
            logger.info("No upcoming fixtures found (may be an international break).")
        logger.info("\nBetfair API connection successful.")
    except Exception as exc:
        logger.error(f"Connection failed: {exc}")
        raise
