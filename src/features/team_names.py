"""
Canonical team-name mapping.

Different data sources spell the same club differently (Betfair returns
"Man Utd", football-data.co.uk writes "Man United", etc.). To keep Elo/xG-Elo
dictionaries, historical rolling stats, and incoming fixtures all joinable on
a single key we canonicalise every team name to the football-data.co.uk form,
since that's what the historical feature table is already built on.

Keep this list in sync whenever a new club joins the Premier League or a data
source starts using a fresh alias — missing entries silently hand out default
Elo (1500) and empty form to that team at inference time.
"""

from __future__ import annotations

# alias -> canonical (football-data.co.uk) name
TEAM_NAME_ALIASES: dict[str, str] = {
    # Betfair / Fotmob short forms
    "Man Utd":              "Man United",
    "Manchester Utd":       "Man United",
    "Manchester United":    "Man United",
    "Nottm Forest":         "Nott'm Forest",
    "Nottingham Forest":    "Nott'm Forest",
    "Nottingham":           "Nott'm Forest",
    "Sheffield Utd":        "Sheffield United",
    "Sheffield Wed":        "Sheffield Weds",
    "Spurs":                "Tottenham",
    "Tottenham Hotspur":    "Tottenham",
    "Wolverhampton":        "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham United":      "West Ham",
    "Newcastle United":     "Newcastle",
    "Leeds United":         "Leeds",
    "Leicester City":       "Leicester",
    "Norwich City":         "Norwich",
    "Swansea City":         "Swansea",
    "Cardiff City":         "Cardiff",
    "Stoke City":           "Stoke",
    "Hull City":            "Hull",
    "Huddersfield Town":    "Huddersfield",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "AFC Bournemouth":      "Bournemouth",
    "Luton Town":           "Luton",
    "Ipswich Town":         "Ipswich",
    "Birmingham City":      "Birmingham",
    "West Bromwich Albion": "West Brom",
    "Blackburn Rovers":     "Blackburn",
    "Bolton Wanderers":     "Bolton",
    "Queens Park Rangers":  "QPR",
    "Wigan Athletic":       "Wigan",
}


def canonical_team_name(name: str | None) -> str:
    """Return the canonical (football-data) form for a team name."""
    if not name:
        return name or ""
    return TEAM_NAME_ALIASES.get(name, name)
