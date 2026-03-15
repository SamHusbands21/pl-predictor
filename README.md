# PL Predictor

Premier League match outcome prediction model using ELO ratings (result-based and xG-based), form features, and an ensemble of XGBoost and Random Forest classifiers. Value bets are identified against Betfair Exchange odds.

## Structure

```
pl-predictor/
├── data/
│   ├── raw/football_data/   # CSVs from football-data.co.uk (not committed)
│   ├── raw/understat/       # xG JSON per season (not committed)
│   └── processed/           # Feature parquet files (not committed)
├── src/
│   ├── collect/             # Data collection scripts
│   ├── features/            # ELO and feature engineering
│   ├── models/              # Training, calibration, evaluation
│   └── pipeline/            # Live daily update entrypoint
├── notebooks/               # EDA, modelling and evaluation notebooks
├── .github/workflows/       # GitHub Actions daily update
└── output/                  # recommendations.json (pushed to sam_website)
```

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your Betfair credentials:
   ```bash
   cp .env.example .env
   ```

3. Place your Betfair SSL certificates in a `certs/` directory:
   - `certs/client-2048.crt`
   - `certs/client-2048.key`
   (Download from betfair.com → Account → API Developer Programme)

## Running the pipeline

### 1. Collect historical data
```bash
python -m src.collect.football_data
python -m src.collect.understat
```

### 2. Build features
```bash
python -m src.features.engineer
```

### 3. Train models
```bash
python -m src.models.train
```

### 4. Run live update (fetches fixtures, generates recommendations.json)
```bash
python -m src.pipeline.live
```

## Data sources

- **Historical results + odds**: [football-data.co.uk](https://www.football-data.co.uk/englandm.php) — free CSVs, includes Betfair exchange odds
- **xG data**: [understat.com](https://understat.com/) via the `understat` Python package — PL from 2014/15
- **Live odds**: Betfair Exchange API via `betfairlightweight`

## Methodology

- **ELO**: Standard Elo updated after each match (K=20, home advantage offset of 75 pts)
- **xG-ELO**: Elo updated using `xG_home / (xG_home + xG_away)` as the "result" — less noisy than actual outcomes
- **Features**: ELO deltas, form (rolling PPG over 5/10 games), rolling xG metrics, rest days, H2H
- **Models**: XGBoost (`multi:softprob`) + calibrated Random Forest, averaged ensemble
- **Value bets**: Bet when `model_prob × decimal_odds > 1.05`
- **Staking**: Flat (1 unit) and Kelly criterion

## Evaluation (test set: 2022/23–2024/25)

Metrics reported:
- Log loss, multi-class Brier score, accuracy
- Calibration reliability diagram
- Flat staking ROI, Kelly P&L, Sharpe ratio
- Compared against naive baselines (always-favourite, uniform prior)
