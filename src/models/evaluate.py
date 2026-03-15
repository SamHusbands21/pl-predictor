"""
Evaluation of model performance on the held-out test set.

Classification metrics:
  - Log loss (primary — penalises miscalibration)
  - Multi-class Brier score
  - Accuracy
  - Confusion matrix
  - Calibration reliability diagram (saved as PNG)

Profitability metrics (using historical Betfair / Bet365 odds):
  - Value bets identified when model_prob × decimal_odds > 1.05
  - Flat staking: 1 unit per bet, ROI and total P&L
  - Kelly staking: fraction = (p × odds − 1) / (odds − 1), capped at 0.25
  - Sharpe ratio of bet-level returns
  - Compared against naive strategies

All results are printed to stdout and saved to output/evaluation_report.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
)

from src.features.engineer import MODEL_FEATURES
from src.models.train import TEST_SEASONS, _load_features, _split, _ensemble_proba

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parents[2] / "output"
EV_THRESHOLD = 1.05  # bet when model_prob × odds > this
MAX_KELLY = 0.25     # cap Kelly fraction


def _brier_multiclass(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Multiclass Brier score: mean of per-class squared errors."""
    n = len(y_true)
    n_classes = proba.shape[1]
    total = 0.0
    for cls in range(n_classes):
        binary_true = (y_true == cls).astype(float)
        total += np.mean((proba[:, cls] - binary_true) ** 2)
    return total / n_classes


def _calibration_diagram(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
    save_path: Path | None = None,
) -> None:
    """Save reliability (calibration) diagram for each outcome class."""
    labels = ["Home win", "Draw", "Away win"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Calibration Reliability Diagrams (Test Set)", fontsize=13)

    for cls, (ax, label) in enumerate(zip(axes, labels)):
        binary_true = (y_true == cls).astype(float)
        p_cls = proba[:, cls]
        bins = np.linspace(0, 1, n_bins + 1)
        bin_means, bin_fracs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p_cls >= lo) & (p_cls < hi)
            if mask.sum() > 0:
                bin_means.append(p_cls[mask].mean())
                bin_fracs.append(binary_true[mask].mean())

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.plot(bin_means, bin_fracs, "o-", label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    path = save_path or OUTPUT_DIR / "calibration.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Calibration diagram saved to {path}")


def _value_bets(
    proba: np.ndarray,
    odds_home: np.ndarray,
    odds_draw: np.ndarray,
    odds_away: np.ndarray,
    y_true: np.ndarray,
    threshold: float = EV_THRESHOLD,
) -> pd.DataFrame:
    """Identify value bets and compute flat / Kelly returns per bet."""
    outcomes = ["home", "draw", "away"]
    odds_arr = np.stack([odds_home, odds_draw, odds_away], axis=1)

    rows = []
    for i in range(len(y_true)):
        for cls, outcome in enumerate(outcomes):
            p = proba[i, cls]
            o = odds_arr[i, cls]
            if np.isnan(o) or o <= 1.0:
                continue
            ev = p * o
            if ev < threshold:
                continue
            # Kelly fraction (standard formula, capped)
            kelly = min(MAX_KELLY, (p * o - 1) / (o - 1))
            won = int(y_true[i] == cls)
            flat_return = o - 1 if won else -1.0
            kelly_return = kelly * (o - 1) if won else -kelly

            rows.append({
                "match_idx": i,
                "outcome": outcome,
                "p_model": round(p, 4),
                "odds": round(o, 2),
                "ev": round(ev, 4),
                "kelly_fraction": round(kelly, 4),
                "won": won,
                "flat_return": round(flat_return, 4),
                "kelly_return": round(kelly_return, 4),
            })

    return pd.DataFrame(rows)


def _profitability_summary(bets: pd.DataFrame, label: str) -> dict:
    if len(bets) == 0:
        return {"label": label, "n_bets": 0}

    flat_returns = bets["flat_return"].values
    kelly_returns = bets["kelly_return"].values
    n = len(bets)

    flat_roi = flat_returns.sum() / n * 100
    kelly_pnl = kelly_returns.sum()
    kelly_sharpe = (
        kelly_returns.mean() / (kelly_returns.std() + 1e-9)
        * np.sqrt(n)
    )

    return {
        "label": label,
        "n_bets": n,
        "win_rate_pct": round(bets["won"].mean() * 100, 1),
        "flat_roi_pct": round(flat_roi, 2),
        "flat_total_pnl": round(flat_returns.sum(), 2),
        "kelly_total_pnl": round(kelly_pnl, 4),
        "kelly_sharpe": round(kelly_sharpe, 3),
    }


def _naive_baseline_roi(
    odds_home: np.ndarray,
    odds_draw: np.ndarray,
    odds_away: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Always bet on the favourite (shortest odds). Flat staking."""
    odds_arr = np.stack([odds_home, odds_draw, odds_away], axis=1)
    valid = ~np.any(np.isnan(odds_arr), axis=1)
    odds_v = odds_arr[valid]
    y_v = y_true[valid]
    fav_cls = np.argmin(odds_v, axis=1)  # lowest odds = favourite
    won = (fav_cls == y_v).astype(float)
    returns = np.where(won, odds_v[np.arange(len(odds_v)), fav_cls] - 1, -1.0)
    return {
        "label": "Always-favourite baseline",
        "n_bets": len(returns),
        "win_rate_pct": round(won.mean() * 100, 1),
        "flat_roi_pct": round(returns.mean() * 100, 2),
        "flat_total_pnl": round(returns.sum(), 2),
    }


def run_evaluation(
    xgb_model,
    rf_model,
    test_df: pd.DataFrame,
) -> dict:
    """Full evaluation on the test set. Returns a results dict."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_df = test_df.sort_values("date").reset_index(drop=True)
    X_test = test_df[MODEL_FEATURES].fillna(0).values
    y_test = test_df["target"].values

    proba = _ensemble_proba(xgb_model, rf_model, X_test)

    # --- Classification metrics ---
    ll = log_loss(y_test, proba)
    brier = _brier_multiclass(y_test, proba)
    acc = accuracy_score(y_test, proba.argmax(axis=1))
    cm = confusion_matrix(y_test, proba.argmax(axis=1))

    logger.info(f"\n=== Classification (test set: {len(test_df)} matches) ===")
    logger.info(f"  Log loss:    {ll:.4f}")
    logger.info(f"  Brier score: {brier:.4f}")
    logger.info(f"  Accuracy:    {acc:.4f}")
    logger.info(f"  Confusion matrix:\n{cm}")

    _calibration_diagram(y_test, proba, save_path=OUTPUT_DIR / "calibration.png")

    # --- Profitability ---
    odds_h = test_df["odds_home"].values.astype(float)
    odds_d = test_df["odds_draw"].values.astype(float)
    odds_a = test_df["odds_away"].values.astype(float)

    model_bets = _value_bets(proba, odds_h, odds_d, odds_a, y_test)
    model_summary = _profitability_summary(model_bets, "Model value bets")
    baseline_summary = _naive_baseline_roi(odds_h, odds_d, odds_a, y_test)

    logger.info(f"\n=== Profitability (EV threshold {EV_THRESHOLD}) ===")
    for k, v in model_summary.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"\n=== Baseline ===")
    for k, v in baseline_summary.items():
        logger.info(f"  {k}: {v}")

    report = {
        "test_set": {
            "n_matches": len(test_df),
            "seasons": sorted(test_df["season"].unique().tolist()),
        },
        "classification": {
            "log_loss": round(ll, 4),
            "brier_score": round(brier, 4),
            "accuracy": round(acc, 4),
            "confusion_matrix": cm.tolist(),
        },
        "profitability": {
            "ev_threshold": EV_THRESHOLD,
            "model": model_summary,
            "baseline_favourite": baseline_summary,
        },
    }

    report_path = OUTPUT_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved to {report_path}")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from src.models.train import load_models, _load_features, _split

    xgb_model, rf_model = load_models()
    df = _load_features()
    _, test_df = _split(df)
    run_evaluation(xgb_model, rf_model, test_df)
