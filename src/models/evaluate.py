"""
Evaluation of model performance on the held-out test set.

Classification metrics:
  - Log loss (primary — penalises miscalibration)
  - Multi-class Brier score
  - ROC-AUC (multiclass OVR, macro-averaged)
  - Accuracy
  - Confusion matrix
  - Precision and recall per class (Home / Draw / Away)
  - Calibration reliability diagram (saved as PNG)

Profitability metrics (using historical Betfair / Bet365 odds):
  - Value bets identified when model_prob × decimal_odds > 1.05
  - Flat staking: 1 unit per bet, ROI and total P&L
  - Kelly staking: fraction = (p × odds − 1) / (odds − 1), capped at 0.25
  - Sharpe ratio of bet-level returns
  - Compared against naive strategies

Feature importance:
  - SHAP TreeExplainer on XGBoost model (saved as PNG)

All results are saved to:
  output/evaluation_report.json   — full detailed report
  output/website_evaluation.json  — simplified schema for the project paper page
  output/calibration.png
  output/shap_summary.png
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
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
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.features.engineer import MODEL_FEATURES
from src.models.train import TEST_SEASONS, _load_features, _split, _ensemble_proba

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parents[2] / "output"
EV_THRESHOLD = 1.25          # bet when model_prob × odds > this
ALLOWED_OUTCOMES = {"home", "away"}  # draw excluded — near-zero recall destroys ROI
MAX_KELLY = 0.25             # cap Kelly fraction

# Human-readable names for every model feature (used in SHAP plot labels)
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "elo_home":          "Home Elo rating",
    "elo_away":          "Away Elo rating",
    "elo_diff":          "Elo difference (Home − Away)",
    "xg_elo_home":       "Home xG-Elo rating",
    "xg_elo_away":       "Away xG-Elo rating",
    "xg_elo_diff":       "xG-Elo difference (Home − Away)",
    "home_ppg_5":        "Home pts/game (last 5)",
    "home_ppg_10":       "Home pts/game (last 10)",
    "away_ppg_5":        "Away pts/game (last 5)",
    "away_ppg_10":       "Away pts/game (last 10)",
    "home_xgf_5":        "Home xG for (last 5)",
    "home_xga_5":        "Home xG against (last 5)",
    "away_xgf_5":        "Away xG for (last 5)",
    "away_xga_5":        "Away xG against (last 5)",
    "home_gf_5":         "Home goals for (last 5)",
    "home_ga_5":         "Home goals against (last 5)",
    "away_gf_5":         "Away goals for (last 5)",
    "away_ga_5":         "Away goals against (last 5)",
    "home_days_rest":    "Home days since last match",
    "away_days_rest":    "Away days since last match",
    "h2h_home_win_rate": "H2H home win rate (last 5 meetings)",
    "home_advantage":    "Home advantage",
}

DISPLAY_NAMES = [FEATURE_DISPLAY_NAMES.get(f, f) for f in MODEL_FEATURES]


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


def _shap_summary_plot(
    xgb_model,
    X_test: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    Compute SHAP values for the XGBoost model and save a horizontal bar chart
    showing mean absolute SHAP importance averaged across all three outcome classes.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP plot. Run: pip install shap")
        return

    logger.info("Computing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    # shap_values: list of [n_samples, n_features] per class, or 3-D array
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # Shape: (n_samples, n_features, n_classes)
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))

    order = np.argsort(mean_abs)
    ordered_names = [DISPLAY_NAMES[i] for i in order]
    ordered_vals  = mean_abs[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(ordered_names))
    bars = ax.barh(y_pos, ordered_vals, color="#6cabdd", edgecolor="none", height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_names, fontsize=9)
    ax.set_xlabel("Mean |SHAP value| — averaged across Home / Draw / Away", fontsize=9)
    ax.set_title("Feature Importance (XGBoost — SHAP)", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()

    path = save_path or OUTPUT_DIR / "shap_summary.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"SHAP summary plot saved to {path}")


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


def _threshold_sweep(
    proba: np.ndarray,
    odds_h: np.ndarray,
    odds_d: np.ndarray,
    odds_a: np.ndarray,
    y_test: np.ndarray,
    thresholds: list[float] | None = None,
    outcomes: list[str] | None = None,
) -> list[dict]:
    """
    Compute flat ROI and bet count at each EV threshold.

    Parameters
    ----------
    outcomes : None = all three; or e.g. ['home', 'away'] to exclude draws.
    """
    if thresholds is None:
        thresholds = [round(1.00 + i * 0.05, 2) for i in range(1, 11)]  # 1.05 – 1.50

    # Pull every bet that has any EV > 1.0 so we can filter cheaply in the loop
    all_bets = _value_bets(proba, odds_h, odds_d, odds_a, y_test, threshold=1.0)
    if outcomes is not None:
        all_bets = all_bets[all_bets["outcome"].isin(outcomes)]

    rows = []
    for t in thresholds:
        bets = all_bets[all_bets["ev"] >= t]
        if len(bets) == 0:
            rows.append({"ev_threshold": t, "n_bets": 0,
                         "flat_roi_pct": None, "flat_pnl": None,
                         "win_rate_pct": None, "kelly_sharpe": None})
            continue
        flat_returns  = bets["flat_return"].values
        kelly_returns = bets["kelly_return"].values
        n = len(bets)
        kelly_sharpe = (
            kelly_returns.mean() / (kelly_returns.std() + 1e-9) * np.sqrt(n)
        )
        rows.append({
            "ev_threshold":  t,
            "n_bets":        n,
            "flat_roi_pct":  round(float(flat_returns.sum() / n * 100), 2),
            "flat_pnl":      round(float(flat_returns.sum()), 2),
            "win_rate_pct":  round(float(bets["won"].mean() * 100), 1),
            "kelly_sharpe":  round(float(kelly_sharpe), 3),
        })
    return rows


def _sweep_plot(
    all_rows: list[dict],
    ha_rows: list[dict],
    save_path: Path,
) -> None:
    """Two-panel plot: ROI vs threshold and bet volume vs threshold."""
    thresholds = [r["ev_threshold"] for r in all_rows]

    def _vals(rows, key):
        return [r.get(key) for r in rows]

    roi_all  = _vals(all_rows, "flat_roi_pct")
    roi_ha   = _vals(ha_rows,  "flat_roi_pct")
    n_all    = _vals(all_rows, "n_bets")
    n_ha     = _vals(ha_rows,  "n_bets")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("EV Threshold Sweep — Test Set (2022/23–2024/25)", fontsize=12,
                 fontweight="bold")

    # --- ROI panel ---
    ax1.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.plot(thresholds, roi_all, "o-", color="#6cabdd", linewidth=2,
             markersize=6, label="All outcomes")
    ax1.plot(thresholds, roi_ha,  "s-", color="#e07b54", linewidth=2,
             markersize=6, label="Home & Away only")
    ax1.set_xlabel("EV threshold", fontsize=9)
    ax1.set_ylabel("Flat ROI (%)", fontsize=9)
    ax1.set_title("ROI vs EV threshold", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(labelsize=8)

    # --- Volume panel ---
    ax2.plot(thresholds, n_all, "o-", color="#6cabdd", linewidth=2,
             markersize=6, label="All outcomes")
    ax2.plot(thresholds, n_ha,  "s-", color="#e07b54", linewidth=2,
             markersize=6, label="Home & Away only")
    ax2.set_xlabel("EV threshold", fontsize=9)
    ax2.set_ylabel("Number of bets", fontsize=9)
    ax2.set_title("Bet volume vs EV threshold", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(labelsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Threshold sweep plot saved to {save_path}")


def _pnl_curve_plot(
    bets: pd.DataFrame,
    test_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Plot cumulative flat P&L over time for the filtered strategy.
    bets rows have match_idx aligned to test_df's integer index.
    """
    if len(bets) == 0:
        logger.warning("No bets for P&L curve — skipping plot.")
        return

    # Attach date to each bet via match_idx
    idx_to_date = test_df["date"].reset_index(drop=True)
    bets = bets.copy()
    bets["date"] = bets["match_idx"].map(idx_to_date)
    bets = bets.sort_values("date").reset_index(drop=True)
    bets["cum_pnl"] = bets["flat_return"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    # Shade positive/negative regions
    x = bets.index.values
    y = bets["cum_pnl"].values
    ax.fill_between(x, y, 0, where=(y >= 0), alpha=0.15, color="#16a34a", interpolate=True)
    ax.fill_between(x, y, 0, where=(y <  0), alpha=0.15, color="#dc2626", interpolate=True)
    ax.plot(x, y, color="#6cabdd", linewidth=2)

    # Annotate final P&L
    final = round(float(y[-1]), 2)
    color = "#16a34a" if final >= 0 else "#dc2626"
    sign = "+" if final >= 0 else ""
    ax.annotate(
        f"Final: {sign}{final} units",
        xy=(x[-1], y[-1]),
        xytext=(-10, 12),
        textcoords="offset points",
        fontsize=9,
        color=color,
        fontweight="bold",
    )

    ax.set_xlabel("Bet number (chronological)", fontsize=9)
    ax.set_ylabel("Cumulative flat P&L (units)", fontsize=9)
    ax.set_title(
        f"Cumulative P&L — EV > {EV_THRESHOLD}, Home & Away only (test set 2022/23–2024/25)",
        fontsize=10,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"P&L curve saved to {save_path}")


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
    ll      = log_loss(y_test, proba)
    brier   = _brier_multiclass(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
    acc     = accuracy_score(y_test, proba.argmax(axis=1))
    cm      = confusion_matrix(y_test, proba.argmax(axis=1))

    precision, recall, _, _ = precision_recall_fscore_support(
        y_test, proba.argmax(axis=1), labels=[0, 1, 2], zero_division=0
    )

    logger.info(f"\n=== Classification (test set: {len(test_df)} matches) ===")
    logger.info(f"  Log loss:    {ll:.4f}")
    logger.info(f"  Brier score: {brier:.4f}")
    logger.info(f"  ROC-AUC:     {roc_auc:.4f}  (multiclass OVR, macro)")
    logger.info(f"  Accuracy:    {acc:.4f}")
    logger.info(f"  Precision — Home: {precision[0]:.3f}  Draw: {precision[1]:.3f}  Away: {precision[2]:.3f}")
    logger.info(f"  Recall    — Home: {recall[0]:.3f}  Draw: {recall[1]:.3f}  Away: {recall[2]:.3f}")
    logger.info(f"  Confusion matrix:\n{cm}")

    _calibration_diagram(y_test, proba, save_path=OUTPUT_DIR / "calibration.png")
    _shap_summary_plot(xgb_model, X_test, save_path=OUTPUT_DIR / "shap_summary.png")

    # --- Profitability (deployed strategy: EV > EV_THRESHOLD, ALLOWED_OUTCOMES only) ---
    odds_h = test_df["odds_home"].values.astype(float)
    odds_d = test_df["odds_draw"].values.astype(float)
    odds_a = test_df["odds_away"].values.astype(float)

    model_bets = _value_bets(proba, odds_h, odds_d, odds_a, y_test,
                             threshold=EV_THRESHOLD)
    model_bets = model_bets[model_bets["outcome"].isin(ALLOWED_OUTCOMES)]
    model_summary = _profitability_summary(model_bets, "Model value bets")
    baseline_summary = _naive_baseline_roi(odds_h, odds_d, odds_a, y_test)

    _pnl_curve_plot(model_bets, test_df, OUTPUT_DIR / "pnl_curve.png")

    logger.info(f"\n=== Profitability (EV threshold {EV_THRESHOLD}) ===")
    for k, v in model_summary.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"\n=== Baseline ===")
    for k, v in baseline_summary.items():
        logger.info(f"  {k}: {v}")

    # --- Threshold sweep ---
    sweep_all = _threshold_sweep(proba, odds_h, odds_d, odds_a, y_test)
    sweep_ha  = _threshold_sweep(proba, odds_h, odds_d, odds_a, y_test,
                                 outcomes=["home", "away"])

    logger.info("\n=== Threshold sweep — All outcomes ===")
    logger.info(f"  {'EV≥':>6}  {'Bets':>6}  {'ROI%':>8}  {'Sharpe':>8}")
    for r in sweep_all:
        roi_str    = f"{r['flat_roi_pct']:+.2f}" if r["flat_roi_pct"] is not None else "  —"
        sharpe_str = f"{r['kelly_sharpe']:+.3f}" if r["kelly_sharpe"] is not None else "  —"
        logger.info(f"  {r['ev_threshold']:>6.2f}  {r['n_bets']:>6}  {roi_str:>8}  {sharpe_str:>8}")

    logger.info("\n=== Threshold sweep — Home & Away only ===")
    logger.info(f"  {'EV≥':>6}  {'Bets':>6}  {'ROI%':>8}  {'Sharpe':>8}")
    for r in sweep_ha:
        roi_str    = f"{r['flat_roi_pct']:+.2f}" if r["flat_roi_pct"] is not None else "  —"
        sharpe_str = f"{r['kelly_sharpe']:+.3f}" if r["kelly_sharpe"] is not None else "  —"
        logger.info(f"  {r['ev_threshold']:>6.2f}  {r['n_bets']:>6}  {roi_str:>8}  {sharpe_str:>8}")

    _sweep_plot(sweep_all, sweep_ha, OUTPUT_DIR / "threshold_sweep.png")

    with open(OUTPUT_DIR / "threshold_sweep.json", "w") as f:
        json.dump({"all_outcomes": sweep_all, "home_away_only": sweep_ha}, f, indent=2)
    logger.info(f"Threshold sweep saved to {OUTPUT_DIR / 'threshold_sweep.json'}")

    report = {
        "test_set": {
            "n_matches": len(test_df),
            "seasons": sorted(test_df["season"].unique().tolist()),
        },
        "classification": {
            "log_loss": round(ll, 4),
            "brier_score": round(brier, 4),
            "roc_auc": round(roc_auc, 4),
            "accuracy": round(acc, 4),
            "precision": {
                "home": round(float(precision[0]), 4),
                "draw": round(float(precision[1]), 4),
                "away": round(float(precision[2]), 4),
            },
            "recall": {
                "home": round(float(recall[0]), 4),
                "draw": round(float(recall[1]), 4),
                "away": round(float(recall[2]), 4),
            },
            "confusion_matrix": cm.tolist(),
        },
        "profitability": {
            "ev_threshold":     EV_THRESHOLD,
            "allowed_outcomes": sorted(list(ALLOWED_OUTCOMES)),
            "model":            model_summary,
            "baseline_favourite": baseline_summary,
        },
        "threshold_sweep": {
            "all_outcomes":    sweep_all,
            "home_away_only":  sweep_ha,
        },
    }

    report_path = OUTPUT_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved to {report_path}")

    # Write simplified schema consumed by the website project paper page
    _write_website_json(report)

    return report


def _write_website_json(report: dict) -> None:
    """
    Write a simplified evaluation.json in the schema expected by betting-paper.html.
    Saved to output/website_evaluation.json; the retrain script copies it to the
    website's data/ folder.
    """
    seasons_list = report["test_set"]["seasons"]
    # Format as "2022/23–2024/25" regardless of how seasons are stored
    def _short(s: str) -> str:
        parts = s.split("/")
        return f"{parts[0]}/{parts[1][-2:]}" if len(parts) == 2 else s

    if seasons_list:
        seasons_str = f"{_short(seasons_list[0])}–{_short(seasons_list[-1])}"
    else:
        seasons_str = "unknown"

    clf = report["classification"]
    prof = report["profitability"]["model"]

    website_json = {
        "_note": "Generated by src/models/evaluate.py. Do not edit manually.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_seasons": seasons_str,
        "n_test_matches": report["test_set"]["n_matches"],
        "roc_auc": clf.get("roc_auc"),
        "brier_score": clf.get("brier_score"),
        "log_loss": clf.get("log_loss"),
        "accuracy": clf.get("accuracy"),
        "precision": clf.get("precision", {"home": None, "draw": None, "away": None}),
        "recall":    clf.get("recall",    {"home": None, "draw": None, "away": None}),
        "strategy": {
            "ev_threshold":     report["profitability"]["ev_threshold"],
            "allowed_outcomes": sorted(list(ALLOWED_OUTCOMES)),
        },
        "gambling": {
            "value_bet_threshold": report["profitability"]["ev_threshold"],
            "flat_roi_pct": prof.get("flat_roi_pct"),
            "flat_bets":    prof.get("n_bets"),
            "flat_pnl":     prof.get("flat_total_pnl"),
            "kelly_pnl":    prof.get("kelly_total_pnl"),
            "sharpe_ratio": prof.get("kelly_sharpe"),
        },
    }

    path = OUTPUT_DIR / "website_evaluation.json"
    with open(path, "w") as f:
        json.dump(website_json, f, indent=2)
    logger.info(f"Website evaluation JSON saved to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from src.models.train import load_models, _load_features, _split

    xgb_model, rf_model = load_models()
    df = _load_features()
    _, test_df = _split(df)
    run_evaluation(xgb_model, rf_model, test_df)
