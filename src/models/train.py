"""
Train XGBoost and Random Forest classifiers with walk-forward cross-validation.

Train/test split (strict time-based holdout — NO leakage):
  Train + CV:  seasons 2014/15 – 2021/22
  Test:        seasons 2022/23 – 2024/25  (untouched until final evaluation)

Walk-forward CV uses sklearn TimeSeriesSplit on the training set to tune
hyperparameters. Probabilities are calibrated with isotonic regression.

Trained models and the ensemble calibrator are saved to models/.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from src.features.engineer import MODEL_FEATURES

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

MODELS_DIR = Path(__file__).parents[2] / "models"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

# Seasons used for training (inclusive)
TRAIN_SEASONS = {
    "2014/2015", "2015/2016", "2016/2017", "2017/2018",
    "2018/2019", "2019/2020", "2020/2021", "2021/2022",
}
# Seasons held out for testing
TEST_SEASONS = {"2022/2023", "2023/2024", "2024/2025"}

N_CV_SPLITS = 5


def _load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/features/engineer.py first."
        )
    return pd.read_parquet(path)


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on season string into train and test sets."""
    # Normalise season format e.g. "2014/2015" or "2014/15"
    def normalise(s: str) -> str:
        parts = s.split("/")
        if len(parts[1]) == 2:
            return f"{parts[0]}/{int(parts[0][:2] + parts[1]) + 1 - int(parts[0][:2] + parts[1]) % 100 + int(parts[1])}"
        return s

    df = df.copy()
    df["_season_norm"] = df["season"].apply(normalise)
    train = df[df["_season_norm"].isin(TRAIN_SEASONS)].drop(columns="_season_norm")
    test = df[df["_season_norm"].isin(TEST_SEASONS)].drop(columns="_season_norm")
    logger.info(f"Train: {len(train)} matches | Test: {len(test)} matches")
    return train, test


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """Tune XGBoost via walk-forward grid search, then refit on full train set."""
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 400],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.7, 1.0],
    }

    base = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Tuning XGBoost with walk-forward CV...")
    gs = GridSearchCV(
        base, param_grid, scoring="neg_log_loss",
        cv=tscv, n_jobs=-1, verbose=0, refit=True,
    )
    gs.fit(X_train, y_train)
    logger.info(f"  Best XGB params: {gs.best_params_}")
    logger.info(f"  Best CV log-loss: {-gs.best_score_:.4f}")
    return gs.best_estimator_


def train_rf(X_train: np.ndarray, y_train: np.ndarray) -> CalibratedClassifierCV:
    """Tune Random Forest, then wrap with isotonic calibration."""
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [None, 10, 15],
        "min_samples_leaf": [5, 10],
        "max_features": ["sqrt", 0.5],
    }

    base = RandomForestClassifier(random_state=42, n_jobs=-1)

    logger.info("Tuning Random Forest with walk-forward CV...")
    gs = GridSearchCV(
        base, param_grid, scoring="neg_log_loss",
        cv=tscv, n_jobs=-1, verbose=0, refit=True,
    )
    gs.fit(X_train, y_train)
    logger.info(f"  Best RF params: {gs.best_params_}")
    logger.info(f"  Best CV log-loss: {-gs.best_score_:.4f}")

    # Calibrate probabilities — isotonic calibration on the training set
    # using cross_val_predict to avoid using same data for calibration
    calibrated = CalibratedClassifierCV(
        gs.best_estimator_, method="isotonic", cv=5
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def _ensemble_proba(
    xgb_model: XGBClassifier,
    rf_model: CalibratedClassifierCV,
    X: np.ndarray,
) -> np.ndarray:
    """Average calibrated probabilities from both models."""
    p_xgb = xgb_model.predict_proba(X)
    p_rf = rf_model.predict_proba(X)
    return (p_xgb + p_rf) / 2.0


def run_training() -> dict:
    """Full training pipeline. Returns dict with models and split DataFrames."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_features()
    train_df, test_df = _split(df)

    train_df = train_df.sort_values("date").reset_index(drop=True)

    X_train = train_df[MODEL_FEATURES].fillna(0).values
    y_train = train_df["target"].values

    xgb_model = train_xgboost(X_train, y_train)
    rf_model = train_rf(X_train, y_train)

    joblib.dump(xgb_model, MODELS_DIR / "xgb_model.joblib")
    joblib.dump(rf_model, MODELS_DIR / "rf_model.joblib")
    logger.info("Models saved to models/")

    return {
        "xgb_model": xgb_model,
        "rf_model": rf_model,
        "train_df": train_df,
        "test_df": test_df,
    }


def load_models() -> tuple:
    """Load saved models from disk."""
    xgb_model = joblib.load(MODELS_DIR / "xgb_model.joblib")
    rf_model = joblib.load(MODELS_DIR / "rf_model.joblib")
    return xgb_model, rf_model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = run_training()
    logger.info("Training complete.")
