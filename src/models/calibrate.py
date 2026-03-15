"""
Probability calibration utilities.

XGBoost's softprob output is generally reasonably calibrated, but we
offer a function to re-calibrate any model's probabilities on a held-out
calibration fold using isotonic regression.

The main ensemble averaging in train.py handles most calibration needs.
This module provides standalone helpers for post-hoc calibration checks.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def calibrate_proba_isotonic(
    proba: np.ndarray,
    y_true: np.ndarray,
    proba_val: np.ndarray,
) -> np.ndarray:
    """
    Fit per-class isotonic calibration on (proba, y_true) and apply to
    proba_val.

    proba:     shape (n_train, 3) — predicted probabilities on cal set
    y_true:    shape (n_train,)   — true class labels (0, 1, 2)
    proba_val: shape (n_val, 3)   — probabilities to calibrate

    Returns calibrated probabilities of shape (n_val, 3), row-normalised.
    """
    n_classes = proba.shape[1]
    calibrated = np.zeros_like(proba_val)

    for cls in range(n_classes):
        binary_y = (y_true == cls).astype(float)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(proba[:, cls], binary_y)
        calibrated[:, cls] = iso.predict(proba_val[:, cls])

    # Renormalise so probabilities sum to 1
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return calibrated / row_sums
