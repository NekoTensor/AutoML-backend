"""
Phase 1 — Data Understanding & Augmentation
--------------------------------------------
Takes a raw CSV path, figures out basic stats, and if the target
class is imbalanced (classification only) generates synthetic samples
using a lightweight from-scratch SMOTE-style interpolation so we don't
need an extra `imbalanced-learn` dependency.

Every phase in this project follows the same contract:

    def run(..., progress=None) -> dict

`progress` is an optional callable: progress(event: dict) -> None
It is called zero or more times during the phase to report live
status back to the orchestrator (which forwards it to the websocket).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


def _emit(progress, **kwargs):
    if progress:
        progress(kwargs)


def _smote_like_oversample(X: np.ndarray, y: np.ndarray, minority_label, n_samples: int, k=5) -> tuple[np.ndarray, np.ndarray]:
    """Minimal SMOTE re-implementation: interpolate between each minority
    sample and one of its k nearest minority neighbours."""
    minority_idx = np.where(y == minority_label)[0]
    if len(minority_idx) < 2:
        # Not enough points to interpolate — just duplicate with jitter
        base = X[minority_idx]
        synth = base[np.random.randint(0, len(base), n_samples)]
        synth += np.random.normal(0, 0.01, synth.shape)
        return synth, np.full(n_samples, minority_label)

    minority_X = X[minority_idx]
    k = min(k, len(minority_idx) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(minority_X)
    _, neighbors = nn.kneighbors(minority_X)

    synth = np.zeros((n_samples, X.shape[1]))
    for i in range(n_samples):
        base_i = np.random.randint(0, len(minority_X))
        neighbor_i = neighbors[base_i][np.random.randint(1, k + 1)]
        gap = np.random.rand()
        synth[i] = minority_X[base_i] + gap * (minority_X[neighbor_i] - minority_X[base_i])

    return synth, np.full(n_samples, minority_label)


def run(dataset_path: str, task_type: str, target_col: str, progress=None) -> dict:
    _emit(progress, phase="phase1", status="running", message="Checking dataset...")

    df = pd.read_csv(dataset_path)
    n_rows, n_cols_total = df.shape
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    feature_cols = [c for c in df.columns if c != target_col]

    # Drop rows with a missing target; impute missing features with column median/mode
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    for c in feature_cols:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "missing")

    # Encode categorical features
    encoders: dict[str, LabelEncoder] = {}
    for c in feature_cols:
        if df[c].dtype.kind not in "biufc":
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            encoders[c] = le

    class_balance = None
    was_augmented = False
    synthetic_count = 0

    if task_type == "classification":
        target_encoder = LabelEncoder()
        df[target_col] = target_encoder.fit_transform(df[target_col].astype(str))
        encoders[target_col] = target_encoder

        counts = df[target_col].value_counts().sort_index()
        total = counts.sum()
        class_balance = {str(k): round(100 * v / total, 1) for k, v in counts.items()}

        majority_count = counts.max()
        minority_label = counts.idxmin()
        minority_count = counts.min()
        imbalance_ratio = minority_count / majority_count

        _emit(
            progress,
            phase="phase1",
            status="stats",
            rows=n_rows,
            features=len(feature_cols),
            target=target_col,
            class_balance=class_balance,
        )

        # Threshold: anything worse than a 65/35 split gets topped up
        if imbalance_ratio < 0.54 and n_rows < 20000:
            n_needed = int(majority_count - minority_count)
            _emit(
                progress,
                phase="phase1",
                status="augmenting",
                message=f"Insufficient class balance detected. Generating {n_needed} synthetic samples...",
                n_samples=n_needed,
            )

            X = df[feature_cols].values.astype(float)
            y = df[target_col].values
            synth_X, synth_y = _smote_like_oversample(X, y, minority_label, n_needed)

            synth_df = pd.DataFrame(synth_X, columns=feature_cols)
            synth_df[target_col] = synth_y
            df = pd.concat([df, synth_df], ignore_index=True)
            was_augmented = True
            synthetic_count = n_needed
    else:
        # Regression: no class balance concept, just report row/feature counts
        _emit(
            progress,
            phase="phase1",
            status="stats",
            rows=n_rows,
            features=len(feature_cols),
            target=target_col,
            class_balance=None,
        )

    _emit(
        progress,
        phase="phase1",
        status="done",
        message="Dataset ready.",
        final_rows=len(df),
        synthetic_added=synthetic_count,
    )

    return {
        "df": df,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "task_type": task_type,
        "encoders": encoders,
        "class_balance": class_balance,
        "was_augmented": was_augmented,
        "synthetic_added": synthetic_count,
        "original_rows": n_rows,
    }
