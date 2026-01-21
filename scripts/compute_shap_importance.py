#!/usr/bin/env python3
"""
Compute global feature importance via SHAP for daily CSV datasets.

Workflow (per date):
  1) Load feature CSV and target CSV
  2) Filter if samples < min_samples
  3) Select subset of features (by name -> indices)
  4) Train RF regressor on train split (optionally with scaling)
  5) Compute SHAP values on test split
Aggregate across all valid dates:
  - Global importance = mean(abs(SHAP)) per feature
  - Save summary figure (SHAP beeswarm + bar plot) and CSV ranking

Expected input format:
  - feature CSV: headerless numeric matrix (rows=samples, cols=features)
  - target  CSV: headerless vector or single-column matrix (rows=samples)

Example:
  python scripts/compute_shap_importance.py \
    --feature-pattern "/data/GK2A/rain_case/MID-LAT/heavy_mid-lat_l1b_l2_feature_{date}.csv" \
    --target-pattern  "/data/GK2A/rain_case/MID-LAT/heavy_mid-lat_target_{date}.csv" \
    --out-dir "./shap_results" \
    --year 2023 --time-suffix "0300" \
    --top-n 20 --min-samples 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import date as Date
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# -----------------------------
# Feature definitions
# -----------------------------

FEATURE_NAMES_TOTAL: List[str] = [
    'R$_{0.47}$', 'R$_{0.51}$', 'R$_{0.64}$', 'R$_{0.86}$', 'R$_{1.37}$', 'R$_{1.61}$',
    'BT$_{3.8}$', 'BT$_{6.2}$', 'BT$_{6.9}$', 'BT$_{7.3}$', 'BT$_{8.6}$', 'BT$_{9.6}$',
    'BT$_{10.4}$', 'BT$_{11.2}$', 'BT$_{12.4}$', 'BT$_{13.3}$',

    'R$_{0.47}$ - R$_{0.51}$', 'R$_{0.47}$ - R$_{0.64}$', 'R$_{0.47}$ - R$_{0.86}$',
    'R$_{0.47}$ - R$_{1.37}$', 'R$_{0.47}$ - R$_{1.61}$',
    'R$_{0.51}$ - R$_{0.64}$', 'R$_{0.51}$ - R$_{0.86}$', 'R$_{0.51}$ - R$_{1.37}$',
    'R$_{0.51}$ - R$_{1.61}$',
    'R$_{0.64}$ - R$_{0.86}$', 'R$_{0.64}$ - R$_{1.37}$', 'R$_{0.64}$ - R$_{1.61}$',
    'R$_{0.86}$ - R$_{1.37}$', 'R$_{0.86}$ - R$_{1.61}$', 'R$_{1.37}$ - R$_{1.61}$',

    'BT$_{3.8}$ - BT$_{6.2}$', 'BT$_{3.8}$ - BT$_{6.9}$', 'BT$_{3.8}$ - BT$_{7.3}$',
    'BT$_{3.8}$ - BT$_{8.6}$', 'BT$_{3.8}$ - BT$_{9.6}$', 'BT$_{3.8}$ - BT$_{10.4}$',
    'BT$_{3.8}$ - BT$_{11.2}$', 'BT$_{3.8}$ - BT$_{12.4}$', 'BT$_{3.8}$ - BT$_{13.3}$',
    'BT$_{6.2}$ - BT$_{6.9}$', 'BT$_{6.2}$ - BT$_{7.3}$', 'BT$_{6.2}$ - BT$_{8.6}$',
    'BT$_{6.2}$ - BT$_{9.6}$', 'BT$_{6.2}$ - BT$_{10.4}$', 'BT$_{6.2}$ - BT$_{11.2}$',
    'BT$_{6.2}$ - BT$_{12.4}$', 'BT$_{6.2}$ - BT$_{13.3}$',
    'BT$_{6.9}$ - BT$_{7.3}$', 'BT$_{6.9}$ - BT$_{8.6}$', 'BT$_{6.9}$ - BT$_{9.6}$',
    'BT$_{6.9}$ - BT$_{10.4}$', 'BT$_{6.9}$ - BT$_{11.2}$', 'BT$_{6.9}$ - BT$_{12.4}$',
    'BT$_{6.9}$ - BT$_{13.3}$',
    'BT$_{7.3}$ - BT$_{8.6}$', 'BT$_{7.3}$ - BT$_{9.6}$', 'BT$_{7.3}$ - BT$_{10.4}$',
    'BT$_{7.3}$ - BT$_{11.2}$', 'BT$_{7.3}$ - BT$_{12.4}$', 'BT$_{7.3}$ - BT$_{13.3}$',
    'BT$_{8.6}$ - BT$_{9.6}$', 'BT$_{8.6}$ - BT$_{10.4}$', 'BT$_{8.6}$ - BT$_{11.2}$',
    'BT$_{8.6}$ - BT$_{12.4}$', 'BT$_{8.6}$ - BT$_{13.3}$',
    'BT$_{9.6}$ - BT$_{10.4}$', 'BT$_{9.6}$ - BT$_{11.2}$', 'BT$_{9.6}$ - BT$_{12.4}$',
    'BT$_{9.6}$ - BT$_{13.3}$',
    'BT$_{10.4}$ - BT$_{11.2}$', 'BT$_{10.4}$ - BT$_{12.4}$', 'BT$_{10.4}$ - BT$_{13.3}$',
    'BT$_{11.2}$ - BT$_{12.4}$', 'BT$_{11.2}$ - BT$_{13.3}$', 'BT$_{12.4}$ - BT$_{13.3}$',

    'CTH', 'CTT', 'CTP', 'COT', 'CER', 'IWP', 'LWP'
]

# Your selected features (subset used for training/SHAP)
DEFAULT_SELECTED_FEATURES: List[str] = [
    'R$_{0.47}$', 'R$_{0.51}$', 'R$_{0.64}$', 'R$_{0.86}$', 'R$_{1.37}$', 'R$_{1.61}$',
    'BT$_{3.8}$', 'BT$_{6.2}$', 'BT$_{6.9}$', 'BT$_{7.3}$', 'BT$_{8.6}$', 'BT$_{9.6}$',
    'BT$_{10.4}$', 'BT$_{11.2}$', 'BT$_{12.4}$', 'BT$_{13.3}$',

    'R$_{0.47}$ - R$_{0.51}$', 'R$_{0.47}$ - R$_{0.64}$', 'R$_{0.47}$ - R$_{0.86}$',
    'R$_{0.47}$ - R$_{1.37}$', 'R$_{0.47}$ - R$_{1.61}$',
    'R$_{0.51}$ - R$_{0.64}$', 'R$_{0.51}$ - R$_{0.86}$', 'R$_{0.51}$ - R$_{1.37}$',
    'R$_{0.51}$ - R$_{1.61}$',
    'R$_{0.64}$ - R$_{0.86}$', 'R$_{0.64}$ - R$_{1.37}$', 'R$_{0.64}$ - R$_{1.61}$',
    'R$_{0.86}$ - R$_{1.37}$', 'R$_{0.86}$ - R$_{1.61}$', 'R$_{1.37}$ - R$_{1.61}$',

    'BT$_{3.8}$ - BT$_{6.2}$', 'BT$_{3.8}$ - BT$_{6.9}$', 'BT$_{3.8}$ - BT$_{7.3}$',
    'BT$_{3.8}$ - BT$_{8.6}$', 'BT$_{3.8}$ - BT$_{9.6}$', 'BT$_{3.8}$ - BT$_{10.4}$',
    'BT$_{3.8}$ - BT$_{11.2}$', 'BT$_{3.8}$ - BT$_{12.4}$', 'BT$_{3.8}$ - BT$_{13.3}$',
    'BT$_{6.2}$ - BT$_{6.9}$', 'BT$_{6.2}$ - BT$_{7.3}$', 'BT$_{6.2}$ - BT$_{8.6}$',
    'BT$_{6.2}$ - BT$_{9.6}$', 'BT$_{6.2}$ - BT$_{10.4}$', 'BT$_{6.2}$ - BT$_{11.2}$',
    'BT$_{6.2}$ - BT$_{12.4}$', 'BT$_{6.2}$ - BT$_{13.3}$',
    'BT$_{6.9}$ - BT$_{7.3}$', 'BT$_{6.9}$ - BT$_{8.6}$', 'BT$_{6.9}$ - BT$_{9.6}$',
    'BT$_{6.9}$ - BT$_{10.4}$', 'BT$_{6.9}$ - BT$_{11.2}$', 'BT$_{6.9}$ - BT$_{12.4}$',
    'BT$_{6.9}$ - BT$_{13.3}$',
    'BT$_{7.3}$ - BT$_{8.6}$', 'BT$_{7.3}$ - BT$_{9.6}$', 'BT$_{7.3}$ - BT$_{10.4}$',
    'BT$_{7.3}$ - BT$_{11.2}$', 'BT$_{7.3}$ - BT$_{12.4}$', 'BT$_{7.3}$ - BT$_{13.3}$',
    'BT$_{8.6}$ - BT$_{9.6}$', 'BT$_{8.6}$ - BT$_{10.4}$', 'BT$_{8.6}$ - BT$_{11.2}$',
    'BT$_{8.6}$ - BT$_{12.4}$', 'BT$_{8.6}$ - BT$_{13.3}$',
    'BT$_{9.6}$ - BT$_{10.4}$', 'BT$_{9.6}$ - BT$_{11.2}$', 'BT$_{9.6}$ - BT$_{12.4}$',
    'BT$_{9.6}$ - BT$_{13.3}$',
    'BT$_{10.4}$ - BT$_{11.2}$', 'BT$_{10.4}$ - BT$_{12.4}$', 'BT$_{10.4}$ - BT$_{13.3}$',
    'BT$_{11.2}$ - BT$_{12.4}$', 'BT$_{11.2}$ - BT$_{13.3}$', 'BT$_{12.4}$ - BT$_{13.3}$',

    'CTT', 'COT', 'CER', 'IWP'
]


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class Config:
    feature_pattern: str
    target_pattern: str
    out_dir: str
    year: int
    time_suffix: str
    top_n: int
    min_samples: int
    test_size: float
    seed: int
    n_estimators: int
    n_jobs: int
    use_scaler: bool
    max_rows_per_day: Optional[int]  # None = use all rows
    selected_features: List[str]


# -----------------------------
# Utilities
# -----------------------------

def setup_logging(verbosity: int) -> None:
    level = logging.INFO if verbosity <= 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def generate_date_strings(year: int, time_suffix: str) -> List[str]:
    """Generate YYYYMMDD{time_suffix} for all days in a given year."""
    dates: List[str] = []
    d = Date(year, 1, 1)
    while d.year == year:
        dates.append(f"{d.year:04d}{d.month:02d}{d.day:02d}{time_suffix}")
        d = Date.fromordinal(d.toordinal() + 1)
    return dates


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def validate_selected_features(selected: List[str], total: List[str]) -> Tuple[List[int], List[str]]:
    """Return indices and a cleaned feature list; raise if any missing."""
    missing = [f for f in selected if f not in total]
    if missing:
        raise ValueError(
            "Some selected_features are not found in FEATURE_NAMES_TOTAL:\n"
            + "\n".join(missing)
        )
    indices = [total.index(f) for f in selected]
    return indices, selected


def load_day_data(feature_file: str, target_file: str, max_rows: Optional[int]) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load a single day of features and target. Assumes headerless CSV."""
    X = pd.read_csv(feature_file, header=None)
    y = pd.read_csv(target_file, header=None).values.ravel()

    if max_rows is not None and len(X) > max_rows:
        # deterministic subset (top rows) for reproducibility
        X = X.iloc[:max_rows, :].reset_index(drop=True)
        y = y[:max_rows]

    if len(X) != len(y):
        raise ValueError(f"Row mismatch: X has {len(X)} rows but y has {len(y)} rows.")

    return X, y


def train_and_shap(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    test_size: float,
    n_estimators: int,
    n_jobs: int,
    use_scaler: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train RF on train split and compute SHAP on test split.
    Returns (shap_values, X_test_used) where X_test_used is the matrix passed to SHAP.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    if use_scaler:
        scaler = StandardScaler()
        X_train_used = scaler.fit_transform(X_train)
        X_test_used = scaler.transform(X_test)
    else:
        X_train_used = X_train.values
        X_test_used = X_test.values

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(X_train_used, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_used)

    return shap_values, X_test_used


def make_summary_figure(
    top_shap_values: np.ndarray,
    top_X_values: np.ndarray,
    top_features: List[str],
    top_mean_abs_shap: np.ndarray,
    out_png: str,
) -> None:
    """
    Save a combined figure:
      - left: SHAP summary (beeswarm)
      - right: mean(|SHAP|) horizontal bar
    """
    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    gs.update(wspace=0.6)

    # Left: SHAP summary plot
    ax0 = plt.subplot(gs[0])
    shap.summary_plot(
        top_shap_values,
        top_X_values,
        feature_names=top_features,
        show=False,
        cmap="bwr",
    )
    plt.sca(ax0)

    # Right: bar plot
    ax1 = plt.subplot(gs[1])
    y_pos = np.arange(len(top_features))
    ax1.barh(y_pos, top_mean_abs_shap, align="center")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_features)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean(|SHAP value|)", fontsize=14)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute SHAP global importance for daily datasets.")
    p.add_argument("--feature-pattern", type=str, required=True,
                   help='Path pattern for feature CSV with "{date}" placeholder.')
    p.add_argument("--target-pattern", type=str, required=True,
                   help='Path pattern for target CSV with "{date}" placeholder.')
    p.add_argument("--out-dir", type=str, default="./shap_results")

    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--time-suffix", type=str, default="0300",
                   help="Appended to YYYYMMDD to form {date} string (e.g., 0300).")

    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--min-samples", type=int, default=100)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--use-scaler", action="store_true",
                   help="Apply StandardScaler before training/SHAP (optional).")

    p.add_argument("--max-rows-per-day", type=int, default=None,
                   help="If set, use only the first N rows per day (for speed/repro).")

    p.add_argument("--selected-features-json", type=str, default=None,
                   help="Optional JSON file containing a list of selected feature names.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    ensure_dir(args.out_dir)

    # Load selected features if provided
    if args.selected_features_json is not None:
        with open(args.selected_features_json, "r", encoding="utf-8") as f:
            selected_features = json.load(f)
        if not isinstance(selected_features, list) or not all(isinstance(x, str) for x in selected_features):
            raise ValueError("--selected-features-json must contain a JSON list of strings.")
    else:
        selected_features = DEFAULT_SELECTED_FEATURES

    cfg = Config(
        feature_pattern=args.feature_pattern,
        target_pattern=args.target_pattern,
        out_dir=args.out_dir,
        year=args.year,
        time_suffix=args.time_suffix,
        top_n=args.top_n,
        min_samples=args.min_samples,
        test_size=args.test_size,
        seed=args.seed,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        use_scaler=bool(args.use_scaler),
        max_rows_per_day=args.max_rows_per_day,
        selected_features=selected_features,
    )

    selected_indices, selected_features_clean = validate_selected_features(
        cfg.selected_features, FEATURE_NAMES_TOTAL
    )

    date_list = generate_date_strings(cfg.year, cfg.time_suffix)
    logging.info("Dates to scan: %d", len(date_list))

    shap_values_all: List[np.ndarray] = []
    X_all: List[np.ndarray] = []

    valid_dates: List[str] = []
    skipped_missing: int = 0
    skipped_small: int = 0
    failed_dates: int = 0

    for d in date_list:
        feature_file = cfg.feature_pattern.format(date=d)
        target_file = cfg.target_pattern.format(date=d)

        if not (os.path.exists(feature_file) and os.path.exists(target_file)):
            skipped_missing += 1
            continue

        try:
            X_raw, y = load_day_data(feature_file, target_file, cfg.max_rows_per_day)
            if len(X_raw) < cfg.min_samples:
                skipped_small += 1
                continue

            # Select columns by indices
            X = X_raw.iloc[:, selected_indices]

            shap_vals, X_test_used = train_and_shap(
                X=X,
                y=y,
                seed=cfg.seed,
                test_size=cfg.test_size,
                n_estimators=cfg.n_estimators,
                n_jobs=cfg.n_jobs,
                use_scaler=cfg.use_scaler,
            )

            shap_values_all.append(shap_vals)
            X_all.append(X_test_used)
            valid_dates.append(d)

        except Exception as e:
            failed_dates += 1
            logging.warning("Failed on date %s: %s", d, str(e))
            continue

    if len(shap_values_all) == 0:
        raise RuntimeError(
            "No valid SHAP results were produced. "
            "Check file patterns, min_samples, and data integrity."
        )

    # Concatenate across dates/samples
    shap_values_concat = np.concatenate(shap_values_all, axis=0)
    X_all_concat = np.concatenate(X_all, axis=0)

    # Global importance
    mean_abs_shap = np.mean(np.abs(shap_values_concat), axis=0)

    top_n = min(cfg.top_n, len(selected_features_clean))
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

    top_features = [selected_features_clean[i] for i in top_indices]
    top_shap_values = shap_values_concat[:, top_indices]
    top_X_values = X_all_concat[:, top_indices]
    top_mean_abs_shap = mean_abs_shap[top_indices]

    # Outputs
    out_png = os.path.join(cfg.out_dir, f"shap_summary_{cfg.year}_{cfg.time_suffix}.png")
    out_csv = os.path.join(cfg.out_dir, f"shap_importance_{cfg.year}_{cfg.time_suffix}.csv")
    out_meta = os.path.join(cfg.out_dir, f"run_metadata_{cfg.year}_{cfg.time_suffix}.json")

    make_summary_figure(
        top_shap_values=top_shap_values,
        top_X_values=top_X_values,
        top_features=top_features,
        top_mean_abs_shap=top_mean_abs_shap,
        out_png=out_png,
    )

    # Save ranking CSV
    summary_df = pd.DataFrame({
        "Rank": np.arange(1, top_n + 1),
        "Feature_Name": top_features,
        "Mean_ABS_SHAP": top_mean_abs_shap,
    })
    summary_df.to_csv(out_csv, index=False)

    # Save metadata for reproducibility
    metadata = {
        "config": {
            **cfg.__dict__,
            "selected_features_count": len(cfg.selected_features),
            "feature_names_total_count": len(FEATURE_NAMES_TOTAL),
        },
        "results": {
            "valid_dates_count": len(valid_dates),
            "valid_dates": valid_dates[:20] + (["..."] if len(valid_dates) > 20 else []),
            "skipped_missing": skipped_missing,
            "skipped_small": skipped_small,
            "failed_dates": failed_dates,
            "total_test_samples_aggregated": int(shap_values_concat.shape[0]),
            "top_n": int(top_n),
        },
        "outputs": {
            "figure_png": os.path.abspath(out_png),
            "ranking_csv": os.path.abspath(out_csv),
        },
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Done.")
    logging.info("Valid dates: %d | skipped missing: %d | skipped small: %d | failed: %d",
                 len(valid_dates), skipped_missing, skipped_small, failed_dates)
    logging.info("Saved: %s", out_png)
    logging.info("Saved: %s", out_csv)
    logging.info("Saved: %s", out_meta)


if __name__ == "__main__":
    main()

