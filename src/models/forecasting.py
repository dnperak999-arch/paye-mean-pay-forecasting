from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import outputs_path


def prepare_forecasting_matrix(pay: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare the (industries x months) matrix for the forecasting module.

    Fixes the common runtime error:
    'TypeError: DataFrame cannot interpolate with object dtype'
    by ensuring:
      - Date is parsed + sorted
      - industry columns are numeric
      - interpolation is performed along the time axis
    """

    # These two columns record survey-style answers about each month's payroll,
    # not industry pay series, so they cannot be used as features.
    col_b = "Was there any payroll error or Data Issue this month"
    col_c = "Majority of the payroll were contributed by people from region North East, North West or East Midlands"
    pay = pay.drop(columns=[col_b, col_c], errors="ignore")

    if "Date" not in pay.columns:
        raise KeyError("Expected a 'Date' column in the pay DataFrame.")

    industries = [c for c in pay.columns if c != "Date"]

    # Ensure Date is datetime and sorted
    df = pay[["Date"] + industries].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Convert all industry columns to numeric (critical for interpolate)
    for col in industries:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Transpose so rows = industries, columns = months
    df_t = df.set_index("Date")[industries].T

    # Force numeric dtype on the transposed matrix (safety)
    df_t = df_t.apply(pd.to_numeric, errors="coerce")

    # Standardise columns to month-start timestamps, then to YYYY-MM strings
    df_t.columns = pd.to_datetime(df_t.columns).to_period("M").to_timestamp()

    # Interpolate across months for each industry (row-wise / time axis)
    df_t = df_t.interpolate(axis=1, method="linear", limit_direction="both")

    # Final safety net for any edge NaNs
    df_t = df_t.ffill(axis=1).bfill(axis=1)

    # Keep the same month label style used elsewhere in the report/code
    df_t.columns = [d.strftime("%Y-%m") for d in df_t.columns]

    return df_t, industries


def loocv_ols_predictions(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-One-Out CV predictions for OLS (manual numpy)."""
    n = X.shape[0]
    y_pred = np.empty(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        X_train = X[mask]
        y_train = y[mask]
        X_test = X[i:i + 1]

        # Add intercept
        Xb_train = np.column_stack([np.ones(X_train.shape[0]), X_train])
        Xb_test = np.column_stack([np.ones(X_test.shape[0]), X_test])

        beta = np.linalg.lstsq(Xb_train, y_train, rcond=None)[0]
        y_pred[i] = float((Xb_test @ beta).ravel()[0])

    return y_pred


def loocv_ridge_predictions(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Leave-One-Out CV predictions for Ridge regression (scikit-learn).

    Standardisation is fitted inside each fold to avoid leakage.
    """
    n = X.shape[0]
    y_pred = np.empty(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        X_train = X[mask]
        y_train = y[mask]
        X_test = X[i:i + 1]

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        model.fit(X_train, y_train)
        y_pred[i] = float(model.predict(X_test)[0])

    return y_pred


@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return mae, rmse, r2


def run_forecasting(pay: pd.DataFrame) -> pd.DataFrame:
    """Forecast June 2025 mean pay across industries (cross-sectional setup).

    What this function does:
    - Builds an industries × months matrix from the PAYE mean pay sheet.
    - Uses the last 12 months (Jun-2024 … May-2025) as features.
    - Predicts the June-2025 value for each industry.

    Evaluation:
    - LOOCV across industries (n=21) for OLS (main model).
    - Baselines for comparison: persistence (use May-2025) and Ridge.
    - Optional train/test split across industries to sanity-check generalisation.

    Note:
    This is *not* a rolling time-series backtest; industries are treated as samples.
    """
    df_t, industries = prepare_forecasting_matrix(pay)

    target_col = "2025-06"
    feature_cols = [c for c in df_t.columns if c != target_col]
    last_12 = feature_cols[-12:]

    # Features: last 12 months (Jun-2024 ... May-2025); Target: Jun-2025
    X_feat = df_t[last_12].values.astype(float)
    y_actual = df_t[target_col].values.astype(float)

    # A) Baseline: persistence (predict Jun-2025 using May-2025)
    y_pred_baseline = X_feat[:, -1].copy()

    # B) OLS (LOOCV)
    y_pred_ols_loocv = loocv_ols_predictions(X_feat, y_actual)
    ols_mae, ols_rmse, ols_r2 = regression_metrics(y_actual, y_pred_ols_loocv)

    # C) Ridge (LOOCV) - portfolio
    y_pred_ridge_loocv = loocv_ridge_predictions(X_feat, y_actual, alpha=1.0)
    ridge_mae, ridge_rmse, ridge_r2 = regression_metrics(y_actual, y_pred_ridge_loocv)

    # Baseline metrics
    base_mae, base_rmse, base_r2 = regression_metrics(y_actual, y_pred_baseline)

    # Print summary metrics
    print("LOOCV evaluation (June 2025 forecast across industries)")
    print(f"OLS (LOOCV)        -> MAE: {ols_mae:.2f} | RMSE: {ols_rmse:.2f} | R^2: {ols_r2:.4f}")
    print(f"Baseline (May=Jun) -> MAE: {base_mae:.2f} | RMSE: {base_rmse:.2f} | R^2: {base_r2:.4f}")
    print(f"Ridge (LOOCV)      -> MAE: {ridge_mae:.2f} | RMSE: {ridge_rmse:.2f} | R^2: {ridge_r2:.4f}")

    # Results table (keep OLS LOOCV as the main predicted column; add extras)
    results = pd.DataFrame(
        {
            "Industry": industries,
            "Actual (Jun-25)": np.round(y_actual, 2),
            "Predicted OLS (LOOCV)": np.round(y_pred_ols_loocv, 2),
            "Predicted Ridge (LOOCV)": np.round(y_pred_ridge_loocv, 2),
            "Predicted Baseline (May-25)": np.round(y_pred_baseline, 2),
        }
    )
    results["Abs Error (OLS)"] = np.round(np.abs(y_pred_ols_loocv - y_actual), 2)
    results["Abs Error (Ridge)"] = np.round(np.abs(y_pred_ridge_loocv - y_actual), 2)
    results["Abs Error (Baseline)"] = np.round(np.abs(y_pred_baseline - y_actual), 2)

    # D) Train/Test split (portfolio)
    # Split industries (rows). This is not time-series CV; it tests cross-sectional generalisation.
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y_actual, test_size=0.30, random_state=42
    )

    # OLS on train -> test
    Xb_train = np.column_stack([np.ones(X_train.shape[0]), X_train])
    Xb_test = np.column_stack([np.ones(X_test.shape[0]), X_test])
    beta = np.linalg.lstsq(Xb_train, y_train, rcond=None)[0]
    y_pred_test_ols = (Xb_test @ beta).ravel()

    # Ridge on train -> test (with scaling)
    ridge_model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    ridge_model.fit(X_train, y_train)
    y_pred_test_ridge = ridge_model.predict(X_test)

    # Baseline on test -> test (use last observed month)
    y_pred_test_base = X_test[:, -1]

    tt_ols = Metrics(*regression_metrics(y_test, y_pred_test_ols))
    tt_ridge = Metrics(*regression_metrics(y_test, y_pred_test_ridge))
    tt_base = Metrics(*regression_metrics(y_test, y_pred_test_base))

    print("Train/test split across industries (30% test) — sanity check")
    print(f"OLS (train→test)   -> MAE: {tt_ols.mae:.2f} | RMSE: {tt_ols.rmse:.2f} | R^2: {tt_ols.r2:.4f}")
    print(f"Ridge (train→test) -> MAE: {tt_ridge.mae:.2f} | RMSE: {tt_ridge.rmse:.2f} | R^2: {tt_ridge.r2:.4f}")
    print(f"Baseline (test)    -> MAE: {tt_base.mae:.2f} | RMSE: {tt_base.rmse:.2f} | R^2: {tt_base.r2:.4f}")

    # Figures (save only; do not show)
    # Bar chart: Actual vs OLS(LOOCV)
    fig, ax = plt.subplots(figsize=(14, 7))
    short_names = [name[:25] for name in industries]
    x = np.arange(len(short_names))
    width = 0.35

    ax.bar(x - width / 2, y_actual, width, label="Actual")
    ax.bar(x + width / 2, y_pred_ols_loocv, width, label="Predicted (OLS LOOCV)")

    ax.set_ylabel("Mean Pay (£)")
    ax.set_title("Actual vs Predicted Mean Pay for June 2025 (All Industries)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(outputs_path("mean_pay_forecast_bar.png"), dpi=300)
    plt.close(fig)

    # Scatter: Actual vs OLS(LOOCV)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(y_actual, y_pred_ols_loocv, s=60, edgecolors="black", linewidth=0.5)

    min_val = min(float(y_actual.min()), float(y_pred_ols_loocv.min()))
    max_val = max(float(y_actual.max()), float(y_pred_ols_loocv.max()))

    ax2.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Perfect prediction")
    ax2.set_xlabel("Actual Mean Pay (£)")
    ax2.set_ylabel("Predicted Mean Pay (£)")
    ax2.set_title(
        f"Predicted vs Actual (June 2025)\n"
        f"OLS LOOCV MAE={ols_mae:.2f}, RMSE={ols_rmse:.2f}, R²={ols_r2:.4f}"
    )
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outputs_path("mean_pay_forecast_scatter.png"), dpi=300)
    plt.close(fig2)

    return results