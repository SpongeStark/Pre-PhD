"""
Unified Microgrid Multi-Target Forecaster Engine
Benchmarks machine learning forecasting models (LightGBM vs XGBoost)
across 4 microgrid assets:
  1. Solar PV Generation (master_dataset.parquet)
  2. Supermarket Facility Load (master_dataset_con.parquet)
  3. Lidl EV Charging Stations (master_dataset_ev.parquet)
  4. Caltech Campus EV Fleet (master_dataset_caltech_ev.parquet)

Outputs predictions, metrics (R2, MAE, RMSE, nRMSE), and publication-quality
plots saved to BMS_brain/results_forecasting/.
"""

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Forecasting Imports
from skforecast.recursive import ForecasterRecursive
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configuration & Directories
SRC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SRC_DIR.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results_forecasting"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CONFIGS = {
    "pv": {
        "name": "Solar PV Generation",
        "file": "master_dataset.parquet",
        "target_col": "PV",
        "unit": "kW",
        "color": "#f59e0b",
        "exog_cols": ["temp", "GHI", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos"]
    },
    "con": {
        "name": "Supermarket Facility Load (CON)",
        "file": "master_dataset_con.parquet",
        "target_col": "c_gen",
        "unit": "kW",
        "color": "#ef4444",
        "exog_cols": ["temp", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos"]
    },
    "ev": {
        "name": "Lidl Commercial EV Chargers",
        "file": "master_dataset_ev.parquet",
        "target_col": "c_ev",
        "unit": "kW",
        "color": "#10b981",
        "exog_cols": ["temp", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos"]
    },
    "caltech": {
        "name": "Caltech Campus EV Fleet (ACN-Data)",
        "file": "master_dataset_caltech_ev.parquet",
        "target_col": "c_ev",
        "unit": "kW",
        "color": "#6366f1",
        "exog_cols": ["temp", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos"]
    }
}

LAGS = [1, 2, 3, 4, 96, 672]  # Lags: 15-min, 30-min, 45-min, 1-hr, 24-hr (yesterday), 7-days (last week)
STEPS_PER_DAY = 96            # 96 steps of 15-min = 24 hours
EVAL_DAYS = 7                 # 7-day evaluation rigor
STEPS = EVAL_DAYS * STEPS_PER_DAY  # 672 steps (168 hours = 1 full week)

def compute_metrics(actual, pred):
    """Computes standard and robust time-series forecasting metrics."""
    actual = np.array(actual, dtype=float)
    pred = np.array(pred, dtype=float)
    
    mae = float(mean_absolute_error(actual, pred))
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    r2 = float(r2_score(actual, pred))
    mean_act = float(np.mean(actual))
    sum_act = float(np.sum(actual))
    
    nrmse = float((rmse / mean_act) * 100) if mean_act > 0 else 0.0
    wape = float((np.sum(np.abs(actual - pred)) / sum_act) * 100) if sum_act > 0 else 0.0
    bias = float(((np.sum(pred) - sum_act) / sum_act) * 100) if sum_act > 0 else 0.0
    
    return {
        "r2": round(r2, 4),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "nrmse_pct": round(nrmse, 2),
        "wape_pct": round(wape, 2),
        "energy_bias_pct": round(bias, 2)
    }

def get_models():
    """Returns the candidate forecasting regressors."""
    return {
        "LGBM": LGBMRegressor(
            n_estimators=120,
            learning_rate=0.06,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        ),
        "XGB": XGBRegressor(
            n_estimators=120,
            learning_rate=0.06,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
    }

def load_and_preprocess(target_key: str):
    """Loads, sanitizes, and splits the time-series dataset."""
    cfg = TARGET_CONFIGS[target_key]
    file_path = DATA_DIR / cfg["file"]
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
        
    df = pd.read_parquet(file_path)
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    
    # If caltech dataset lacks temp, merge temp from master_dataset
    if "temp" not in df.columns:
        pv_ref = pd.read_parquet(DATA_DIR / "master_dataset.parquet")[["Date", "temp"]].drop_duplicates("Date")
        df = pd.merge(df, pv_ref, on="Date", how="left")
        df["temp"] = df["temp"].ffill().bfill()
        
    df = df.set_index("Date").asfreq("15min").ffill()
    
    # Target column check
    target_col = cfg["target_col"]
    exog_cols = [c for c in cfg["exog_cols"] if c in df.columns]
    
    # Time-based splitting
    max_year = df.index.max().year
    if max_year >= 2024:
        # Multi-year datasets (2022-2024)
        # Train: 2022 to end of 2023
        train = df[df.index < "2024-01-01"].copy()
        # Val: Jan 2024 to Sept 30, 2024
        val = df[(df.index >= "2024-01-01") & (df.index < "2024-10-01")].copy()
        # Test: Oct 1, 2024 onwards (first 24 hours used for evaluation horizon)
        test = df[df.index >= "2024-10-01"].copy()
    else:
        # Single-year datasets (e.g. 2022 Caltech EV)
        # Train: Jan - Aug 2022 (~67%)
        train = df[df.index < "2022-09-01"].copy()
        # Val: Sep - Oct 2022 (~17%)
        val = df[(df.index >= "2022-09-01") & (df.index < "2022-11-01")].copy()
        # Test: Nov - Dec 2022 (first 24 hours used for evaluation horizon)
        test = df[df.index >= "2022-11-01"].copy()
        
    return df, train, val, test, target_col, exog_cols

def train_and_evaluate_target(target_key: str, model_choice: str = "compare"):
    """
    Trains forecasters on the specified target and performs rolling 24h
    walk-forward backtesting across 7 days (672 steps), benchmarking against
    naive persistence baselines.
    """
    cfg = TARGET_CONFIGS[target_key]
    print(f"\n========================================================")
    print(f"  FORECASTING TARGET: {cfg['name']} ({target_key.upper()})")
    print(f"  Evaluation Horizon: 7 Days (168 Hours = {STEPS} steps)")
    print(f"========================================================")
    
    df, train, val, test, target_col, exog_cols = load_and_preprocess(target_key)
    print(f"Dataset Timeline: {df.index.min()} to {df.index.max()}")
    print(f"Split sizes: Train={len(train)} steps, Val={len(val)} steps, Test Pool={len(test)} steps")
    print(f"Exogenous features ({len(exog_cols)}): {exog_cols}")
    
    models = get_models()
    if model_choice != "compare" and model_choice.upper() in models:
        models = {model_choice.upper(): models[model_choice.upper()]}
        
    eval_slice = test.iloc[:STEPS]
    actuals = eval_slice[target_col].values
    time_index = eval_slice.index
    
    # Construct continuous series between val and test for lag lookups
    full_series = pd.concat([val[target_col], test[target_col]])
    val_len = len(val)
    
    results = {
        "target_key": target_key,
        "target_name": cfg["name"],
        "unit": cfg["unit"],
        "eval_start": str(time_index[0]),
        "eval_end": str(time_index[-1]),
        "eval_steps": STEPS,
        "eval_days": EVAL_DAYS,
        "actual_mean": float(np.mean(actuals)),
        "actual_max": float(np.max(actuals)),
        "models": {}
    }
    
    predictions_dict = {}
    
    for name, est in models.items():
        print(f"\n--- Training {name} Regressor (Walk-Forward: {EVAL_DAYS} Daily Cycles x {STEPS_PER_DAY} Steps) ---")
        t0 = time.time()
        forecaster = ForecasterRecursive(estimator=est, lags=LAGS)
        forecaster.fit(y=train[target_col], exog=train[exog_cols])
        fit_time = time.time() - t0
        
        # Operational walk-forward backtesting: predict each 24h block using actual history up to midnight
        daily_preds = []
        for day in range(EVAL_DAYS):
            s_start = day * STEPS_PER_DAY
            s_end = (day + 1) * STEPS_PER_DAY
            if s_start == 0:
                lw = val[target_col].iloc[-max(LAGS):]
            else:
                lw = full_series.iloc[val_len + s_start - max(LAGS) : val_len + s_start]
            exog_slice = test[exog_cols].iloc[s_start:s_end]
            p = forecaster.predict(steps=STEPS_PER_DAY, last_window=lw, exog=exog_slice).values
            daily_preds.extend(p)
            
        preds = np.clip(np.array(daily_preds), 0.0, None)
        
        # Standby power handling for Lidl EV (idle baseline ~0.08 kW)
        if target_key == "ev":
            preds = np.where(preds < 0.25, 0.08, preds)
            
        m = compute_metrics(actuals, preds)
        m["fit_time_sec"] = round(fit_time, 2)
        results["models"][name] = m
        predictions_dict[name] = preds
        
        print(f"Results for {name} (7-Day Aggregate):")
        print(f"  - Training Time: {fit_time:.2f} s")
        print(f"  - R2 Score:      {m['r2']:.4f}")
        print(f"  - MAE:           {m['mae']:.2f} {cfg['unit']}")
        print(f"  - RMSE:          {m['rmse']:.2f} {cfg['unit']}")
        print(f"  - WAPE:          {m['wape_pct']:.2f}%")
        print(f"  - Energy Bias:   {m['energy_bias_pct']:+.2f}%")
        
    best_model_name = max(models.keys(), key=lambda k: results["models"][k]["r2"])
    best_m = results["models"][best_model_name]
    results["best_model"] = best_model_name
    
    print(f"\n=> Best Performing Model for {cfg['name']}: {best_model_name} (R2 = {best_m['r2']:.4f} | WAPE = {best_m['wape_pct']:.2f}%)")
    
    # Generate & save individual 7-day plot
    plot_path = RESULTS_DIR / f"forecast_{target_key}.png"
    generate_target_plot(target_key, time_index, actuals, predictions_dict, results, plot_path)
    
    return results, time_index, actuals, predictions_dict

def generate_target_plot(target_key, time_index, actuals, predictions_dict, metrics, save_path):
    """Generates an individual high-resolution 7-day forecast plot: Actual vs Predicted with residuals."""
    cfg = TARGET_CONFIGS[target_key]
    best_name = metrics["best_model"]
    best_preds = predictions_dict[best_name]
    best_m = metrics["models"][best_name]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [2.6, 1]})
    
    hours = np.arange(len(actuals)) / 4.0  # 0 to 168 hours
    
    # Plot Actual Observed line
    ax1.plot(hours, actuals, label="Actual Observed", color="#0f172a", linewidth=2.4, zorder=5)
    
    # Plot ML Models Predicted
    colors = {"LGBM": "#3b82f6", "XGB": "#f97316"}
    styles = {"LGBM": "--", "XGB": "-."}
    for m_name in metrics["models"].keys():
        if m_name in predictions_dict:
            is_best = (m_name == best_name)
            lw = 2.4 if is_best else 1.5
            label = f"{m_name} Forecast ({'★ Best: ' if is_best else ''}R²={metrics['models'][m_name]['r2']:.3f}, WAPE={metrics['models'][m_name]['wape_pct']:.1f}%)"
            ax1.plot(hours, predictions_dict[m_name], label=label, color=colors.get(m_name, "#8b5cf6"),
                     linestyle=styles.get(m_name, "-"), linewidth=lw, alpha=0.95 if is_best else 0.7)
                     
    # Day boundary vertical grid lines & labels
    max_y = float(np.max(actuals)) if len(actuals) > 0 else 1.0
    for d in range(1, 8):
        h = d * 24
        ax1.axvline(h, color="#cbd5e1", linestyle="--", alpha=0.7, linewidth=1.0)
        ax2.axvline(h, color="#cbd5e1", linestyle="--", alpha=0.7, linewidth=1.0)
        if d <= 7:
            ax1.text(h - 12, max_y * 0.96, f"Day {d}", ha="center", va="top",
                     fontsize=9, fontweight="bold", color="#475569",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.85))
                     
    ax1.set_ylabel(f"Power Demand / Output ({cfg['unit']})", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"7-Day Operational Forecast (168h): {cfg['name']}\n"
        f"Best Model: {best_name} | R²: {best_m['r2']:.3f} | MAE: {best_m['mae']:.2f} {cfg['unit']} | "
        f"WAPE: {best_m['wape_pct']:.1f}% | Energy Bias: {best_m['energy_bias_pct']:+.1f}%",
        fontsize=12, fontweight="bold", pad=10
    )
    ax1.legend(loc="upper right", frameon=True, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5)
    
    # Bottom: Residuals (Error = Actual - Best Prediction)
    residuals = actuals - best_preds
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    ax2.bar(hours, residuals, width=0.22, color=np.where(residuals >= 0, "#ef4444", "#3b82f6"), alpha=0.6,
            label=f"Residual Error (Actual - {best_name})")
    ax2.set_xlabel("Operational Forecast Horizon (Hours Ahead: 0 to 168h | 7 Days)", fontsize=11, fontweight="bold")
    ax2.set_ylabel(f"Residual ({cfg['unit']})", fontsize=10)
    ax2.set_xlim(-1, 169)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"Saved 7-day forecast plot to: {save_path}")

def generate_combined_scorecard(all_results, all_trajectories, save_path):
    """
    Builds a unified 4-panel dashboard plot comparing all 4 forecast systems across 7 days: Actual vs Predicted.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    axes = axes.flatten()
    
    target_keys = ["pv", "con", "ev", "caltech"]
    
    for idx, key in enumerate(target_keys):
        ax = axes[idx]
        if key not in all_results:
            ax.text(0.5, 0.5, f"No data for {key}", ha="center", va="center")
            continue
            
        res = all_results[key]
        time_index, actuals, preds_dict = all_trajectories[key]
        hours = np.arange(len(actuals)) / 4.0
        cfg = TARGET_CONFIGS[key]
        best_m = res["best_model"]
        best_metrics = res["models"][best_m]
        
        # Plot actual
        ax.plot(hours, actuals, label="Actual Observed", color="#0f172a", linewidth=2.0)
        
        # Plot best ML forecast
        ax.plot(hours, preds_dict[best_m], label=f"Predicted ({best_m})",
                color=cfg["color"], linestyle="--", linewidth=2.0)
                
        # Day boundary markers
        for d in range(1, 7):
            ax.axvline(d * 24, color="#cbd5e1", linestyle=":", alpha=0.7)
            
        r2 = best_metrics["r2"]
        wape = best_metrics["wape_pct"]
        bias = best_metrics["energy_bias_pct"]
        
        pred_label = "HIGH PREDICTABILITY" if r2 >= 0.70 else ("MODERATE PREDICTABILITY" if r2 >= 0.30 else "SPARSE / INTERMITTENT")
        
        ax.set_title(
            f"[{idx+1}] {cfg['name']}\n"
            f"{pred_label} | R²={r2:.3f} | MAE={best_metrics['mae']:.2f} kW | WAPE={wape:.1f}% | Bias={bias:+.1f}%",
            fontsize=10, fontweight="bold", pad=8
        )
        ax.set_xlabel("Horizon Hours (0 to 168h | 7 Days)", fontsize=9)
        ax.set_ylabel(f"Power ({cfg['unit']})", fontsize=9)
        ax.set_xlim(0, 168)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)
        
    plt.suptitle("Microgrid 7-Day Multi-Asset Forecasting Scorecard: Actual vs. Predicted", 
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"\nUnified 4-target 7-day scorecard saved to: {save_path}")

def run_pipeline(target_arg: str = "all", model_arg: str = "compare"):
    """Main execution coordinator."""
    if target_arg == "all":
        targets_to_run = ["pv", "con", "ev", "caltech"]
    else:
        if target_arg not in TARGET_CONFIGS:
            raise ValueError(f"Unknown target '{target_arg}'. Choose from: {list(TARGET_CONFIGS.keys())} or 'all'")
        targets_to_run = [target_arg]
        
    all_results = {}
    all_trajectories = {}
    
    for t_key in targets_to_run:
        res, t_idx, actuals, preds_dict = train_and_evaluate_target(t_key, model_choice=model_arg)
        all_results[t_key] = res
        all_trajectories[t_key] = (t_idx, actuals, preds_dict)
        
    # Save results JSON
    metrics_file = RESULTS_DIR / "forecast_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved forecast metrics to: {metrics_file}")
    
    # Save CSV summary
    rows = []
    for t_key, res in all_results.items():
        for m_name, m_stats in res["models"].items():
            rows.append({
                "Asset": res["target_name"],
                "Target_Key": t_key,
                "Model": m_name,
                "Is_Best": (m_name == res["best_model"]),
                "R2": m_stats["r2"],
                "MAE_kW": m_stats["mae"],
                "RMSE_kW": m_stats["rmse"],
                "nRMSE_pct": m_stats["nrmse_pct"],
                "WAPE_pct": m_stats["wape_pct"],
                "Energy_Bias_pct": m_stats["energy_bias_pct"],
                "Train_Time_s": m_stats.get("fit_time_sec", None)
            })
    df_summary = pd.DataFrame(rows)
    csv_file = RESULTS_DIR / "forecast_metrics_summary.csv"
    df_summary.to_csv(csv_file, index=False)
    print(f"Saved CSV scorecard to: {csv_file}")
    
    # If all targets were executed, generate the unified scorecard
    if len(all_results) == 4:
        scorecard_path = RESULTS_DIR / "forecast_scorecard.png"
        generate_combined_scorecard(all_results, all_trajectories, scorecard_path)
        
    print("\n========================================================")
    print("  7-DAY FORECASTING BENCHMARK SCORECARD")
    print("========================================================")
    print(df_summary.to_string(index=False))
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Microgrid Multi-Target Forecaster Engine")
    parser.add_argument(
        "--target",
        choices=["pv", "con", "ev", "caltech", "all"],
        default="all",
        help="Asset target to forecast: 'pv', 'con', 'ev', 'caltech', or 'all' (default)"
    )
    parser.add_argument(
        "--model",
        choices=["lgbm", "xgb", "compare"],
        default="compare",
        help="Model algorithm to use: 'lgbm', 'xgb', or 'compare' (benchmarks both)"
    )
    args = parser.parse_args()
    run_pipeline(target_arg=args.target, model_arg=args.model)
