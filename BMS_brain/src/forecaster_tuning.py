"""
Systematic Hyperparameter Tuning & Model Optimization Engine
============================================================
Performs systematic Bayesian optimization (Optuna TPE Sampler) on:
  - Tree Depth (max_depth, num_leaves)
  - Learning Rate (learning_rate)
  - L1 Regularization (reg_alpha)
  - L2 Regularization (reg_lambda)
  - Subsampling (subsample, colsample_bytree)
  - Estimators (n_estimators)

Across both LightGBM (LGBMRegressor) and XGBoost (XGBRegressor) on 4 microgrid assets:
  1. Solar PV Generation (pv)
  2. Supermarket Facility Load (con)
  3. Lidl Commercial EV Chargers (ev)
  4. Caltech Campus EV Fleet (caltech)

Outputs:
  - Best parameters: results_forecasting/best_hyperparameters.json
  - Comparison table: results_forecasting/tuning_comparison_results.csv
  - Comparison summary: results_forecasting/tuning_comparison_summary.json
  - Dashboard plots: results_forecasting/tuning_comparison_dashboard.png
  - Prediction overlay plots: results_forecasting/tuning_overlay_{target}.png
"""

import sys
import json
import time
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Enforce UTF-8 stdout encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Optuna Bayesian Optimization
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Machine Learning & Forecasting
from skforecast.recursive import ForecasterRecursive
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Project paths
SRC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SRC_DIR.parent.resolve()
RESULTS_DIR = ROOT_DIR / "results_forecasting"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = Path(r"C:\Users\alesk\.gemini\antigravity\brain\4fec5526-7196-4acd-9976-88084cad6a69")

from forecaster_engine import (
    load_and_preprocess,
    compute_metrics,
    TARGET_CONFIGS,
    LAGS,
    STEPS_PER_DAY
)

# Default baseline hyperparameters (as previously fixed in forecaster_engine.py)
DEFAULT_PARAMS = {
    "LGBM": {
        "n_estimators": 120,
        "learning_rate": 0.06,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1
    },
    "XGB": {
        "n_estimators": 120,
        "learning_rate": 0.06,
        "max_depth": 5,
        "random_state": 42,
        "n_jobs": -1
    }
}

def evaluate_walk_forward(forecaster, full_series, exog_df, split_idx, num_days, steps_per_day, target_key):
    """
    Simulates operational rolling 24-hour walk-forward forecasting.
    Updates historical last_window before each midnight dispatch prediction.
    """
    preds = []
    for day in range(num_days):
        s_start = day * steps_per_day
        s_end = (day + 1) * steps_per_day
        lw = full_series.iloc[split_idx + s_start - max(LAGS) : split_idx + s_start]
        exog_slice = exog_df.iloc[s_start:s_end]
        p = forecaster.predict(steps=steps_per_day, last_window=lw, exog=exog_slice).values
        preds.extend(p)
        
    preds = np.clip(np.array(preds), 0.0, None)
    if target_key == "ev":
        preds = np.where(preds < 0.25, 0.08, preds)
    return preds

def optimize_target_model(target_key: str, model_name: str, n_trials: int = 20, val_days: int = 5):
    """
    Executes an Optuna study to find the best hyperparameters on the validation split.
    """
    cfg = TARGET_CONFIGS[target_key]
    df, train, val, test, target_col, exog_cols = load_and_preprocess(target_key)
    
    val_steps = val_days * STEPS_PER_DAY
    val_slice = val.iloc[-val_steps:]
    actual_val = val_slice[target_col].values
    
    full_val_series = pd.concat([train[target_col], val[target_col]])
    val_start_idx = len(train) + len(val) - val_steps
    
    print(f"\n>>> Running Optuna Sweep for {cfg['name']} [{model_name}] ({n_trials} trials, {val_days} val days) <<<")

    def objective(trial):
        if model_name == "LGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 220, step=20),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.10),
                "max_depth": trial.suggest_int("max_depth", 5, 9),
                "num_leaves": trial.suggest_int("num_leaves", 31, 127),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 2.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.75, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.75, 1.0),
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1
            }
            reg = LGBMRegressor(**params)
        else: # XGBoost
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 220, step=20),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.10),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 2.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.75, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.75, 1.0),
                "random_state": 42,
                "n_jobs": -1
            }
            reg = XGBRegressor(**params)
            
        forecaster = ForecasterRecursive(estimator=reg, lags=LAGS)
        forecaster.fit(y=train[target_col], exog=train[exog_cols])
        
        val_preds = evaluate_walk_forward(
            forecaster, full_val_series, val_slice[exog_cols],
            val_start_idx, val_days, STEPS_PER_DAY, target_key
        )
        
        mae = mean_absolute_error(actual_val, val_preds)
        return mae

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    tune_time = time.time() - t0
    
    best_params = study.best_params
    print(f"  Optimization completed in {tune_time:.2f} s")
    print(f"  Best Validation MAE: {study.best_value:.3f} {cfg['unit']}")
    print(f"  Best Hyperparameters: {best_params}")
    
    return best_params, study

def run_tuning_benchmark(targets=None, models=None, n_trials=20, eval_days=7):
    """
    Tunes models across targets and performs 7-day unseen test set evaluation
    comparing Default vs. Tuned models.
    """
    if targets is None:
        targets = ["pv", "con", "ev", "caltech"]
    if models is None:
        models = ["LGBM", "XGB"]
        
    eval_steps = eval_days * STEPS_PER_DAY
    comparison_records = []
    best_hyperparams_all = {}
    test_predictions = {}
    
    print("\n========================================================")
    print(" SYSTEMATIC HYPERPARAMETER SWEEP & BENCHMARK (OPTUNA)")
    print(f" Targets: {targets} | Models: {models} | Trials: {n_trials}")
    print(f" Test Evaluation: {eval_days} Days ({eval_steps} Steps)")
    print("========================================================")
    
    for t_key in targets:
        cfg = TARGET_CONFIGS[t_key]
        df, train, val, test, target_col, exog_cols = load_and_preprocess(t_key)
        test_slice = test.iloc[:eval_steps]
        actual_test = test_slice[target_col].values
        test_index = test_slice.index
        
        full_test_series = pd.concat([val[target_col], test[target_col]])
        val_len = len(val)
        
        best_hyperparams_all[t_key] = {}
        test_predictions[t_key] = {
            "actual": actual_test,
            "timestamps": test_index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "models": {}
        }
        
        for m_name in models:
            # 1. Run Optuna Hyperparameter Optimization
            best_p, study = optimize_target_model(t_key, m_name, n_trials=n_trials)
            best_hyperparams_all[t_key][m_name] = best_p
            
            # 2. Evaluate Default Model on Unseen Test Set
            print(f"\n--- Testing DEFAULT {m_name} on Unseen 7-Day Split [{cfg['name']}] ---")
            default_p = DEFAULT_PARAMS[m_name].copy()
            if m_name == "LGBM":
                default_reg = LGBMRegressor(**default_p)
            else:
                default_reg = XGBRegressor(**default_p)
                
            t0 = time.time()
            forecaster_default = ForecasterRecursive(estimator=default_reg, lags=LAGS)
            forecaster_default.fit(y=train[target_col], exog=train[exog_cols])
            default_fit_time = time.time() - t0
            
            preds_default = evaluate_walk_forward(
                forecaster_default, full_test_series, test_slice[exog_cols],
                val_len, eval_days, STEPS_PER_DAY, t_key
            )
            metrics_default = compute_metrics(actual_test, preds_default)
            metrics_default["fit_time_sec"] = round(default_fit_time, 2)
            
            # 3. Evaluate Tuned Model on Unseen Test Set
            print(f"--- Testing TUNED {m_name} on Unseen 7-Day Split [{cfg['name']}] ---")
            tuned_p = best_p.copy()
            if m_name == "LGBM":
                tuned_p.update({"random_state": 42, "verbose": -1, "n_jobs": -1})
                tuned_reg = LGBMRegressor(**tuned_p)
            else:
                tuned_p.update({"random_state": 42, "n_jobs": -1})
                tuned_reg = XGBRegressor(**tuned_p)
                
            t0 = time.time()
            forecaster_tuned = ForecasterRecursive(estimator=tuned_reg, lags=LAGS)
            forecaster_tuned.fit(y=train[target_col], exog=train[exog_cols])
            tuned_fit_time = time.time() - t0
            
            preds_tuned = evaluate_walk_forward(
                forecaster_tuned, full_test_series, test_slice[exog_cols],
                val_len, eval_days, STEPS_PER_DAY, t_key
            )
            metrics_tuned = compute_metrics(actual_test, preds_tuned)
            metrics_tuned["fit_time_sec"] = round(tuned_fit_time, 2)
            
            # 4. Deltas and Improvements
            mae_improvement_pct = ((metrics_default["mae"] - metrics_tuned["mae"]) / metrics_default["mae"]) * 100.0 if metrics_default["mae"] > 0 else 0.0
            rmse_improvement_pct = ((metrics_default["rmse"] - metrics_tuned["rmse"]) / metrics_default["rmse"]) * 100.0 if metrics_default["rmse"] > 0 else 0.0
            r2_delta = metrics_tuned["r2"] - metrics_default["r2"]
            wape_delta = metrics_default["wape_pct"] - metrics_tuned["wape_pct"]
            
            print(f"  Default => R2: {metrics_default['r2']:.4f} | MAE: {metrics_default['mae']:.2f} | WAPE: {metrics_default['wape_pct']:.2f}%")
            print(f"  Tuned   => R2: {metrics_tuned['r2']:.4f} | MAE: {metrics_tuned['mae']:.2f} | WAPE: {metrics_tuned['wape_pct']:.2f}%")
            print(f"  Gain    => Delta_MAE: {mae_improvement_pct:+.2f}% | Delta_WAPE: {wape_delta:+.2f}% | Delta_R2: {r2_delta:+.4f}")
            
            record = {
                "target_key": t_key,
                "target_name": cfg["name"],
                "model": m_name,
                "default_r2": metrics_default["r2"],
                "tuned_r2": metrics_tuned["r2"],
                "r2_delta": round(r2_delta, 4),
                "default_mae": metrics_default["mae"],
                "tuned_mae": metrics_tuned["mae"],
                "mae_improvement_pct": round(mae_improvement_pct, 2),
                "default_rmse": metrics_default["rmse"],
                "tuned_rmse": metrics_tuned["rmse"],
                "rmse_improvement_pct": round(rmse_improvement_pct, 2),
                "default_wape_pct": metrics_default["wape_pct"],
                "tuned_wape_pct": metrics_tuned["wape_pct"],
                "wape_reduction_pct": round(wape_delta, 2),
                "default_fit_time_sec": metrics_default["fit_time_sec"],
                "tuned_fit_time_sec": metrics_tuned["fit_time_sec"],
                "best_params": json.dumps(best_p)
            }
            comparison_records.append(record)
            
            test_predictions[t_key]["models"][m_name] = {
                "default": preds_default.tolist(),
                "tuned": preds_tuned.tolist(),
                "metrics_default": metrics_default,
                "metrics_tuned": metrics_tuned
            }
            
    # Export Results
    comp_df = pd.DataFrame(comparison_records)
    csv_path = RESULTS_DIR / "tuning_comparison_results.csv"
    comp_df.to_csv(csv_path, index=False)
    print(f"\nSaved tuning comparison table to: {csv_path}")
    
    json_path = RESULTS_DIR / "best_hyperparameters.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_hyperparams_all, f, indent=2)
    print(f"Saved best hyperparameters to: {json_path}")
    
    summary_path = RESULTS_DIR / "tuning_comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "eval_days": eval_days,
            "n_trials": n_trials,
            "comparisons": comparison_records,
            "best_hyperparameters": best_hyperparams_all
        }, f, indent=2)
    print(f"Saved tuning summary JSON to: {summary_path}")
    
    # Generate Visual Dashboards
    plot_tuning_dashboard(comp_df, RESULTS_DIR / "tuning_comparison_dashboard.png")
    
    for t_key in targets:
        plot_prediction_overlay(
            t_key, test_predictions[t_key],
            RESULTS_DIR / f"tuning_overlay_{t_key}.png"
        )
        
    return comp_df, best_hyperparams_all

def plot_tuning_dashboard(comp_df: pd.DataFrame, save_path: Path):
    """
    Generates a 4-panel dashboard visualizing the impact of Optuna hyperparameter optimization.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.grid(True, linestyle="--", alpha=0.3, color="#64748b")
        ax.tick_params(colors="#cbd5e1", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#334155")
            
    fig.suptitle("Systematic Hyperparameter Optimization: Default vs. Optuna-Tuned Forecasters (7-Day Unseen Test)",
                 color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)

    targets = comp_df["target_key"].unique()
    target_labels = [TARGET_CONFIGS[t]["name"] for t in targets]
    x = np.arange(len(targets))
    width = 0.20

    # Panel 1: R2 Score Comparison (LGBM & XGB)
    ax1 = axes[0, 0]
    lgbm_def_r2 = comp_df[comp_df["model"] == "LGBM"]["default_r2"].values
    lgbm_tun_r2 = comp_df[comp_df["model"] == "LGBM"]["tuned_r2"].values
    xgb_def_r2 = comp_df[comp_df["model"] == "XGB"]["default_r2"].values
    xgb_tun_r2 = comp_df[comp_df["model"] == "XGB"]["tuned_r2"].values
    
    r1 = ax1.bar(x - 1.5*width, lgbm_def_r2, width, label="LGBM Default", color="#64748b", alpha=0.8)
    r2 = ax1.bar(x - 0.5*width, lgbm_tun_r2, width, label="LGBM Tuned", color="#38bdf8")
    r3 = ax1.bar(x + 0.5*width, xgb_def_r2, width, label="XGB Default", color="#94a3b8", alpha=0.8)
    r4 = ax1.bar(x + 1.5*width, xgb_tun_r2, width, label="XGB Tuned", color="#fbbf24")
    
    ax1.set_ylabel("R² Determination Coefficient", color="#e2e8f0", fontweight="bold")
    ax1.set_title("Forecast Accuracy (R² Score: Default vs. Tuned)", color="#f8fafc", fontweight="bold", pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(target_labels, fontsize=9)
    ax1.set_ylim(-0.1, 1.05)
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)

    # Panel 2: WAPE (%) Error Reduction
    ax2 = axes[0, 1]
    lgbm_wape_red = comp_df[comp_df["model"] == "LGBM"]["wape_reduction_pct"].values
    xgb_wape_red = comp_df[comp_df["model"] == "XGB"]["wape_reduction_pct"].values
    
    r5 = ax2.bar(x - width/2, lgbm_wape_red, width, label="LGBM WAPE Reduction", color="#38bdf8")
    r6 = ax2.bar(x + width/2, xgb_wape_red, width, label="XGB WAPE Reduction", color="#fbbf24")
    ax2.set_ylabel("Error Reduction ΔWAPE (Percentage Points)", color="#e2e8f0", fontweight="bold")
    ax2.set_title("Weighted Absolute Percentage Error (WAPE) Reduction", color="#f8fafc", fontweight="bold", pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(target_labels, fontsize=9)
    ax2.axhline(0, color="#64748b", linestyle="-", linewidth=0.8)
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
    for r in r5:
        h = r.get_height()
        ax2.annotate(f"{h:+.1f}%", (r.get_x() + r.get_width()/2, h), ha="center", va="bottom" if h>=0 else "top", color="#38bdf8", fontsize=8, fontweight="bold")
    for r in r6:
        h = r.get_height()
        ax2.annotate(f"{h:+.1f}%", (r.get_x() + r.get_width()/2, h), ha="center", va="bottom" if h>=0 else "top", color="#fbbf24", fontsize=8, fontweight="bold")

    # Panel 3: MAE Relative Improvement (%)
    ax3 = axes[1, 0]
    lgbm_mae_imp = comp_df[comp_df["model"] == "LGBM"]["mae_improvement_pct"].values
    xgb_mae_imp = comp_df[comp_df["model"] == "XGB"]["mae_improvement_pct"].values
    
    r7 = ax3.bar(x - width/2, lgbm_mae_imp, width, label="LGBM ΔMAE (%)", color="#34d399")
    r8 = ax3.bar(x + width/2, xgb_mae_imp, width, label="XGB ΔMAE (%)", color="#f59e0b")
    ax3.set_ylabel("Relative MAE Improvement (%)", color="#e2e8f0", fontweight="bold")
    ax3.set_title("Mean Absolute Error (MAE) Improvement over Default", color="#f8fafc", fontweight="bold", pad=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(target_labels, fontsize=9)
    ax3.axhline(0, color="#64748b", linestyle="-", linewidth=0.8)
    ax3.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
    for r in r7:
        h = r.get_height()
        ax3.annotate(f"{h:+.1f}%", (r.get_x() + r.get_width()/2, h), ha="center", va="bottom" if h>=0 else "top", color="#34d399", fontsize=8, fontweight="bold")
    for r in r8:
        h = r.get_height()
        ax3.annotate(f"{h:+.1f}%", (r.get_x() + r.get_width()/2, h), ha="center", va="bottom" if h>=0 else "top", color="#f59e0b", fontsize=8, fontweight="bold")

    # Panel 4: Training Fit Time Overhead
    ax4 = axes[1, 1]
    lgbm_time = comp_df[comp_df["model"] == "LGBM"]["tuned_fit_time_sec"].values
    xgb_time = comp_df[comp_df["model"] == "XGB"]["tuned_fit_time_sec"].values
    
    r9 = ax4.bar(x - width/2, lgbm_time, width, label="LGBM Fit Time", color="#a855f7")
    r10 = ax4.bar(x + width/2, xgb_time, width, label="XGB Fit Time", color="#ec4899")
    ax4.set_ylabel("Fit Time (seconds)", color="#e2e8f0", fontweight="bold")
    ax4.set_title("Computational Cost (Single Full Retrain Time)", color="#f8fafc", fontweight="bold", pad=8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(target_labels, fontsize=9)
    ax4.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
    for r in r9:
        ax4.annotate(f"{r.get_height():.2f}s", (r.get_x() + r.get_width()/2, r.get_height()), ha="center", va="bottom", color="#a855f7", fontsize=8)
    for r in r10:
        ax4.annotate(f"{r.get_height():.2f}s", (r.get_x() + r.get_width()/2, r.get_height()), ha="center", va="bottom", color="#ec4899", fontsize=8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved tuning comparison dashboard to: {save_path}")
    
    if ARTIFACTS_DIR.exists():
        dst = ARTIFACTS_DIR / "tuning_comparison_dashboard.png"
        shutil.copy2(save_path, dst)
        print(f"Copied dashboard to artifacts: {dst.name}")

def plot_prediction_overlay(target_key: str, data: dict, save_path: Path):
    """
    Plots a high-resolution timeseries overlay of Actual vs. Default vs. Tuned
    predictions over the 7-day evaluation window.
    """
    cfg = TARGET_CONFIGS[target_key]
    timestamps = pd.to_datetime(data["timestamps"])
    actual = np.array(data["actual"])
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.grid(True, linestyle="--", alpha=0.3, color="#64748b")
        ax.tick_params(colors="#cbd5e1", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")
            
    fig.suptitle(f"7-Day Walk-Forward Forecast: Default vs. Tuned Models [{cfg['name']}]",
                 color="#f8fafc", fontsize=13, fontweight="bold", y=0.98)

    # Subplot 1: LightGBM
    ax1 = axes[0]
    ax1.plot(timestamps, actual, color="#cbd5e1", linewidth=1.6, label=f"Actual Ground Truth ({cfg['unit']})")
    lgbm_data = data["models"]["LGBM"]
    ax1.plot(timestamps, lgbm_data["default"], color="#94a3b8", linestyle="--", linewidth=1.2, alpha=0.8,
             label=f"Default LGBM (R²={lgbm_data['metrics_default']['r2']:.3f}, MAE={lgbm_data['metrics_default']['mae']:.2f})")
    ax1.plot(timestamps, lgbm_data["tuned"], color="#38bdf8", linewidth=1.8,
             label=f"Optuna Tuned LGBM (R²={lgbm_data['metrics_tuned']['r2']:.3f}, MAE={lgbm_data['metrics_tuned']['mae']:.2f})")
    ax1.set_ylabel(f"Power ({cfg['unit']})", color="#e2e8f0", fontweight="bold")
    ax1.set_title("LightGBM Regressor (Default vs. Tuned)", color="#f8fafc", fontsize=10, fontweight="bold", pad=6)
    ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)

    # Subplot 2: XGBoost
    ax2 = axes[1]
    ax2.plot(timestamps, actual, color="#cbd5e1", linewidth=1.6, label=f"Actual Ground Truth ({cfg['unit']})")
    xgb_data = data["models"]["XGB"]
    ax2.plot(timestamps, xgb_data["default"], color="#94a3b8", linestyle="--", linewidth=1.2, alpha=0.8,
             label=f"Default XGB (R²={xgb_data['metrics_default']['r2']:.3f}, MAE={xgb_data['metrics_default']['mae']:.2f})")
    ax2.plot(timestamps, xgb_data["tuned"], color="#fbbf24", linewidth=1.8,
             label=f"Optuna Tuned XGB (R²={xgb_data['metrics_tuned']['r2']:.3f}, MAE={xgb_data['metrics_tuned']['mae']:.2f})")
    ax2.set_ylabel(f"Power ({cfg['unit']})", color="#e2e8f0", fontweight="bold")
    ax2.set_xlabel("Operational Timeline (15-min Intervals)", color="#e2e8f0", fontweight="bold")
    ax2.set_title("XGBoost Regressor (Default vs. Tuned)", color="#f8fafc", fontsize=10, fontweight="bold", pad=6)
    ax2.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved prediction overlay to: {save_path}")
    
    if ARTIFACTS_DIR.exists():
        dst = ARTIFACTS_DIR / f"tuning_overlay_{target_key}.png"
        shutil.copy2(save_path, dst)
        print(f"Copied overlay to artifacts: {dst.name}")

def main():
    parser = argparse.ArgumentParser(description="Systematic Forecaster Hyperparameter Optimization")
    parser.add_argument("--target", choices=["pv", "con", "ev", "caltech", "all"], default="all", help="Target asset")
    parser.add_argument("--model", choices=["LGBM", "XGB", "both"], default="both", help="Model family")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials per model (default: 20)")
    parser.add_argument("--eval-days", type=int, default=7, help="Evaluation horizon on unseen test set in days (default: 7)")
    args = parser.parse_args()
    
    targets = ["pv", "con", "ev", "caltech"] if args.target == "all" else [args.target]
    models = ["LGBM", "XGB"] if args.model == "both" else [args.model]
    
    run_tuning_benchmark(targets=targets, models=models, n_trials=args.trials, eval_days=args.eval_days)

if __name__ == "__main__":
    main()
