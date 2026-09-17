"""
Model Predictive Control (MPC) Microgrid Optimizer
===================================================
Solves 24-hour lookahead non-linear operational dispatch for microgrid assets:
  1. Supermarket Facility Load (c_gen) + Solar PV
  2. Lidl Commercial EV Chargers (c_ev) + Solar PV
  3. Caltech Campus EV Fleet (c_ev) + Solar PV

Formulation:
  - Discrete-time NLP over N_p = 96 steps (15-min intervals, 24-hour horizon)
  - Pyomo algebraic modeling with IPOPT interior-point solver
  - Hardware sizing parameters (E_B_max, P_B_max) imported from battery_sizing_results.json
  - Day-ahead forecast inputs (P_PV^f, P_load^f) from trained ML forecaster engine
  - Dynamic calendar aging (C_SoC) & 4th-degree cyclic DoD wear (C_DoD)
  - Piecewise State-of-Charge power derating constraints
  - Real-time forecast mismatch analysis (P_Error) and operational case classification
"""

import os
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

# Pyomo NLP modeling
import pyomo.environ as pyo

# Project imports
SRC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SRC_DIR.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results_mpc"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from forecaster_engine import load_and_preprocess, get_models, LAGS, STEPS_PER_DAY
from skforecast.recursive import ForecasterRecursive
from rte_wholesale_market import RTEWholesaleMarketClient

# Default Hardware Sizing Fallback (overridden by battery_sizing_results.json)
DEFAULT_SIZING = {
    "supermarket": {"E_B_max": 264.46, "P_B_max": 69.31, "name": "Supermarket Facility (c_gen)"},
    "ev": {"E_B_max": 1.29, "P_B_max": 0.65, "name": "Lidl Commercial EV Chargers (c_ev)"},
    "caltech_ev": {"E_B_max": 31.36, "P_B_max": 14.07, "name": "Caltech Campus EV Fleet (c_ev)"}
}

def load_battery_sizing(mode: str):
    """Loads established hardware sizing parameters from battery_sizing_results.json."""
    json_path = SRC_DIR / "battery_sizing_results.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if mode in data:
                return data[mode]
        except Exception as e:
            print(f"Warning: Could not read {json_path}: {e}")
    return DEFAULT_SIZING.get(mode, DEFAULT_SIZING["supermarket"])

def fetch_or_load_tariffs(year: int, start_date: str, steps: int = 96):
    """Loads historic electricity wholesale prices + TURPE & taxes, interpolated to 15-min."""
    cache_path = DATA_DIR / f"historic_wholesale_prices_{year}.parquet"
    client = RTEWholesaleMarketClient()
    if not cache_path.exists():
        try:
            client.fetch_historic_prices_for_year(year)
        except Exception as e:
            print(f"Warning: Failed to fetch {year} wholesale prices ({e}). Using synthetic ToU.")
            
    if cache_path.exists():
        df_p = pd.read_parquet(cache_path)
        df_p["Date"] = pd.to_datetime(df_p["start_time"]).dt.tz_localize(None)
        df_p = df_p.sort_values("Date").drop_duplicates("Date")
        
        # Build 15-min time index for the requested evaluation window
        start_ts = pd.to_datetime(start_date)
        dt_idx = pd.date_range(start_ts, periods=steps, freq="15min")
        df_eval = pd.DataFrame({"Date": dt_idx})
        
        df_merged = pd.merge_asof(
            df_eval,
            df_p[["Date", "total_price_eur_kwh"]],
            on="Date",
            direction="nearest"
        )
        tariffs = df_merged["total_price_eur_kwh"].fillna(0.12).values
        return tariffs, dt_idx
    else:
        # Synthetic French Time-of-Use tariff fallback
        start_ts = pd.to_datetime(start_date)
        dt_idx = pd.date_range(start_ts, periods=steps, freq="15min")
        hours = dt_idx.hour
        tariffs = np.where((hours >= 17) & (hours <= 21), 0.22, 0.12)
        return tariffs, dt_idx

def get_forecast_and_actual_trajectories(mode: str, start_date: str, horizon: int = 96):
    """
    Retrieves the 24-hour day-ahead forecast and ground truth actuals for both PV and Load.
    Uses trained LightGBM or XGBoost models as determined by the forecaster engine.
    """
    print(f"\n--- Generating Day-Ahead Forecasts for Mode: {mode.upper()} ({start_date}) ---")
    
    # Target identifiers
    load_target_key = "con" if mode == "supermarket" else ("ev" if mode == "ev" else "caltech")
    
    # 1. Load PV Forecast and Actual
    if mode == "caltech_ev":
        # For Caltech 2022 timeline, train PV up to Oct 2022 and predict Nov 1, 2022
        df_pv, _, _, _, target_col_pv, exog_cols_pv = load_and_preprocess("pv")
        train_pv = df_pv[df_pv.index < "2022-10-01"]
        val_pv = df_pv[(df_pv.index >= "2022-10-01") & (df_pv.index < start_date)]
        test_pv = df_pv[df_pv.index >= start_date]
        
        forecaster_pv = ForecasterRecursive(estimator=get_models()["XGB"], lags=LAGS)
        forecaster_pv.fit(y=train_pv[target_col_pv], exog=train_pv[exog_cols_pv])
        lw_pv = val_pv[target_col_pv].iloc[-max(LAGS):]
        pv_forecast = forecaster_pv.predict(steps=horizon, last_window=lw_pv, exog=test_pv[exog_cols_pv].iloc[:horizon]).values
        pv_forecast = np.clip(pv_forecast, 0.0, None)
        pv_actual = test_pv[target_col_pv].iloc[:horizon].values
        time_index = test_pv.index[:horizon]
    else:
        # Standard 2024 test split starting at start_date (2024-10-01)
        df_pv, train_pv, val_pv, test_pv, target_col_pv, exog_cols_pv = load_and_preprocess("pv")
        forecaster_pv = ForecasterRecursive(estimator=get_models()["XGB"], lags=LAGS)
        forecaster_pv.fit(y=train_pv[target_col_pv], exog=train_pv[exog_cols_pv])
        lw_pv = val_pv[target_col_pv].iloc[-max(LAGS):]
        pv_forecast = forecaster_pv.predict(steps=horizon, last_window=lw_pv, exog=test_pv[exog_cols_pv].iloc[:horizon]).values
        pv_forecast = np.clip(pv_forecast, 0.0, None)
        pv_actual = test_pv[target_col_pv].iloc[:horizon].values
        time_index = test_pv.index[:horizon]
        
    # 2. Load Asset Demand Forecast and Actual
    df_load, train_load, val_load, test_load, target_col_load, exog_cols_load = load_and_preprocess(load_target_key)
    best_regressor = "LGBM" if load_target_key in ["con", "caltech"] else "XGB"
    forecaster_load = ForecasterRecursive(estimator=get_models()[best_regressor], lags=LAGS)
    forecaster_load.fit(y=train_load[target_col_load], exog=train_load[exog_cols_load])
    lw_load = val_load[target_col_load].iloc[-max(LAGS):]
    load_forecast = forecaster_load.predict(steps=horizon, last_window=lw_load, exog=test_load[exog_cols_load].iloc[:horizon]).values
    load_forecast = np.clip(load_forecast, 0.0, None)
    if load_target_key == "ev":
        load_forecast = np.where(load_forecast < 0.25, 0.08, load_forecast)
    load_actual = test_load[target_col_load].iloc[:horizon].values
    
    print(f"Successfully generated {horizon}-step trajectories:")
    print(f"  PV Generation  : Forecast Mean={pv_forecast.mean():.2f} kW | Actual Mean={pv_actual.mean():.2f} kW")
    print(f"  Asset Load     : Forecast Mean={load_forecast.mean():.2f} kW | Actual Mean={load_actual.mean():.2f} kW")
    
    return time_index, pv_forecast, pv_actual, load_forecast, load_actual

def solve_mpc_optimization(
    mode: str,
    pv_forecast: np.ndarray,
    load_forecast: np.ndarray,
    tariffs: np.ndarray,
    sizing_params: dict,
    dt: float = 0.25,
    soc_init: float = 0.50
):
    """
    Formulates and solves the operational Non-Linear Model Predictive Control (NLP)
    using Pyomo and IPOPT.
    """
    N_p = len(pv_forecast)
    E_B_max = float(sizing_params["E_B_max"])
    P_B_max = float(sizing_params["P_B_max"])
    
    # System Technical Efficiencies
    eta_pvinverter = 0.99
    eta_mppt = 0.99
    eta_acdc = 0.99
    eta_dcac = 0.99
    eta_c = 0.95
    eta_d = 0.95
    
    # Operating SoC and Inverter Derating Parameters
    SoC_min = 0.20
    SoC_max = 0.90
    D_ch = 0.90
    D_dis = 0.20
    
    # Economic & Degradation Parameters
    C_0 = 500.0 * E_B_max + 100.0 * P_B_max  # Battery replacement capital cost [EUR]
    CF_max = 0.20                            # Max allowable capacity fade (20%)
    alpha = 0.05
    beta = 0.01
    
    # Build Pyomo Non-Linear Model
    model = pyo.ConcreteModel(name=f"MPC_{mode}")
    model.K = pyo.RangeSet(0, N_p - 1)
    
    # Decision Variables
    model.P_grid = pyo.Var(model.K, domain=pyo.NonNegativeReals, initialize=5.0)
    model.P_ch = pyo.Var(model.K, domain=pyo.NonNegativeReals, bounds=(0, P_B_max), initialize=0.0)
    model.P_dis = pyo.Var(model.K, domain=pyo.NonNegativeReals, bounds=(0, P_B_max), initialize=0.0)
    model.E_B = pyo.Var(model.K, domain=pyo.NonNegativeReals, bounds=(SoC_min * E_B_max, SoC_max * E_B_max), initialize=soc_init * E_B_max)
    model.P_curt = pyo.Var(model.K, domain=pyo.NonNegativeReals, initialize=0.0)
    
    # Constraints
    # 1. DC Bus Power Balance:
    # (P_PV^f * eta_mppt / eta_pvinverter) + P_grid * eta_acdc + P_dis = (P_load^f / eta_dcac) + P_ch + P_curt
    def power_balance_rule(m, k):
        pv_gen = (pv_forecast[k] / eta_pvinverter) * eta_mppt
        load_dem = load_forecast[k] / eta_dcac
        return pv_gen + m.P_grid[k] * eta_acdc + m.P_dis[k] == load_dem + m.P_ch[k] + m.P_curt[k]
    model.PowerBalance = pyo.Constraint(model.K, rule=power_balance_rule)
    
    # 2. Battery Energy Dynamics:
    def energy_dynamics_rule(m, k):
        delta_e = (m.P_ch[k] * eta_c - m.P_dis[k] / eta_d) * dt
        if k == 0:
            return m.E_B[k] == soc_init * E_B_max + delta_e
        else:
            return m.E_B[k] == m.E_B[k-1] + delta_e
    model.EnergyDynamics = pyo.Constraint(model.K, rule=energy_dynamics_rule)
    
    # 3. Dynamic Inverter Power Derating as function of SoC:
    # P_ch <= P_B_max / (1 - D_ch) * (1 - SoC)
    def derating_charge_rule(m, k):
        soc = m.E_B[k] / E_B_max
        return m.P_ch[k] <= (P_B_max / (1.0 - D_ch)) * (1.0 - soc)
    model.DeratingCharge = pyo.Constraint(model.K, rule=derating_charge_rule)
    
    # P_dis <= P_B_max / D_dis * SoC
    def derating_discharge_rule(m, k):
        soc = m.E_B[k] / E_B_max
        return m.P_dis[k] <= (P_B_max / D_dis) * soc
    model.DeratingDischarge = pyo.Constraint(model.K, rule=derating_discharge_rule)
    
    # Objective Function: Min Sum [ Grid_Cost + C_SoC + C_DoD + 1e-4 * P_curt ]
    def objective_rule(m):
        cost_terms = []
        for k in m.K:
            # 1. Grid Import Cost
            grid_cost = m.P_grid[k] * dt * tariffs[k]
            
            # 2. Calendar Aging Cost:
            # C_SoC = C_0 * (alpha * SoC - beta) / (CF_max * 15 * 8760) * dt
            soc = m.E_B[k] / E_B_max
            c_soc = C_0 * (alpha * soc - beta) / (CF_max * 15.0 * 8760.0) * dt
            
            # 3. Cyclic Depth-of-Discharge Wear Cost:
            # delta_L = P_dis * dt / E_B_max
            # N_tot = (1.06*delta_L^4 - 2.80*delta_L^3 + 2.66*delta_L^2 - 1.07*delta_L + 0.17)*1e5
            # C_DoD = C_0 * delta_L / N_tot
            delta_L = (m.P_dis[k] * dt) / E_B_max
            n_tot = (1.06 * (delta_L**4) - 2.80 * (delta_L**3) + 2.66 * (delta_L**2) - 1.07 * delta_L + 0.17) * 1e5
            c_dod = C_0 * (delta_L / n_tot)
            
            # PV Curtailment soft penalty
            curt_cost = 1e-4 * m.P_curt[k]
            
            cost_terms.append(grid_cost + c_soc + c_dod + curt_cost)
        return sum(cost_terms)
    model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    
    # Solve with IPOPT
    ipopt_exe = shutil.which('ipopt') or r'C:\Users\alesk\miniconda3\Library\bin\ipopt.EXE'
    solver = pyo.SolverFactory('ipopt', executable=ipopt_exe)
    solver.options['max_iter'] = 3000
    solver.options['tol'] = 1e-6
    solver.options['print_level'] = 0  # Clean output
    
    print(f"\nSolving Pyomo NLP ({N_p} steps) with IPOPT solver...")
    t0 = time.time()
    results = solver.solve(model, tee=False)
    solve_time = time.time() - t0
    print(f"IPOPT Solve completed in {solve_time:.3f} s (Status: {results.solver.status}, Termination: {results.solver.termination_condition})")
    
    # Extract scheduled trajectory
    p_grid_sch = np.array([pyo.value(model.P_grid[k]) for k in model.K])
    p_ch_sch = np.array([pyo.value(model.P_ch[k]) for k in model.K])
    p_dis_sch = np.array([pyo.value(model.P_dis[k]) for k in model.K])
    e_b_sch = np.array([pyo.value(model.E_B[k]) for k in model.K])
    soc_sch = e_b_sch / E_B_max
    p_curt_sch = np.array([pyo.value(model.P_curt[k]) for k in model.K])
    
    # Compute interval degradation costs
    c_soc_arr = np.array([C_0 * (alpha * soc_sch[k] - beta) / (CF_max * 15.0 * 8760.0) * dt for k in range(N_p)])
    delta_L_arr = (p_dis_sch * dt) / E_B_max
    n_tot_arr = (1.06 * (delta_L_arr**4) - 2.80 * (delta_L_arr**3) + 2.66 * (delta_L_arr**2) - 1.07 * delta_L_arr + 0.17) * 1e5
    c_dod_arr = np.array([C_0 * (delta_L_arr[k] / n_tot_arr[k]) for k in range(N_p)])
    
    total_grid_cost = float(np.sum(p_grid_sch * dt * tariffs))
    total_soc_cost = float(np.sum(c_soc_arr))
    total_dod_cost = float(np.sum(c_dod_arr))
    total_deg_cost = total_soc_cost + total_dod_cost
    total_obj = total_grid_cost + total_deg_cost
    
    dispatch_df = pd.DataFrame({
        "step": range(N_p),
        "P_PV_forecast_kW": pv_forecast,
        "P_load_forecast_kW": load_forecast,
        "tariff_eur_kwh": tariffs,
        "P_grid_sch_kW": p_grid_sch,
        "P_ch_sch_kW": p_ch_sch,
        "P_dis_sch_kW": p_dis_sch,
        "E_B_sch_kWh": e_b_sch,
        "SoC_sch": soc_sch,
        "P_curt_sch_kW": p_curt_sch,
        "C_SoC_eur": c_soc_arr,
        "C_DoD_eur": c_dod_arr,
        "C_deg_total_eur": c_soc_arr + c_dod_arr
    })
    
    summary = {
        "solve_time_sec": round(solve_time, 3),
        "total_grid_cost_eur": round(total_grid_cost, 2),
        "total_calendar_aging_cost_eur": round(total_soc_cost, 4),
        "total_cyclic_wear_cost_eur": round(total_dod_cost, 4),
        "total_battery_degradation_cost_eur": round(total_deg_cost, 4),
        "total_mpc_operational_cost_eur": round(total_obj, 2),
        "grid_energy_imported_kwh": round(float(np.sum(p_grid_sch * dt)), 2),
        "battery_discharged_kwh": round(float(np.sum(p_dis_sch * dt)), 2),
        "battery_charged_kwh": round(float(np.sum(p_ch_sch * dt)), 2),
        "curtailed_pv_kwh": round(float(np.sum(p_curt_sch * dt)), 2),
        "final_soc": round(float(soc_sch[-1]), 4)
    }
    
    return dispatch_df, summary

def analyze_forecast_mismatch(
    dispatch_df: pd.DataFrame,
    pv_actual: np.ndarray,
    load_actual: np.ndarray,
    sizing_params: dict,
    dt: float = 0.25
):
    """
    Evaluates real-time mismatch between forecast schedule and ground-truth measurements.
    Categorizes intervals into Deficit (Cases 1, 2, 3) and Surplus (Cases 1, 2, 3) scenarios.
    """
    eta_mppt = 0.99
    eta_dcac = 0.99
    
    pv_f = dispatch_df["P_PV_forecast_kW"].values
    load_f = dispatch_df["P_load_forecast_kW"].values
    
    # Deviations (Actual - Forecast)
    delta_pv = pv_actual - pv_f
    delta_load = load_actual - load_f
    
    # Net Microgrid Error on DC Bus
    p_error = (delta_pv * eta_mppt) - (delta_load / eta_dcac)
    
    # Scenario Classification
    scenario_types = []
    case_details = []
    
    # Real-time Compensated Simulation:
    # Tracks how a real-time secondary controller handles the error mismatch:
    p_ch_rt = []
    p_dis_rt = []
    p_grid_rt = []
    p_curt_rt = []
    soc_rt = []
    
    E_B_max = sizing_params["E_B_max"]
    P_B_max = sizing_params["P_B_max"]
    SoC_min = 0.20
    SoC_max = 0.90
    eta_c = 0.95
    eta_d = 0.95
    
    curr_e = 0.50 * E_B_max
    
    for k in range(len(p_error)):
        dpv = delta_pv[k]
        dld = delta_load[k]
        err = p_error[k]
        
        # Categorize
        if err < -1e-3:
            s_type = "Deficit"
            if dpv < 0 and dld > 0:
                c_det = "Case 1 (Double Deficit: PV < Fcast & Load > Fcast)"
            elif dpv < 0 and dld <= 0:
                c_det = "Case 2 (PV Deficit Dominates: Both Lower)"
            else:
                c_det = "Case 3 (Load Surge Dominates: Both Higher)"
        elif err > 1e-3:
            s_type = "Surplus"
            if dpv > 0 and dld < 0:
                c_det = "Case 1 (Double Surplus: PV > Fcast & Load < Fcast)"
            elif dpv <= 0 and dld < 0:
                c_det = "Case 2 (Load Drop Dominates: Both Lower)"
            else:
                c_det = "Case 3 (PV Gain Dominates: Both Higher)"
        else:
            s_type = "Balanced"
            c_det = "Exact Forecast Match"
            
        scenario_types.append(s_type)
        case_details.append(c_det)
        
        # Real-time power balance compensation logic
        sch_dis = dispatch_df["P_dis_sch_kW"].iloc[k]
        sch_ch = dispatch_df["P_ch_sch_kW"].iloc[k]
        sch_grid = dispatch_df["P_grid_sch_kW"].iloc[k]
        
        if err < 0:  # Missing power on DC bus
            deficit = abs(err)
            curr_soc = curr_e / E_B_max
            # Can battery discharge more?
            max_avail_dis = min(P_B_max, (P_B_max / 0.20) * curr_soc)
            add_dis = min(deficit, max(0.0, max_avail_dis - sch_dis), (curr_e - SoC_min * E_B_max) * eta_d / dt)
            dis_actual = sch_dis + add_dis
            rem_deficit = deficit - add_dis
            grid_actual = sch_grid + rem_deficit / 0.99
            ch_actual = max(0.0, sch_ch)
            curt_actual = 0.0
        else:  # Surplus power on DC bus
            surplus = err
            curr_soc = curr_e / E_B_max
            # Can battery charge more?
            max_avail_ch = min(P_B_max, (P_B_max / (1.0 - 0.90)) * (1.0 - curr_soc))
            add_ch = min(surplus, max(0.0, max_avail_ch - sch_ch), (SoC_max * E_B_max - curr_e) / (eta_c * dt))
            ch_actual = sch_ch + add_ch
            rem_surplus = surplus - add_ch
            curt_actual = rem_surplus
            dis_actual = max(0.0, sch_dis)
            grid_actual = sch_grid
            
        # Update RT battery state
        curr_e = curr_e + (ch_actual * eta_c - dis_actual / eta_d) * dt
        curr_e = max(SoC_min * E_B_max, min(SoC_max * E_B_max, curr_e))
        
        p_ch_rt.append(ch_actual)
        p_dis_rt.append(dis_actual)
        p_grid_rt.append(grid_actual)
        p_curt_rt.append(curt_actual)
        soc_rt.append(curr_e / E_B_max)
        
    analysis_df = dispatch_df.copy()
    analysis_df["P_PV_actual_kW"] = pv_actual
    analysis_df["P_load_actual_kW"] = load_actual
    analysis_df["Delta_P_PV_kW"] = delta_pv
    analysis_df["Delta_P_load_kW"] = delta_load
    analysis_df["P_Error_kW"] = p_error
    analysis_df["Scenario"] = scenario_types
    analysis_df["Case_Detail"] = case_details
    analysis_df["P_grid_rt_comp_kW"] = p_grid_rt
    analysis_df["P_ch_rt_comp_kW"] = p_ch_rt
    analysis_df["P_dis_rt_comp_kW"] = p_dis_rt
    analysis_df["P_curt_rt_comp_kW"] = p_curt_rt
    analysis_df["SoC_rt_comp"] = soc_rt
    
    # Error Statistics
    counts = pd.Series(scenario_types).value_counts().to_dict()
    case_counts = pd.Series(case_details).value_counts().to_dict()
    
    mismatch_stats = {
        "mean_p_error_kw": round(float(np.mean(p_error)), 2),
        "mae_p_error_kw": round(float(np.mean(np.abs(p_error))), 2),
        "max_surplus_kw": round(float(np.max(p_error)), 2),
        "max_deficit_kw": round(float(abs(np.min(p_error))), 2),
        "deficit_intervals": int(counts.get("Deficit", 0)),
        "surplus_intervals": int(counts.get("Surplus", 0)),
        "balanced_intervals": int(counts.get("Balanced", 0)),
        "case_distribution": case_counts
    }
    
    return analysis_df, mismatch_stats

def plot_mpc_results(
    mode: str,
    time_index: pd.DatetimeIndex,
    analysis_df: pd.DataFrame,
    sizing_params: dict,
    summary: dict,
    mismatch_stats: dict,
    save_path: Path
):
    """
    Generates a 4-panel publication-quality dashboard visualization:
      Panel 1: Scheduled DC Power Balance (PV, Load, Grid, Battery)
      Panel 2: Battery State of Charge (SoC) & Inverter Derating Constraints
      Panel 3: Dynamic Battery Degradation Penalties (Calendar C_SoC vs Cyclic C_DoD)
      Panel 4: Real-Time Forecast Mismatch P_Error & Operational Case Scenarios
    """
    hours = np.arange(len(time_index)) / 4.0  # 0 to 24 hours
    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.4, 1.3, 1.6]})
    
    # ---------------- PANEL 1: POWER BALANCE ----------------
    ax1 = axes[0]
    ax1.plot(hours, analysis_df["P_load_forecast_kW"], label="Forecast Load", color="#ef4444", linewidth=2.0, linestyle="-")
    ax1.plot(hours, analysis_df["P_PV_forecast_kW"], label="Forecast PV Gen", color="#f59e0b", linewidth=2.0, linestyle="-")
    ax1.plot(hours, analysis_df["P_grid_sch_kW"], label="Scheduled Grid Import", color="#3b82f6", linewidth=2.0, linestyle="--")
    ax1.plot(hours, analysis_df["P_ch_sch_kW"], label="Scheduled Battery Charge", color="#10b981", linewidth=1.8, linestyle=":")
    ax1.plot(hours, analysis_df["P_dis_sch_kW"], label="Scheduled Battery Discharge", color="#8b5cf6", linewidth=1.8, linestyle="-.")
    if analysis_df["P_curt_sch_kW"].max() > 0.01:
        ax1.plot(hours, analysis_df["P_curt_sch_kW"], label="PV Curtailment", color="#6b7280", linewidth=1.5, linestyle="--")
        
    ax1.set_ylabel("Power (kW)", fontsize=10, fontweight="bold")
    ax1.set_title(
        f"Model Predictive Control (24h Day-Ahead Schedule): {sizing_params['name']}\n"
        f"Optimal Sizing: {sizing_params['E_B_max']} kWh / {sizing_params['P_B_max']} kW | "
        f"Total Grid Cost: EUR {summary['total_grid_cost_eur']:.2f} | "
        f"Battery Degradation: EUR {summary['total_battery_degradation_cost_eur']:.4f} | "
        f"Solve Time: {summary['solve_time_sec']:.2f}s (IPOPT)",
        fontsize=12, fontweight="bold", pad=8
    )
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", ncol=3, fontsize=9, framealpha=0.9)
    
    # ---------------- PANEL 2: BATTERY SOC & DERATING ----------------
    ax2 = axes[1]
    ax2.plot(hours, analysis_df["SoC_sch"] * 100.0, label="Scheduled SoC (%)", color="#0284c7", linewidth=2.2)
    ax2.plot(hours, analysis_df["SoC_rt_comp"] * 100.0, label="RT Compensated SoC (%)", color="#0369a1", linewidth=1.8, linestyle="--", alpha=0.8)
    ax2.axhline(90.0, color="#ef4444", linestyle=":", label="SoC Max (90%)", linewidth=1.2)
    ax2.axhline(20.0, color="#ef4444", linestyle=":", label="SoC Min (20%)", linewidth=1.2)
    ax2.fill_between(hours, 20.0, 90.0, color="#0284c7", alpha=0.08)
    ax2.set_ylabel("State of Charge (%)", fontsize=10, fontweight="bold")
    ax2.set_ylim(10, 100)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", ncol=4, fontsize=9, framealpha=0.9)
    
    # ---------------- PANEL 3: DYNAMIC DEGRADATION PENALTIES ----------------
    ax3 = axes[2]
    ax3.plot(hours, analysis_df["C_SoC_eur"] * 1000.0, label="Calendar Aging Cost C_SoC (mEUR/15m)", color="#d97706", linewidth=1.8)
    ax3.plot(hours, analysis_df["C_DoD_eur"] * 1000.0, label="Cyclic DoD Wear Cost C_DoD (mEUR/15m)", color="#dc2626", linewidth=1.8)
    ax3.set_ylabel("Wear Cost (mEUR/step)", fontsize=10, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=9, framealpha=0.9)
    
    # ---------------- PANEL 4: FORECAST MISMATCH & CASES ----------------
    ax4 = axes[3]
    p_err = analysis_df["P_Error_kW"].values
    surplus_bars = np.where(p_err > 0, p_err, 0)
    deficit_bars = np.where(p_err < 0, p_err, 0)
    
    ax4.bar(hours, surplus_bars, width=0.22, color="#10b981", alpha=0.8, label=f"Surplus Error (PV > Load Mismatch: {mismatch_stats['surplus_intervals']} steps)")
    ax4.bar(hours, deficit_bars, width=0.22, color="#f43f5e", alpha=0.8, label=f"Deficit Error (PV < Load Mismatch: {mismatch_stats['deficit_intervals']} steps)")
    ax4.axhline(0, color="#475569", linewidth=1.0)
    ax4.set_ylabel("P_Error (kW)", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Hour of Operation (0 to 24h)", fontsize=11, fontweight="bold")
    ax4.set_title(
        f"Real-Time Forecast Error Realization: Net DC Mismatch $P_{{Error}}(k) = (\\Delta P_{{PV}} \\cdot \\eta_{{mppt}}) - (\\Delta P_{{load}} / \\eta_{{dc/ac}})$\n"
        f"Mean Mismatch: {mismatch_stats['mean_p_error_kw']:+.2f} kW | "
        f"Max Surplus: +{mismatch_stats['max_surplus_kw']:.2f} kW | "
        f"Max Deficit: -{mismatch_stats['max_deficit_kw']:.2f} kW | "
        f"Deficit: {mismatch_stats['deficit_intervals']} / Surplus: {mismatch_stats['surplus_intervals']} / Balanced: {mismatch_stats['balanced_intervals']}",
        fontsize=11, fontweight="bold", pad=6
    )
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right", ncol=2, fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved publication-quality MPC chart to {save_path}")

def run_mpc_pipeline(mode: str, start_date: str = None, horizon: int = 96, soc_init: float = 0.50):
    """Complete execution pipeline for an individual microgrid MPC asset."""
    print(f"\n=================================================================")
    print(f"  RUNNING MODEL PREDICTIVE CONTROL OPTIMIZER: {mode.upper()}")
    print(f"=================================================================")
    
    # 1. Hardware Sizing Parameters
    sizing = load_battery_sizing(mode)
    print(f"Loaded Hardware Sizing Configuration:")
    print(f"  Asset Name        : {sizing['name']}")
    print(f"  Battery Capacity  : {sizing['E_B_max']} kWh")
    print(f"  Inverter Power    : {sizing['P_B_max']} kW")
    print(f"  Annual Cost Basis : EUR {sizing.get('annualized_cost_eur', 0):,.2f}")
    
    # 2. Horizon Date Determination
    if start_date is None:
        start_date = "2022-11-01" if mode == "caltech_ev" else "2024-10-01"
    year = pd.to_datetime(start_date).year
    
    # 3. Forecast and Ground Truth Trajectories
    time_index, pv_f, pv_act, load_f, load_act = get_forecast_and_actual_trajectories(mode, start_date, horizon)
    
    # 4. Electricity Tariffs
    tariffs, dt_idx = fetch_or_load_tariffs(year, start_date, steps=horizon)
    
    # 5. Non-Linear MPC Optimization
    dispatch_df, summary = solve_mpc_optimization(
        mode=mode,
        pv_forecast=pv_f,
        load_forecast=load_f,
        tariffs=tariffs,
        sizing_params=sizing,
        dt=0.25,
        soc_init=soc_init
    )
    
    # 6. Real-Time Mismatch and Case Analysis
    analysis_df, mismatch_stats = analyze_forecast_mismatch(
        dispatch_df=dispatch_df,
        pv_actual=pv_act,
        load_actual=load_act,
        sizing_params=sizing,
        dt=0.25
    )
    
    # 7. Persistence & Visualizations
    results_json_path = RESULTS_DIR / f"mpc_metrics_{mode}.json"
    results_csv_path = RESULTS_DIR / f"mpc_timeseries_{mode}.csv"
    plot_path = RESULTS_DIR / f"mpc_schedule_{mode}.png"
    
    # Export full timeseries
    analysis_df["timestamp"] = time_index.strftime("%Y-%m-%d %H:%M:%S")
    analysis_df.to_csv(results_csv_path, index=False)
    
    combined_metrics = {
        "mode": mode,
        "name": sizing["name"],
        "date": start_date,
        "horizon_steps": horizon,
        "horizon_hours": horizon * 0.25,
        "sizing": sizing,
        "summary": summary,
        "mismatch_stats": mismatch_stats,
        "artifacts": {
            "timeseries_csv": str(results_csv_path.name),
            "plot_png": str(plot_path.name)
        }
    }
    
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, indent=2)
    print(f"Saved MPC summary metrics to {results_json_path}")
    
    # Generate 4-panel dashboard plot
    plot_mpc_results(mode, time_index, analysis_df, sizing, summary, mismatch_stats, plot_path)
    
    print(f"\n--- MPC OPTIMIZATION COMPLETE [{mode.upper()}] ---")
    print(f"  Grid Imported Energy : {summary['grid_energy_imported_kwh']:.2f} kWh (Cost: EUR {summary['total_grid_cost_eur']:.2f})")
    print(f"  Battery Discharge    : {summary['battery_discharged_kwh']:.2f} kWh")
    print(f"  Calendar Aging Cost  : EUR {summary['total_calendar_aging_cost_eur']:.4f}")
    print(f"  Cyclic Wear Cost     : EUR {summary['total_cyclic_wear_cost_eur']:.4f}")
    print(f"  Total Degradation    : EUR {summary['total_battery_degradation_cost_eur']:.4f}")
    print(f"  Mean Error Mismatch  : {mismatch_stats['mean_p_error_kw']:+.2f} kW (MAE: {mismatch_stats['mae_p_error_kw']:.2f} kW)")
    print(f"  Operational Cases    : {mismatch_stats['deficit_intervals']} Deficit / {mismatch_stats['surplus_intervals']} Surplus / {mismatch_stats['balanced_intervals']} Balanced")
    
    return combined_metrics

def main():
    parser = argparse.ArgumentParser(description="Operational Non-Linear MPC Microgrid Optimizer")
    parser.add_argument(
        "--mode",
        choices=["supermarket", "ev", "caltech_ev"],
        default="supermarket",
        help="Target asset mode: 'supermarket', 'ev', or 'caltech_ev'"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Evaluation start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=96,
        help="Prediction horizon in steps (default: 96 = 24h)"
    )
    parser.add_argument(
        "--soc-init",
        type=float,
        default=0.50,
        help="Initial battery state of charge (default: 0.50)"
    )
    args = parser.parse_args()
    
    run_mpc_pipeline(
        mode=args.mode,
        start_date=args.date,
        horizon=args.horizon,
        soc_init=args.soc_init
    )

if __name__ == "__main__":
    main()
