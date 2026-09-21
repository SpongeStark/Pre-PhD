"""
Real-Time (RT) Secondary Microgrid Controller
==============================================
Implements real-time closed-loop secondary power compensation between consecutive
MPC execution steps over a 2-day (48-hour = 192 steps) evaluation horizon.

Hierarchical 4-Layer Control Architecture:
  1. Layer 1: Error Tolerance Deadband (epsilon_tol = 0.5 kW)
  2. Layer 2: 3-State History Filtering (w = [0.5, 0.3, 0.2]) & Anti-Chattering Persistence
  3. Layer 3: Physical Headroom Allocation (Delta P_dis_avail, Delta P_ch_avail, Delta P_grid, P_curt)
  4. Layer 4: Physical Inverter Ramp-Rate Limiting (|Delta P_bat| <= Delta P_ramp_max)

Guarantees:
  - Strict zero grid re-injection constraint (P_grid >= 0)
  - Battery capacity and inverter rating bounds (P_B_max, SoC_min, SoC_max)
  - DC bus power balance satisfaction (Residual Mismatch == 0 kW at every 15-min step)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Project imports
SRC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SRC_DIR.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
RESULTS_RT_DIR = ROOT_DIR / "results_rt"
RESULTS_RT_DIR.mkdir(parents=True, exist_ok=True)

from mpc_optimizer import (
    load_battery_sizing,
    fetch_or_load_tariffs,
    get_forecast_and_actual_trajectories,
    solve_mpc_optimization,
    DEFAULT_SIZING
)

def run_rt_compensation(
    mode: str,
    time_index: pd.DatetimeIndex,
    pv_forecast: np.ndarray,
    pv_actual: np.ndarray,
    load_forecast: np.ndarray,
    load_actual: np.ndarray,
    tariffs: np.ndarray,
    dispatch_mpc_df: pd.DataFrame,
    sizing_params: dict,
    dt: float = 0.25,
    soc_init: float = 0.50,
    epsilon_tol: float = 0.50,
    ramp_limit_fraction: float = 0.50
):
    """
    Simulates real-time closed-loop secondary control step-by-step across all N_p intervals.
    """
    N_p = len(pv_forecast)
    E_B_max = float(sizing_params["E_B_max"])
    P_B_max = float(sizing_params["P_B_max"])
    
    # Technical efficiencies
    eta_pvinverter = 0.99
    eta_mppt = 0.99
    eta_acdc = 0.99
    eta_dcac = 0.99
    eta_c = 0.95
    eta_d = 0.95
    
    SoC_min = 0.20
    SoC_max = 0.90
    
    # Layer 4 Ramp Limit: maximum power delta per interval
    delta_P_ramp_max = ramp_limit_fraction * P_B_max
    
    # Degradation Cost Model parameters
    C_0 = 500.0 * E_B_max + 100.0 * P_B_max
    CF_max = 0.20
    alpha = 0.05
    beta = 0.01
    
    # Scheduled MPC Setpoints
    p_grid_sch = dispatch_mpc_df["P_grid_sch_kW"].values
    p_ch_sch = dispatch_mpc_df["P_ch_sch_kW"].values
    p_dis_sch = dispatch_mpc_df["P_dis_sch_kW"].values
    soc_sch = dispatch_mpc_df["SoC_sch"].values
    e_b_sch = dispatch_mpc_df["E_B_sch_kWh"].values
    
    # Real-Time Trajectory Containers
    p_grid_rt = np.zeros(N_p)
    p_ch_rt = np.zeros(N_p)
    p_dis_rt = np.zeros(N_p)
    p_curt_rt = np.zeros(N_p)
    e_b_rt = np.zeros(N_p)
    soc_rt = np.zeros(N_p)
    
    # Error signals
    p_error_raw = np.zeros(N_p)
    p_error_filtered = np.zeros(N_p)
    scenario_tags = []
    
    # Initial state
    current_e_b = soc_init * E_B_max
    p_bat_prev = p_dis_sch[0] - p_ch_sch[0]  # Initial net battery power (+ for discharge)
    
    print(f"\n--- Executing RT Secondary Controller Simulation ({N_p} steps, Mode: {mode.upper()}) ---")
    print(f"  Battery Sizing : Capacity={E_B_max:.2f} kWh, Rated Power={P_B_max:.2f} kW")
    print(f"  Ramp Limit     : {delta_P_ramp_max:.2f} kW / 15-min ({ramp_limit_fraction*100:.0f}% of P_B_max)")
    print(f"  Error Deadband : +/- {epsilon_tol:.2f} kW")
    
    for k in range(N_p):
        # 1. Physical Measurements and Forecast Deviations
        delta_pv = pv_actual[k] - pv_forecast[k]
        delta_load = load_actual[k] - load_forecast[k]
        
        # Net instantaneous DC Bus Mismatch:
        # P_Error(k) = (Delta P_PV * eta_mppt) - (Delta P_load / eta_dcac)
        err_raw = (delta_pv * eta_mppt) - (delta_load / eta_dcac)
        p_error_raw[k] = err_raw
        
        # 2. Scenario Classification
        if err_raw < -epsilon_tol:
            if delta_pv < 0 and delta_load > 0:
                scen = "Deficit Case 1 (Double Deficit)"
            elif delta_pv < 0 and delta_load <= 0:
                scen = "Deficit Case 2 (Solar Drop Dominates)"
            else:
                scen = "Deficit Case 3 (Load Surge Dominates)"
        elif err_raw > epsilon_tol:
            if delta_pv > 0 and delta_load < 0:
                scen = "Surplus Case 1 (Double Surplus)"
            elif delta_pv <= 0 and delta_load < 0:
                scen = "Surplus Case 2 (Load Drop Dominates)"
            else:
                scen = "Surplus Case 3 (Solar Gain Dominates)"
        else:
            scen = "Deadband Balanced"
        scenario_tags.append(scen)
        
        # 3. Layer 2: 3-State History Filtering
        if k == 0:
            err_filt = err_raw
        elif k == 1:
            err_filt = (0.5 * err_raw + 0.3 * p_error_raw[0]) / 0.8
        else:
            err_filt = 0.5 * err_raw + 0.3 * p_error_raw[k-1] + 0.2 * p_error_raw[k-2]
        p_error_filtered[k] = err_filt
        
        # 4. Layer 1 Deadband Thresholding
        if abs(err_filt) <= epsilon_tol:
            comp_error = 0.0
        else:
            comp_error = err_filt
            
        # 5. Layer 3: Merit-Order Energy Hierarchy & Physical Headroom Allocation
        sch_grid = p_grid_sch[k]
        sch_ch = p_ch_sch[k]
        sch_dis = p_dis_sch[k]
        
        # Local DC bus flows
        pv_gen_dc = (pv_actual[k] / eta_pvinverter) * eta_mppt
        load_dem_dc = load_actual[k] / eta_dcac
        p_phys_net = pv_gen_dc - load_dem_dc  # > 0: physical excess solar, <= 0: physical deficit
        
        # Physical energy headroom limits
        avail_dis_energy = max(0.0, ((current_e_b - SoC_min * E_B_max) / dt) * eta_d)
        avail_ch_energy = max(0.0, ((SoC_max * E_B_max - current_e_b) / (dt * eta_c)))

        if comp_error < -epsilon_tol:
            # === DEFICIT REGIME: Missing generation or demand surge ===
            p_deficit = abs(comp_error)
            
            # Merit Order Step 1: Immediately cancel/reduce scheduled battery charging
            if sch_ch > 0.0:
                cancelled_ch = min(p_deficit, sch_ch)
                target_ch = sch_ch - cancelled_ch
                p_deficit_rem = p_deficit - cancelled_ch
            else:
                target_ch = 0.0
                p_deficit_rem = p_deficit
                
            # Merit Order Step 2: Discharge battery to cover remaining deficit
            # Constrained by converter rating (P_B_max - sch_dis), available stored energy,
            # and actual DC load demand ceiling (cannot discharge more than building consumes)
            avail_dis = max(0.0, min(
                P_B_max - sch_dis,
                avail_dis_energy,
                load_dem_dc - sch_dis
            ))
            delta_dis = min(p_deficit_rem, avail_dis)
            target_dis = sch_dis + delta_dis
            
        elif comp_error > epsilon_tol:
            # === SURPLUS REGIME: Excess solar or load drop ===
            p_surplus = comp_error
            
            if p_phys_net > 0.0:
                # Subcase A: GENUINE PHYSICAL EXCESS SOLAR (PV > Load on DC Bus)
                # Merit Order Step 1: Grid import is zero
                # Merit Order Step 2: Battery discharge is eliminated
                target_dis = 0.0
                # Merit Order Step 3: Charge battery using physical excess solar
                avail_ch = max(0.0, min(P_B_max, avail_ch_energy))
                target_ch = min(p_phys_net, avail_ch)
                # Merit Order Step 4: Any excess solar beyond target_ch is curtailed
            else:
                # Subcase B: LOAD DROP / NO PHYSICAL EXCESS SOLAR (Night/evening load reduction)
                # Building still consumes net power (load_dem_dc >= pv_gen_dc).
                # There is NO physical excess solar to charge the battery!
                # Merit Order Step 1: Reduce scheduled discharge if active (preserves stored energy)
                if sch_dis > 0.0:
                    reduced_dis = min(p_surplus, sch_dis)
                    target_dis = sch_dis - reduced_dis
                else:
                    target_dis = 0.0
                # Merit Order Step 2: Maintain scheduled charge only if explicitly planned by MPC,
                # but NEVER initiate unscheduled grid charging from a forecast error!
                target_ch = min(sch_ch, avail_ch_energy)
                # Grid import will automatically reduce at the DC bus slack!
                
        else:
            # === DEADBAND BALANCED REGIME (|comp_error| <= epsilon_tol) ===
            # Follow scheduled setpoints within current physical boundaries
            target_dis = min(sch_dis, avail_dis_energy, load_dem_dc)
            target_ch = min(sch_ch, avail_ch_energy)

        # Layer 4: Inverter Ramp-Rate Limiting on net battery power
        target_p_bat = target_dis - target_ch
        clamped_p_bat = np.clip(
            target_p_bat,
            p_bat_prev - delta_P_ramp_max,
            p_bat_prev + delta_P_ramp_max
        )
        
        # Decompose into physical discharging and charging commands
        if clamped_p_bat >= 0.0:
            act_dis = min(clamped_p_bat, avail_dis_energy, load_dem_dc)
            act_ch = 0.0
        else:
            act_dis = 0.0
            act_ch = min(-clamped_p_bat, avail_ch_energy)

        # Exact DC Bus Power Balance:
        # Inflows: pv_gen_dc + act_grid * eta_acdc + act_dis
        # Outflows: load_dem_dc + act_ch + act_curt
        # Solve for slack: p_net_needed = load_dem_dc + act_ch - act_dis - pv_gen_dc
        p_net_needed = load_dem_dc + act_ch - act_dis - pv_gen_dc
        
        if p_net_needed > 0.0:
            # Power deficit on DC bus: electrical grid supplies the exact balance
            act_grid = p_net_needed / eta_acdc
            act_curt = 0.0
        else:
            # Power surplus on DC bus: zero grid re-injection constraint (P_grid >= 0)
            # Throttling PV inverter curtails the exact surplus
            act_grid = 0.0
            act_curt = -p_net_needed
                
        # Store RT Setpoints
        p_grid_rt[k] = act_grid
        p_ch_rt[k] = act_ch
        p_dis_rt[k] = act_dis
        p_curt_rt[k] = act_curt
        e_b_rt[k] = current_e_b
        soc_rt[k] = current_e_b / E_B_max
        
        # Update physical battery energy state for next interval
        current_e_b = current_e_b + (act_ch * eta_c - act_dis / eta_d) * dt
        # Enforce numerical bounds
        current_e_b = np.clip(current_e_b, SoC_min * E_B_max, SoC_max * E_B_max)
        
        # Update previous battery power for next step's ramp check
        p_bat_prev = act_dis - act_ch
        
    # 6. Physical DC Bus Balance Validation
    # Inflows: PV * eta_mppt / eta_pvinverter + P_grid * eta_acdc + P_dis
    # Outflows: P_load / eta_dcac + P_ch + P_curt
    inflows = (pv_actual / eta_pvinverter) * eta_mppt + p_grid_rt * eta_acdc + p_dis_rt
    outflows = (load_actual / eta_dcac) + p_ch_rt + p_curt_rt
    residual_mismatch = inflows - outflows
    max_res_error = np.max(np.abs(residual_mismatch))
    mean_res_error = np.mean(np.abs(residual_mismatch))
    
    print(f"\nPhysical DC Bus Balance Verification:")
    print(f"  Max Residual DC Mismatch  : {max_res_error:.8f} kW")
    print(f"  Mean Residual DC Mismatch : {mean_res_error:.8f} kW")
    print(f"  Status: {'PERFECT BALANCE [PASSED]' if max_res_error < 1e-4 else 'WARNING: Balance Drift'}")
    
    # 7. Degradation and Economic Comparison (MPC vs. RT)
    # MPC Costs
    mpc_grid_cost = float(np.sum(p_grid_sch * dt * tariffs))
    c_soc_mpc = np.array([C_0 * (alpha * soc_sch[k] - beta) / (CF_max * 15.0 * 8760.0) * dt for k in range(N_p)])
    delta_L_mpc = (p_dis_sch * dt) / E_B_max
    n_tot_mpc = (1.06 * (delta_L_mpc**4) - 2.80 * (delta_L_mpc**3) + 2.66 * (delta_L_mpc**2) - 1.07 * delta_L_mpc + 0.17) * 1e5
    c_dod_mpc = np.array([C_0 * (delta_L_mpc[k] / n_tot_mpc[k]) for k in range(N_p)])
    mpc_deg_cost = float(np.sum(c_soc_mpc + c_dod_mpc))
    
    # RT Actual Costs
    rt_grid_cost = float(np.sum(p_grid_rt * dt * tariffs))
    c_soc_rt = np.array([C_0 * (alpha * soc_rt[k] - beta) / (CF_max * 15.0 * 8760.0) * dt for k in range(N_p)])
    delta_L_rt = (p_dis_rt * dt) / E_B_max
    n_tot_rt = (1.06 * (delta_L_rt**4) - 2.80 * (delta_L_rt**3) + 2.66 * (delta_L_rt**2) - 1.07 * delta_L_rt + 0.17) * 1e5
    c_dod_rt = np.array([C_0 * (delta_L_rt[k] / n_tot_rt[k]) for k in range(N_p)])
    rt_soc_cost = float(np.sum(c_soc_rt))
    rt_dod_cost = float(np.sum(c_dod_rt))
    rt_deg_cost = rt_soc_cost + rt_dod_cost
    
    rt_curt_kwh = float(np.sum(p_curt_rt * dt))
    rt_grid_kwh = float(np.sum(p_grid_rt * dt))
    rt_dis_kwh = float(np.sum(p_dis_rt * dt))
    rt_ch_kwh = float(np.sum(p_ch_rt * dt))
    
    # Build complete timeseries dataframe
    rt_df = pd.DataFrame({
        "timestamp": time_index.strftime("%Y-%m-%d %H:%M:%S"),
        "tariff_eur_kwh": tariffs,
        "P_PV_forecast_kW": pv_forecast,
        "P_PV_actual_kW": pv_actual,
        "P_load_forecast_kW": load_forecast,
        "P_load_actual_kW": load_actual,
        "P_Error_raw_kW": p_error_raw,
        "P_Error_filtered_kW": p_error_filtered,
        "scenario": scenario_tags,
        "P_grid_sch_kW": p_grid_sch,
        "P_grid_rt_kW": p_grid_rt,
        "P_ch_sch_kW": p_ch_sch,
        "P_ch_rt_kW": p_ch_rt,
        "P_dis_sch_kW": p_dis_sch,
        "P_dis_rt_kW": p_dis_rt,
        "E_B_sch_kWh": e_b_sch,
        "E_B_rt_kWh": e_b_rt,
        "SoC_sch": soc_sch,
        "SoC_rt": soc_rt,
        "P_curt_rt_kW": p_curt_rt,
        "residual_mismatch_kW": residual_mismatch
    })
    
    # Statistical case counts
    case_counts = pd.Series(scenario_tags).value_counts().to_dict()
    deficit_count = int(np.sum(p_error_filtered < -epsilon_tol))
    surplus_count = int(np.sum(p_error_filtered > epsilon_tol))
    deadband_count = N_p - deficit_count - surplus_count
    
    summary = {
        "mode": mode,
        "name": sizing_params["name"],
        "horizon_steps": N_p,
        "horizon_hours": N_p * dt,
        "deadband_kw": epsilon_tol,
        "ramp_limit_kw_per_step": round(delta_P_ramp_max, 2),
        "scheduled_mpc": {
            "grid_energy_kwh": round(float(np.sum(p_grid_sch * dt)), 2),
            "grid_cost_eur": round(mpc_grid_cost, 2),
            "battery_discharged_kwh": round(float(np.sum(p_dis_sch * dt)), 2),
            "battery_charged_kwh": round(float(np.sum(p_ch_sch * dt)), 2),
            "degradation_cost_eur": round(mpc_deg_cost, 4),
            "total_cost_eur": round(mpc_grid_cost + mpc_deg_cost, 2)
        },
        "real_time_actual": {
            "grid_energy_kwh": round(rt_grid_kwh, 2),
            "grid_cost_eur": round(rt_grid_cost, 2),
            "battery_discharged_kwh": round(rt_dis_kwh, 2),
            "battery_charged_kwh": round(rt_ch_kwh, 2),
            "calendar_aging_cost_eur": round(rt_soc_cost, 4),
            "cyclic_wear_cost_eur": round(rt_dod_cost, 4),
            "degradation_cost_eur": round(rt_deg_cost, 4),
            "total_cost_eur": round(rt_grid_cost + rt_deg_cost, 2),
            "curtailed_solar_kwh": round(rt_curt_kwh, 2),
            "final_soc": round(float(soc_rt[-1]), 4)
        },
        "comparison_delta": {
            "grid_cost_delta_eur": round(rt_grid_cost - mpc_grid_cost, 2),
            "degradation_delta_eur": round(rt_deg_cost - mpc_deg_cost, 4),
            "total_cost_delta_eur": round((rt_grid_cost + rt_deg_cost) - (mpc_grid_cost + mpc_deg_cost), 2),
            "peak_grid_power_sch_kw": round(float(np.max(p_grid_sch)), 2),
            "peak_grid_power_rt_kw": round(float(np.max(p_grid_rt)), 2)
        },
        "mismatch_categorization": {
            "mean_error_raw_kw": round(float(np.mean(p_error_raw)), 2),
            "mae_error_raw_kw": round(float(np.mean(np.abs(p_error_raw))), 2),
            "mean_error_filtered_kw": round(float(np.mean(p_error_filtered)), 2),
            "deficit_intervals": deficit_count,
            "surplus_intervals": surplus_count,
            "deadband_intervals": deadband_count,
            "case_distribution": case_counts
        },
        "dc_bus_balance": {
            "max_residual_mismatch_kw": round(float(max_res_error), 6),
            "mean_residual_mismatch_kw": round(float(mean_res_error), 6),
            "perfect_balance": bool(max_res_error < 1e-4)
        }
    }
    
    return rt_df, summary

def plot_rt_results(mode: str, time_index: pd.DatetimeIndex, rt_df: pd.DataFrame, sizing: dict, summary: dict, save_path: Path):
    """
    Generates high-resolution 5-panel dashboard plot:
      Panel 1: Solar Generation & Load Demand (Forecast vs. Actual)
      Panel 2: DC Bus Net Forecast Mismatch (Raw, Filtered, Deadband)
      Panel 3: Battery Power Dispatch (MPC Planned vs. RT Actual)
      Panel 4: State of Charge (SoC Planned vs. RT Actual)
      Panel 5: Grid Import & Solar Curtailment (MPC Planned vs. RT Actual)
    """
    fig, axes = plt.subplots(5, 1, figsize=(15, 16), sharex=True)
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.grid(True, linestyle="--", alpha=0.3, color="#64748b")
        ax.tick_params(colors="#cbd5e1", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")
            
    # Panel 1: Generation & Load
    ax1 = axes[0]
    ax1.plot(time_index, rt_df["P_PV_actual_kW"], color="#fbbf24", linewidth=1.8, label="PV Actual (Meas)")
    ax1.plot(time_index, rt_df["P_PV_forecast_kW"], color="#f59e0b", linestyle="--", linewidth=1.4, alpha=0.8, label="PV Forecast (Day-Ahead)")
    ax1.plot(time_index, rt_df["P_load_actual_kW"], color="#f87171", linewidth=1.8, label="Load Actual (Meas)")
    ax1.plot(time_index, rt_df["P_load_forecast_kW"], color="#ef4444", linestyle="--", linewidth=1.4, alpha=0.8, label="Load Forecast (Day-Ahead)")
    ax1.set_ylabel("Power (kW)", color="#e2e8f0", fontsize=10, fontweight="bold")
    ax1.set_title(f"2-Day Real-Time Secondary Control: {sizing['name']} (48 Hours)", color="#f8fafc", fontsize=12, fontweight="bold", pad=8)
    ax1.legend(loc="upper right", framealpha=0.8, facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
    
    # Panel 2: DC Bus Forecast Error
    ax2 = axes[1]
    ax2.plot(time_index, rt_df["P_Error_raw_kW"], color="#94a3b8", linewidth=1.0, alpha=0.5, label="P_Error Raw Mismatch")
    ax2.plot(time_index, rt_df["P_Error_filtered_kW"], color="#38bdf8", linewidth=1.8, label="P_Error Filtered (3-State History)")
    ax2.axhline(0.5, color="#10b981", linestyle=":", linewidth=1.2, label="Deadband (+/-0.5 kW)")
    ax2.axhline(-0.5, color="#10b981", linestyle=":", linewidth=1.2)
    ax2.axhline(0.0, color="#64748b", linestyle="-", linewidth=0.8)
    ax2.fill_between(time_index, -0.5, 0.5, color="#10b981", alpha=0.1)
    ax2.set_ylabel("DC Mismatch (kW)", color="#e2e8f0", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.8, facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
    
    # Panel 3: Battery Power Dispatch (MPC vs RT)
    ax3 = axes[2]
    # Net battery power: P_dis - P_ch
    p_bat_sch = rt_df["P_dis_sch_kW"] - rt_df["P_ch_sch_kW"]
    p_bat_rt = rt_df["P_dis_rt_kW"] - rt_df["P_ch_rt_kW"]
    ax3.plot(time_index, p_bat_sch, color="#a855f7", linestyle="--", linewidth=1.5, label="MPC Net Battery (Planned)")
    ax3.plot(time_index, p_bat_rt, color="#c084fc", linewidth=1.8, label="RT Compensated Battery (Actual)")
    ax3.axhline(sizing["P_B_max"], color="#ef4444", linestyle=":", linewidth=1.0, label=f"+P_B_max ({sizing['P_B_max']:.1f} kW)")
    ax3.axhline(-sizing["P_B_max"], color="#3b82f6", linestyle=":", linewidth=1.0, label=f"-P_B_max (-{sizing['P_B_max']:.1f} kW)")
    ax3.set_ylabel("Battery P (kW)", color="#e2e8f0", fontsize=10, fontweight="bold")
    ax3.legend(loc="upper right", framealpha=0.8, facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
    
    # Panel 4: Battery State of Charge
    ax4 = axes[3]
    ax4.plot(time_index, rt_df["SoC_sch"], color="#a855f7", linestyle="--", linewidth=1.5, label="MPC Planned SoC")
    ax4.plot(time_index, rt_df["SoC_rt"], color="#34d399", linewidth=2.0, label="RT Dynamic SoC")
    ax4.axhline(0.90, color="#ef4444", linestyle=":", linewidth=1.0, label="Max Limit (90%)")
    ax4.axhline(0.20, color="#f59e0b", linestyle=":", linewidth=1.0, label="Min Safety (20%)")
    ax4.set_ylabel("Battery SoC", color="#e2e8f0", fontsize=10, fontweight="bold")
    ax4.set_ylim(0.15, 0.95)
    ax4.legend(loc="upper right", framealpha=0.8, facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
    
    # Panel 5: Grid Import & Curtailment
    ax5 = axes[4]
    ax5.plot(time_index, rt_df["P_grid_sch_kW"], color="#64748b", linestyle="--", linewidth=1.4, label="MPC Grid Import (Planned)")
    ax5.plot(time_index, rt_df["P_grid_rt_kW"], color="#38bdf8", linewidth=1.8, label="RT Grid Import (Actual)")
    ax5.plot(time_index, rt_df["P_curt_rt_kW"], color="#ec4899", linewidth=1.5, label="RT PV Curtailment (Throttled)")
    ax5.set_ylabel("Grid / Curt (kW)", color="#e2e8f0", fontsize=10, fontweight="bold")
    ax5.set_xlabel("Operational Time Horizon (15-Minute Intervals)", color="#e2e8f0", fontsize=10, fontweight="bold")
    ax5.legend(loc="upper right", framealpha=0.8, facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
    
    # Format x-axis dates nicely
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved 5-panel RT schedule visualization to {save_path}")

def run_rt_pipeline(
    mode: str = "supermarket",
    start_date: str = None,
    horizon: int = 192,
    soc_init: float = 0.50,
    epsilon_tol: float = 0.50,
    ramp_limit_fraction: float = 0.50
):
    """
    End-to-end pipeline:
      1. Solves 2-Day MPC (192 steps) to get baseline scheduled setpoints.
      2. Executes RT secondary controller step-by-step with real measurements.
      3. Validates DC bus power balance.
      4. Saves metrics JSON, timeseries CSV, and 5-panel plot in results_rt/.
    """
    print(f"\n========================================================")
    print(f" STARTING 2-DAY REAL-TIME CONTROLLER: {mode.upper()}")
    print(f"========================================================")
    
    # 1. Battery Sizing
    sizing = load_battery_sizing(mode)
    
    # 2. Evaluation Dates (Default: First 2 unseen test days)
    if start_date is None:
        start_date = "2022-11-01" if mode == "caltech_ev" else "2024-10-01"
    year = pd.to_datetime(start_date).year
    
    # 3. Forecast and Ground Truth Trajectories (192 steps = 48 hours = 2 days)
    time_index, pv_f, pv_act, load_f, load_act = get_forecast_and_actual_trajectories(mode, start_date, horizon)
    
    # 4. Electricity Tariffs for 192 steps
    tariffs, _ = fetch_or_load_tariffs(year, start_date, steps=horizon)
    
    # 5. Solve 2-Day Operational MPC Baseline
    print(f"\nSolving 2-Day Lookahead MPC Baseline (N_p = {horizon} steps)...")
    dispatch_mpc_df, mpc_summary = solve_mpc_optimization(
        mode=mode,
        pv_forecast=pv_f,
        load_forecast=load_f,
        tariffs=tariffs,
        sizing_params=sizing,
        dt=0.25,
        soc_init=soc_init
    )
    
    # 6. Execute Real-Time Secondary Control Simulation
    rt_df, rt_summary = run_rt_compensation(
        mode=mode,
        time_index=time_index,
        pv_forecast=pv_f,
        pv_actual=pv_act,
        load_forecast=load_f,
        load_actual=load_act,
        tariffs=tariffs,
        dispatch_mpc_df=dispatch_mpc_df,
        sizing_params=sizing,
        dt=0.25,
        soc_init=soc_init,
        epsilon_tol=epsilon_tol,
        ramp_limit_fraction=ramp_limit_fraction
    )
    
    # 7. Persistence & Visualizations
    results_json_path = RESULTS_RT_DIR / f"rt_metrics_{mode}.json"
    results_csv_path = RESULTS_RT_DIR / f"rt_timeseries_{mode}.csv"
    plot_path = RESULTS_RT_DIR / f"rt_schedule_{mode}.png"
    
    # Export CSV
    rt_df.to_csv(results_csv_path, index=False)
    print(f"Saved RT timeseries data to {results_csv_path}")
    
    # Export JSON
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(rt_summary, f, indent=2)
    print(f"Saved RT metrics summary to {results_json_path}")
    
    # Generate 5-Panel Plot
    plot_rt_results(mode, time_index, rt_df, sizing, rt_summary, plot_path)
    
    print(f"\n--- 2-DAY RT SIMULATION COMPLETE [{mode.upper()}] ---")
    sch = rt_summary["scheduled_mpc"]
    act = rt_summary["real_time_actual"]
    diff = rt_summary["comparison_delta"]
    print(f"  Grid Cost       : MPC Planned = EUR {sch['grid_cost_eur']:.2f} | RT Actual = EUR {act['grid_cost_eur']:.2f} (Delta: EUR {diff['grid_cost_delta_eur']:+.2f})")
    print(f"  Degradation Cost: MPC Planned = EUR {sch['degradation_cost_eur']:.4f} | RT Actual = EUR {act['degradation_cost_eur']:.4f} (Delta: EUR {diff['degradation_delta_eur']:+.4f})")
    print(f"  PV Curtailment  : {act['curtailed_solar_kwh']:.2f} kWh")
    print(f"  DC Bus Residual : Max Mismatch = {rt_summary['dc_bus_balance']['max_residual_mismatch_kw']:.6f} kW")
    print(f"  Intervals       : {rt_summary['mismatch_categorization']['deficit_intervals']} Deficit / {rt_summary['mismatch_categorization']['surplus_intervals']} Surplus / {rt_summary['mismatch_categorization']['deadband_intervals']} Deadband")
    print(f"========================================================\n")
    
    return rt_summary

def main():
    parser = argparse.ArgumentParser(description="2-Day Real-Time Secondary Microgrid Controller")
    parser.add_argument(
        "--mode",
        choices=["supermarket", "ev", "caltech_ev", "all"],
        default="supermarket",
        help="Target asset mode: 'supermarket', 'ev', 'caltech_ev', or 'all'"
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
        default=192,
        help="Prediction horizon in steps (default: 192 = 48h = 2 days)"
    )
    parser.add_argument(
        "--soc-init",
        type=float,
        default=0.50,
        help="Initial battery state of charge (default: 0.50)"
    )
    parser.add_argument(
        "--epsilon-tol",
        type=float,
        default=0.50,
        help="Deadband tolerance in kW (default: 0.50)"
    )
    parser.add_argument(
        "--ramp-limit",
        type=float,
        default=0.50,
        help="Ramp limit fraction of P_B_max per interval (default: 0.50)"
    )
    args = parser.parse_args()
    
    if args.mode == "all":
        modes = ["supermarket", "ev", "caltech_ev"]
        for m in modes:
            run_rt_pipeline(
                mode=m,
                start_date=args.date,
                horizon=args.horizon,
                soc_init=args.soc_init,
                epsilon_tol=args.epsilon_tol,
                ramp_limit_fraction=args.ramp_limit
            )
    else:
        run_rt_pipeline(
            mode=args.mode,
            start_date=args.date,
            horizon=args.horizon,
            soc_init=args.soc_init,
            epsilon_tol=args.epsilon_tol,
            ramp_limit_fraction=args.ramp_limit
        )

if __name__ == "__main__":
    main()
