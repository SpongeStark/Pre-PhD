"""
MPC & Real-Time Equations Evolution Tracker
===========================================
Consolidates, computes, and visualizes the exact mathematical equations of the MPC
and Real-Time secondary controller:

1. Objective Function & Economic Cost Evolution:
   - Instantaneous & Cumulative Grid Import Cost: Cost_grid(k) = P_grid(k) * dt * tariff(k)
   - Total Combined Operational Cost: J(k) = Cost_grid(k) + C_deg(k) + eps_curt * P_curt(k)

2. Battery Degradation & Aging Mechanics:
   - Calendar Aging Cost: C_SoC(k) = C_0 * (alpha * SoC(k) - beta) / (CF_max * 15 * 8760) * dt
   - Fractional Depth-of-Discharge: Delta_L(k) = P_dis(k) * dt / E_B_max
   - 4th-degree Polynomial Cycle-Life: N_tot(Delta_L) = (1.06*dL^4 - 2.80*dL^3 + 2.66*dL^2 - 1.07*dL + 0.17) * 1e5
   - Cyclic DoD Wear Cost: C_DoD(k) = C_0 * Delta_L(k) / N_tot(Delta_L(k))
   - Total Degradation Cost: C_deg(k) = C_SoC(k) + C_DoD(k)
   - Cumulative Capacity Fade (%): CF(k) = [Sum(C_deg) / C_0] * CF_max * 100%
   - Equivalent Full Cycles (EFC): EFC(k) = Sum(P_dis * dt) / E_B_max

3. Dynamic Inverter Power Derating Envelope:
   - Charging Derating Limit: P_ch_max(SoC) = [P_B_max / (1 - D_ch)] * (1 - SoC)
   - Discharging Derating Limit: P_dis_max(SoC) = [P_B_max / D_dis] * SoC

4. DC Bus Power Balance & Exact Slack Residual:
   - Inflows = (P_PV / eta_pvinv) * eta_mppt + P_grid * eta_acdc + P_dis
   - Outflows = (P_load / eta_dcac) + P_ch + P_curt
   - Residual Mismatch = Inflows - Outflows == 0.000000 kW

Outputs:
  - CSV files: results_mpc/mpc_equations_evolution_{mode}.csv
  - JSON summaries: results_mpc/mpc_equations_summary_{mode}.json
  - Dashboards: results_mpc/mpc_equations_dashboard_{mode}.png
  - Comparative plot: results_mpc/mpc_equations_comparison.png
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Project root paths
SRC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SRC_DIR.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
RESULTS_MPC_DIR = ROOT_DIR / "results_mpc"
RESULTS_MPC_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_RT_DIR = ROOT_DIR / "results_rt"

from mpc_optimizer import load_battery_sizing, fetch_or_load_tariffs

def compute_equations_timeseries(mode: str, sizing: dict, dt: float = 0.25):
    """
    Computes step-by-step numerical trajectories for all MPC and RT equations.
    Reads from existing results_rt/rt_timeseries_{mode}.csv.
    """
    csv_path = RESULTS_RT_DIR / f"rt_timeseries_{mode}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing RT timeseries CSV at {csv_path}. Run rt_controller.py first.")
        
    df = pd.read_csv(csv_path)
    time_index = pd.DatetimeIndex(df["timestamp"])
    N_p = len(df)
    
    E_B_max = float(sizing["E_B_max"])
    P_B_max = float(sizing["P_B_max"])
    
    # Technical constants
    eta_pvinverter = 0.99
    eta_mppt = 0.99
    eta_acdc = 0.99
    eta_dcac = 0.99
    eta_c = 0.95
    eta_d = 0.95
    
    # Aging & economic parameters
    C_0 = 500.0 * E_B_max + 100.0 * P_B_max
    CF_max = 0.20
    alpha = 0.05
    beta = 0.01
    D_ch = 0.90
    D_dis = 0.20
    
    tariffs = df["tariff_eur_kwh"].values
    
    # --- 1. MPC Planned Equations ---
    p_grid_sch = df["P_grid_sch_kW"].values
    p_ch_sch = df["P_ch_sch_kW"].values
    p_dis_sch = df["P_dis_sch_kW"].values
    soc_sch = df["SoC_sch"].values
    
    # Grid Cost
    cost_grid_sch_step = p_grid_sch * dt * tariffs
    cost_grid_sch_cum = np.cumsum(cost_grid_sch_step)
    
    # Calendar Aging Cost
    c_soc_sch_step = np.array([C_0 * (alpha * soc_sch[k] - beta) / (CF_max * 15.0 * 8760.0) * dt for k in range(N_p)])
    c_soc_sch_cum = np.cumsum(c_soc_sch_step)
    
    # Cyclic DoD Wear Cost
    delta_L_sch = (p_dis_sch * dt) / E_B_max
    n_tot_sch = np.array([
        (1.06 * (dL**4) - 2.80 * (dL**3) + 2.66 * (dL**2) - 1.07 * dL + 0.17) * 1e5 if dL > 1e-6 else 1e8
        for dL in delta_L_sch
    ])
    c_dod_sch_step = np.array([C_0 * (delta_L_sch[k] / n_tot_sch[k]) if delta_L_sch[k] > 1e-6 else 0.0 for k in range(N_p)])
    c_dod_sch_cum = np.cumsum(c_dod_sch_step)
    
    # Total Degradation & Objective
    c_deg_sch_step = c_soc_sch_step + c_dod_sch_step
    c_deg_sch_cum = np.cumsum(c_deg_sch_step)
    total_cost_sch_cum = cost_grid_sch_cum + c_deg_sch_cum
    
    # Capacity Fade & EFC
    cap_fade_sch_pct = (c_deg_sch_cum / C_0) * CF_max * 100.0
    efc_sch_cum = np.cumsum(p_dis_sch * dt) / E_B_max
    
    # --- 2. Real-Time Actual Equations ---
    p_grid_rt = df["P_grid_rt_kW"].values
    p_ch_rt = df["P_ch_rt_kW"].values
    p_dis_rt = df["P_dis_rt_kW"].values
    soc_rt = df["SoC_rt"].values
    p_curt_rt = df["P_curt_rt_kW"].values
    
    cost_grid_rt_step = p_grid_rt * dt * tariffs
    cost_grid_rt_cum = np.cumsum(cost_grid_rt_step)
    
    c_soc_rt_step = np.array([C_0 * (alpha * soc_rt[k] - beta) / (CF_max * 15.0 * 8760.0) * dt for k in range(N_p)])
    c_soc_rt_cum = np.cumsum(c_soc_rt_step)
    
    delta_L_rt = (p_dis_rt * dt) / E_B_max
    n_tot_rt = np.array([
        (1.06 * (dL**4) - 2.80 * (dL**3) + 2.66 * (dL**2) - 1.07 * dL + 0.17) * 1e5 if dL > 1e-6 else 1e8
        for dL in delta_L_rt
    ])
    c_dod_rt_step = np.array([C_0 * (delta_L_rt[k] / n_tot_rt[k]) if delta_L_rt[k] > 1e-6 else 0.0 for k in range(N_p)])
    c_dod_rt_cum = np.cumsum(c_dod_rt_step)
    
    c_deg_rt_step = c_soc_rt_step + c_dod_rt_step
    c_deg_rt_cum = np.cumsum(c_deg_rt_step)
    total_cost_rt_cum = cost_grid_rt_cum + c_deg_rt_cum
    
    cap_fade_rt_pct = (c_deg_rt_cum / C_0) * CF_max * 100.0
    efc_rt_cum = np.cumsum(p_dis_rt * dt) / E_B_max
    
    # --- 3. Dynamic Power Derating Envelope (CC-CV & Deep Discharge Limits) ---
    p_ch_derating_limit_sch = (P_B_max / (1.0 - D_ch)) * np.maximum(0.0, 1.0 - soc_sch)
    p_dis_derating_limit_sch = (P_B_max / D_dis) * np.maximum(0.0, soc_sch)
    
    p_ch_derating_limit_rt = (P_B_max / (1.0 - D_ch)) * np.maximum(0.0, 1.0 - soc_rt)
    p_dis_derating_limit_rt = (P_B_max / D_dis) * np.maximum(0.0, soc_rt)
    
    # --- 4. Exact DC Bus Power Balance & Slack ---
    pv_act = df["P_PV_actual_kW"].values
    load_act = df["P_load_actual_kW"].values
    
    inflows_dc = (pv_act / eta_pvinverter) * eta_mppt + p_grid_rt * eta_acdc + p_dis_rt
    outflows_dc = (load_act / eta_dcac) + p_ch_rt + p_curt_rt
    residual_mismatch = inflows_dc - outflows_dc
    
    # Consolidate into DataFrame
    eq_df = pd.DataFrame({
        "timestamp": time_index.strftime("%Y-%m-%d %H:%M:%S"),
        "tariff_eur_kwh": tariffs,
        
        # Grid Costs
        "cost_grid_sch_step_eur": cost_grid_sch_step,
        "cost_grid_sch_cum_eur": cost_grid_sch_cum,
        "cost_grid_rt_step_eur": cost_grid_rt_step,
        "cost_grid_rt_cum_eur": cost_grid_rt_cum,
        
        # Calendar Aging
        "c_soc_sch_step_eur": c_soc_sch_step,
        "c_soc_sch_cum_eur": c_soc_sch_cum,
        "c_soc_rt_step_eur": c_soc_rt_step,
        "c_soc_rt_cum_eur": c_soc_rt_cum,
        
        # Cyclic DoD Wear
        "delta_L_sch": delta_L_sch,
        "N_tot_sch_cycles": n_tot_sch,
        "c_dod_sch_step_eur": c_dod_sch_step,
        "c_dod_sch_cum_eur": c_dod_sch_cum,
        "delta_L_rt": delta_L_rt,
        "N_tot_rt_cycles": n_tot_rt,
        "c_dod_rt_step_eur": c_dod_rt_step,
        "c_dod_rt_cum_eur": c_dod_rt_cum,
        
        # Total Degradation & Combined Operational Costs
        "c_deg_sch_step_eur": c_deg_sch_step,
        "c_deg_sch_cum_eur": c_deg_sch_cum,
        "c_deg_rt_step_eur": c_deg_rt_step,
        "c_deg_rt_cum_eur": c_deg_rt_cum,
        "total_operational_cost_sch_cum_eur": total_cost_sch_cum,
        "total_operational_cost_rt_cum_eur": total_cost_rt_cum,
        
        # Capacity Fade & Equivalent Full Cycles
        "capacity_fade_sch_pct": cap_fade_sch_pct,
        "capacity_fade_rt_pct": cap_fade_rt_pct,
        "efc_sch_cycles": efc_sch_cum,
        "efc_rt_cycles": efc_rt_cum,
        
        # Power & Derating Envelopes
        "SoC_sch": soc_sch,
        "SoC_rt": soc_rt,
        "P_ch_sch_kW": p_ch_sch,
        "P_ch_rt_kW": p_ch_rt,
        "P_ch_derating_limit_kW": p_ch_derating_limit_rt,
        "P_dis_sch_kW": p_dis_sch,
        "P_dis_rt_kW": p_dis_rt,
        "P_dis_derating_limit_kW": p_dis_derating_limit_rt,
        
        # DC Bus Slack Residual
        "inflows_dc_kW": inflows_dc,
        "outflows_dc_kW": outflows_dc,
        "residual_mismatch_kW": residual_mismatch
    })
    
    summary = {
        "mode": mode,
        "asset_name": sizing["name"],
        "battery_capital_cost_c0_eur": round(C_0, 2),
        "capacity_kwh": E_B_max,
        "rated_power_kw": P_B_max,
        "horizon_steps": N_p,
        "horizon_hours": N_p * dt,
        "grid_costs": {
            "mpc_planned_total_eur": round(float(cost_grid_sch_cum[-1]), 2),
            "rt_actual_total_eur": round(float(cost_grid_rt_cum[-1]), 2),
            "delta_eur": round(float(cost_grid_rt_cum[-1] - cost_grid_sch_cum[-1]), 2)
        },
        "degradation_costs": {
            "mpc_calendar_aging_eur": round(float(c_soc_sch_cum[-1]), 4),
            "rt_calendar_aging_eur": round(float(c_soc_rt_cum[-1]), 4),
            "mpc_cyclic_wear_eur": round(float(c_dod_sch_cum[-1]), 4),
            "rt_cyclic_wear_eur": round(float(c_dod_rt_cum[-1]), 4),
            "mpc_total_degradation_eur": round(float(c_deg_sch_cum[-1]), 4),
            "rt_total_degradation_eur": round(float(c_deg_rt_cum[-1]), 4),
            "delta_eur": round(float(c_deg_rt_cum[-1] - c_deg_sch_cum[-1]), 4)
        },
        "health_mechanics": {
            "rt_equivalent_full_cycles": round(float(efc_rt_cum[-1]), 3),
            "rt_capacity_fade_pct": round(float(cap_fade_rt_pct[-1]), 5),
            "max_interval_dod_pct": round(float(np.max(delta_L_rt) * 100.0), 2),
            "mean_interval_dod_pct": round(float(np.mean(delta_L_rt[delta_L_rt > 0]) * 100.0) if np.any(delta_L_rt > 0) else 0.0, 2)
        },
        "total_operational_cost": {
            "mpc_planned_total_eur": round(float(total_cost_sch_cum[-1]), 2),
            "rt_actual_total_eur": round(float(total_cost_rt_cum[-1]), 2),
            "delta_eur": round(float(total_cost_rt_cum[-1] - total_cost_sch_cum[-1]), 2)
        },
        "dc_bus_conservation": {
            "max_residual_error_kw": round(float(np.max(np.abs(residual_mismatch))), 8),
            "mean_residual_error_kw": round(float(np.mean(np.abs(residual_mismatch))), 8),
            "perfect_balance": bool(np.max(np.abs(residual_mismatch)) < 1e-4)
        }
    }
    
    return time_index, eq_df, summary

def plot_equations_dashboard(mode: str, time_index: pd.DatetimeIndex, eq_df: pd.DataFrame, sizing: dict, summary: dict, save_path: Path):
    """
    Generates a comprehensive 6-panel visualization tracking the exact MPC equations.
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 14), sharex=True)
    fig.patch.set_facecolor("#0f172a")
    
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.grid(True, linestyle="--", alpha=0.3, color="#64748b")
        ax.tick_params(colors="#cbd5e1", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")
            
    fig.suptitle(f"MPC & RT Equations Mathematical Evolution: {sizing['name']} (48 Hours)",
                 color="#f8fafc", fontsize=15, fontweight="bold", y=0.98)

    # --- Panel 1: Economic Grid Cost Evolution ---
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time_index, eq_df["tariff_eur_kwh"], color="#fbbf24", linestyle=":", linewidth=1.2, alpha=0.6, label="Electricity Tariff (EUR/kWh)")
    ax1_twin.set_ylabel("Tariff (EUR/kWh)", color="#fbbf24", fontsize=9)
    ax1_twin.tick_params(colors="#fbbf24", labelsize=8)
    
    ax1.plot(time_index, eq_df["cost_grid_sch_cum_eur"], color="#94a3b8", linestyle="--", linewidth=1.6, label="MPC Planned Grid Cost (Cum)")
    ax1.plot(time_index, eq_df["cost_grid_rt_cum_eur"], color="#38bdf8", linewidth=2.0, label="RT Actual Grid Cost (Cum)")
    ax1.set_ylabel("Cumulative Grid Cost (EUR)", color="#38bdf8", fontsize=10, fontweight="bold")
    ax1.set_title("Equation 1: Grid Electricity Cost Integration [Cost_grid = Integral(P_grid * tariff)]", color="#f8fafc", fontsize=10, fontweight="bold", pad=8)
    ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)

    # --- Panel 2: Battery Degradation Cost Breakdown ---
    ax2 = axes[0, 1]
    ax2.plot(time_index, eq_df["c_soc_rt_cum_eur"], color="#fbbf24", linewidth=1.6, label="Calendar Aging (C_SoC Cum)")
    ax2.plot(time_index, eq_df["c_dod_rt_cum_eur"], color="#f43f5e", linewidth=1.6, label="Cyclic DoD Wear (C_DoD Cum)")
    ax2.plot(time_index, eq_df["c_deg_rt_cum_eur"], color="#a855f7", linewidth=2.2, label="Total Battery Degradation (C_deg Cum)")
    ax2.plot(time_index, eq_df["c_deg_sch_cum_eur"], color="#94a3b8", linestyle="--", linewidth=1.4, label="MPC Planned Degradation (Cum)")
    ax2.set_ylabel("Cumulative Degradation (EUR)", color="#a855f7", fontsize=10, fontweight="bold")
    ax2.set_title("Equation 2: Battery Degradation Cost [C_deg = C_SoC(t) + C_DoD(Delta_L)]", color="#f8fafc", fontsize=10, fontweight="bold", pad=8)
    ax2.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)

    # --- Panel 3: DoD per Interval & Polynomial Cycle-Life ---
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    ax3.bar(time_index, eq_df["delta_L_rt"] * 100.0, width=0.008, color="#f43f5e", alpha=0.6, label="Interval DoD Delta_L (% of E_B_max)")
    ax3_twin.plot(time_index, eq_df["N_tot_rt_cycles"] / 1000.0, color="#10b981", linewidth=1.5, label="Cycle Life Capability N_tot(Delta_L) [kCycles]")
    ax3.set_ylabel("Depth of Discharge (%)", color="#f43f5e", fontsize=10, fontweight="bold")
    ax3_twin.set_ylabel("Permissible Cycles (x1,000)", color="#10b981", fontsize=9)
    ax3_twin.tick_params(colors="#10b981", labelsize=8)
    ax3.set_title("Equation 3: Cyclic Aging Mechanics [N_tot = Poly4(Delta_L)]", color="#f8fafc", fontsize=10, fontweight="bold", pad=8)
    ax3.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
    ax3_twin.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)

    # --- Panel 4: Capacity Fade & Equivalent Full Cycles ---
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    ax4.plot(time_index, eq_df["capacity_fade_rt_pct"], color="#f59e0b", linewidth=2.0, label="Cumulative Capacity Fade (%)")
    ax4.plot(time_index, eq_df["capacity_fade_sch_pct"], color="#94a3b8", linestyle="--", linewidth=1.4, label="MPC Planned Capacity Fade (%)")
    ax4_twin.plot(time_index, eq_df["efc_rt_cycles"], color="#06b6d4", linewidth=1.8, label="Equivalent Full Cycles (EFC)")
    ax4.set_ylabel("Capacity Fade (%)", color="#f59e0b", fontsize=10, fontweight="bold")
    ax4_twin.set_ylabel("EFC (Full Cycles)", color="#06b6d4", fontsize=9)
    ax4_twin.tick_params(colors="#06b6d4", labelsize=8)
    ax4.set_title("Equation 4: Long-Term Battery Health [Fade = (Sum C_deg / C_0) * CF_max]", color="#f8fafc", fontsize=10, fontweight="bold", pad=8)
    ax4.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
    ax4_twin.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)

    # --- Panel 5: Inverter Power Derating Envelope ---
    ax5 = axes[2, 0]
    ax5.plot(time_index, eq_df["P_ch_rt_kW"], color="#3b82f6", linewidth=1.6, label="Actual Charge P_ch (kW)")
    ax5.plot(time_index, eq_df["P_ch_derating_limit_kW"], color="#60a5fa", linestyle=":", linewidth=1.4, label="Charging Ceiling P_ch_max(SoC)")
    ax5.plot(time_index, eq_df["P_dis_rt_kW"], color="#ef4444", linewidth=1.6, label="Actual Discharge P_dis (kW)")
    ax5.plot(time_index, eq_df["P_dis_derating_limit_kW"], color="#f87171", linestyle=":", linewidth=1.4, label="Discharge Ceiling P_dis_max(SoC)")
    ax5.axhline(sizing["P_B_max"], color="#94a3b8", linestyle="--", linewidth=1.0, alpha=0.5, label=f"Rated P_B_max ({sizing['P_B_max']:.1f} kW)")
    ax5.set_ylabel("Inverter Power (kW)", color="#e2e8f0", fontsize=10, fontweight="bold")
    ax5.set_title("Equation 5: Inverter Derating Constraints [P_ch <= f(1-SoC), P_dis <= f(SoC)]", color="#f8fafc", fontsize=10, fontweight="bold", pad=8)
    ax5.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)

    # --- Panel 6: DC Bus Slack Conservation & Residual Error ---
    ax6 = axes[2, 1]
    ax6.plot(time_index, eq_df["inflows_dc_kW"], color="#34d399", linewidth=1.8, label="Total DC Inflows (PV_dc + Grid_dc + Dis)")
    ax6.plot(time_index, eq_df["outflows_dc_kW"], color="#ec4899", linestyle="--", linewidth=1.6, label="Total DC Outflows (Load_dc + Ch + Curt)")
    ax6_twin = ax6.twinx()
    ax6_twin.plot(time_index, eq_df["residual_mismatch_kW"], color="#e2e8f0", linewidth=0.8, alpha=0.7, label="Residual Mismatch (kW)")
    ax6_twin.set_ylabel("Residual (kW)", color="#e2e8f0", fontsize=9)
    ax6_twin.set_ylim(-0.01, 0.01)
    ax6_twin.tick_params(colors="#e2e8f0", labelsize=8)
    ax6.set_ylabel("DC Power Flow (kW)", color="#34d399", fontsize=10, fontweight="bold")
    ax6.set_title("Equation 6: DC Bus Power Conservation [Residual = Inflows - Outflows == 0 kW]", color="#f8fafc", fontsize=10, fontweight="bold", pad=8)
    ax6.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
    ax6_twin.legend(loc="lower right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)

    # Format x-axes
    for ax in axes[2, :]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        ax.set_xlabel("Time Horizon (15-min intervals)", color="#e2e8f0", fontsize=10, fontweight="bold")
        
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved 6-panel equations dashboard to {save_path}")

def plot_cross_asset_comparison(summaries: dict, save_path: Path):
    """
    Generates a cross-asset comparative bar chart illustrating the equation results
    across Supermarket, Caltech EV, and Lidl EV.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.grid(True, linestyle="--", alpha=0.3, color="#64748b")
        ax.tick_params(colors="#cbd5e1", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#334155")
            
    fig.suptitle("Cross-Asset MPC Equation Metrics Comparison (48-Hour Canonical Simulation)",
                 color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)

    modes = ["supermarket", "caltech_ev", "ev"]
    labels = ["Supermarket", "Caltech EV Fleet", "Lidl EV Chargers"]
    x = np.arange(len(modes))
    width = 0.35
    
    # 1. Grid Cost: MPC Planned vs RT Actual
    ax1 = axes[0, 0]
    mpc_grid = [summaries[m]["grid_costs"]["mpc_planned_total_eur"] for m in modes]
    rt_grid = [summaries[m]["grid_costs"]["rt_actual_total_eur"] for m in modes]
    r1 = ax1.bar(x - width/2, mpc_grid, width, label="MPC Planned", color="#94a3b8")
    r2 = ax1.bar(x + width/2, rt_grid, width, label="RT Actual", color="#38bdf8")
    ax1.set_ylabel("Grid Cost (EUR)", color="#e2e8f0", fontweight="bold")
    ax1.set_title("Total Grid Electricity Cost", color="#f8fafc", fontweight="bold", pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1")
    for r in r1:
        ax1.annotate(f"€{r.get_height():.2f}", (r.get_x() + r.get_width()/2, r.get_height()), ha="center", va="bottom", color="#94a3b8", fontsize=8)
    for r in r2:
        ax1.annotate(f"€{r.get_height():.2f}", (r.get_x() + r.get_width()/2, r.get_height()), ha="center", va="bottom", color="#38bdf8", fontsize=8, fontweight="bold")

    # 2. Battery Degradation: Calendar vs Cyclic Wear
    ax2 = axes[0, 1]
    cal_deg = [summaries[m]["degradation_costs"]["rt_calendar_aging_eur"] for m in modes]
    cyc_deg = [summaries[m]["degradation_costs"]["rt_cyclic_wear_eur"] for m in modes]
    r3 = ax2.bar(x - width/2, cal_deg, width, label="Calendar Aging (C_SoC)", color="#fbbf24")
    r4 = ax2.bar(x + width/2, cyc_deg, width, label="Cyclic Wear (C_DoD)", color="#f43f5e")
    ax2.set_ylabel("Degradation Cost (EUR)", color="#e2e8f0", fontweight="bold")
    ax2.set_title("RT Degradation Breakdown (Calendar vs Cyclic)", color="#f8fafc", fontweight="bold", pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1")
    for r in r3:
        ax2.annotate(f"€{r.get_height():.3f}", (r.get_x() + r.get_width()/2, r.get_height()), ha="center", va="bottom", color="#fbbf24", fontsize=8)
    for r in r4:
        ax2.annotate(f"€{r.get_height():.3f}", (r.get_x() + r.get_width()/2, r.get_height()), ha="center", va="bottom", color="#f43f5e", fontsize=8)

    # 3. Equivalent Full Cycles (EFC)
    ax3 = axes[1, 0]
    efc_vals = [summaries[m]["health_mechanics"]["rt_equivalent_full_cycles"] for m in modes]
    r5 = ax3.bar(x, efc_vals, width*1.2, color="#06b6d4")
    ax3.set_ylabel("Cycles (EFC)", color="#e2e8f0", fontweight="bold")
    ax3.set_title("Battery Utilization (Equivalent Full Cycles)", color="#f8fafc", fontweight="bold", pad=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    for r in r5:
        ax3.annotate(f"{r.get_height():.2f} EFC", (r.get_x() + r.get_width()/2, r.get_height()), ha="center", va="bottom", color="#06b6d4", fontsize=9, fontweight="bold")

    # 4. Cumulative Capacity Fade (%)
    ax4 = axes[1, 1]
    fade_vals = [summaries[m]["health_mechanics"]["rt_capacity_fade_pct"] for m in modes]
    r6 = ax4.bar(x, fade_vals, width*1.2, color="#a855f7")
    ax4.set_ylabel("Capacity Fade (%)", color="#e2e8f0", fontweight="bold")
    ax4.set_title("2-Day Cumulative Capacity Fade (%)", color="#f8fafc", fontweight="bold", pad=8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    for r in r6:
        ax4.annotate(f"{r.get_height():.4f}%", (r.get_x() + r.get_width()/2, r.get_height()), ha="center", va="bottom", color="#a855f7", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved cross-asset comparison plot to {save_path}")

def run_tracker():
    """Consolidates MPC equations across all assets, generates CSVs, JSONs, and plots."""
    print("\n========================================================")
    print(" MPC & REAL-TIME EQUATIONS MATHEMATICAL TRACKER")
    print("========================================================")
    
    modes = ["supermarket", "caltech_ev", "ev"]
    summaries = {}
    artifact_dir = Path(r"C:\Users\alesk\.gemini\antigravity\brain\4fec5526-7196-4acd-9976-88084cad6a69")
    
    for mode in modes:
        print(f"\n--- Processing Equations Evolution: {mode.upper()} ---")
        sizing = load_battery_sizing(mode)
        time_index, eq_df, summary = compute_equations_timeseries(mode, sizing)
        summaries[mode] = summary
        
        # Export CSV
        csv_path = RESULTS_MPC_DIR / f"mpc_equations_evolution_{mode}.csv"
        eq_df.to_csv(csv_path, index=False)
        print(f"  Exported timeseries: {csv_path}")
        
        # Export JSON
        json_path = RESULTS_MPC_DIR / f"mpc_equations_summary_{mode}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  Exported summary JSON: {json_path}")
        
        # Plot 6-Panel Dashboard
        plot_path = RESULTS_MPC_DIR / f"mpc_equations_dashboard_{mode}.png"
        plot_equations_dashboard(mode, time_index, eq_df, sizing, summary, plot_path)
        
        # Copy to artifact dir if exists
        if artifact_dir.exists():
            dst = artifact_dir / f"mpc_equations_dashboard_{mode}.png"
            shutil.copy2(plot_path, dst)
            print(f"  Copied to artifacts: {dst.name}")
            
    # Generate cross-asset summary plot
    comp_path = RESULTS_MPC_DIR / "mpc_equations_comparison.png"
    plot_cross_asset_comparison(summaries, comp_path)
    if artifact_dir.exists():
        dst = artifact_dir / "mpc_equations_comparison.png"
        shutil.copy2(comp_path, dst)
        print(f"  Copied comparison plot to artifacts: {dst.name}")
        
    print("\n========================================================")
    print(" ALL EQUATION EVOLUTION DATA & DASHBOARDS GENERATED!")
    print("========================================================\n")

if __name__ == "__main__":
    run_tracker()
