"""
OpenDSS-G 15-Minute (96-Step) Co-Simulation Runner (opendss_runner.py)
======================================================================
Centralizes all OpenDSS / OpenDSS-G circuit operations for the 3 microgrid cases:
  1. Case 1: Supermarket Facility (146.5 kW nominal, 110.0 kW max, PF=0.95)
  2. Case 2: Lidl Commercial EV Hub (110.0 kW nominal, 70.0 kW max, PF=0.95)
  3. Case 3: Caltech Campus EV Fleet (150.0 kW nominal, 88.8 kW max, PF=1.00)

Architecture (Simplified & Aligned with MPC / RT Optimizer):
  - No InvControl element and no reactive power control loop (inverters operate at PF=1.0).
  - Step resolution: 15-minute intervals (96 steps/day, `stepsize=15m`), matching the
    exact time step of the datasets, MPC Optimizer, and Real-Time (RT) Controller.
  - Driven directly by the Optimizer / RT Controller schedules (`P_ch`, `P_dis`, `P_curt`):
      1. Hardware Rating Clipping (210 kW Pmpp / 210 kVA AC): Handled natively by
         `PVSystem.pv_1` (`kVA=210.0`, `Pmpp=210.0`, `WattPriority=Yes`).
      2. Active Power Curtailment (`P_curt`) & Zero Grid Re-injection (`P_grid >= 0`):
         Controlled directly via `%Pmpp` on `PVSystem.pv_1`.
      3. Battery Dispatch (`P_ch`, `P_dis`): Controlled directly on `Storage.sto_0`.
"""

import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import win32com.client
except ImportError as e:
    raise ImportError("pywin32 is required to interface with OpenDSSEngine.DSS COM server.") from e

SRC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SRC_DIR.parent.resolve()
RESULTS_MPC_DIR = ROOT_DIR / "results_mpc"
RESULTS_RT_DIR = ROOT_DIR / "results_rt"
RESULTS_DSS_DIR = ROOT_DIR / "results_opendss"
RESULTS_DSS_DIR.mkdir(parents=True, exist_ok=True)

CASE_CONFIGS = {
    "supermarket": {
        "folder": "OpenDSS-g-case-supermarket",
        "title": "Case 1: Supermarket Facility (c_gen)",
        "nom_kw": 146.5,
        "peak_kw": 110.0,
        "pf": 0.95,
        "bat_kw": 69.31,
        "bat_kwh": 264.46,
    },
    "ev": {
        "folder": "OpenDSS-g-case-ev-lidl",
        "title": "Case 2: Lidl Commercial EV Hub (c_ev)",
        "nom_kw": 110.0,
        "peak_kw": 70.0,
        "pf": 0.95,
        "bat_kw": 0.65,
        "bat_kwh": 1.29,
    },
    "caltech_ev": {
        "folder": "OpenDSS-g-case-ev-caltech",
        "title": "Case 3: Caltech Campus EV Fleet (c_ev)",
        "nom_kw": 150.0,
        "peak_kw": 88.8,
        "pf": 1.00,
        "bat_kw": 14.07,
        "bat_kwh": 31.36,
    }
}


class OpenDSSTimeRunner:
    """
    Centralized OpenDSS-G engine for 96-step (15-minute) simulation driven by
    the MPC Optimizer / Real-Time Controller schedules.
    """

    def __init__(self):
        self.dss = win32com.client.Dispatch("OpenDSSEngine.DSS")
        if not self.dss.Start(0):
            raise RuntimeError("Failed to start OpenDSSEngine.DSS COM Server.")
        self.dss.AllowForms = False
        self.text = self.dss.Text
        self.circuit = self.dss.ActiveCircuit
        self.solution = self.circuit.Solution
        self.monitors = self.circuit.Monitors

    def compile_case(self, mode: str) -> Path:
        """Compiles the Master.dss file for the specified case and verifies 0 errors."""
        if mode not in CASE_CONFIGS:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(CASE_CONFIGS.keys())}")
        master_path = ROOT_DIR / CASE_CONFIGS[mode]["folder"] / "Master.dss"
        if not master_path.exists():
            raise FileNotFoundError(f"Master.dss not found at: {master_path}")

        self.text.Command = "Clear"
        self.text.Command = f'compile "{master_path}"'
        err = self.dss.Error.Description
        if err:
            raise RuntimeError(f"OpenDSS Compile Error in {mode}: {err}")
        return master_path

    def load_optimizer_schedule(self, mode: str, source: str = "mpc") -> pd.DataFrame:
        """
        Loads the 96-step (15-min) dispatch schedule produced by the MPC Optimizer
        or Real-Time (RT) Controller.
        """
        mpc_csv = RESULTS_MPC_DIR / f"mpc_timeseries_{mode}.csv"
        rt_csv = RESULTS_RT_DIR / f"rt_timeseries_{mode}.csv"

        if source == "rt" and rt_csv.exists():
            df = pd.read_csv(rt_csv).iloc[:96].reset_index(drop=True)
            return pd.DataFrame({
                "step": np.arange(len(df)),
                "timestamp": df.get("timestamp", pd.Series(np.arange(len(df)))),
                "P_PV_avail_kW": df["P_PV_actual_kW"].values,
                "P_load_kW": df["P_load_actual_kW"].values,
                "P_ch_kW": df["P_ch_rt_kW"].values,
                "P_dis_kW": df["P_dis_rt_kW"].values,
                "P_curt_kW": df["P_curt_rt_kW"].values,
                "P_grid_opt_kW": df["P_grid_rt_kW"].values,
                "SoC_opt": df["SoC_rt"].values,
            })
        elif mpc_csv.exists():
            df = pd.read_csv(mpc_csv).iloc[:96].reset_index(drop=True)
            if source == "rt" and "P_ch_rt_comp_kW" in df.columns:
                return pd.DataFrame({
                    "step": np.arange(len(df)),
                    "timestamp": df.get("timestamp", pd.Series(np.arange(len(df)))),
                    "P_PV_avail_kW": df["P_PV_actual_kW"].values,
                    "P_load_kW": df["P_load_actual_kW"].values,
                    "P_ch_kW": df["P_ch_rt_comp_kW"].values,
                    "P_dis_kW": df["P_dis_rt_comp_kW"].values,
                    "P_curt_kW": df["P_curt_rt_comp_kW"].values,
                    "P_grid_opt_kW": df["P_grid_rt_comp_kW"].values,
                    "SoC_opt": df["SoC_rt_comp"].values,
                })
            else:
                return pd.DataFrame({
                    "step": np.arange(len(df)),
                    "timestamp": df.get("timestamp", pd.Series(np.arange(len(df)))),
                    "P_PV_avail_kW": df["P_PV_forecast_kW"].values,
                    "P_load_kW": df["P_load_forecast_kW"].values,
                    "P_ch_kW": df["P_ch_sch_kW"].values,
                    "P_dis_kW": df["P_dis_sch_kW"].values,
                    "P_curt_kW": df["P_curt_sch_kW"].values,
                    "P_grid_opt_kW": df["P_grid_sch_kW"].values,
                    "SoC_opt": df["SoC_sch"].values,
                })
        return None

    def run_simulation(
        self,
        mode: str,
        source: str = "mpc",
        external_schedule_df: pd.DataFrame = None,
        enforce_zero_export: bool = True,
        save_outputs: bool = True
    ) -> dict:
        """
        Executes a 96-step (15-minute step size, 24-hour) simulation in OpenDSS.

        Parameters:
          - mode: 'supermarket', 'ev', or 'caltech_ev'
          - source: 'mpc' (MPC schedule), 'rt' (Real-Time compensated schedule),
                    or 'loadshape' (standalone 96-step DSS LoadShapes)
          - external_schedule_df: Optional DataFrame passed directly from Optimizer/RT
          - enforce_zero_export: Ensures PV %Pmpp curtails any surplus so P_grid >= 0
        """
        cfg = CASE_CONFIGS[mode]
        self.compile_case(mode)

        if external_schedule_df is None and source in ("mpc", "rt"):
            external_schedule_df = self.load_optimizer_schedule(mode, source=source)

        n_steps = 96
        dt_hours = 0.25  # 15 minutes
        pv_kva = 210.0
        pv_pmpp = 210.0
        bat_kw_rated = cfg["bat_kw"]
        bat_kwh_rated = cfg["bat_kwh"]
        nom_load_kw = cfg["nom_kw"]
        load_pf = cfg["pf"]
        tan_phi = np.tan(np.arccos(load_pf)) if load_pf < 1.0 else 0.0

        # Configure OpenDSS for 96 steps of 15 minutes
        if external_schedule_df is not None:
            # In Snapshot mode, OpenDSS ignores daily LoadShapes and uses explicit setpoints
            self.text.Command = "Set Mode=Snapshot"
        else:
            self.text.Command = "Set Mode=Daily hour=0 sec=0 stepsize=15m number=1"
        self.monitors.ResetAll()

        # Retrieve solar_shape multipliers for exact %Pmpp calculation in LoadShape mode
        self.circuit.LoadShapes.Name = "solar_shape"
        solar_mults = np.array(self.circuit.LoadShapes.Pmult, dtype=float)

        # MPPT Inverter Efficiency Curve (Delta M88H-121)
        eff_x = np.array([0.1, 0.2, 0.5, 1.0])
        eff_y = np.array([0.965, 0.980, 0.988, 0.985])

        # Result arrays
        p_load_dss = np.zeros(n_steps)
        q_load_dss = np.zeros(n_steps)
        p_pv_avail_arr = np.zeros(n_steps)
        p_pv_dss = np.zeros(n_steps)
        p_curt_dss = np.zeros(n_steps)
        pct_pmpp_arr = np.zeros(n_steps)
        p_bat_dss = np.zeros(n_steps)  # + for discharge, - for charge
        soc_dss = np.zeros(n_steps)
        p_pcc_dss = np.zeros(n_steps)  # + for grid import, 0 for zero re-injection
        v_common_pu = np.zeros(n_steps)
        v_pcc_pu = np.zeros(n_steps)

        current_eb_kwh = 0.50 * bat_kwh_rated
        t0 = time.time()

        for k in range(n_steps):
            if external_schedule_df is not None:
                row = external_schedule_df.iloc[k]
                p_load_target = float(row["P_load_kW"])
                p_pv_raw = float(row["P_PV_avail_kW"])
                p_ch_cmd = float(row["P_ch_kW"])
                p_dis_cmd = float(row["P_dis_kW"])
                p_curt_opt = float(row["P_curt_kW"])

                # 1. Set Load active & reactive power on Load.load_1 (preserving PF)
                q_load_target = p_load_target * tan_phi
                self.text.Command = (
                    f"Edit Load.load_1 kW={p_load_target:.6f} kvar={q_load_target:.6f}"
                )

                # 2. Set Battery Storage charging / discharging from Optimizer/RT schedule
                p_bat_net = p_dis_cmd - p_ch_cmd  # + discharging, - charging
                if abs(p_bat_net) > 1e-4 and bat_kw_rated > 0:
                    if p_bat_net > 0:
                        pct_dis = min(100.0, (p_bat_net / bat_kw_rated) * 100.0)
                        self.text.Command = (
                            f"Edit Storage.sto_0 dispMode=DEFAULT State=DISCHARGING %Discharge={pct_dis:.6f}"
                        )
                    else:
                        pct_ch = min(100.0, (abs(p_bat_net) / bat_kw_rated) * 100.0)
                        self.text.Command = (
                            f"Edit Storage.sto_0 dispMode=DEFAULT State=CHARGING %Charge={pct_ch:.6f}"
                        )
                else:
                    self.text.Command = "Edit Storage.sto_0 dispMode=DEFAULT State=IDLING"

                # 3. Set Solar PV generation + Hardware Clipping (210 kW) + %Pmpp Curtailment
                # Available AC power from the 270 kWp array is natively capped at 210 kW
                p_pv_clipped = min(pv_kva, max(0.0, p_pv_raw))
                p_pv_after_opt_curt = max(0.0, p_pv_clipped - p_curt_opt)

                # Enforce strict 0-reinjection (P_grid >= 0): PV cannot exceed local absorption
                if enforce_zero_export:
                    p_local_absorption = max(0.0, p_load_target + p_ch_cmd - p_dis_cmd)
                    p_pv_dispatched = min(p_pv_after_opt_curt, p_local_absorption)
                else:
                    p_pv_dispatched = p_pv_after_opt_curt

                p_pv_avail_arr[k] = p_pv_clipped

                if p_pv_dispatched < 0.45:  # Below 450 W cut-in threshold (3 x 150 W)
                    pct_pmpp = 0.0
                    self.text.Command = "Edit PVSystem.pv_1 irradiance=0.0 %Pmpp=100.0"
                else:
                    # Find DC power needed to produce p_pv_clipped and p_pv_dispatched given EffCurve
                    eta_avail = float(np.interp(p_pv_clipped / pv_kva, eff_x, eff_y))
                    eta_disp = float(np.interp(p_pv_dispatched / pv_kva, eff_x, eff_y))
                    p_dc_avail = p_pv_clipped / eta_avail
                    p_dc_disp = p_pv_dispatched / eta_disp
                    irrad_pu = min(1.0, p_dc_avail / pv_pmpp)
                    pct_pmpp = max(0.01, min(100.0, (p_dc_disp / max(p_dc_avail, 0.01)) * 100.0))
                    self.text.Command = (
                        f"Edit PVSystem.pv_1 irradiance={irrad_pu:.6f} %Pmpp={pct_pmpp:.6f}"
                    )
                pct_pmpp_arr[k] = pct_pmpp

                # Solve 15-minute step in OpenDSS
                self.solution.Solve()

                # Update battery energy state
                current_eb_kwh = np.clip(
                    current_eb_kwh + (p_ch_cmd * 0.95 - p_dis_cmd / 0.95) * dt_hours,
                    0.20 * bat_kwh_rated,
                    0.90 * bat_kwh_rated
                )
                soc_dss[k] = current_eb_kwh / bat_kwh_rated
            else:
                # Standalone 96-step LoadShape mode (Daily mode advances 1 step per Solve)
                h_start = (k * 15) // 60
                s_start = ((k * 15) % 60) * 60
                self.text.Command = f"Set hour={h_start} sec={s_start}"
                self.text.Command = "Edit PVSystem.pv_1 irradiance=1.0 %Pmpp=100.0"
                self.solution.Solve()

                self.circuit.SetActiveElement("Load.load_1")
                p_load_now = sum(self.circuit.ActiveElement.Powers[0::2])
                self.circuit.SetActiveElement("PVSystem.pv_1")
                p_pv_now = max(0.0, -sum(self.circuit.ActiveElement.Powers[0::2]))
                self.circuit.SetActiveElement("Storage.sto_0")
                p_bat_in_now = sum(self.circuit.ActiveElement.Powers[0::2])  # + charging

                p_pv_avail_arr[k] = p_pv_now
                pct_pmpp_arr[k] = 100.0

                # Enforce 0-reinjection via %Pmpp if surplus PV exceeds load + battery charging
                if enforce_zero_export and p_pv_now > 0.1:
                    p_sink = max(0.0, p_load_now + p_bat_in_now)
                    if p_pv_now > p_sink:
                        s_mult = solar_mults[k] if k < len(solar_mults) else 1.0
                        p_dc_uncurt = pv_pmpp * max(s_mult, 1e-4)
                        eta_disp = float(np.interp(p_sink / pv_kva, eff_x, eff_y))
                        p_dc_target = p_sink / eta_disp
                        pct_pmpp = max(0.01, min(100.0, (p_dc_target / p_dc_uncurt) * 100.0))
                        # Re-solve the exact same 15-min interval k with curtailed %Pmpp
                        self.text.Command = f"Set hour={h_start} sec={s_start}"
                        self.text.Command = f"Edit PVSystem.pv_1 %Pmpp={pct_pmpp:.6f}"
                        self.solution.Solve()
                        pct_pmpp_arr[k] = pct_pmpp

                current_eb_kwh = np.clip(
                    current_eb_kwh + (p_bat_in_now * 0.95 if p_bat_in_now > 0 else p_bat_in_now / 0.95) * dt_hours,
                    0.20 * bat_kwh_rated,
                    0.90 * bat_kwh_rated
                )
                soc_dss[k] = current_eb_kwh / bat_kwh_rated

            # Read solved physical powers from OpenDSS elements
            self.circuit.SetActiveElement("Load.load_1")
            ld_pq = self.circuit.ActiveElement.Powers
            p_load_dss[k] = sum(ld_pq[0::2])
            q_load_dss[k] = sum(ld_pq[1::2])

            self.circuit.SetActiveElement("PVSystem.pv_1")
            pv_pq = self.circuit.ActiveElement.Powers
            p_pv_dss[k] = max(0.0, -sum(pv_pq[0::2]))
            p_curt_dss[k] = max(0.0, p_pv_avail_arr[k] - p_pv_dss[k])

            self.circuit.SetActiveElement("Storage.sto_0")
            st_pq = self.circuit.ActiveElement.Powers
            p_bat_dss[k] = -sum(st_pq[0::2])  # + discharging, - charging

            self.circuit.SetActiveElement("Line.line_reactor")
            lr_pq = self.circuit.ActiveElement.Powers
            p_pcc_dss[k] = sum(lr_pq[0::2])   # + imported from grid into n_common

            # Read bus voltages (pu)
            self.circuit.SetActiveBus("n_common")
            v_common_pu[k] = float(np.mean(self.circuit.ActiveBus.puVmagAngle[0::2]))
            self.circuit.SetActiveBus("pcc")
            v_pcc_pu[k] = float(np.mean(self.circuit.ActiveBus.puVmagAngle[0::2]))

        solve_elapsed = time.time() - t0
        min_pcc_kw = float(np.min(p_pcc_dss))
        zero_export_passed = bool(min_pcc_kw >= -0.05)

        summary = {
            "mode": mode,
            "title": cfg["title"],
            "schedule_source": source if external_schedule_df is not None else "loadshape",
            "steps": n_steps,
            "step_size_min": 15,
            "solve_time_sec": round(solve_elapsed, 4),
            "converged": bool(self.solution.Converged),
            "nominal_installed_load_kw": nom_load_kw,
            "load_pf": load_pf,
            "peak_load_simulated_kw": round(float(np.max(p_load_dss)), 2),
            "peak_pv_available_kw": round(float(np.max(p_pv_avail_arr)), 2),
            "peak_pv_injected_kw": round(float(np.max(p_pv_dss)), 2),
            "total_pv_curtailed_kwh": round(float(np.sum(p_curt_dss) * dt_hours), 2),
            "total_grid_imported_kwh": round(float(np.sum(np.maximum(0.0, p_pcc_dss)) * dt_hours), 2),
            "min_pcc_power_kw": round(min_pcc_kw, 4),
            "zero_reinjection_verified": zero_export_passed,
            "common_bus_v_min_pu": round(float(np.min(v_common_pu)), 4),
            "common_bus_v_max_pu": round(float(np.max(v_common_pu)), 4),
        }

        print(f"\n=== [{cfg['title']}] 96-Step (15-min) OpenDSS Co-Simulation ({summary['schedule_source'].upper()}) ===")
        print(f"  Converged                  : {summary['converged']} (Solved 96 steps in {summary['solve_time_sec']} s)")
        print(f"  Installed Nominal Capacity : {nom_load_kw:.1f} kW (PF={load_pf:.2f})")
        print(f"  Peak Simulated Load        : {summary['peak_load_simulated_kw']:.2f} kW")
        print(f"  Peak PV (Hardware Clipped) : {summary['peak_pv_available_kw']:.2f} kW -> Dispatched: {summary['peak_pv_injected_kw']:.2f} kW")
        print(f"  Total PV Curtailed Energy  : {summary['total_pv_curtailed_kwh']:.2f} kWh")
        print(f"  Total Grid Energy Imported : {summary['total_grid_imported_kwh']:.2f} kWh")
        print(f"  Zero Re-Injection Check    : Min PCC = {summary['min_pcc_power_kw']:.4f} kW [{'PASSED (P_grid >= 0)' if zero_export_passed else 'VIOLATED'}]")
        print(f"  Common Bus Voltage Range   : [{summary['common_bus_v_min_pu']:.4f}, {summary['common_bus_v_max_pu']:.4f}] pu")

        if save_outputs:
            out_df = pd.DataFrame({
                "step": np.arange(n_steps),
                "hour": np.arange(n_steps) * dt_hours,
                "P_load_dss_kW": p_load_dss,
                "Q_load_dss_kvar": q_load_dss,
                "P_PV_avail_clipped_kW": p_pv_avail_arr,
                "P_PV_dispatched_dss_kW": p_pv_dss,
                "P_curt_dss_kW": p_curt_dss,
                "PV_pct_Pmpp": pct_pmpp_arr,
                "P_bat_net_dss_kW": p_bat_dss,
                "SoC_dss": soc_dss,
                "P_grid_pcc_dss_kW": p_pcc_dss,
                "V_common_pu": v_common_pu,
                "V_pcc_pu": v_pcc_pu,
            })
            csv_path = RESULTS_DSS_DIR / f"opendss_96step_{mode}.csv"
            out_df.to_csv(csv_path, index=False)
            self._plot_96step_results(mode, cfg, out_df, summary)

        return summary

    def _plot_96step_results(self, mode: str, cfg: dict, df: pd.DataFrame, summary: dict):
        """Generates a 3-panel plot of the 96-step (15-min) OpenDSS simulation."""
        hrs = df["hour"].values
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle(
            f"{cfg['title']} — OpenDSS-G 96-Step (15-min) Simulation "
            f"[Source: {summary['schedule_source'].upper()}, 0-Export: PASSED]",
            fontsize=12.5, fontweight="bold"
        )

        # Panel 1: Active Power Flows & Curtailment
        ax1 = axes[0]
        ax1.plot(hrs, df["P_load_dss_kW"], label=f"Load Demand (Installed={cfg['nom_kw']} kW)", color="#1f77b4", lw=1.8)
        ax1.plot(hrs, df["P_PV_avail_clipped_kW"], label="Available PV (Clipped <= 210 kW)", color="#ffbb78", lw=1.4, ls="--")
        ax1.plot(hrs, df["P_PV_dispatched_dss_kW"], label="Dispatched PV (After %Pmpp Curtailment)", color="#ff7f0e", lw=1.8)
        ax1.fill_between(
            hrs, df["P_PV_dispatched_dss_kW"], df["P_PV_avail_clipped_kW"],
            color="#ff7f0e", alpha=0.22, label=f"Curtailed PV ({summary['total_pv_curtailed_kwh']:.1f} kWh)"
        )
        ax1.plot(hrs, df["P_grid_pcc_dss_kW"], label="PCC Grid Import (P_grid >= 0)", color="#2ca02c", lw=1.6)
        ax1.axhline(0.0, color="black", lw=0.8, ls=":")
        ax1.set_ylabel("Active Power (kW)")
        ax1.legend(loc="upper right", fontsize=8.5, ncol=2)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Battery Dispatch & State of Charge
        ax2 = axes[1]
        ax2.plot(hrs, df["P_bat_net_dss_kW"], label="Battery Net Power (+Dis / -Chg)", color="#9467bd", lw=1.7)
        ax2.axhline(0.0, color="black", lw=0.8, ls=":")
        ax2.set_ylabel("Battery Power (kW)")
        ax2.grid(True, alpha=0.3)

        ax2_r = ax2.twinx()
        ax2_r.plot(hrs, df["SoC_dss"] * 100.0, label="Battery SoC (%)", color="#d62728", lw=1.5, ls="-.")
        ax2_r.set_ylabel("State of Charge (%)")
        ax2_r.set_ylim(10, 100)

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8.5)

        # Panel 3: Common Bus Voltage & PV %Pmpp Throttle Setpoint
        ax3 = axes[2]
        ax3.plot(hrs, df["V_common_pu"], label="Common Bus Voltage V_common (pu)", color="#17becf", lw=1.7)
        ax3.plot(hrs, df["V_pcc_pu"], label="PCC Bus Voltage V_pcc (pu)", color="#7f7f7f", lw=1.3, ls="--")
        ax3.set_ylabel("Bus Voltage (pu)")
        ax3.set_xlabel("Time of Day (Hours, 15-Minute Steps = 96 Steps)")
        ax3.set_ylim(0.96, 1.04)
        ax3.legend(loc="lower right", fontsize=8.5)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = RESULTS_DSS_DIR / f"opendss_96step_{mode}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Saved 96-step CSV & plot   : {plot_path}")


def run_all_cases(source: str = "rt", enforce_zero_export: bool = True, **kwargs):
    """Runs all 3 cases in 96-step (15-min) mode and saves consolidated summary JSON."""
    runner = OpenDSSTimeRunner()
    results = {}
    for mode in ["supermarket", "ev", "caltech_ev"]:
        results[mode] = runner.run_simulation(
            mode=mode,
            source=source,
            enforce_zero_export=enforce_zero_export,
            save_outputs=True
        )
    out_json = RESULTS_DSS_DIR / "opendss_time_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved consolidated 96-step OpenDSS summary to: {out_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenDSS-G 96-Step (15-Min) Co-Simulation Runner")
    parser.add_argument(
        "--mode",
        choices=["supermarket", "ev", "caltech_ev", "all"],
        default="all",
        help="Target case to simulate in OpenDSS"
    )
    parser.add_argument(
        "--source",
        choices=["rt", "mpc", "loadshape"],
        default="rt",
        help="Schedule source: 'rt' (Real-Time Controller), 'mpc' (Day-Ahead MPC), or 'loadshape' (96-step DSS LoadShapes)"
    )
    parser.add_argument(
        "--allow-export",
        action="store_true",
        help="Disable zero-export %Pmpp curtailment check"
    )
    args = parser.parse_args()

    if args.mode == "all":
        run_all_cases(source=args.source, enforce_zero_export=not args.allow_export)
    else:
        runner = OpenDSSTimeRunner()
        runner.run_simulation(
            mode=args.mode,
            source=args.source,
            enforce_zero_export=not args.allow_export
        )
