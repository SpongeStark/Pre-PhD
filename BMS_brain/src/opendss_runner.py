"""
OpenDSS-G Time-Mode Simulation & Inverter Control Runner (opendss_runner.py)
============================================================================
Centralizes all OpenDSS / OpenDSS-G circuit operations for the 3 microgrid cases:
  1. Case 1: Supermarket Facility (146.5 kW nominal, 110.0 kW max, PF=0.95)
  2. Case 2: Lidl Commercial EV Hub (110.0 kW nominal, 70.0 kW max, PF=0.95)
  3. Case 3: Caltech Campus EV Fleet (150.0 kW nominal, 88.8 kW max, PF=1.00)

Key Capabilities:
  - Executes high-resolution Time-Mode simulation (`Set Mode=Time stepsize=1s number=1`),
    advancing second-by-second across the 24-hour horizon (86,400 steps at 1s).
  - Enforces the 3 levels of Inverter Control (3 x Delta M88H-121, 210 kVA AC / 270 kWp DC):
      1. Hardware Rating Clipping (270 kWp DC -> 210 kW AC via WattPriority=Yes)
         + Autonomous Overvoltage Curtailment (InvControl VOLTWATT on 210 kVA base).
      2. Active Power Factor (PF) Compensation: Inverters dynamically inject capacitive
         reactive power (kvar) to compensate inductive load demand (Q_load), driving
         PCC Power Factor toward unity (PF_pcc -> 1.00) while respecting the apparent
         power circle |S| <= kVA.
      3. External BMS / Optimizer Curtailment (P_curt) & Battery Storage Dispatch (P_ch, P_dis):
         Can run either standalone with the case LoadShapes or coupled with the MPC / RT
         Controller schedules to enforce strict zero grid export (P_grid >= 0).
"""

import os
import sys
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
    Centralized OpenDSS-G engine for snapshot verification, 1-second Time-Mode QSTS
    simulation, active PF compensation, and BMS Optimizer co-simulation.
    """

    def __init__(self):
        self.dss = win32com.client.Dispatch("OpenDSSEngine.DSS")
        if not self.dss.Start(0):
            raise RuntimeError("Failed to start OpenDSSEngine.DSS COM Server.")
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

    @staticmethod
    def _extract_monitor_pq(monitors, mon_name: str):
        """Extracts 3-phase total P (kW) and Q (kvar) time-series from an OpenDSS Monitor (mode=1)."""
        monitors.Name = mon_name
        p1 = np.array(monitors.Channel(1), dtype=float)
        q1 = np.array(monitors.Channel(2), dtype=float)
        p2 = np.array(monitors.Channel(3), dtype=float)
        q2 = np.array(monitors.Channel(4), dtype=float)
        p3 = np.array(monitors.Channel(5), dtype=float)
        q3 = np.array(monitors.Channel(6), dtype=float)
        return (p1 + p2 + p3), (q1 + q2 + q3)

    def run_time_simulation(
        self,
        mode: str,
        dt_sec: int = 1,
        hours: int = 24,
        pf_compensation: bool = True,
        zero_export_curtailment: bool = False,
        external_schedule_df: pd.DataFrame = None,
        save_plot: bool = True
    ) -> dict:
        """
        Runs the OpenDSS circuit in `Mode=Time` advancing by `dt_sec` (default 1 second)
        across `hours` (default 24h = 86,400 steps at 1s).

        Parameters:
          - mode: 'supermarket', 'ev', or 'caltech_ev'
          - dt_sec: Step size in seconds (default 1s)
          - hours: Total horizon in hours (default 24h)
          - pf_compensation: If True, inverters dynamically inject reactive power (kvar)
                             to compensate load inductive reactive demand (PF -> 1.0 at PCC).
          - zero_export_curtailment: If True, BMS throttles PV `%Pmpp` whenever surplus PV
                                     exceeds load + battery charging so P_grid >= 0.
          - external_schedule_df: Optional DataFrame from MPC/RT controller with columns
                                  ['P_ch_kW', 'P_dis_kW', 'P_curt_kW'] to co-simulate.
        """
        cfg = CASE_CONFIGS[mode]
        self.compile_case(mode)

        total_seconds = int(hours * 3600)
        n_steps = total_seconds // dt_sec
        steps_per_hour = 3600 // dt_sec

        # Configure OpenDSS 1-Second Time-Stepping Mode advancing by dt_sec per Solve
        self.text.Command = f"Set Mode=Daily hour=0 sec=0 stepsize={dt_sec}s number=1"
        if pf_compensation and cfg["pf"] < 1.0:
            # Pre-initialize reactive power compensation for the initial hour before recording
            q_init = cfg["nom_kw"] * 0.3808 * np.tan(np.arccos(cfg["pf"]))
            self.text.Command = f"Edit PVSystem.pv_1 kvar={q_init:.4f}"
        self.monitors.ResetAll()

        # Precompute reactive tangent factor for load PF compensation
        load_pf = cfg["pf"]
        tan_phi = np.tan(np.arccos(load_pf)) if load_pf < 1.0 else 0.0

        # PV & Battery Inverter ratings
        pv_kva = 210.0
        pv_pmpp = 270.0
        pv_kvar_max = 126.0  # Delta M88H PF=0.80 capability
        bat_kw_rated = cfg["bat_kw"]
        bat_kva = round(bat_kw_rated * 1.25, 2)
        bat_kvar_max = round(bat_kw_rated * 0.75, 2)

        # Control update cadence: update inverter setpoints every 15-min (900s) block
        ctrl_interval_sec = 900
        ctrl_step_mod = max(1, ctrl_interval_sec // dt_sec)
        n_blocks = n_steps // ctrl_step_mod

        t0 = time.time()
        cur_pct_pmpp = 100.0

        for blk in range(n_blocks):
            # 1. External BMS Battery Dispatch Setpoint (if provided)
            if external_schedule_df is not None:
                idx_15m = min(blk, len(external_schedule_df) - 1)
                row = external_schedule_df.iloc[idx_15m]
                p_bat_net = float(row.get("P_dis_kW", 0.0)) - float(row.get("P_ch_kW", 0.0))
                if abs(p_bat_net) > 0.01 and bat_kw_rated > 0:
                    if p_bat_net > 0:
                        pct_dis = min(100.0, (p_bat_net / bat_kw_rated) * 100.0)
                        self.text.Command = f"Edit Storage.sto_0 dispMode=DEFAULT State=DISCHARGING %Discharge={pct_dis:.4f}"
                    else:
                        pct_ch = min(100.0, (abs(p_bat_net) / bat_kw_rated) * 100.0)
                        self.text.Command = f"Edit Storage.sto_0 dispMode=DEFAULT State=CHARGING %Charge={pct_ch:.4f}"
                else:
                    self.text.Command = "Edit Storage.sto_0 dispMode=DEFAULT State=IDLING"

            # Solve the first 1-second step of this block to sample current operating point
            self.solution.Number = 1
            self.solution.Solve()

            # Read instantaneous Load P & Q, PV P, and Battery P
            self.circuit.SetActiveElement("Load.load_1")
            load_pq = self.circuit.ActiveElement.Powers
            p_load_now = sum(load_pq[0::2])
            q_load_now = sum(load_pq[1::2])

            self.circuit.SetActiveElement("PVSystem.pv_1")
            pv_pq = self.circuit.ActiveElement.Powers
            p_pv_gen_now = -sum(pv_pq[0::2])  # Positive when generating

            self.circuit.SetActiveElement("Storage.sto_0")
            bat_pq = self.circuit.ActiveElement.Powers
            p_bat_now = sum(bat_pq[0::2])     # Positive when charging, negative when discharging

            # 2. Enforce Zero Grid Re-injection Curtailment if enabled
            if zero_export_curtailment and p_pv_gen_now > 0.1:
                p_local_sink = max(0.0, p_load_now + p_bat_now)
                if p_pv_gen_now > p_local_sink:
                    cur_pct_pmpp = max(0.5, min(100.0, (p_local_sink / pv_pmpp) * 100.0 / 0.985))
                    self.text.Command = f"Edit PVSystem.pv_1 %Pmpp={cur_pct_pmpp:.4f}"
                else:
                    cur_pct_pmpp = 100.0
                    self.text.Command = "Edit PVSystem.pv_1 %Pmpp=100.0"

            # 3. Active Power Factor (PF) Compensation by Inverters
            if pf_compensation:
                q_target = max(0.0, q_load_now)
                # Available reactive headroom on PV inverter under WattPriority (|S| <= 210 kVA)
                p_pv_clipped = min(pv_kva, max(0.0, p_pv_gen_now))
                q_pv_headroom = min(pv_kvar_max, float(np.sqrt(max(0.0, pv_kva**2 - p_pv_clipped**2))))
                q_pv_cmd = min(q_target, q_pv_headroom)

                # Remaining reactive demand compensated by Battery inverter
                q_rem = max(0.0, q_target - q_pv_cmd)
                p_bat_abs = min(bat_kva, abs(p_bat_now))
                q_bat_headroom = min(bat_kvar_max, float(np.sqrt(max(0.0, bat_kva**2 - p_bat_abs**2))))
                q_bat_cmd = min(q_rem, q_bat_headroom)

                self.text.Command = f"Edit PVSystem.pv_1 kvar={q_pv_cmd:.4f}"
                self.text.Command = f"Edit Storage.sto_0 kvar={q_bat_cmd:.4f}"

            # Advance the remaining (ctrl_step_mod - 1) 1-second steps in this block
            if ctrl_step_mod > 1:
                self.solution.Number = ctrl_step_mod - 1
                self.solution.Solve()

        solve_elapsed = time.time() - t0

        # Extract full high-resolution 1-second trajectories from OpenDSS Monitors
        p_pcc, q_pcc = self._extract_monitor_pq(self.monitors, "inv_flow")
        p_pv, q_pv = self._extract_monitor_pq(self.monitors, "pv_mon")
        p_load, q_load = self._extract_monitor_pq(self.monitors, "load_mon")

        # Storage monitor (mode=7 records states, or use power flow balance)
        self.monitors.Name = "bat_mon"
        n_bat_chans = self.monitors.NumChannels
        if n_bat_chans >= 3:
            kwh_stored = np.array(self.monitors.Channel(1), dtype=float)
        else:
            kwh_stored = np.full(len(p_pcc), cfg["bat_kwh"] * 0.5)

        # Sign conventions:
        # PV generation as positive kW/kvar injected into n_common
        p_pv_gen = -p_pv
        q_pv_inj = -q_pv
        # Net battery power from node balance on common bus
        p_bat_gen = -(p_pcc + p_load - p_pv_gen)

        # Compute PCC Power Factor during active load hours
        s_pcc = np.sqrt(p_pcc**2 + q_pcc**2)
        pf_pcc = np.where(s_pcc > 1.0, np.abs(p_pcc) / s_pcc, 1.0)

        # Snapshot voltage check at Common Bus
        self.circuit.SetActiveBus("n_common")
        v_pu_angles = self.circuit.ActiveBus.puVmagAngle
        v_common_pu = float(np.mean(v_pu_angles[0::2]))

        dt_hours = dt_sec / 3600.0
        summary = {
            "mode": mode,
            "title": cfg["title"],
            "time_step_sec": dt_sec,
            "total_steps": int(len(p_pcc)),
            "solve_time_sec": round(solve_elapsed, 3),
            "converged": bool(self.solution.Converged),
            "nominal_load_kw": cfg["nom_kw"],
            "peak_load_kw": round(float(np.max(p_load)), 2),
            "peak_load_kvar": round(float(np.max(q_load)), 2),
            "peak_pv_ac_kw": round(float(np.max(p_pv_gen)), 2),
            "peak_pv_kvar_injected": round(float(np.max(q_pv_inj)), 2),
            "pcc_max_kvar_drawn": round(float(np.max(np.abs(q_pcc))), 2),
            "mean_pcc_pf": round(float(np.mean(pf_pcc)), 4),
            "common_bus_voltage_pu": round(v_common_pu, 4),
            "total_load_energy_kwh": round(float(np.sum(p_load) * dt_hours), 2),
            "total_pv_energy_kwh": round(float(np.sum(p_pv_gen) * dt_hours), 2),
            "total_pcc_net_energy_kwh": round(float(np.sum(p_pcc) * dt_hours), 2),
        }

        print(f"\n=== [{cfg['title']}] Time-Mode Simulation ({dt_sec}s step, {len(p_pcc):,} steps) ===")
        print(f"  Converged               : {summary['converged']} (Solved in {summary['solve_time_sec']} s)")
        print(f"  Installed Load Capacity : {summary['nominal_load_kw']:.1f} kW (PF={cfg['pf']:.2f})")
        print(f"  Simulated Peak Demand   : {summary['peak_load_kw']:.2f} kW | {summary['peak_load_kvar']:.2f} kvar")
        print(f"  Peak PV Output (Clipped): {summary['peak_pv_ac_kw']:.2f} kW AC (from 270.0 kWp DC)")
        print(f"  Inverter Q Compensation : {summary['peak_pv_kvar_injected']:.2f} kvar max injected")
        print(f"  PCC Residual Reactive Q : {summary['pcc_max_kvar_drawn']:.2f} kvar | Mean PCC PF = {summary['mean_pcc_pf']:.4f}")
        print(f"  Common Bus Voltage      : {summary['common_bus_voltage_pu']:.4f} pu")

        if save_plot:
            self._plot_time_results(
                mode, cfg, dt_sec, p_load, q_load, p_pv_gen, q_pv_inj, p_pcc, q_pcc, pf_pcc
            )

        return summary

    def _plot_time_results(
        self, mode: str, cfg: dict, dt_sec: int,
        p_load: np.ndarray, q_load: np.ndarray,
        p_pv_gen: np.ndarray, q_pv_inj: np.ndarray,
        p_pcc: np.ndarray, q_pcc: np.ndarray, pf_pcc: np.ndarray
    ):
        """Generates a 3-panel diagnostic plot of the 1-second Time-Mode simulation."""
        t_hours = np.arange(len(p_load)) * (dt_sec / 3600.0)

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle(f"{cfg['title']} — OpenDSS-G Time-Mode ({dt_sec}s Step, 3x Delta M88H Inverters)", fontsize=13, fontweight="bold")

        # Panel 1: Active Power Flows & 210 kW Inverter Clipping
        ax1 = axes[0]
        ax1.plot(t_hours, p_load, label=f"Load Demand (Nom={cfg['nom_kw']} kW, Peak={np.max(p_load):.1f} kW)", color="#1f77b4", lw=1.6)
        ax1.plot(t_hours, p_pv_gen, label="PV AC Output (Clipped at 210 kW)", color="#ff7f0e", lw=1.6)
        ax1.axhline(210.0, color="#d62728", ls="--", lw=1.1, label="Inverter Rating Limit (210 kVA)")
        ax1.plot(t_hours, p_pcc, label="Net PCC Power (Grid Exchange)", color="#2ca02c", lw=1.3, alpha=0.85)
        ax1.set_ylabel("Active Power (kW)")
        ax1.legend(loc="upper right", fontsize=8.5)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Reactive Power Compensation (Q_load vs Q_inv vs Q_pcc)
        ax2 = axes[1]
        ax2.plot(t_hours, q_load, label=f"Load Reactive Demand Q_load (PF={cfg['pf']})", color="#9467bd", lw=1.6)
        ax2.plot(t_hours, q_pv_inj, label="Inverter Reactive Injection Q_inv (PF Comp)", color="#e377c2", lw=1.5, ls="--")
        ax2.plot(t_hours, q_pcc, label="Residual Grid PCC Reactive Power Q_pcc", color="#17becf", lw=1.4)
        ax2.set_ylabel("Reactive Power (kvar)")
        ax2.legend(loc="upper right", fontsize=8.5)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Compensated PCC Power Factor
        ax3 = axes[2]
        ax3.plot(t_hours, pf_pcc, label="Compensated PCC Power Factor", color="#2ca02c", lw=1.6)
        ax3.axhline(cfg["pf"], color="#7f7f7f", ls=":", lw=1.2, label=f"Uncompensated Load PF ({cfg['pf']})")
        ax3.set_ylim(0.85, 1.02)
        ax3.set_ylabel("Power Factor")
        ax3.set_xlabel("Simulation Time (Hours, 1-Second Resolution)")
        ax3.legend(loc="lower right", fontsize=8.5)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        out_plot = RESULTS_DSS_DIR / f"opendss_time_1s_{mode}.png"
        plt.savefig(out_plot, dpi=150)
        plt.close(fig)
        print(f"  Saved Time-Mode plot    : {out_plot}")


def run_all_cases(dt_sec: int = 1, pf_compensation: bool = True, zero_export: bool = False):
    """Runs all 3 microgrid cases in Time-Mode and saves consolidated summary JSON."""
    runner = OpenDSSTimeRunner()
    results = {}
    for mode in ["supermarket", "ev", "caltech_ev"]:
        results[mode] = runner.run_time_simulation(
            mode=mode,
            dt_sec=dt_sec,
            hours=24,
            pf_compensation=pf_compensation,
            zero_export_curtailment=zero_export,
            save_plot=True
        )
    out_json = RESULTS_DSS_DIR / "opendss_time_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved consolidated OpenDSS Time-Mode summary to: {out_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenDSS-G 1-Second Time-Mode & Inverter Control Runner")
    parser.add_argument(
        "--mode",
        choices=["supermarket", "ev", "caltech_ev", "all"],
        default="all",
        help="Target case to simulate in OpenDSS Time-Mode"
    )
    parser.add_argument(
        "--dt-sec",
        type=int,
        default=1,
        help="Time step in seconds for Mode=Time simulation (default: 1s)"
    )
    parser.add_argument(
        "--no-pf-comp",
        action="store_true",
        help="Disable active inverter reactive power PF compensation"
    )
    parser.add_argument(
        "--zero-export",
        action="store_true",
        help="Enable autonomous zero-export active power curtailment (%Pmpp throttling)"
    )
    args = parser.parse_args()

    if args.mode == "all":
        run_all_cases(
            dt_sec=args.dt_sec,
            pf_compensation=not args.no_pf_comp,
            zero_export=args.zero_export
        )
    else:
        runner = OpenDSSTimeRunner()
        runner.run_time_simulation(
            mode=args.mode,
            dt_sec=args.dt_sec,
            pf_compensation=not args.no_pf_comp,
            zero_export_curtailment=args.zero_export
        )
