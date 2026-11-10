# -*- coding: utf-8 -*-
"""
OpenDSS-G Time-Mode Controller via DSTCP Protocol (opendss_dstcp_controller.py)
================================================================================
Communicates directly with the running OpenDSS-G TCP Server (127.0.0.1:6345)
using the DSTCP protocol (`openDSSGTCP.py`) — NO COM (`win32com`) interface used.

Closed-Loop 24-Hour (96-Step) Native Time-Mode (`stepIn`) Architecture:
  1. Establish Connection & Configure Time Mode Once:
     `set time=(0,0) mode=Daily controlmode=time stepsize=15m number=1 maxiterations=50 maxcontroliter=100`
     OpenDSS-G drives `Load.load_1` (`daily=target_load_shape`) and `PVSystem.pv_1` (`daily=solar_shape`)
     natively from the 96-step `LoadShape.dss` of the selected `start_date`.
  2. At each 15-minute step k in 0..95:
     - Step A (Advance Time): Call `ODGTCP.stepIn()` (`23;`) to advance 15 minutes in OpenDSS-G.
     - Step B (Request Values from OpenDSS-G):
       Read 3-phase branch/element powers (`getLinesPowers`), Battery SoC (`? Storage.sto_0.%stored`),
       and bus voltages in p.u. (`getVoltsPU`).
     - Step C (Compare with MPC Schedule & Compute RT Control):
       Compare OpenDSS-G's requested values against row `k` of the 24-hour Day-Ahead MPC Schedule
       (`mpc_df.iloc[k]`) inside `RealTimeStepCompensator` to compute the compensated battery
       charging/discharging power and the zero-reinjection `%Pmpp` ceiling.
     - Step D (Update Battery & %Pmpp in OpenDSS-G):
       Send `Storage.sto_0` (`state`, `%charge` / `%discharge`) and `PVSystem.pv_1` (`%Pmpp`) commands.
"""

import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import openDSSGTCP as dssTCP
from openDSSGTCP import parse_float_token
from mpc_optimizer import (
    load_battery_sizing,
    resolve_valid_start_date,
    run_mpc_pipeline,
)
from rt_controller import RealTimeStepCompensator

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
    },
}


class OpenDSSGDSTCPController:
    """
    Time-Mode Controller that drives OpenDSS-G over TCP/IP (DSTCP protocol on port 6345)
    using `stepIn()` across 96 steps (15-minute intervals = 24 hours).
    """

    def __init__(self, tcp_ip="127.0.0.1", tcp_port=6345, buffer_size=80000, step_delay=0.45):
        self.tcp_ip = tcp_ip
        self.tcp_port = tcp_port
        self.buffer_size = buffer_size
        self.step_delay = step_delay
        self.odg = dssTCP.OpenDSSGTCP(tcp_ip, tcp_port, buffer_size)

        # Delta M88H-121 MPPT efficiency curve points
        self.eff_x = np.array([0.1, 0.2, 0.5, 1.0])
        self.eff_y = np.array([0.965, 0.980, 0.988, 0.985])
        self.pv_kva = 210.0
        self.pv_pmpp = 210.0

    def connect(self):
        """Starts the TCP connection to OpenDSS-G on 127.0.0.1:6345."""
        self.odg.startTCPConnection()

    def close(self):
        """Closes the TCP connection to OpenDSS-G."""
        self.odg.closeTCPConnection()

    def _get_sim_time_sec(self) -> float:
        """Queries OpenDSS-G simulation clock in seconds via DSTCP Function 01 (`seconds`)."""
        try:
            toks = self.odg.getTimeSeconds()
            if toks:
                return parse_float_token(toks[0])
        except Exception:
            pass
        return -1.0

    def _advance_substep(self, prev_sim_sec: float) -> float:
        """Advances OpenDSS-G by one 7.5-minute (450s) stage via stepIn() and verifies clock progression."""
        curr_sim_sec = prev_sim_sec
        for attempt in range(5):
            self.odg.stepIn()
            time.sleep(self.step_delay if attempt == 0 else 0.40)
            curr_sim_sec = self._get_sim_time_sec()
            if curr_sim_sec < 0 or prev_sim_sec < 0 or abs(curr_sim_sec - prev_sim_sec) > 1.0:
                return curr_sim_sec
        return curr_sim_sec

    def _query_float_array(self, dss_query: str) -> np.ndarray:
        """Queries a bracketed float array from OpenDSS-G via Function 25."""
        resp_tokens = self.odg.sendCommand(dss_query)
        raw = " ".join(resp_tokens)
        raw = raw.replace("[", " ").replace("]", " ")
        nums = [parse_float_token(tok) for tok in raw.split() if tok.strip()]
        return np.array(nums, dtype=float)

    def sync_case_loadshapes_for_date(self, mode: str, start_date: str = None):
        """
        Populates the OpenDSS-G `LoadShape.dss` file for `mode` with the actual 96-step (24-hour)
        consumption and solar PV trajectories of `start_date`, and ensures the 96-step Day-Ahead
        MPC schedule (`mpc_timeseries_{mode}.csv`) is ready for step-by-step comparison.
        """
        cfg = CASE_CONFIGS[mode]
        start_date = resolve_valid_start_date(mode, start_date)
        mpc_csv = RESULTS_MPC_DIR / f"mpc_timeseries_{mode}.csv"
        mpc_json = RESULTS_MPC_DIR / f"mpc_metrics_{mode}.json"

        need_mpc_run = True
        if mpc_csv.exists() and mpc_json.exists():
            try:
                with open(mpc_json, "r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                df_check = pd.read_csv(mpc_csv)
                if meta.get("date") == start_date and len(df_check) >= 96 and "P_PV_actual_kW" in df_check.columns:
                    need_mpc_run = False
            except Exception:
                need_mpc_run = True

        if need_mpc_run:
            print(f"\n[DSTCP Pre-Flight] Generating 24-Hour (96-step) Day-Ahead MPC Schedule & Actuals for {mode.upper()} on {start_date}...")
            run_mpc_pipeline(mode=mode, start_date=start_date, horizon=96)

        mpc_df = pd.read_csv(mpc_csv).iloc[:96].reset_index(drop=True)
        pv_actual = mpc_df["P_PV_actual_kW"].values
        load_actual = mpc_df["P_load_actual_kW"].values
        p_ch_sch = mpc_df["P_ch_sch_kW"].values
        p_dis_sch = mpc_df["P_dis_sch_kW"].values

        nom_kw = float(cfg["nom_kw"])
        bat_kw = max(1e-3, float(cfg["bat_kw"]))

        # Compute 96 normalized multipliers for LoadShape.dss
        m_load = np.round(np.clip(load_actual / nom_kw, 0.0, None), 6)
        m_solar = np.zeros(96, dtype=float)
        for k in range(96):
            p_ac = float(pv_actual[k])
            if p_ac < 0.05:
                m_solar[k] = 0.0
            else:
                p_ac_clip = min(self.pv_kva, p_ac)
                eta_k = float(np.interp(p_ac_clip / self.pv_kva, self.eff_x, self.eff_y))
                p_dc_k = p_ac_clip / eta_k
                m_solar[k] = round(min(1.0, p_dc_k / self.pv_pmpp), 6)

        m_bess = np.round(np.clip((p_dis_sch - p_ch_sch) / bat_kw, -1.0, 1.0), 4)

        # Duplicate each 15-minute value into two 7.5-minute (450s) sub-steps (npts=192, sinterval=450):
        # Stage 1 (7.5 min): Sense uncurtailed values | Stage 2 (15.0 min): Actuate & Record results
        m_load_192 = np.repeat(m_load, 2)
        m_solar_192 = np.repeat(m_solar, 2)
        m_bess_192 = np.repeat(m_bess, 2)

        str_load = " ".join(f"{val:.6f}" for val in m_load_192)
        str_solar = " ".join(f"{val:.6f}" for val in m_solar_192)
        str_bess = " ".join(f"{val:.4f}" for val in m_bess_192)

        loadshape_path = ROOT_DIR / cfg["folder"] / "LoadShape.dss"
        dss_content = (
            f"! --------------------------------------------------------------------------\n"
            f"! LoadShape.dss - {cfg['title']} | Date: {start_date} (192 sub-steps of 7.5m = 24h)\n"
            f"! Actual Peak Load: {load_actual.max():.2f} kW (Nominal: {nom_kw:.1f} kW) | Actual Peak PV: {pv_actual.max():.2f} kW (Pmpp: {self.pv_pmpp:.1f} kW)\n"
            f"! --------------------------------------------------------------------------\n\n"
            f"! 1. Target Demand Profile ({start_date} Actual Load / {nom_kw:.1f} kW)\n"
            f'New "LoadShape.target_load_shape" npts=192 sinterval=450 mult=[{str_load}]\n\n'
            f"! 2. Solar PV Generation Shape ({start_date} Actual PV / {self.pv_pmpp:.1f} kW Pmpp)\n"
            f'New "LoadShape.solar_shape" npts=192 sinterval=450 mult=[{str_solar}]\n\n'
            f"! 3. Battery Storage Scheduled Dispatch Shape ({start_date} MPC Schedule)\n"
            f'New "LoadShape.bess_dispatch" npts=192 sinterval=450 mult=[{str_bess}]\n'
        )
        with open(loadshape_path, "w", encoding="utf-8") as lf:
            lf.write(dss_content)

        print(f"  [LoadShape Sync] Updated {loadshape_path.name} for {start_date} (192x7.5m sub-steps = 24h, Peak Load={load_actual.max():.2f} kW, Peak PV={pv_actual.max():.2f} kW)")
        return mpc_df, m_load, m_solar, m_bess, start_date

    def run_96_steps(
        self,
        mode: str = "supermarket",
        source: str = "rt",
        start_date: str = None,
        compile_master: bool = True,
        refresh_gui: bool = False,
        show_plots: bool = False,
    ) -> dict:
        """
        Runs the 96-step (15-minute step = 24 hours) simulation in OpenDSS-G via DSTCP
        using the native Time/Daily mode + `stepIn()` workflow (Option A):
          1. Establish connection & configure `mode=Daily controlmode=time stepsize=15m number=1`
          2. Loop k = 0..95:
             - Advance 15 min (`ODGTCP.stepIn()`)
             - Request values from OpenDSS-G (Powers, Battery %stored, Bus Voltages pu)
             - Compare with step k of the MPC schedule & compute RT control
             - Update Battery (`Storage.sto_0`) and PV `%Pmpp` (`PVSystem.pv_1`) in OpenDSS-G
        """
        cfg = CASE_CONFIGS[mode]
        master_path = ROOT_DIR / cfg["folder"] / "Master.dss"
        sizing = load_battery_sizing(mode)

        # 1. Sync LoadShape.dss for `start_date` and load the 96-step Day-Ahead MPC schedule
        mpc_df, m_load_arr, m_solar_arr, m_bess_arr, start_date = self.sync_case_loadshapes_for_date(mode, start_date)

        # Instantiate the 4-Layer Real-Time Secondary Controller for step-by-step comparison
        rt_compensator = RealTimeStepCompensator(
            sizing_params=sizing,
            dt=0.25,
            soc_init=0.50,
            epsilon_tol=0.50,
            ramp_limit_fraction=1.00,
            initial_p_bat=float(mpc_df["P_dis_sch_kW"].iloc[0] - mpc_df["P_ch_sch_kW"].iloc[0])
        )

        self.connect()
        try:
            if compile_master and master_path.exists():
                self.odg.sendCommand(f'compile "{master_path}"')
                time.sleep(self.step_delay)

            # Update the 192-substep (7.5m = 450s) LoadShapes in OpenDSS-G memory
            str_load = " ".join(f"{val:.6f}" for val in np.repeat(m_load_arr, 2))
            str_solar = " ".join(f"{val:.6f}" for val in np.repeat(m_solar_arr, 2))
            str_bess = " ".join(f"{val:.4f}" for val in np.repeat(m_bess_arr, 2))
            self.odg.sendCommand(f"Edit LoadShape.target_load_shape npts=192 sinterval=450 mult=[{str_load}]")
            self.odg.sendCommand(f"Edit LoadShape.solar_shape npts=192 sinterval=450 mult=[{str_solar}]")
            self.odg.sendCommand(f"Edit LoadShape.bess_dispatch npts=192 sinterval=450 mult=[{str_bess}]")

            # Configure OpenDSS-G in Daily Time mode with 7.5-minute (450s) sub-step size:
            # 96 steps x 2 stages (7.5m sense + 7.5m actuate) = 192 sub-steps = 24 hours
            self.odg.sendCommand(
                "set time=(0,0) mode=Daily controlmode=time stepsize=450s number=1 maxiterations=50 maxcontroliter=100"
            )
            self.odg.sendCommand("SwtControl.sw_inv.Action=Close")
            self.odg.sendCommand(
                f"Edit Load.load_1 kW={cfg['nom_kw']} pf={cfg['pf']} daily=target_load_shape"
            )
            self.odg.sendCommand(
                "Edit PVSystem.pv_1 kVA=210.0 Pmpp=210.0 %Pmpp=100.0 %Cutin=0 %Cutout=0 irradiance=1 daily=solar_shape"
            )
            self.odg.sendCommand(
                f"Edit Storage.sto_0 kWrated={cfg['bat_kw']} kWhrated={cfg['bat_kwh']} %stored=50 %reserve=0 dispMode=DEFAULT state=IDLING"
            )
            time.sleep(0.1)

            # Read line names (Function 33) and node names (Function 43) once
            line_names = [ln.lower() for ln in self.odg.getLineNames()]
            node_names = [nd.lower() for nd in self.odg.getNodeNames()]

            idx_line_pcc = line_names.index("line_reactor") if "line_reactor" in line_names else 0
            idx_line_pv = line_names.index("line_pv") if "line_pv" in line_names else 2
            idx_line_bat = line_names.index("line_bat") if "line_bat" in line_names else 3
            idx_line_load = line_names.index("line_load") if "line_load" in line_names else 4

            idx_node_pcc = node_names.index("pcc.1") if "pcc.1" in node_names else 6
            idx_node_common = node_names.index("n_common.1") if "n_common.1" in node_names else 9
            idx_node_pv = node_names.index("n_pv.1") if "n_pv.1" in node_names else 12
            idx_node_bat = node_names.index("n_bat.1") if "n_bat.1" in node_names else 15
            idx_node_load = node_names.index("n_load.1") if "n_load.1" in node_names else 18

            # Use 96-step solar_shape multipliers (1 per 15-min interval k)
            solar_mults = m_solar_arr

            n_steps = 96
            dt_hours = 0.25  # 15 minutes (two 7.5-minute stages)
            nom_kw = cfg["nom_kw"]
            bat_kw_rated = cfg["bat_kw"]
            bat_kwh_rated = cfg["bat_kwh"]
            soc_min = 20.0
            soc_max = 90.0

            # Tracking arrays (96 steps = every 15 minutes after Stage 2 actuation)
            PVkW_avail = []
            PVkW = []
            CurtkW = []
            PctPmpp = []
            EVkW = []       # Load active power (kW)
            PbkW = []       # Battery power (+ discharging, - charging)
            PCCkW = []      # Grid power at PCC (+ importing from grid)
            SOC_bkW = []    # Battery SoC (%)
            PB_t = []       # Balance de potencia prod-cons (PV_avail - Load)

            PVkV = []
            EVkV = []
            BATkV = []
            PCCkV = []
            COMMONkV = []

            def sum_3ph_term(tokens, line_idx, term=1):
                base = line_idx * 12 + (0 if term == 1 else 6)
                return sum(parse_float_token(tokens[base + offset]) for offset in (0, 2, 4))

            sim_eb_kwh = 0.50 * bat_kwh_rated
            prev_sim_sec = self._get_sim_time_sec()
            t0 = time.time()

            print(f"\n==========================================================================")
            print(f" Starting OpenDSS-G Two-Stage 96-Step Control ({cfg['title']})")
            print(f" Date: {start_date} | TCP: {self.tcp_ip}:{self.tcp_port} | Stage 1 (7.5m Sense) + Stage 2 (15m Actuate) | Mode: {source.upper()}")
            print(f"==========================================================================")

            for k in range(n_steps):
                # ------------------------------------------------------------------
                # STAGE 1 (Minute +7.5m): Advance 7.5m with uncurtailed PV & Sense
                # ------------------------------------------------------------------
                soc_start_pct = (sim_eb_kwh / bat_kwh_rated) * 100.0 if bat_kwh_rated > 0 else 50.0
                self.odg.sendCommand("PVSystem.pv_1.%Pmpp=100.0")
                self.odg.sendCommand("Storage.sto_0.state=IDLING")

                prev_sim_sec = self._advance_substep(prev_sim_sec)

                # Recover uncurtailed available PV and Load measurements from OpenDSS-G at +7.5m
                lp_probe = self.odg.getLinesPowers()
                p_pv_probe = max(0.0, sum_3ph_term(lp_probe, idx_line_pv, term=2))
                p_load_meas = abs(sum_3ph_term(lp_probe, idx_line_load, term=2))

                s_mult = float(solar_mults[k])
                p_dc_uncurt = self.pv_pmpp * s_mult
                if p_dc_uncurt < 0.05:
                    p_pv_shape_k = 0.0
                else:
                    eta_uncurt = float(np.interp(min(1.0, p_dc_uncurt / self.pv_kva), self.eff_x, self.eff_y))
                    p_pv_shape_k = min(self.pv_kva, p_dc_uncurt * eta_uncurt)
                p_pv_avail_k = max(p_pv_probe, p_pv_shape_k)
                soc_meas_pct = soc_start_pct

                # ------------------------------------------------------------------
                # CONTROL COMPUTATION: Compare Stage 1 values with Step k of MPC
                # ------------------------------------------------------------------
                row_mpc = mpc_df.iloc[k]
                rt_curt_cmd = 0.0
                if source == "rt":
                    rt_out = rt_compensator.compute_step(
                        k=k,
                        pv_meas_kw=p_pv_avail_k,
                        load_meas_kw=p_load_meas,
                        pv_fcast_kw=float(row_mpc["P_PV_forecast_kW"]),
                        load_fcast_kw=float(row_mpc["P_load_forecast_kW"]),
                        sch_ch_kw=float(row_mpc["P_ch_sch_kW"]),
                        sch_dis_kw=float(row_mpc["P_dis_sch_kW"]),
                        sch_grid_kw=float(row_mpc["P_grid_sch_kW"]),
                        current_eb_kwh=sim_eb_kwh
                    )
                    p_ch_cmd = rt_out["act_ch_kw"]
                    p_dis_cmd = rt_out["act_dis_kw"]
                    rt_curt_cmd = rt_out["act_curt_kw"]
                    sim_eb_kwh = rt_out["e_b_end_kwh"]
                elif source == "mpc":
                    sch_ch = float(row_mpc["P_ch_sch_kW"])
                    sch_dis = float(row_mpc["P_dis_sch_kW"])
                    max_ch_soc_kw = max(0.0, ((soc_max - soc_meas_pct) / 100.0) * bat_kwh_rated / (0.95 * dt_hours))
                    max_dis_soc_kw = max(0.0, ((soc_meas_pct - soc_min) / 100.0) * bat_kwh_rated * 0.95 / dt_hours)
                    p_ch_cmd = min(sch_ch, bat_kw_rated, max_ch_soc_kw)
                    p_dis_cmd = min(sch_dis, bat_kw_rated, max_dis_soc_kw, p_load_meas)
                    sim_eb_kwh = float(np.clip(
                        sim_eb_kwh + (p_ch_cmd * 0.95 - p_dis_cmd / 0.95) * dt_hours,
                        (soc_min / 100.0) * bat_kwh_rated,
                        (soc_max / 100.0) * bat_kwh_rated
                    ))
                else:
                    # Mode A ('loadshape'): Local Greedy BMS Rule on requested OpenDSS-G measurements
                    PBt = p_pv_avail_k - p_load_meas
                    if PBt >= 0.0:
                        max_ch_soc_kw = max(0.0, ((soc_max - soc_meas_pct) / 100.0) * bat_kwh_rated / (0.95 * dt_hours))
                        p_ch_cmd = min(PBt, bat_kw_rated, max_ch_soc_kw)
                        p_dis_cmd = 0.0
                    else:
                        max_dis_soc_kw = max(0.0, ((soc_meas_pct - soc_min) / 100.0) * bat_kwh_rated * 0.95 / dt_hours)
                        p_dis_cmd = min(abs(PBt), bat_kw_rated, max_dis_soc_kw)
                        p_ch_cmd = 0.0
                    sim_eb_kwh = float(np.clip(
                        sim_eb_kwh + (p_ch_cmd * 0.95 - p_dis_cmd / 0.95) * dt_hours,
                        (soc_min / 100.0) * bat_kwh_rated,
                        (soc_max / 100.0) * bat_kwh_rated
                    ))

                soc_end_pct = (sim_eb_kwh / bat_kwh_rated) * 100.0 if bat_kwh_rated > 0 else 50.0
                soc_mid_pct = 0.5 * (soc_start_pct + soc_end_pct)

                # ------------------------------------------------------------------
                # STAGE 2 (Minute +15.0m): Send Control Setpoints, stepIn(), & Record
                # ------------------------------------------------------------------
                # 1. Pre-load midpoint SoC first so Stage 2's 7.5m integration lands on soc_end_pct,
                #    then send state + %charge/%discharge (after %stored so state is never reset to IDLING)
                self.odg.sendCommand(f"Storage.sto_0.%stored={soc_mid_pct:.4f}")
                p_bat_net_cmd = p_dis_cmd - p_ch_cmd
                if abs(p_bat_net_cmd) > 1e-4 and bat_kw_rated > 0:
                    if p_bat_net_cmd > 0:
                        pct_dis = min(100.0, (p_bat_net_cmd / bat_kw_rated) * 100.0)
                        self.odg.sendCommand(f"Edit Storage.sto_0 state=DISCHARGING %discharge={pct_dis:.4f}")
                    else:
                        pct_ch = min(100.0, (abs(p_bat_net_cmd) / bat_kw_rated) * 100.0)
                        self.odg.sendCommand(f"Edit Storage.sto_0 state=CHARGING %charge={pct_ch:.4f}")
                else:
                    self.odg.sendCommand("Storage.sto_0.state=IDLING")

                # 2. Compute and send %Pmpp ceiling for Stage 2
                p_local_sink = max(0.0, p_load_meas + p_ch_cmd - p_dis_cmd)
                need_curtailment = (rt_curt_cmd > 0.10) if source == "rt" else (p_pv_avail_k > p_local_sink + 0.10)
                if need_curtailment and p_pv_avail_k > p_local_sink:
                    pct_pmpp = max(0.01, min(100.0, (p_local_sink / self.pv_pmpp) * 100.0))
                else:
                    pct_pmpp = 100.0
                self.odg.sendCommand(f"PVSystem.pv_1.%Pmpp={pct_pmpp:.4f}")

                # 3. Advance the second 7.5m stage via stepIn() to complete the 15-minute step
                prev_sim_sec = self._advance_substep(prev_sim_sec)
                if refresh_gui:
                    self.odg.RefreshGUI()

                # 4. Read settled post-actuation powers & voltages at minute +15.0m and record
                lp_tokens = self.odg.getLinesPowers()
                p_pcc_meas = sum_3ph_term(lp_tokens, idx_line_pcc, term=1)
                p_pv_meas = max(0.0, sum_3ph_term(lp_tokens, idx_line_pv, term=2))
                p_bat_meas = sum_3ph_term(lp_tokens, idx_line_bat, term=2)
                p_load_meas = abs(sum_3ph_term(lp_tokens, idx_line_load, term=2))
                soc_meas_pct = soc_end_pct

                vpu_tokens = self.odg.getVoltsPU()
                v_pcc = parse_float_token(vpu_tokens[idx_node_pcc])
                v_common = parse_float_token(vpu_tokens[idx_node_common])
                v_pv = parse_float_token(vpu_tokens[idx_node_pv])
                v_bat = parse_float_token(vpu_tokens[idx_node_bat])
                v_load = parse_float_token(vpu_tokens[idx_node_load])

                # Record step metrics
                p_curt_meas = max(0.0, p_pv_avail_k - p_pv_meas)
                PVkW_avail.append(p_pv_avail_k)
                PVkW.append(p_pv_meas)
                CurtkW.append(p_curt_meas)
                PctPmpp.append(pct_pmpp)
                EVkW.append(p_load_meas)
                PbkW.append(p_bat_meas)
                PCCkW.append(p_pcc_meas)
                SOC_bkW.append(soc_meas_pct)
                PB_t.append(p_pv_avail_k - p_load_meas)

                PVkV.append(v_pv)
                EVkV.append(v_load)
                BATkV.append(v_bat)
                PCCkV.append(v_pcc)
                COMMONkV.append(v_common)

                if k % 8 == 0 or k == n_steps - 1 or p_curt_meas > 1.0:
                    h_disp = (k * 15) // 60
                    m_disp = (k * 15) % 60
                    print(
                        f"  Step {k+1:02d}/96 [{h_disp:02d}:{m_disp:02d}] | "
                        f"Load={p_load_meas:6.2f} kW | PV_avail={p_pv_avail_k:6.2f} kW -> "
                        f"PV_inj={p_pv_meas:6.2f} kW (Curt={p_curt_meas:6.2f} kW, %Pmpp={pct_pmpp:5.1f}%) | "
                        f"Bat={p_bat_meas:+6.2f} kW (SoC={soc_meas_pct:5.1f}%) | "
                        f"PCC={p_pcc_meas:+6.2f} kW | V_com={v_common:.4f} pu"
                    )

            elapsed = time.time() - t0
        finally:
            self.close()

        # Build results DataFrame & summary
        hours_arr = np.arange(n_steps) * dt_hours
        df_res = pd.DataFrame({
            "step": np.arange(n_steps),
            "hour": hours_arr,
            "date": start_date,
            "P_load_kW": EVkW,
            "P_PV_avail_clipped_kW": PVkW_avail,
            "P_PV_dispatched_kW": PVkW,
            "P_curt_kW": CurtkW,
            "PV_pct_Pmpp": PctPmpp,
            "P_bat_net_kW": PbkW,
            "SoC_pct": SOC_bkW,
            "P_grid_pcc_kW": PCCkW,
            "P_balance_prod_cons_kW": PB_t,
            "V_pv_pu": PVkV,
            "V_load_pu": EVkV,
            "V_bat_pu": BATkV,
            "V_pcc_pu": PCCkV,
            "V_common_pu": COMMONkV,
        })

        csv_out = RESULTS_DSS_DIR / f"dstcp_96step_{mode}.csv"
        df_res.to_csv(csv_out, index=False)

        min_pcc = float(np.min(PCCkW))
        summary = {
            "protocol": "DSTCP (127.0.0.1:6345) — Time-Mode stepIn()",
            "mode": mode,
            "title": cfg["title"],
            "date": start_date,
            "source": source,
            "steps": n_steps,
            "step_size_min": 15,
            "elapsed_sec": round(elapsed, 2),
            "installed_load_kw": nom_kw,
            "peak_load_kw": round(float(np.max(EVkW)), 2),
            "peak_pv_avail_clipped_kw": round(float(np.max(PVkW_avail)), 2),
            "peak_pv_dispatched_kw": round(float(np.max(PVkW)), 2),
            "total_pv_curtailed_kwh": round(float(np.sum(CurtkW) * dt_hours), 2),
            "total_grid_imported_kwh": round(float(np.sum(np.maximum(0.0, PCCkW)) * dt_hours), 2),
            "min_pcc_kw": round(min_pcc, 4),
            "zero_reinjection_passed": bool(min_pcc >= -0.05),
            "v_common_min_pu": round(float(np.min(COMMONkV)), 4),
            "v_common_max_pu": round(float(np.max(COMMONkV)), 4),
        }

        print(f"\n=== [{cfg['title']} | {start_date}] OpenDSS-G DSTCP 96-Step Summary ===")
        print(f"  Protocol                   : {summary['protocol']}")
        print(f"  Control Source             : {source.upper()} | Date: {start_date}")
        print(f"  Peak Simulated Load        : {summary['peak_load_kw']:.2f} kW (Installed: {nom_kw:.1f} kW)")
        print(f"  Peak PV (Clipped -> Disp.) : {summary['peak_pv_avail_clipped_kw']:.2f} kW -> {summary['peak_pv_dispatched_kw']:.2f} kW")
        print(f"  Total Curtailed PV Energy  : {summary['total_pv_curtailed_kwh']:.2f} kWh")
        print(f"  Total Grid Imported Energy : {summary['total_grid_imported_kwh']:.2f} kWh")
        print(f"  Zero Re-Injection Check    : Min PCC = {summary['min_pcc_kw']:.4f} kW [{'PASSED' if summary['zero_reinjection_passed'] else 'VIOLATED'}]")
        print(f"  Common Bus Voltage Range   : [{summary['v_common_min_pu']:.4f}, {summary['v_common_max_pu']:.4f}] pu")
        print(f"  Saved 96-step DSTCP CSV    : {csv_out}")

        # Save per-case JSON metrics for the Web UI dashboard
        metrics_json = RESULTS_DSS_DIR / f"dstcp_metrics_{mode}.json"
        with open(metrics_json, "w", encoding="utf-8") as mf:
            json.dump(summary, mf, indent=2)

        self._plot_dstcp_results(mode, cfg, df_res, summary, show_plots=show_plots)
        return summary

    def _plot_dstcp_results(self, mode: str, cfg: dict, df: pd.DataFrame, summary: dict, show_plots: bool = False):
        """
        Generates:
          1. A 2x2 Classic Microgrid Figure (`dstcp_classic_4figs_{mode}.png`)
          2. A 3-panel Detailed Curtailment & Zero-Export Dashboard (`dstcp_96step_{mode}.png`).
        """
        hrs = df["hour"].values
        date_str = summary.get("date", "")
        src_str = summary.get("source", "").upper()

        # --- Plot 1: Classic 4-Figure Layout (Power MG, Voltage MG, Balance Prod-Cons, Voltage PCC) ---
        fig1, axes1 = plt.subplots(2, 2, figsize=(14, 9))
        fig1.suptitle(
            f"{cfg['title']} — OpenDSS-G Time-Mode stepIn() ({date_str} | Mode: {src_str} | 96 Steps = 24h)",
            fontsize=13, fontweight="bold"
        )

        # Subplot 1 (Figure 1): Power MG1
        ax_p = axes1[0, 0]
        ax_p.plot(hrs, df["P_PV_avail_clipped_kW"], color="#ffbb78", ls="--", lw=1.4, label="PV Available (Clipped <= 210 kW)")
        ax_p.plot(hrs, df["P_PV_dispatched_kW"], color="red", lw=1.8, label="PV Dispatched (After Curtailment)")
        ax_p.plot(hrs, df["P_load_kW"], color="green", lw=1.8, label="Load Demand")
        ax_p.plot(hrs, df["P_bat_net_kW"], color="orange", lw=1.8, label="Battery (+Dis / -Chg)")
        ax_p.plot(hrs, df["P_grid_pcc_kW"], color="#1f77b4", lw=1.5, ls="-.", label="PCC Grid Import (>= 0)")
        ax_p.axhline(0.0, color="black", lw=0.8, ls=":")
        ax_p.set_title("Figure 1: Power MG1 (kW)")
        ax_p.set_xlabel(f"Time (h) on {date_str} [96 Steps of 15 min]")
        ax_p.set_ylabel("Power (kW)")
        ax_p.grid(True, linestyle="-.", alpha=0.5)
        ax_p.legend(loc="upper right", fontsize=8)

        # Subplot 2 (Figure 2): Voltage MG (PV, Load, Battery)
        ax_v = axes1[0, 1]
        ax_v.plot(hrs, df["V_pv_pu"], label="V_PV (n_pv)", lw=1.6)
        ax_v.plot(hrs, df["V_load_pu"], label="V_Load (n_load)", lw=1.6)
        ax_v.plot(hrs, df["V_bat_pu"], label="V_Bat (n_bat)", lw=1.6)
        ax_v.set_title("Figure 2: Voltage MG (pu)")
        ax_v.set_xlabel(f"Time (h) on {date_str} [96 Steps of 15 min]")
        ax_v.set_ylabel("Voltage (pu)")
        ax_v.grid(True, linestyle="-.", alpha=0.5)
        ax_v.legend(loc="lower right", fontsize=8.5)

        # Subplot 3 (Figure 3): Balance de Potencia Prod-Cons & Curtailed Power
        ax_b = axes1[1, 0]
        ax_b.plot(hrs, df["P_balance_prod_cons_kW"], color="#2ca02c", lw=1.8, label="Net Balance (PV_avail - Load)")
        ax_b.plot(hrs, df["P_curt_kW"], color="#d62728", lw=1.8, label=f"Curtailed PV ({summary['total_pv_curtailed_kwh']:.1f} kWh)")
        ax_b.fill_between(hrs, 0, df["P_curt_kW"], color="#d62728", alpha=0.18)
        ax_b.axhline(0.0, color="black", lw=0.8, ls=":")
        ax_b.set_title("Figure 3: Balance de Potencia Prod-Cons & Curtailment")
        ax_b.set_xlabel(f"Time (h) on {date_str} [96 Steps of 15 min]")
        ax_b.set_ylabel("P (kW)")
        ax_b.grid(True, linestyle="-.", alpha=0.5)
        ax_b.legend(loc="upper right", fontsize=8.5)

        # Subplot 4 (Figure 4): Voltage PCC MG & Battery SoC
        ax_pcc = axes1[1, 1]
        ax_pcc.plot(hrs, df["V_pcc_pu"], color="#17becf", lw=1.8, label="V_PCC (pu)")
        ax_pcc.plot(hrs, df["V_common_pu"], color="#7f7f7f", lw=1.4, ls="--", label="V_Common (pu)")
        ax_pcc.set_title("Figure 4: Voltage PCC MG & Battery SoC")
        ax_pcc.set_xlabel(f"Time (h) on {date_str} [96 Steps of 15 min]")
        ax_pcc.set_ylabel("Voltage (pu)")
        ax_pcc.grid(True, linestyle="-.", alpha=0.5)

        ax_soc = ax_pcc.twinx()
        ax_soc.plot(hrs, df["SoC_pct"], color="#9467bd", lw=1.6, ls="-.", label="Battery SoC (%)")
        ax_soc.set_ylabel("Battery SoC (%)")
        ax_soc.set_ylim(10, 100)

        l1, lb1 = ax_pcc.get_legend_handles_labels()
        l2, lb2 = ax_soc.get_legend_handles_labels()
        ax_pcc.legend(l1 + l2, lb1 + lb2, loc="lower right", fontsize=8.5)

        plt.tight_layout()
        classic_path = RESULTS_DSS_DIR / f"dstcp_classic_4figs_{mode}.png"
        fig1.savefig(classic_path, dpi=150)

        # --- Plot 2: 3-Panel Detailed Co-Simulation Dashboard ---
        fig2, axes2 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig2.suptitle(
            f"{cfg['title']} — OpenDSS-G Time-Mode stepIn() ({date_str} | Mode: {src_str}) "
            f"[Min PCC = {summary['min_pcc_kw']:.3f} kW]",
            fontsize=12.5, fontweight="bold"
        )

        ax1 = axes2[0]
        ax1.plot(hrs, df["P_load_kW"], label=f"Load Demand (Peak={summary['peak_load_kw']:.1f} kW)", color="#1f77b4", lw=1.8)
        ax1.plot(hrs, df["P_PV_avail_clipped_kW"], label="Available PV (Clipped <= 210 kW)", color="#ffbb78", lw=1.4, ls="--")
        ax1.plot(hrs, df["P_PV_dispatched_kW"], label="Dispatched PV (After %Pmpp Curtailment)", color="#ff7f0e", lw=1.8)
        ax1.fill_between(
            hrs, df["P_PV_dispatched_kW"], df["P_PV_avail_clipped_kW"],
            color="#ff7f0e", alpha=0.22, label=f"Curtailed PV ({summary['total_pv_curtailed_kwh']:.1f} kWh)"
        )
        ax1.plot(hrs, df["P_grid_pcc_kW"], label="PCC Grid Import (P_grid >= 0)", color="#2ca02c", lw=1.6)
        ax1.axhline(0.0, color="black", lw=0.8, ls=":")
        ax1.set_ylabel("Active Power (kW)")
        ax1.legend(loc="upper right", fontsize=8.5, ncol=2)
        ax1.grid(True, alpha=0.3)

        ax2 = axes2[1]
        ax2.plot(hrs, df["P_bat_net_kW"], label="Battery Net Power (+Dis / -Chg)", color="#9467bd", lw=1.7)
        ax2.axhline(0.0, color="black", lw=0.8, ls=":")
        ax2.set_ylabel("Battery Power (kW)")
        ax2.grid(True, alpha=0.3)

        ax2_r = ax2.twinx()
        ax2_r.plot(hrs, df["SoC_pct"], label="Battery SoC (%)", color="#d62728", lw=1.5, ls="-.")
        ax2_r.set_ylabel("State of Charge (%)")
        ax2_r.set_ylim(10, 100)
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8.5)

        ax3 = axes2[2]
        ax3.plot(hrs, df["V_common_pu"], label="Common Bus V_common (pu)", color="#17becf", lw=1.7)
        ax3.plot(hrs, df["V_pcc_pu"], label="PCC Bus V_pcc (pu)", color="#7f7f7f", lw=1.3, ls="--")
        ax3.set_ylabel("Bus Voltage (pu)")
        ax3.set_xlabel(f"Time of Day on {date_str} (Hours, 96 Steps of 15 Minutes)")
        ax3.set_ylim(0.96, 1.04)
        ax3.grid(True, alpha=0.3)

        ax3_r = ax3.twinx()
        ax3_r.plot(hrs, df["PV_pct_Pmpp"], label="PV Inverter Ceiling %Pmpp (%)", color="#e377c2", lw=1.2, ls=":")
        ax3_r.set_ylabel("PV %Pmpp (%)")
        ax3_r.set_ylim(0, 105)
        l3_1, lb3_1 = ax3.get_legend_handles_labels()
        l3_2, lb3_2 = ax3_r.get_legend_handles_labels()
        ax3.legend(l3_1 + l3_2, lb3_1 + lb3_2, loc="lower right", fontsize=8.5)

        plt.tight_layout()
        dash_path = RESULTS_DSS_DIR / f"dstcp_96step_{mode}.png"
        fig2.savefig(dash_path, dpi=150)

        if show_plots:
            plt.show()
        else:
            plt.close(fig1)
            plt.close(fig2)

        print(f"  Saved Classic 4-Fig Plot   : {classic_path}")
        print(f"  Saved 3-Panel Dashboard    : {dash_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenDSS-G 96-Step Time-Mode (`stepIn`) Controller via DSTCP (Port 6345)")
    parser.add_argument(
        "--mode",
        choices=["supermarket", "ev", "caltech_ev", "all"],
        default="supermarket",
        help="Microgrid case to simulate in OpenDSS-G over DSTCP"
    )
    parser.add_argument(
        "--source",
        choices=["rt", "mpc", "loadshape"],
        default="rt",
        help="Control mode: 'rt' (Closed-loop MPC + 4-Layer RT Control), 'mpc' (MPC Schedule), or 'loadshape' (Local Greedy BMS)"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Evaluation date YYYY-MM-DD (default: 2024-10-24 for 2024 cases, 2022-11-01 for Caltech)"
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.45,
        help="Delay after each 7.5m stepIn() sub-step in seconds (default: 0.45s = 0.90s per 15m interval)"
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Use currently compiled circuit in OpenDSS-G without sending compile command"
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Open interactive matplotlib windows at the end of the 96-step run"
    )
    args = parser.parse_args()

    ctrl = OpenDSSGDSTCPController(step_delay=args.step_delay)
    if args.mode == "all":
        all_summaries = {}
        for m in ["supermarket", "ev", "caltech_ev"]:
            all_summaries[m] = ctrl.run_96_steps(
                mode=m,
                source=args.source,
                start_date=args.date,
                compile_master=not args.no_compile,
                show_plots=args.show_plots
            )
        summary_json = RESULTS_DSS_DIR / "dstcp_96step_summary.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nSaved all-case DSTCP summary to: {summary_json}")
    else:
        ctrl.run_96_steps(
            mode=args.mode,
            source=args.source,
            start_date=args.date,
            compile_master=not args.no_compile,
            show_plots=args.show_plots
        )
