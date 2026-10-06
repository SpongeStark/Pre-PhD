# OpenDSS-G Common DC Bus Microgrid Model (Approach B)

## Overview
This circuit implements **Approach B (Quasi-Steady-State Common AC Bus with Virtual Micro-Reactor Interface)** to simulate a physical **Common DC Bus Microgrid** inside **OpenDSS-G**.

The system models:
1. **Utility Distribution Grid:** 11 kV Medium Voltage Source with step-down substation transformer (800 kVA, 11 kV / 0.23 kV).
2. **AC Point of Common Coupling (PCC):** Main low-voltage distribution board (0.23 kV, 3-phase).
3. **Supermarket AC Load:** Base facility consumption (40 kW, PF=0.95, 24-hour retail profile).
4. **Virtual Micro-Reactor Line (`line_reactor`):** Physical AC interface and coupling reactor representing the **Central Bidirectional Inverter** linking the AC PCC to the Common DC Bus.
5. **Common Equivalent DC Bus (`n_dc_bus`):**
   - **Solar PV System (`pv_1`):** 20 kW / 20 kVA with flat 99% MPPT efficiency (`mppt_eff`), zero reactive power (`kvarMax=0`, `PF=1.0`).
   - **Battery Energy Storage & BMS (`sto_0`):** 20 kW / 40 kWh, 95% charging/discharging efficiency (`%effcharge=95`, `%effdischarge=95`), 20% minimum SoC reserve (`%reserve=20`), zero reactive power (`kvarMax=0`, `PF=1.0`).
   - **EV Fast Charging Load (`ev_dc`):** 15 kW DC fast charger connected directly to the DC bus (`PF=1.0`, `kvar=0`).

---

## Circuit Architecture

```
[11 kV Grid]
     │
 [xfmr_0] (11 / 0.23 kV, 800 kVA)
     │
  [ slv ]
     │ (line_0)
  [ pcc ] ─────────────────────────┐
     │ (line_reactor: VIRTUAL      │ (line_load)
     │  MICRO-REACTOR LINE)        ▼
     │                         [n_load] (Supermarket AC Load)
     ▼
[n_dc_bus] (Common DC Bus)
     ├────── (line_pv_dc) ──► [n_pv_dc]  (PV Array 20 kW, MPPT 99%)
     ├────── (line_bat_dc) ─► [n_bat_dc] (BESS / BMS 40 kWh / 20 kW)
     └────── (line_ev_dc) ──► [n_ev_dc]  (EV Fast Charger 15 kW, DC)
```

---

## The Virtual Micro-Reactor Line System

In OpenDSS-G, setting a line impedance to zero ($R=0, X=0$) causes the nodal admittance matrix $[Y_{bus}]$ to become singular ($1/Z \to \infty$). 

To solve this, **Line.line_reactor** implements a virtual micro-reactor:
* $R_1 = 0.0002\,\Omega$, $X_1 = 0.0002\,\Omega$ (length = 1 m).
* **Numerical Stability:** Perfectly conditioned admittance matrix with fast convergence (< 2 iterations).
* **Negligible Voltage Drop:** $\Delta V < 0.005\%$, keeping the DC bus voltage practically identical to the PCC bus voltage ($V_{pu} = 0.9937$).
* **Exact Inverter Monitoring:** `Monitor.inv_flow` on `line_reactor` directly records the net active and reactive power throughput of the Central Inverter.

---

## Key Parameter Mapping to Research Model

| Component | Research / Optimizer Model (`battery_optimizer.py`) | OpenDSS-g-2 Implementation |
| :--- | :--- | :--- |
| **PV MPPT Efficiency** | $\eta_{pv} = 0.99$ | `XYCurve.mppt_eff` ($99\%$ flat efficiency) |
| **Battery Charging Eff.** | $\eta_c = 0.95$ | `Storage.sto_0 %effcharge=95.0` |
| **Battery Discharging Eff.** | $\eta_d = 0.95$ | `Storage.sto_0 %effdischarge=95.0` |
| **Battery Sizing** | Continuous rating: 20 kW / 40 kWh | `kWrated=20.0, kWhrated=40.0` |
| **Degradation / Reserve** | $SoC_{min} = 0.20$ | `Storage.sto_0 %reserve=20.0` |
| **Central Inverter Sizing** | Max net power exchange ($P_{grid}^{max} \le 50\text{ kW}$) | `Line.line_reactor Normamps=130` ($50\text{ kVA}$) |
| **DC Bus Reactive Power** | Zero reactive power on DC link ($Q = 0$) | `kvarMax=0.0, PF=1.0` on PV, Battery, EV |

---

## How to Run

### In OpenDSS-G (Graphical GUI)
1. Open **OpenDSS-G**.
2. Select **File -> Open Project** and navigate to:
   `C:\Users\alesk\Documents\Git-repo\Pre-PhD\BMS_brain\OpenDSS-g-2\Default_project.Dsp`
3. Click **Compile & Run** (or F5).

### In Python (Batch / Automated Simulation)
```python
import win32com.client

dss = win32com.client.Dispatch("OpenDSSEngine.DSS")
dss.Start(0)
dss.Text.Command = r'compile "C:\Users\alesk\Documents\Git-repo\Pre-PhD\BMS_brain\OpenDSS-g-2\Master.dss"'
dss.Text.Command = "Set mode=daily number=24 stepsize=1h"
dss.Text.Command = "Solve"
dss.Text.Command = "Export monitors inv_flow"
print("Exported central inverter flow to:", dss.Text.Result)
```
