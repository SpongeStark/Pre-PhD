# Case 2: Lidl EV Charging Stations (`c_ev`)

## Unified Microgrid Architecture
* **Topology:** Standard Common Bus architecture (100% identical grid, transformer, PV, and Battery across all 3 cases).
* **Load Node:** `n_load.1.2.3` connected to Common Bus (`n_common`).
* **Active Load:** Lidl retail store EV charging stations (`c_ev`).
  * Nominal Power: $P_{nom} = 22.0\text{ kW}$
  * Power Factor: $PF = 1.00, Q = 0.0\text{ kvar}$ (pure active DC charging load)
  * Daily Profile: `target_load_shape` (24h Lidl EV curve from 2022 dataset)

## Fixed Hardware Sizing (Held Constant Across All 3 Cases)
* **Solar PV (`pv_1`):** $20\text{ kW}$ array ($20\text{ kVA}$), flat $99\%$ MPPT efficiency, $Q = 0$.
* **Battery & BMS (`sto_0`):** $20\text{ kW} / 40\text{ kWh}$ BESS, $\eta_c = 95\%, \eta_d = 95\%, SoC_{min} = 20\%$.
* **Central Inverter (`line_reactor`):** Virtual Micro-Reactor Line linking Common Bus to Grid PCC.

## How to Open in OpenDSS-G
Open OpenDSS-G, click **File -> Open Project**, and open:
`C:\Users\alesk\Documents\Git-repo\Pre-PhD\BMS_brain\OpenDSS-g-case-ev-lidl\Default_project.Dsp`
