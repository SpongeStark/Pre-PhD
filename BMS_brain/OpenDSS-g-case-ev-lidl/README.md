# Case 2: Lidl EV Charging Stations (`c_ev`)

## Microgrid Specifications (400V European Base)
* **Plant Site:** Lidl Chilly-Mazarin (`DR26-3591-CHILLY MAZARIN`)
* **Voltage Base:** $400\text{ V}_{AC}$ line-to-line ($230\text{ V}$ line-to-neutral, $0.4\text{ kV}$)
* **Transformer:** $11\text{ kV} / 0.4\text{ kV}$, $800\text{ kVA}$, Delta-Wye

## Load Configuration (The Only Load: `Load.load_1` at `n_load`)
* **Charging Hardware:** Commercial dual-socket EV charging terminals (Schneider EVlink Parking / Legrand Green'up architecture)
* **Kiosk Rating:** $2 \times 22\text{ kW}$ at $400\text{ V}_{AC}$ ($3\text{Ph}+\text{N}+\text{T}$, $32\text{ A}$ per socket)
* **Installed Connectors:** 5 charging points (`LFR3591EVCP01` to `05`)
* **Earthing Scheme:** TT / TN grounding system
* **Power Factor:** $PF = 0.95$ lagging (per datasheet: `facteur de puissance > 0.95`, drawing $Q \approx 23.0\text{ kvar}$ at $70\text{ kW}$)
* **Nominal Active Power:** $P_{nom} = 110.0\text{ kW}$ (installed charging hub / dedicated transformer capacity: $5 \times 22\text{ kW}$)
* **Maximum Operational Demand:** $P_{max} = 70.0\text{ kW}$ (peak diversity factor $= 70.0 / 110.0 = 63.6\%$)
* **Daily Shape:** `target_load_shape` (24h Lidl EV profile scaled to $110.0\text{ kW}$ base)
* **Protection & Certification:** IP55 (terminal), IP44 (sockets), IK10, ZE Ready & CE certified

## Battery / BMS Sizing (Input from Optimization System)
* **Energy Capacity:** $E_B = 1.29\text{ kWh}$
* **Rated Power:** $P_B = 0.65\text{ kW}$
* **Initial Energy:** $0.65\text{ kWh}$ ($50\%$ SoC)
* **Reserve:** $SoC_{min} = 20\%$ reserve
* **Efficiencies:** $\eta_c = 95\%, \eta_d = 95\%$

## PV Inverter System (3 x Delta M88H Inverters Aggregated)
* **AC Inverter Rating:** $3 \times 70\text{ kW} = \mathbf{210\text{ kVA}}$ (`kVA=210.0`)
* **DC Panel Peak Rating:** $3 \times 90\text{ kW}_p = \mathbf{270\text{ kW}}_p$ (`Pmpp=270.0`)
* **Inverter Loading Ratio (ILR):** $270 / 210 = 1.286$
* **Inverter Power Clipping:** Naturally clips active power at $210\text{ kW}$ (`WattPriority=Yes`)
* **Reactive Power / PF Compensation:** `kvarMax=126.0`, `kvarMaxAbs=126.0` ($PF = 0.80$ lead/lag capability), `VarFollowInverter=No` for active load reactive power compensation
* **Smart Inverter Control:** `InvControl.pv_ctrl` (`DERList=[PVSystem.pv_1]`, `mode=VOLTWATT`, `VoltwattYAxis=KVARATINGPU`) for overvoltage curtailment
* **Startup Power:** $3 \times 150\text{ W} = 450\text{ W}$ (`%Cutin=0.21`)

## Central Inverter Interface & Time-Mode Runner
* **Virtual Micro-Reactor Line:** `Line.line_reactor` ($R=0.20\,\Omega/\text{km}, X=0.20\,\Omega/\text{km}$, $L=1\text{ m}$, $Normamps=400\text{ A}$)
* **Time-Mode Simulation Engine:** Executed at 1-second resolution ($86,400\text{ steps/day}$) via `BMS_brain/src/opendss_runner.py`

