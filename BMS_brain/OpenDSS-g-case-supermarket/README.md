# Case 1: Supermarket Facility Load (`c_gen`)

## Microgrid Specifications (400V European Base)
* **Plant Site:** Lidl Chilly-Mazarin (`DR26-3591-CHILLY MAZARIN`)
* **Voltage Base:** $400\text{ V}_{AC}$ line-to-line ($230\text{ V}$ line-to-neutral, $0.4\text{ kV}$)
* **Transformer:** $11\text{ kV} / 0.4\text{ kV}$, $800\text{ kVA}$, Delta-Wye

## Load Configuration (The Only Load: `Load.load_1` at `n_load`)
* **Target Load:** Supermarket retail facility base consumption (`c_gen`)
* **Nominal Active Power:** $P_{nom} = 146.5\text{ kW}$ (nominal connection / contracted capacity)
* **Maximum Operational Demand:** $P_{max} = 110.0\text{ kW}$ (peak load factor $= 110.0 / 146.5 = 75.1\%$)
* **Power Factor:** $PF = 0.95$ lagging (inductive AC commercial load, $Q \approx 36.2\text{ kvar}$ at $110\text{ kW}$)
* **Daily Shape:** `target_load_shape` (24h supermarket curve scaled to $146.5\text{ kW}$ base)

## Battery / BMS Sizing (Input from Optimization System)
* **Energy Capacity:** $E_B = 264.46\text{ kWh}$
* **Rated Power:** $P_B = 69.31\text{ kW}$ ($C\text{-rate} = 0.5$)
* **Initial Energy:** $132.23\text{ kWh}$ ($50\%$ SoC)
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

