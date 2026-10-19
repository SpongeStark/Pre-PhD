# Case 3: Caltech Campus EV Fleet Load (`c_ev`)

## Microgrid Specifications (400V European Base)
* **Plant Site:** Lidl Chilly-Mazarin (`DR26-3591-CHILLY MAZARIN`)
* **Voltage Base:** $400\text{ V}_{AC}$ line-to-line ($230\text{ V}$ line-to-neutral, $0.4\text{ kV}$)
* **Transformer:** $11\text{ kV} / 0.4\text{ kV}$, $800\text{ kVA}$, Delta-Wye

## Load Configuration (The Only Load: `Load.load_1` at `n_load`)
* **Facility / Site:** California Parking Garage, Caltech campus, Pasadena, CA (from open research testbed **ACN-Data**)
* **Installed EVSE Hardware:** 54 Level-2 EVSEs (AeroVironment EVSE-RS / ClipperCreek / PowerFlex) + 1 DC Fast Charger
* **Individual Port Rating:** $208\text{ V}_{AC}$ line-to-line, $32\text{ A}$ single-phase $\approx 6.66\text{ kW}$ (up to $7.2\text{ kW}$) per EVSE
* **Upstream Infrastructure:** $150\text{ kVA}$ dedicated service transformer (480V $\Delta$ to 208Y/120V)
* **Oversubscription Management:** $54 \times 6.66\text{ kW} \approx 360\text{ kW}$ total potential, dynamically throttled/scheduled by Adaptive Charging Network (ACN) algorithms to avoid overloading the $150\text{ kVA}$ transformer
* **Power Factor:** $PF = 1.00$ ($Q = 0.0\text{ kvar}$, unity power factor via vehicle on-board active PFC rectifiers)
* **Nominal Active Power:** $P_{nom} = 150.0\text{ kW}$ (garage service transformer connection capacity)
* **Maximum Operational Demand:** $P_{max} = 88.8\text{ kW}$ (observed concurrent fleet peak, diversity factor $= 88.8 / 150.0 = 59.2\%$)
* **Daily Shape:** `target_load_shape` (24h Caltech EV profile scaled to $150.0\text{ kW}$ base)

## Battery / BMS Sizing (Input from Optimization System)
* **Energy Capacity:** $E_B = 31.36\text{ kWh}$
* **Rated Power:** $P_B = 14.07\text{ kW}$ ($C\text{-rate} = 0.5$)
* **Initial Energy:** $15.68\text{ kWh}$ ($50\%$ SoC)
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

