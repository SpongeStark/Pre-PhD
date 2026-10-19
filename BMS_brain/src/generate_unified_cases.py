import os
from pathlib import Path

base_dir = Path(r"c:\Users\alesk\Documents\Git-repo\Pre-PhD\BMS_brain")

# Load and Battery specifications for each case
cases = [
    {
        "folder": "OpenDSS-g-case-supermarket",
        "name": "Case 1: Supermarket Facility Load (c_gen)",
        "load_kw": 146.5,
        "load_pf": 0.95,
        "shape_mult": "0.3808 0.3683 0.3647 0.3597 0.3815 0.7000 0.7509 0.4789 0.3163 0.2538 0.1994 0.1641 0.1750 0.1663 0.2001 0.2356 0.2982 0.3569 0.4327 0.4801 0.4455 0.4000 0.4039 0.3856",
        "load_desc": "Supermarket facility (146.5 kW nominal capacity, 110.0 kW max demand, PF=0.95)",
        # Battery from battery_sizing_results.json (supermarket)
        "bat_kw": 69.31,
        "bat_kwh": 264.46,
        "bat_stored": 132.23
    },
    {
        "folder": "OpenDSS-g-case-ev-lidl",
        "name": "Case 2: Lidl EV Charging Stations (c_ev)",
        "load_kw": 110.0,
        "load_pf": 0.95,
        "shape_mult": "0.0526 0.0350 0.0251 0.0181 0.0139 0.0179 0.0285 0.0606 0.2852 0.5471 0.5871 0.5905 0.6364 0.3916 0.4244 0.3776 0.4091 0.4369 0.4340 0.2390 0.1939 0.1686 0.1371 0.0930",
        "load_desc": "Lidl commercial EV charging hub (110.0 kW installed capacity, 70.0 kW peak demand, PF=0.95)",
        # Battery from battery_sizing_results.json (ev)
        "bat_kw": 0.65,
        "bat_kwh": 1.29,
        "bat_stored": 0.65
    },
    {
        "folder": "OpenDSS-g-case-ev-caltech",
        "name": "Case 3: Caltech Campus EV Fleet (c_ev)",
        "load_kw": 150.0,
        "load_pf": 1.00,
        "shape_mult": "0.0418 0.0343 0.0262 0.0211 0.0181 0.0157 0.0311 0.0554 0.1890 0.4698 0.5920 0.5192 0.3985 0.3566 0.3065 0.2445 0.1927 0.1388 0.1253 0.1280 0.1165 0.0969 0.0599 0.0478",
        "load_desc": "Caltech campus workplace EV fleet (150.0 kW transformer capacity, 88.8 kW peak demand, PF=1.0)",
        # Battery from battery_sizing_results.json (caltech_ev)
        "bat_kw": 14.07,
        "bat_kwh": 31.36,
        "bat_stored": 15.68
    }
]

bus_coords = """sbus, 14, 600
slv, 900, 600
pcc, 1700, 600
n_common, 2500, 600
n_pv, 2500, 100
n_bat, 3300, 600
n_load, 2500, 1100
"""

gis_coords = """sbus, 0, 0
slv, 0, 0
pcc, 0, 0
n_common, 0, 0
n_pv, 0, 0
n_bat, 0, 0
n_load, 0, 0
"""

# European LV distribution standard: 400 V line-to-line (0.4 kV), matching Delta M88H 400 VCA
voltage_bases = """Set Voltagebases=[11.0, 0.4]
CalcVoltagebases
"""

growth_shape = """New "GrowthShape.default" npts=2 year=[1 20] mult=[1.025 1.025]
"""

spectrum = """New "Spectrum.default" NumHarm=7 harmonic=[1 3 5 7 9 11 13] %mag=[100 33 20 14 11 9 7] angle=[0 0 0 0 0 0 0]
New "Spectrum.defaultload" NumHarm=7 harmonic=[1 3 5 7 9 11 13] %mag=[100 1.5 20 14 1 9 7] angle=[0 180 180 180 180 180 180]
New "Spectrum.defaultgen" NumHarm=7 harmonic=[1 3 5 7 9 11 13] %mag=[100 5 3 1.5 1 0.7 0.5] angle=[0 0 0 0 0 0 0]
New "Spectrum.defaultvsource" NumHarm=1 harmonic=[1] %mag=[100] angle=[0]
New "Spectrum.linear" NumHarm=1 harmonic=[1] %mag=[100] angle=[0]
New "Spectrum.pwm6" NumHarm=13 harmonic=[1 3 5 7 9 11 13 15 17 19 21 23 25] %mag=[100 4.4 76.5 62.7 2.9 24.8 12.7 0.5 7.1 8.4 0.9 4.4 3.3] angle=[-103 -5 28 -180 -33 -59 79 36 -253 -124 3 -30 86]
New "Spectrum.dc6" NumHarm=10 harmonic=[1 3 5 7 9 11 13 15 17 19] %mag=[100 1.2 33.6 1.6 0.4 8.7 1.2 0.3 4.5 1.3] angle=[-75 28 156 29 -91 49 54 148 -57 -46]
"""

tcc_curve = """New "TCC_Curve.a" npts=5 C_array=[1 2.5 4.5 8 14] T_array=[0.15 0.07 0.05 0.045 0.045]
New "TCC_Curve.d" npts=5 C_array=[1 2.5 4.5 8 14] T_array=[6 0.7 0.2 0.06 0.02]
New "TCC_Curve.tlink" npts=7 C_array=[2 2.1 3 4 6 22 50] T_array=[300 100 10.1 4 1.4 0.1 0.02]
New "TCC_Curve.klink" npts=6 C_array=[2 2.2 3 4 6 30] T_array=[300 20 4 1.3 0.41 0.02]
New "TCC_Curve.uv1547" npts=2 C_array=[0.5 0.9] T_array=[0.166 2]
New "TCC_Curve.ov1547" npts=2 C_array=[1.1 1.2] T_array=[2 0.166]
"""

swt_control = """New "SwtControl.sw_inv" SwitchedObj=Line.line_reactor State=closed Action=close
"""

vsource = """Edit "Vsource.source" bus1=sbus.1.2.3 bus2=sbus.0.0.0 basekv=11.0 Isc1=1500 Isc3=3000 pu=1.000000
"""

# Transformer secondary updated to 0.4 kV (400 V) matching Chilly-Mazarin plant
transformer = """New "Transformer.xfmr_0" windings=2 XHL=10 buses=[sbus.1.2.3, slv.1.2.3] conns=[delta, wye] kVs=[11.0, 0.4] kVAs=[800, 800] taps=[1.0, 1.0] wdg=1 %R=0.2 wdg=2 %R=0.2
"""

# Distribution lines and central inverter virtual micro-reactor
lines = """! 1. Substation LV Main Feeder: slv to pcc (400V)
New "Line.line_0" phases=3 bus1=slv.1.2.3 bus2=pcc.1.2.3 r1=0.09 x1=0.08 r0=0.27 x0=0.24 length=0.05 units=km Normamps=1200 Emergamps=1500

! 2. VIRTUAL MICRO-REACTOR LINE (Central Bidirectional Inverter Interface): pcc to n_common
! Sized for 210 kW PV + Battery exchange at 400V (Normamps=400)
New "Line.line_reactor" phases=3 bus1=pcc.1.2.3 bus2=n_common.1.2.3 r1=0.20 x1=0.20 r0=0.20 x0=0.20 length=0.001 units=km Normamps=400 Emergamps=500

! 3. Internal Branches connected to the Common Bus:
New "Line.line_pv" phases=3 bus1=n_common.1.2.3 bus2=n_pv.1.2.3 r1=0.10 x1=0.10 r0=0.10 x0=0.10 length=0.001 units=km Normamps=400 Emergamps=500
New "Line.line_bat" phases=3 bus1=n_common.1.2.3 bus2=n_bat.1.2.3 r1=0.10 x1=0.10 r0=0.10 x0=0.10 length=0.001 units=km Normamps=200 Emergamps=250
New "Line.line_load" phases=3 bus1=n_common.1.2.3 bus2=n_load.1.2.3 r1=0.10 x1=0.10 r0=0.10 x0=0.10 length=0.001 units=km Normamps=250 Emergamps=300
"""

# PV System: 3 full Delta M88H inverters aggregated
# Sizing: 3 x 70 kW = 210 kVA AC rating, 3 x 90 kWp = 270 kWp DC peak rating
# Inverter Loading Ratio ILR = 270 / 210 = 1.286 (clipping enabled above 210 kW)
# %Cutin = 0.21% (3 x 150 W startup power)
# Reactive capability: PF 0.80 lead/lag -> kvarMax = 210 * 0.6 = 126.0 kvar
pv_system = """! 1. MPPT Inverter Efficiency Curve (Delta M88H-121: Peak 98.8%, Euro 98.5%)
New "XYCurve.mppt_eff" npts=4 xarray=[0.1 0.2 0.5 1.0] yarray=[0.965 0.980 0.988 0.985]

! 2. Solar PV System on Common Bus (3 x Delta M88H Inverters Aggregated)
! Supports active power clipping (210 kW) + reactive PF compensation (up to 126 kvar)
New "PVSystem.pv_1" phases=3 bus1=n_pv.1.2.3 kv=0.40 kVA=210.0 Pmpp=270.0 %Pmpp=100.0 %Cutin=0.21 %Cutout=0.10 EffCurve=mppt_eff kvarMax=126.0 kvarMaxAbs=126.0 kvar=0.0 VarFollowInverter=No ControlMode=GFL WattPriority=Yes daily=solar_shape duty=solar_shape
"""

# InvControl: Volt-Watt Active Power Curtailment as in EPRI PVSystem_InvControl.pdf
inv_control = """! --------------------------------------------------------------------------
! InvControl.dss - Smart Inverter Volt-Watt Power Curtailment / Inverter Clipping
! Based on EPRI Reference: PVSystem_InvControl.pdf
! - DERList explicitly targets PVSystem.pv_1
! - VoltwattYAxis=KVARATINGPU anchors curtailment to the 210 kVA AC rating
! --------------------------------------------------------------------------
New "XYCurve.voltwatt_curve" npts=4 xarray=[1.00 1.03 1.06 1.10] yarray=[1.00 1.00 0.20 0.00]
New "XYCurve.voltvar_curve" npts=6 xarray=[0.90 0.95 0.98 1.02 1.05 1.10] yarray=[1.00 1.00 0.00 0.00 -1.00 -1.00]
New "InvControl.pv_ctrl" DERList=[PVSystem.pv_1] mode=VOLTWATT voltage_curvex_ref=rated voltwatt_curve=voltwatt_curve VoltwattYAxis=KVARATINGPU DeltaP_factor=0.45
"""

monitors = """New "Monitor.inv_flow" element=Line.line_reactor terminal=1 mode=1 PPolar=no enabled=true
New "Monitor.grid_pcc" element=Line.line_0 terminal=1 mode=1 PPolar=no enabled=true
New "Monitor.bat_mon" element=Storage.sto_0 terminal=1 mode=7 enabled=true
New "Monitor.pv_mon" element=PVSystem.pv_1 terminal=1 mode=1 PPolar=no enabled=true
New "Monitor.load_mon" element=Load.load_1 terminal=1 mode=1 PPolar=no enabled=true
"""

energy_meter = """New "EnergyMeter.em_pcc" element=Line.line_0 terminal=1 enabled=true
New "EnergyMeter.em_inv" element=Line.line_reactor terminal=1 enabled=true
"""

master = """Clear
New Circuit.microgrid_common_bus

Redirect LoadShape.dss
Redirect GrowthShape.dss
Redirect TCC_Curve.dss
Redirect Spectrum.dss

Redirect Vsource.dss
Redirect Transformer.dss
Redirect Line.dss
Redirect Loads.dss
Redirect PVSystem.dss
Redirect Storage.dss
Redirect InvControl.dss
Redirect SwtControl.dss

Redirect Monitor.dss
Redirect EnergyMeter.dss

MakeBusList
Redirect BusVoltageBases.dss
Buscoords BusCoords.dss
GIScoords GISCoords.dss

Set Mode=Snapshot
Solve
"""

dsp = """V2.0;;;
1\tELEMENT\t \tNODE\t0\t600\t0\tsbus\t14\t600\t14\t600\t \t \t \t 
2\tELEMENT\t \tNODE\t886\t600\t0\tslv\t900\t600\t900\t600\t \t \t \t 
3\tELEMENT\t \tNODE\t1686\t600\t0\tpcc\t1700\t600\t1700\t600\t \t \t \t 
4\tELEMENT\t \tNODE\t2486\t600\t0\tn_common\t2500\t600\t2500\t600\t \t \t \t 
5\tELEMENT\t \tNODE\t2486\t100\t0\tn_pv\t2500\t100\t2500\t100\t \t \t \t 
6\tELEMENT\t \tNODE\t3286\t600\t0\tn_bat\t3300\t600\t3300\t600\t \t \t \t 
7\tELEMENT\t \tNODE\t2486\t1100\t0\tn_load\t2500\t1100\t2500\t1100\t \t \t \t 
8\tLINE\tPhases=3\t \t0\t0\t0\tline_0\t900\t600\t1700\t600\tslv.1.2.3\tpcc.1.2.3\t900,600;1700,600;1700,600--2\t 
9\tLINE\tPhases=3\t \t0\t0\t0\tline_reactor\t1700\t600\t2500\t600\tpcc.1.2.3\tn_common.1.2.3\t1700,600;2500,600;2500,600--2\t 
10\tLINE\tPhases=3\t \t0\t0\t0\tline_pv\t2500\t600\t2500\t100\tn_common.1.2.3\tn_pv.1.2.3\t2500,600;2500,100;2500,100--2\t 
11\tLINE\tPhases=3\t \t0\t0\t0\tline_bat\t2500\t600\t3300\t600\tn_common.1.2.3\tn_bat.1.2.3\t2500,600;3300,600;3300,600--2\t 
12\tLINE\tPhases=3\t \t0\t0\t0\tline_load\t2500\t600\t2500\t1100\tn_common.1.2.3\tn_load.1.2.3\t2500,600;2500,1100;2500,1100--2\t 
13\tSYSTEM\t3\tMAIN_NODE\t14\t600\t0\t \t14\t600\t20000\t20000\tsbus.1.2.3\t \t \t 
14\tELEMENT\tPhases=3--\tTRANSFORMER_YY\t14\t600\t0\txfmr_0\t14\t600\t900\t600\tsbus.1.2.3\tslv.1.2.3\t \t 
"""

for c in cases:
    cdir = base_dir / c["folder"]
    cdir.mkdir(parents=True, exist_ok=True)
    
    (cdir / "BusCoords.dss").write_text(bus_coords, encoding="utf-8")
    (cdir / "GISCoords.dss").write_text(gis_coords, encoding="utf-8")
    (cdir / "BusVoltageBases.dss").write_text(voltage_bases, encoding="utf-8")
    (cdir / "GrowthShape.dss").write_text(growth_shape, encoding="utf-8")
    (cdir / "Spectrum.dss").write_text(spectrum, encoding="utf-8")
    (cdir / "TCC_Curve.dss").write_text(tcc_curve, encoding="utf-8")
    (cdir / "SwtControl.dss").write_text(swt_control, encoding="utf-8")
    (cdir / "Vsource.dss").write_text(vsource, encoding="utf-8")
    (cdir / "Transformer.dss").write_text(transformer, encoding="utf-8")
    (cdir / "Line.dss").write_text(lines, encoding="utf-8")
    (cdir / "PVSystem.dss").write_text(pv_system, encoding="utf-8")
    (cdir / "InvControl.dss").write_text(inv_control, encoding="utf-8")
    (cdir / "Monitor.dss").write_text(monitors, encoding="utf-8")
    (cdir / "EnergyMeter.dss").write_text(energy_meter, encoding="utf-8")
    (cdir / "Master.dss").write_text(master, encoding="utf-8")
    (cdir / "Default_project.Dsp").write_text(dsp, encoding="utf-8")
    
    # Storage sized from battery dimensioning optimizer for the specific case
    bat_kva = round(c['bat_kw'] * 1.25, 2)
    bat_kvar = round(c['bat_kw'] * 0.75, 2)
    storage_content = f"""! --------------------------------------------------------------------------
! Storage.dss - {c['name']}
! Sizing direct input from Battery Dimensioning System:
! Rated Power: {c['bat_kw']} kW, Energy Capacity: {c['bat_kwh']} kWh
! --------------------------------------------------------------------------
New "Storage.sto_0" phases=3 bus1=n_bat.1.2.3 kv=0.40 kVA={bat_kva} kWrated={c['bat_kw']} kWhrated={c['bat_kwh']} kWhstored={c['bat_stored']} %reserve=20.0 %effcharge=95.0 %effdischarge=95.0 %IdlingkW=0.1 kvarMax={bat_kvar} kvarMaxAbs={bat_kvar} kvar=0.0 WattPriority=Yes daily=bess_dispatch duty=bess_dispatch dispMode=Follow State=IDLING
"""
    (cdir / "Storage.dss").write_text(storage_content, encoding="utf-8")
    
    # Case-specific Loads.dss (strictly 1 load on n_load)
    loads_content = f"""! --------------------------------------------------------------------------
! Loads.dss - {c['name']}
! STRICTLY ONE LOAD CONNECTED TO THE COMMON BUS LOAD NODE (n_load)
! --------------------------------------------------------------------------
New "Load.load_1" phases=3 bus1=n_load.1.2.3 kV=0.40 kW={c['load_kw']} pf={c['load_pf']} model=1 daily=target_load_shape duty=target_load_shape
"""
    (cdir / "Loads.dss").write_text(loads_content, encoding="utf-8")
    
    # Case-specific LoadShape.dss
    load_shape_content = f"""! --------------------------------------------------------------------------
! LoadShape.dss - {c['name']}
! --------------------------------------------------------------------------

! 1. Target Demand Profile ({c['load_desc']})
New "LoadShape.target_load_shape" npts=24 interval=1.0 mult=[{c['shape_mult']}]

! 2. Solar PV Generation Shape (3 x Delta M88H Inverters)
New "LoadShape.solar_shape" npts=24 interval=1.0 mult=[0.0 0.0 0.0 0.0 0.0099 0.0854 0.2496 0.4567 0.6810 0.8529 0.9604 0.9973 1.0000 0.9602 0.8699 0.7222 0.5223 0.3243 0.1514 0.0318 0.0006 0.0 0.0 0.0]

! 3. Battery Storage Dispatch Shape
New "LoadShape.bess_dispatch" npts=24 interval=1.0 mult=[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.4 -0.8 -1.0 -0.9 -0.6 0.0 0.0 0.3 0.8 1.0 0.9 0.7 0.2 0.0 0.0]
"""
    (cdir / "LoadShape.dss").write_text(load_shape_content, encoding="utf-8")
    print(f"Successfully generated 3-inverter circuit for: {c['folder']}")
