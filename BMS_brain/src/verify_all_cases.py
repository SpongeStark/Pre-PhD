import win32com.client

dss = win32com.client.Dispatch('OpenDSSEngine.DSS')
dss.AllowForms = False
dss.Start(0)
text = dss.Text

cases = [
    ('Case 1: Supermarket', r'c:\Users\alesk\Documents\Git-repo\Pre-PhD\BMS_brain\OpenDSS-g-case-supermarket\Master.dss'),
    ('Case 2: Lidl EV', r'c:\Users\alesk\Documents\Git-repo\Pre-PhD\BMS_brain\OpenDSS-g-case-ev-lidl\Master.dss'),
    ('Case 3: Caltech EV', r'c:\Users\alesk\Documents\Git-repo\Pre-PhD\BMS_brain\OpenDSS-g-case-ev-caltech\Master.dss')
]

for name, master in cases:
    text.Command = f'compile "{master}"'
    err = dss.Error.Description
    c = dss.ActiveCircuit
    c.Solution.Solve()
    conv_snap = c.Solution.Converged
    p_snap, q_snap = c.TotalPower
    num_loads = c.Loads.Count
    load_name = c.Loads.Name
    
    # 24-hour daily simulation
    text.Command = 'Set mode=daily number=24 stepsize=1h'
    text.Command = 'Solve'
    conv_daily = c.Solution.Converged
    
    c.Loads.First
    print(f'=== {name} ===')
    print(f'  Compile: {"ERROR: " + err if err else "0 Errors (OK)"}')
    print(f'  Snapshot Solve Converged: {conv_snap}')
    print(f'  24h Daily Simulation Converged: {conv_daily}')
    print(f'  Load Object: {c.Loads.Name} connected to bus {c.ActiveCktElement.BusNames[0]}')
    print(f'  Total Net Power (kW, kvar): ({p_snap:.2f}, {q_snap:.2f})')
