import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rte_wholesale_market import RTEWholesaleMarketClient

# ==========================================
# 1. PARAMETERS & REAL DATA LOADING
# ==========================================

# Define project root and data paths
root_proj = Path(__file__).parent.parent
pv_data_path = root_proj / "data/master_dataset.parquet"
con_data_path = root_proj / "data/master_dataset_con.parquet"

# Load data
print("Loading real PV and Consumption data...")
df_pv = pd.read_parquet(pv_data_path)
df_con = pd.read_parquet(con_data_path)

# Filter for the year 2022
df_pv = df_pv[df_pv['Date'].dt.year == 2022]
df_con = df_con[df_con['Date'].dt.year == 2022]

# Merge the datasets on 'Date' to align them
df = pd.merge(df_pv, df_con, on='Date', how='inner')
df = df.sort_values(by='Date').reset_index(drop=True)

# Time parameters
T = len(df) # Full year 2022
dt = 0.25 # 15 minutes time step
time_steps = range(T)

# Extract Profiles
pv_profile = df['PV'].fillna(0).values
load_profile = df['c_gen'].fillna(0).values

# Electricity Tariff (Time-of-Use / Real-Time RTE API)
print("\nFetching real-time energy prices from RTE API...")
try:
    client = RTEWholesaleMarketClient()
    rt_profile = client.get_representative_24h_profile()
    
    if rt_profile:
        print("Successfully retrieved 24-hour real-time price profile from RTE API.")
        # Map to the dataset based on hour and minute of the day
        tariff_profile = df['Date'].apply(lambda x: rt_profile.get((x.hour, x.minute), 0.15)).values
        
        # Log pricing statistics
        print(f"RTE Tariff stats - Min: {tariff_profile.min():.4f} €/kWh, Max: {tariff_profile.max():.4f} €/kWh, Mean: {tariff_profile.mean():.4f} €/kWh")
    else:
        print("RTE API returned empty profile. Falling back to synthetic Time-of-Use tariff.")
        hours = df['Date'].dt.hour
        tariff_profile = np.where((hours >= 16) & (hours <= 21), 0.25, 0.15)
except Exception as e:
    print(f"Failed to fetch from RTE API ({e}). Falling back to synthetic Time-of-Use tariff.")
    hours = df['Date'].dt.hour
    tariff_profile = np.where((hours >= 16) & (hours <= 21), 0.25, 0.15)


# Economic Parameters (Sourced from the provided IEEE paper)
life_years = 15
discount_rate = 0.04
CRF = (discount_rate * (1 + discount_rate)**life_years) / ((1 + discount_rate)**life_years - 1) # Capital Recovery Factor

C_E = 500.0 # Battery capacity cost [EUR/kWh]
C_P = 200.0 # Battery inverter/power cost [EUR/kW]
c_deg = 0.02 # Linearized marginal degradation cost [EUR/kWh throughput]

# Technical Parameters
eta_pv = 0.98       # PV MPPT efficiency
eta_acdc = 0.95     # Grid rectifier efficiency
eta_dcac = 0.95     # EV/Load inverter efficiency
eta_c = 0.95        # Battery charging efficiency
eta_d = 0.95        # Battery discharging efficiency

SoC_min = 0.2       # Minimum State of Charge
SoC_max = 0.9       # Maximum State of Charge
c_rate = 0.5        # Max C-rate (e.g., 0.5C means a 10kWh battery can output max 5kW)

# ==========================================
# 2. OPTIMIZATION MODEL SETUP
# ==========================================

# Initialize the LP problem (Minimization)
model = pulp.LpProblem("Battery_Sizing_Optimization", pulp.LpMinimize)

# Decision Variables
E_B_max = pulp.LpVariable("E_B_max", lowBound=0, cat='Continuous') # Installed Capacity [kWh]
P_B_max = pulp.LpVariable("P_B_max", lowBound=0, cat='Continuous') # Installed Power [kW]

# Time-series Variables
E_B = pulp.LpVariable.dicts("E_B", time_steps, lowBound=0, cat='Continuous')
P_ch = pulp.LpVariable.dicts("P_ch", time_steps, lowBound=0, cat='Continuous')
P_dis = pulp.LpVariable.dicts("P_dis", time_steps, lowBound=0, cat='Continuous')
P_grid = pulp.LpVariable.dicts("P_grid", time_steps, lowBound=0, cat='Continuous')
P_curt = pulp.LpVariable.dicts("P_curt", time_steps, lowBound=0, cat='Continuous')

# ==========================================
# 3. OBJECTIVE FUNCTION
# ==========================================
# Minimize: Annualized CAPEX + OPEX (Grid costs + Linear Degradation costs)
# Note: Since T is exactly 1 year (2022), OPEX is already annual.
capex = CRF * (C_E * E_B_max + C_P * P_B_max)
opex_annual = pulp.lpSum([P_grid[t] * dt * tariff_profile[t] + c_deg * P_dis[t] * dt for t in time_steps])
model += capex + opex_annual

# ==========================================
# 4. CONSTRAINTS
# ==========================================

# Hardware limit: Rated power cannot exceed Capacity * C-rate
model += P_B_max <= c_rate * E_B_max, "C_rate_limit"

for t in time_steps:
    # 1. Power Balance
    model += (pv_profile[t] * eta_pv + P_grid[t] * eta_acdc + P_dis[t] == 
              (load_profile[t] / eta_dcac) + P_ch[t] + P_curt[t]), f"Power_Balance_{t}"
    
    # 1.5 Curtailment Limit (cannot curtail more than available PV)
    model += P_curt[t] <= pv_profile[t] * eta_pv, f"Curtailment_Limit_{t}"
    
    # 2. Energy Dynamics
    if t == 0:
        # Initial condition (assume battery starts at 50% capacity)
        model += E_B[t] == 0.5 * E_B_max + (P_ch[t] * eta_c - P_dis[t] * (1.0 / eta_d)) * dt, f"Energy_Dyn_Init_{t}"
    else:
        model += E_B[t] == E_B[t-1] + (P_ch[t] * eta_c - P_dis[t] * (1.0 / eta_d)) * dt, f"Energy_Dyn_{t}"
        
    # 3. State of Charge Bounds (Linearized using E_B_max)
    model += E_B[t] >= SoC_min * E_B_max, f"SoC_Min_{t}"
    model += E_B[t] <= SoC_max * E_B_max, f"SoC_Max_{t}"
    
    # 4. Power Bounds
    model += P_ch[t] <= P_B_max, f"P_ch_Max_{t}"
    model += P_dis[t] <= P_B_max, f"P_dis_Max_{t}"

# ==========================================
# 5. SOLVE & EXTRACT RESULTS
# ==========================================

print("Solving the sizing optimization...")
model.solve(pulp.PULP_CBC_CMD(msg=0)) # Using PuLP's default open-source CBC solver

if pulp.LpStatus[model.status] == 'Optimal':
    print("\n--- OPTIMAL HARDWARE SIZING ---")
    print(f"Battery Capacity (E_B_max): {E_B_max.varValue:.2f} kWh")
    print(f"Battery Rated Power (P_B_max): {P_B_max.varValue:.2f} kW")
    print(f"Total Annualized Cost (CAPEX + OPEX): €{pulp.value(model.objective):.2f}")
    
    # Extract timeseries results into a DataFrame for easy plotting later
    results = pd.DataFrame({
        'PV_Gen_kW': pv_profile,
        'Load_kW': load_profile,
        'Tariff_EUR_kWh': tariff_profile,
        'E_B_kWh': [E_B[t].varValue for t in time_steps],
        'P_ch_kW': [P_ch[t].varValue for t in time_steps],
        'P_dis_kW': [P_dis[t].varValue for t in time_steps],
        'P_grid_kW': [P_grid[t].varValue for t in time_steps],
        'P_curt_kW': [P_curt[t].varValue for t in time_steps]
    })
    
    print("\nFirst 5 hours of operation:")
    print(results.head())

    # --- Plotting ---
    print("\nGenerating plots for the first 3 days to verify optimal behavior...")
    
    # We plot a slice of the data (e.g., first 3 days = 96 steps * 3) so the chart isn't too crowded
    plot_steps = min(T, 96 * 3) 
    res_plot = results.iloc[:plot_steps].copy()
    
    # Create a time axis for plotting (Hours)
    t_axis = np.arange(plot_steps) / 4.0 
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Power Balance
    axes[0].plot(t_axis, res_plot['Load_kW'], label='Load Demand', color='red', linestyle='--')
    axes[0].plot(t_axis, res_plot['PV_Gen_kW'], label='PV Generation', color='orange')
    axes[0].plot(t_axis, res_plot['P_grid_kW'], label='Grid Import', color='black')
    axes[0].set_ylabel('Power (kW)')
    axes[0].set_title('Power Balance (Load, PV, Grid Import)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)
    
    # Plot 2: Battery Dynamics
    axes[1].plot(t_axis, res_plot['P_ch_kW'], label='Charge (kW)', color='blue')
    axes[1].plot(t_axis, res_plot['P_dis_kW'], label='Discharge (kW)', color='green')
    axes[1].plot(t_axis, res_plot['E_B_kWh'], label='Stored Energy (kWh)', color='purple', linewidth=2)
    # Add a dashed line representing the optimal battery capacity found
    axes[1].axhline(y=E_B_max.varValue, color='purple', linestyle=':', label='Max Capacity Limit')
    axes[1].set_ylabel('Power / Energy')
    axes[1].set_title('Battery Operation & Energy State (SoC)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)
    
    # Plot 3: Tariff and Curtailment
    ax3 = axes[2]
    ax3.plot(t_axis, res_plot['Tariff_EUR_kWh'], label='Tariff (€/kWh)', color='grey')
    ax3.set_ylabel('Tariff (€/kWh)')
    ax3.set_xlabel('Time (Hours)')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(t_axis, res_plot['P_curt_kW'], label='PV Curtailed (kW)', color='orange', alpha=0.5, linestyle='-.')
    ax3_twin.set_ylabel('Curtailment (kW)')
    
    ax3.set_title('Electricity Tariff & PV Curtailment')
    
    # Combine legend for ax3 and ax3_twin
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3_twin.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.tight_layout()
    # Save the plot to a file in case the environment cannot display it
    output_plot_path = Path(__file__).parent / "optimization_results.png"
    plt.savefig(output_plot_path)
    print(f"Optimization plots successfully saved to: {output_plot_path}")
    plt.close()
else:
    print("Solver could not find an optimal solution. Check constraints.")