# Import necessary libraries
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define the root path of the project
root_proj = Path(__file__).parent.parent

# Add the 'src' directory to the system path
if str(root_proj/"src") not in sys.path:
    sys.path.append(str(root_proj/"src"))

# ==========================================
# PROCESS EV CHARGING DATA (YEARS 2022, 2023, 2024)
# ==========================================
print("Loading EV charging consumption Excel file...")
excel_path = root_proj / "data/Responsive Utility meter.xlsx"

# This loads a dictionary where keys are sheet names and values are DataFrames
sheets = pd.read_excel(excel_path, sheet_name=["2022", "2023", "2024"], skiprows=1)

all_dts = []

for year_str, df_act in sheets.items():
    year = int(year_str)
    print(f"\n--- Processing EV Data for Year {year} ---")
    
    # --- 1. Clean and Transform EV Data ---
    # Extract the target EV charging column "8_ELEC : Bornes électriques (97)"
    ev_col = "8_ELEC : Bornes électriques (97)"
    if ev_col in df_act.columns:
        df_act['c_ev'] = df_act[ev_col]
    else:
        # Fallback search for EV column matching "Bornes" or "electric"
        matching_cols = [c for c in df_act.columns if "Bornes" in str(c) or "8_ELEC" in str(c)]
        if matching_cols:
            print(f"Using found EV column: {matching_cols[0]}")
            df_act['c_ev'] = df_act[matching_cols[0]]
        else:
            raise KeyError(f"Column '{ev_col}' not found in sheet {year_str}")
    
    # --- FIX CORRUPTED EXCEL DATES ---
    # The '2022' sheet accidentally contains '2023' written in its Date column.
    if year == 2022:
        df_act["Date"] = df_act["Date"].astype(str).str.replace("2023-", "2022-")
    
    # Convert 'Date' to datetime and subtract 1 hour to align with UTC
    df_act["Date_utc"] = pd.to_datetime(
        df_act["Date"],
        format="%Y-%m-%d %H:%M"
    ) - pd.Timedelta(hours=1) 

    # Calculate UTC offset for Europe/Paris
    df_act['offset'] = (df_act["Date_utc"].dt.tz_localize("UTC").dt.tz_convert("Europe/Paris").dt.tz_localize(None) - df_act["Date_utc"]).dt.total_seconds()//3600
    
    # Drop original naive Date and rename Date_utc back to Date
    df_act.drop(columns=["Date"], inplace=True)
    df_act = df_act.rename(columns={"Date_utc": "Date"})

    # --- 2. Load and process weather data (Temperature) ---
    temp_file = root_proj / f"data/orly_{year}_hourly_weather.csv"
    if not temp_file.exists():
        print(f"Warning: File {temp_file} not found. Skipping weather merge for {year}.")
    else:
        temp = pd.read_csv(temp_file, sep=",")
        temp["Date_utc"] = pd.to_datetime(temp["time"], utc=True).dt.tz_localize(None)

        # Filter and resample to 15-min
        temp_wanted = temp[temp["Date_utc"] <= f"{year}-12-31 00:00:00"].set_index("Date_utc")
        dts_t = temp_wanted.resample("15min").ffill().reset_index()
        dts_t = dts_t[["Date_utc", "temp"]]
        
        # Merge weather (temp) onto EV data
        df_act = df_act.merge(dts_t, left_on="Date", right_on="Date_utc", how="left")
        df_act.drop(columns=["Date_utc"], inplace=True)

    # --- 3. Feature Engineering (Temporal features) ---
    df_act['hour'] = df_act['Date'].dt.hour
    df_act['dayofyear'] = df_act['Date'].dt.dayofyear
    df_act['month'] = df_act['Date'].dt.month
    df_act['dayofweek'] = df_act['Date'].dt.dayofweek

    # Cyclical encodings
    df_act['hour_sin'] = np.sin(2*np.pi*df_act['hour']/24)
    df_act['hour_cos'] = np.cos(2*np.pi*df_act['hour']/24)
    df_act['doy_sin'] = np.sin(2*np.pi*df_act['dayofyear']/365)
    df_act['doy_cos'] = np.cos(2*np.pi*df_act['dayofyear']/365)
    df_act['dow_sin'] = np.sin(2*np.pi*df_act['dayofweek']/7)
    df_act['dow_cos'] = np.cos(2*np.pi*df_act['dayofweek']/7)
    
    # Filter essential columns to keep parquet clean
    cols_to_keep = ['Date', 'offset', 'c_ev', 'temp', 'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos', 'dow_sin', 'dow_cos']
    available_cols = [c for c in cols_to_keep if c in df_act.columns]
    df_act = df_act[available_cols]

    # Handle negative values or anomalies in EV consumption data
    df_act['c_ev'] = df_act['c_ev'].clip(lower=0)
    # The max normal EV charging consumption is typically <100kW, so spikes >=100kW (e.g., 120kW outlier in 2023) are sensor artifacts
    outlier_mask = df_act['c_ev'] >= 100
    if outlier_mask.any():
        outlier_vals = df_act.loc[outlier_mask, 'c_ev'].tolist()
        print(f"Detected {len(outlier_vals)} EV outlier spike(s) in {year} (values: {outlier_vals}). Cleaning with interpolation...")
        df_act.loc[outlier_mask, 'c_ev'] = np.nan
        df_act['c_ev'] = df_act['c_ev'].interpolate(method='linear')

    all_dts.append(df_act)

# --- 4. Concatenate and Save ---
if all_dts:
    print("\nConcatenating all EV datasets...")
    master_dataset_ev = pd.concat(all_dts, ignore_index=True)
    
    # Plot the full concatenated dataset to visually verify
    plt.figure(figsize=(15, 5))
    plt.plot(master_dataset_ev['Date'], master_dataset_ev['c_ev'], label='EV Consumption (c_ev)', color='green')
    plt.title('Master Dataset: EV Power Consumption (2022-2024)')
    plt.xlabel('Date')
    plt.ylabel('EV Consumption (kW)')
    plt.legend()
    plt.grid(True)
    output_plot_path = Path(__file__).parent / "data_cleaner_EV_results.png"
    plt.savefig(output_plot_path)
    print(f"EV Data Cleaner plots successfully saved to: {output_plot_path}")
    plt.close()
    
    # Save the master EV dataset as parquet
    output_path = root_proj / "data/master_dataset_ev.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    master_dataset_ev.to_parquet(output_path)
    print(f"EV master dataset saved successfully to {output_path}")
    print(f"Total rows: {len(master_dataset_ev)}")
else:
    print("No data was processed.")
