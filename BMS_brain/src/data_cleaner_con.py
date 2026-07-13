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
# PROCESS CONSUMPTION DATA (YEARS 2022, 2023, 2024)
# ==========================================
print("Loading consumption Excel file...")
excel_path = root_proj / "data/Responsive Utility meter.xlsx"

# This loads a dictionary where keys are sheet names and values are DataFrames
sheets = pd.read_excel(excel_path, sheet_name=["2022", "2023", "2024"], skiprows=1)

all_dts = []

for year_str, df_act in sheets.items():
    year = int(year_str)
    print(f"\n--- Processing Consumption Data for Year {year} ---")
    
    # --- 1. Clean and Transform Consumption Data ---
    # Extract the target consumption column
    df_act['c_gen'] = df_act["0_ELEC : Total Fournisseur mesuré (06)"]  
    
    # --- FIX CORRUPTED EXCEL DATES ---
    # The '2022' sheet accidentally contains '2023' written in its Date column.
    if year == 2022:
        df_act["Date"] = df_act["Date"].astype(str).str.replace("2023-", "2022-")
    
    # Convert 'Date' to datetime and subtract 1 hour
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
        
        # Merge weather (temp) onto consumption data
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
    
    # Filter only the essential columns to keep the parquet clean
    cols_to_keep = ['Date', 'offset', 'c_gen', 'temp', 'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos', 'dow_sin', 'dow_cos']
    available_cols = [c for c in cols_to_keep if c in df_act.columns]
    df_act = df_act[available_cols]

    # Handle anomalous consumption spikes (outliers)
    # The max normal consumption is ~80-100kW, so a spike over 150kW is a known sensor artifact
    outlier_mask = df_act['c_gen'] > 150
    if outlier_mask.any():
        df_act.loc[outlier_mask, 'c_gen'] = np.nan
        df_act['c_gen'] = df_act['c_gen'].interpolate(method='linear')

    all_dts.append(df_act)

# --- 4. Concatenate and Save ---
if all_dts:
    print("\nConcatenating all datasets...")
    master_dataset_con = pd.concat(all_dts, ignore_index=True)
    
    # Plot the full concatenated dataset to visually verify
    plt.figure(figsize=(15, 5))
    plt.plot(master_dataset_con['Date'], master_dataset_con['c_gen'], label='Consumption (c_gen)', color='purple')
    plt.title('Master Dataset: Power Consumption (2022-2024)')
    plt.xlabel('Date')
    plt.ylabel('Consumption')
    plt.legend()
    plt.grid(True)
    output_plot_path = Path(__file__).parent / "data_cleaner_con_results.png"
    plt.savefig(output_plot_path)
    print(f"Consumption Data Cleaner plots successfully saved to: {output_plot_path}")
    plt.close()
    
    # Save the master dataset as parquet
    output_path = root_proj / "data/master_dataset_con.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    master_dataset_con.to_parquet(output_path)
    print(f"Consumption master dataset saved successfully to {output_path}")
    print(f"Total rows: {len(master_dataset_con)}")
else:
    print("No data was processed.")
