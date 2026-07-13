# Import necessary libraries
import sys
import requests
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Define the root path of the project
root_proj = Path("/Users/alesk/Documents/Git-repo/Pre-PhD/BMS_brain")

# Add the 'src' directory to the system path
if str(root_proj/"src") not in sys.path:
    sys.path.append(str(root_proj/"src"))

# ==========================================
# PROCESS YEARS 2022, 2023, 2024
# ==========================================
years = [2022, 2023, 2024]
all_dts = []

for year in years:
    print(f"\n--- Processing Year {year} ---")

    # --- 1. Load and process system data (PV generation) ---
    file_name = root_proj / f"data/DTS_LIDL_{year}.csv"
    if not file_name.exists():
        print(f"Warning: File {file_name} not found. Skipping {year}.")
        continue

    dts1 = pd.read_csv(file_name, sep=";")

    dts1["Date_utc"] = pd.to_datetime(
        dts1["Date"],
        format="%d/%m/%Y %H:%M"
    ) - pd.Timedelta(hours=1) 

    dts1['offset'] = (dts1["Date_utc"].dt.tz_localize("UTC").dt.tz_convert("Europe/Paris").dt.tz_localize(None) - dts1["Date_utc"]).dt.total_seconds()//3600

    # --- 2. Load and process weather data (Temperature) ---
    temp_file = root_proj / f"data/orly_{year}_hourly_weather.csv"
    if not temp_file.exists():
        print(f"Warning: File {temp_file} not found. Skipping {year}.")
        continue

    temp = pd.read_csv(temp_file, sep=",")
    temp["Date_utc"] = pd.to_datetime(temp["time"], utc=True).dt.tz_localize(None)

    temp_wanted = temp[temp["Date_utc"] <= f"{year}-12-31 00:00:00"].set_index("Date_utc")
    dts_t = temp_wanted.resample("15min").ffill().reset_index()
    dts_t = dts_t[["Date_utc", "temp"]]

    # --- 3. Load and process GHI data (From Open-Meteo API) ---
    print(f"Downloading {year} GHI data from Open-Meteo...")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude=48.73&longitude=2.42&"
        f"start_date={year}-01-01&end_date={year}-12-31&"
        f"hourly=shortwave_radiation&"
        f"timezone=UTC"
    )
    response = requests.get(url)
    data = response.json()

    df_ghi = pd.DataFrame({
        'Date_utc': pd.to_datetime(data['hourly']['time']),
        'GHI': data['hourly']['shortwave_radiation']
    })
    df_ghi = df_ghi.set_index('Date_utc')

    # Resample GHI from hourly to 15-minute using interpolation
    df_ghi_15m = df_ghi[['GHI']].resample('15min').interpolate(method='linear').reset_index()

    # --- 4. Merge PV, Temperature, and GHI data ---
    dts = dts1.merge(dts_t, on="Date_utc", how="left")
    dts = dts.merge(df_ghi_15m, on="Date_utc", how="left")

    dts = dts[['Date_utc', 'offset', 'PV', 'temp', 'GHI']]
    dts = dts.rename(columns={"Date_utc": "Date"})

    # --- 5. Feature Engineering (Temporal features) ---
    dts['hour'] = dts['Date'].dt.hour
    dts['dayofyear'] = dts['Date'].dt.dayofyear
    dts['month'] = dts['Date'].dt.month
    dts['dayofweek'] = dts['Date'].dt.dayofweek

    dts['hour_sin'] = np.sin(2*np.pi*dts['hour']/24)
    dts['hour_cos'] = np.cos(2*np.pi*dts['hour']/24)
    dts['doy_sin'] = np.sin(2*np.pi*dts['dayofyear']/365)
    dts['doy_cos'] = np.cos(2*np.pi*dts['dayofyear']/365)
    dts['dow_sin'] = np.sin(2*np.pi*dts['dayofweek']/7)
    dts['dow_cos'] = np.cos(2*np.pi*dts['dayofweek']/7)

    # ==============================================================================
    # --- 6. OVERCOME CURTAILMENT (Dynamic by Year) ---
    # ==============================================================================
    print(f"Reconstructing curtailed data for {year}...")
    dts['is_curtailed'] = False

    if year == 2022:
        curtail_1 = (dts['Date'] >= '2022-04-01') & (dts['Date'] <= '2022-05-05')
        curtail_2 = (dts['Date'] >= '2022-07-03') & (dts['Date'] <= '2022-09-07')
        dts.loc[curtail_1 | curtail_2, 'is_curtailed'] = True
    elif year == 2023:
        curtail_1 = (dts['Date'] >= '2023-06-22') & (dts['Date'] <= '2023-09-29')
        dts.loc[curtail_1, 'is_curtailed'] = True

    # Define features used to train the Random Forest
    ml_features = ['GHI', 'temp']

    if dts['is_curtailed'].sum() > 0:
        # Split into clean data (to train) and curtailed data (to fix)
        clean_data = dts[~dts['is_curtailed']].dropna(subset=ml_features + ['PV'])
        curtailed_data = dts[dts['is_curtailed']].copy()
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(clean_data[ml_features], clean_data['PV'])
        
        # Predict and overwrite the bad PV values
        dts.loc[dts['is_curtailed'], 'PV'] = model.predict(curtailed_data[ml_features])
        print(f"Curtailment successfully repaired for {year}.")
    else:
        print(f"No curtailment periods defined or found for {year}.")
    # ==============================================================================

    # Append processed dataset to the list
    all_dts.append(dts)

# --- 7. Concatenate and Save ---
if all_dts:
    print("\nConcatenating all datasets...")
    master_dataset = pd.concat(all_dts, ignore_index=True)
    
    # Plot the full concatenated dataset to visually verify
    plt.figure(figsize=(15, 5))
    plt.plot(master_dataset['Date'], master_dataset['PV'], label='PV Generation (Cleaned & Imputed)')
    plt.title('Master Dataset: PV Generation (2022-2024)')
    plt.xlabel('Date')
    plt.ylabel('PV')
    plt.legend()
    plt.grid(True)
    output_plot_path = Path(__file__).parent / "data_cleaner_results.png"
    plt.savefig(output_plot_path)
    print(f"Data Cleaner plots successfully saved to: {output_plot_path}")
    plt.close()
    
    output_path = root_proj / "data/master_dataset.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    master_dataset.to_parquet(output_path)
    print(f"Master dataset saved successfully to {output_path}")
    print(f"Total rows: {len(master_dataset)}")
else:
    print("No data was processed.")