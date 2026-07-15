# Import necessary libraries
import sys
import requests
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Define the root path of the project
root_proj = Path(__file__).parent.parent.resolve()

# Add the 'src' directory to the system path
if str(root_proj/"src") not in sys.path:
    sys.path.append(str(root_proj/"src"))

# ==========================================
# SET YOUR YEAR HERE (2022 or 2023)
# ==========================================
year = 2023


# --- 1. Load and process system data (PV generation) ---
file_name = root_proj / f"data/DTS_LIDL_{year}.csv"
dts1 = pd.read_csv(file_name, sep=";")

dts1["Date_utc"] = pd.to_datetime(
    dts1["Date"],
    format="%d/%m/%Y %H:%M"
) - pd.Timedelta(hours=1) 

dts1['offset'] = (dts1["Date_utc"].dt.tz_localize("UTC").dt.tz_convert("Europe/Paris").dt.tz_localize(None) - dts1["Date_utc"]).dt.total_seconds()//3600

# --- 2. Load and process weather data (Temperature) ---
temp = pd.read_csv(root_proj / f"data/orly_{year}_hourly_weather.csv", sep=",")
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
# --- 6. OVERCOME CURTAILMENT (Disabled/Removed) ---
# ==============================================================================
print(f"Curtailment reconstruction is disabled for {year}.")
dts['is_curtailed'] = False
# ==============================================================================


# --- 7. Train/Test Split ---
try:
    initial = dts.index[dts['Date'] == pd.Timestamp(f'{year}-10-01 01:00:00', tz='Europe/Paris').tz_convert("UTC").tz_localize(None)][0]
    
    df_train = dts.iloc[:initial]
    df_test = dts.iloc[initial:]
    
    # Plot the split to visually verify
    plt.figure(figsize=(14, 5))
    plt.plot(df_train.Date, df_train['PV'], label='Train (Cleaned & Imputed)')
    plt.plot(df_test.Date, df_test['PV'], label='Test')
    plt.grid(visible=True)
    plt.ylabel('PV Generation (kW)')
    plt.xlabel('Date (UTC)')
    plt.title(f'Train-Test Split for PV Generation Data ({year})')
    plt.legend()
    output_plot_path = Path(__file__).parent / f"curtailment_results_{year}.png"
    plt.savefig(output_plot_path)
    print(f"Curtailment plots successfully saved to: {output_plot_path}")
    plt.close()
except IndexError:
    print("Warning: The exact October 1st split date was not found in the dataset. Skipping plot.")