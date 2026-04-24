import pandas as pd
import requests

# Orly Airport Coordinates
latitude = 48.73
longitude = 2.42

# Open-Meteo Historical Weather API URL
url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}&longitude={longitude}&"
    f"start_date=2023-01-01&end_date=2023-12-31&"
    f"hourly=shortwave_radiation&"
    f"timezone=UTC"
)

print("Downloading 2023 GHI data for Orly Airport...")
response = requests.get(url)
data = response.json()

# Extract time and GHI (shortwave_radiation)
df_2023 = pd.DataFrame({
    'Date_utc': pd.to_datetime(data['hourly']['time']),
    'GHI': data['hourly']['shortwave_radiation']
})

# Save to CSV
output_file = "Orly_GHI_2023.csv"
df_2023.to_csv(output_file, index=False)
print(f"Success! Data saved to {output_file}")
print(df_2023.head())