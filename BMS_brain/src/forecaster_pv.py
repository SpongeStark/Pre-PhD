import sys
import pandas as pd
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Forecasting Imports
try:
    from skforecast.direct import ForecasterDirect
except ImportError:
    from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect as ForecasterDirect

from skforecast.preprocessing import RollingFeatures
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_split_data(config: dict) -> tuple:
    """Loads the master dataset and splits it based on config dates."""
    
    # 1. Load the single, clean dataset produced by data_cleaner.py
    # Resolve the project root relative to this file (src/forecaster.py -> BMS_brain)
    root_proj = Path(__file__).parent.parent
    data_path = root_proj / config['paths']['processed_data']
        
    df = pd.read_parquet(data_path)
    
    # Ensure it is sorted by time just in case!
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Remove any duplicate timestamps (e.g., from Daylight Saving Time transitions or overlaps)
    df = df.drop_duplicates(subset=["Date"], keep="last")
    
    # skforecast strictly requires the index to be a DatetimeIndex (with frequency) or a RangeIndex.
    # We will set the 'Date' column as the index and specify the 15-minute frequency.
    df = df.set_index("Date")
    # In case there are missing 15-minute intervals, asfreq inserts them. We forward-fill to avoid NaNs.
    df = df.asfreq("15min").ffill()
    
    # 2. Extract split dates from config
    val_start = pd.to_datetime(config['forecasting']['split_dates']['val_start'])
    test_start = pd.to_datetime(config['forecasting']['split_dates']['test_start'])
    
    # 3. Apply the splits using the new DatetimeIndex
    # Train: 2022 to end of 2023
    df_train = df[df.index < val_start].copy()
    
    # Validation: Jan 2024 to Sept 30, 2024
    df_val = df[(df.index >= val_start) & (df.index < test_start)].copy()
    
    # Test (Simulation): Oct 1, 2024 onwards
    df_test = df[df.index >= test_start].copy()
    
    print(f"Data Split Complete:")
    print(f"Train size: {len(df_train)} rows")
    print(f"Val size: {len(df_val)} rows")
    print(f"Test size: {len(df_test)} rows")
    
    return df_train, df_val, df_test

def run(config: dict):
    print("Starting Forecaster Pipeline...")
    
    # Step 1: Load and split the newly complete dataset
    df_train, df_val, df_test = load_and_split_data(config)
    
    # Initialize Rolling Features
    rolling = RollingFeatures(stats=['max', 'mean', 'std'], window_sizes=96)

    # 2. Define the exact lags to use
    # 1 to 4 = last hour. 96 = yesterday. 672 = last week.
    target_lags = [1, 2, 3, 4, 96, 672] 

    # 3. Initialize the Direct Forecaster
    # forecaster = ForecasterDirect(
    #     estimator       = LGBMRegressor(random_state=15926, verbose=-1, n_estimators=200),
    #     steps           = 96,           # Hardcode the 24-hour horizon (96 x 15-mins)
    #     lags            = target_lags, 
    #     window_features = rolling
    # )
    forecaster= ForecasterDirect(
    estimator = XGBRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=5,           # XGBoost specific: controls tree complexity
        random_state=15926, 
        n_jobs=-1              # Use all CPU cores
    ),
    steps           = 96,
    lags            = target_lags, 
    window_features = rolling
    )

    # Define your exogenous columns
    exog_cols = [
        'temp', 'GHI', # Weather
        'hour_sin', 'hour_cos', 
        'doy_sin', 'doy_cos', 
        'dow_sin', 'dow_cos'
    ]

    print("\nTraining forecaster on training set...")
    forecaster.fit(
        y    = df_train['PV'], 
        exog = df_train[exog_cols]
    )
    
    # --- Save the trained model ---
    import joblib
    root_proj = Path(__file__).parent.parent
    model_out_path = root_proj / config['paths']['model_output']
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(forecaster, model_out_path)
    print(f"Model successfully saved to {model_out_path}")
    
    # --- Evaluation ---
    print("\nPredicting for the first 24 hours (96 steps) of the Test Set...")
    
    # We will forecast the first 96 steps of df_test. 
    # We need the most recent historical PV data (from df_val) to feed into the lags.
    last_window_data = df_val['PV'].iloc[-max(target_lags):]
    
    # The exogenous data for the next 96 steps
    df_test_first_day = df_test.iloc[:96]
    exog_future = df_test_first_day[exog_cols]
    
    # Generate predictions
    predictions = forecaster.predict(
        steps = 96,
        last_window = last_window_data,
        exog = exog_future
    )
    
    # Calculate error metrics
    actuals = df_test_first_day['PV'].values
    preds = predictions.values
    
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2 = r2_score(actuals, preds)
    
    print("\n--- Forecasting Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} kW")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kW")
    print(f"R2 Score: {r2:.3f}")
    
    # Plot comparison using the index (which is now our Date)
    plt.figure(figsize=(14, 6))
    plt.plot(df_test_first_day.index, actuals, label='Actual PV Generation', linewidth=2, color='blue')
    plt.plot(df_test_first_day.index, preds, label='Predicted PV Generation', linestyle='--', linewidth=2, color='orange')
    plt.title('Forecaster Evaluation: Actual vs. Predicted (First 24 hours of Test Set)')
    plt.xlabel('Date')
    plt.ylabel('PV Generation (kW)')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Entry Point for Individual Testing ---
if __name__ == "__main__":
    # To run this individually, we load the config.yaml and pass it to run()
    config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        print(f"Error: Could not find {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    run(config)
