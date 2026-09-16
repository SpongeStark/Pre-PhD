import json
import pandas as pd
import numpy as np
from pathlib import Path

def process_caltech_sessions(
    raw_parquet_path: Path,
    out_parquet_path: Path,
    target_year: int = 2022
):
    print(f"Reading raw Caltech sessions from {raw_parquet_path}...")
    df_raw = pd.read_parquet(raw_parquet_path)
    print(f"Total raw sessions: {len(df_raw)}")
    
    # 1. Clean & filter valid sessions
    df_clean = df_raw[df_raw['kWhDelivered'] > 0.01].copy()
    print(f"Sessions with kWhDelivered > 0.01: {len(df_clean)}")
    
    # Parse datetimes (all original times are UTC RFC 1123 strings)
    df_clean['t_conn'] = pd.to_datetime(df_clean['connectionTime'], utc=True)
    df_clean['t_disc'] = pd.to_datetime(df_clean['disconnectTime'], utc=True)
    df_clean['t_done'] = pd.to_datetime(df_clean['doneChargingTime'], utc=True)
    
    # Drop rows with invalid connectionTime
    df_clean = df_clean.dropna(subset=['t_conn'])
    
    # Effective charging end time
    # Prefer doneChargingTime if valid and after connectionTime
    valid_done = df_clean['t_done'].notna() & (df_clean['t_done'] > df_clean['t_conn'])
    df_clean['t_end'] = np.where(valid_done, df_clean['t_done'], df_clean['t_disc'])
    
    # If t_end is still invalid or <= t_conn, use t_conn + fallback duration based on kWh
    bad_end = df_clean['t_end'].isna() | (df_clean['t_end'] <= df_clean['t_conn'])
    # Assume 6.6 kW charging rate for fallback
    fallback_hours = np.clip(df_clean.loc[bad_end, 'kWhDelivered'] / 6.6, 0.25, 12.0)
    df_clean.loc[bad_end, 't_end'] = df_clean.loc[bad_end, 't_conn'] + pd.to_timedelta(fallback_hours, unit='h')
    
    # Enforce maximum physical power constraint of Level 2 EVSEs (approx 7.2 kW)
    dur_hours = (df_clean['t_end'] - df_clean['t_conn']).dt.total_seconds() / 3600.0
    implied_kw = df_clean['kWhDelivered'] / dur_hours
    too_fast = implied_kw > 7.2
    if too_fast.any():
        adjusted_hours = df_clean.loc[too_fast, 'kWhDelivered'] / 7.2
        df_clean.loc[too_fast, 't_end'] = df_clean.loc[too_fast, 't_conn'] + pd.to_timedelta(adjusted_hours, unit='h')
        dur_hours = (df_clean['t_end'] - df_clean['t_conn']).dt.total_seconds() / 3600.0
        
    df_clean['duration_hours'] = dur_hours
    df_clean['avg_kw'] = df_clean['kWhDelivered'] / df_clean['duration_hours']
    
    print(f"Sanitized sessions: {len(df_clean)}")
    print(f"Total delivered energy: {df_clean['kWhDelivered'].sum():,.2f} kWh")
    print(f"Avg session duration: {df_clean['duration_hours'].mean():.2f} h")
    print(f"Avg session power: {df_clean['avg_kw'].mean():.2f} kW (Max: {df_clean['avg_kw'].max():.2f} kW)")
    
    # 2. Localize to Pacific Time (site local time)
    df_clean['t_start_local'] = df_clean['t_conn'].dt.tz_convert('America/Los_Angeles')
    df_clean['t_end_local'] = df_clean['t_end'].dt.tz_convert('America/Los_Angeles')
    
    # 3. Create 15-minute grid timeline for 2019
    timeline_2019 = pd.date_range(
        start='2019-01-01 00:00:00',
        end='2019-12-31 23:45:00',
        freq='15min',
        tz='America/Los_Angeles'
    )
    N_steps = len(timeline_2019)
    energy_kwh = np.zeros(N_steps)
    step_sec = 15 * 60
    t0_ref = timeline_2019[0]
    
    print("\nResampling sessions onto 15-minute intervals...")
    for _, row in df_clean.iterrows():
        t_s = row['t_start_local']
        t_e = row['t_end_local']
        kwh = row['kWhDelivered']
        dur_s = (t_e - t_s).total_seconds()
        if dur_s <= 0 or kwh <= 0:
            continue
        p = kwh / (dur_s / 3600.0)
        
        idx_s = int((t_s - t0_ref).total_seconds() // step_sec)
        idx_e = int((t_e - t0_ref).total_seconds() // step_sec)
        idx_s = max(0, min(N_steps - 1, idx_s))
        idx_e = max(0, min(N_steps - 1, idx_e))
        
        for b in range(idx_s, idx_e + 1):
            bin_s = timeline_2019[b]
            bin_e = bin_s + pd.Timedelta(minutes=15)
            overlap_s = max(t_s, bin_s)
            overlap_e = min(t_e, bin_e)
            if overlap_e > overlap_s:
                overlap_h = (overlap_e - overlap_s).total_seconds() / 3600.0
                energy_kwh[b] += p * overlap_h
                
    power_kw_2019 = energy_kwh / 0.25 # kW
    print(f"2019 15-min Load Profile:")
    print(f"  Total energy: {energy_kwh.sum():,.2f} kWh")
    print(f"  Peak power:   {power_kw_2019.max():.2f} kW")
    print(f"  Mean power:   {power_kw_2019.mean():.2f} kW")
    
    # 4. Project onto target year (2022) matching day-of-week and time-of-day
    print(f"\nProjecting load profile onto {target_year} calendar...")
    df_2019 = pd.DataFrame({
        'dt_2019': timeline_2019,
        'c_ev': power_kw_2019,
        'month': timeline_2019.month,
        'day': timeline_2019.day,
        'dayofweek': timeline_2019.dayofweek,
        'time': [t.time() for t in timeline_2019]
    })
    
    # Generate 2022 target timeline (35,040 steps of 15 min, naive or local)
    timeline_target = pd.date_range(
        start=f'{target_year}-01-01 00:00:00',
        end=f'{target_year}-12-31 23:45:00',
        freq='15min'
    )
    
    # Day-of-week aligned mapping:
    # Match each target day to the closest calendar day in 2019 with the exact same day of week
    target_dates = pd.Series(timeline_target.date).unique()
    source_dates = pd.Series(timeline_2019.date).unique()
    
    source_df_dates = pd.DataFrame({
        'date': source_dates,
        'doy': [d.timetuple().tm_yday for d in source_dates],
        'dow': [d.weekday() for d in source_dates]
    })
    
    date_mapping = {}
    for td in target_dates:
        t_doy = td.timetuple().tm_yday
        t_dow = td.weekday()
        # Find 2019 date with same dow closest to t_doy
        same_dow = source_df_dates[source_df_dates['dow'] == t_dow]
        closest_idx = (same_dow['doy'] - t_doy).abs().idxmin()
        date_mapping[td] = same_dow.loc[closest_idx, 'date']
        
    # Build the aligned 2022 series
    source_profile_by_datetime = dict(zip(
        zip(timeline_2019.date, [t.time() for t in timeline_2019]),
        power_kw_2019
    ))
    
    c_ev_target = np.zeros(len(timeline_target))
    for i, dt in enumerate(timeline_target):
        s_date = date_mapping[dt.date()]
        c_ev_target[i] = source_profile_by_datetime.get((s_date, dt.time()), 0.0)
        
    df_out = pd.DataFrame({
        'Date': timeline_target,
        'c_ev': c_ev_target
    })
    
    # Add cyclical features matching master_dataset_ev.parquet schema
    hour = df_out['Date'].dt.hour + df_out['Date'].dt.minute / 60.0
    doy = df_out['Date'].dt.dayofyear
    dow = df_out['Date'].dt.dayofweek
    
    df_out['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df_out['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df_out['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df_out['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    df_out['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    df_out['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
    
    print(f"\nFinal {target_year} Caltech EV Load Profile:")
    print(f"  Shape:        {df_out.shape}")
    print(f"  Total energy: {(df_out['c_ev'].sum() * 0.25):,.2f} kWh")
    print(f"  Peak power:   {df_out['c_ev'].max():.2f} kW")
    print(f"  Mean power:   {df_out['c_ev'].mean():.2f} kW")
    
    df_out.to_parquet(out_parquet_path, index=False)
    print(f"Successfully saved to {out_parquet_path}")

if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    raw_path = root_dir / "data/caltech_ev_sessions_2019.parquet"
    out_path = root_dir / "data/master_dataset_caltech_ev.parquet"
    process_caltech_sessions(raw_path, out_path, target_year=2022)
