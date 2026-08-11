import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for professional publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Define root project path
root_proj = Path(__file__).parent.parent.resolve()

# Output folder in figs/seasonality plots/
output_dir = root_proj / "figs" / "seasonality plots"
output_dir.mkdir(parents=True, exist_ok=True)

# Path to data clean scripts and datasets
pv_data_path = root_proj / "data/master_dataset.parquet"
con_data_path = root_proj / "data/master_dataset_con.parquet"
ev_data_path = root_proj / "data/master_dataset_ev.parquet"

# Ensure datasets exist, or run cleaner scripts if missing
def load_or_generate_data():
    if not pv_data_path.exists():
        print("PV dataset not found. Running data_cleaner.py...")
        import data_cleaner
    if not con_data_path.exists():
        print("Consumption dataset not found. Running data_cleaner_con.py...")
        import data_cleaner_con
    if not ev_data_path.exists():
        print("EV dataset not found. Running data_cleaner_EV.py...")
        import data_cleaner_EV

    print("Loading cleaned datasets...")
    df_pv = pd.read_parquet(pv_data_path)
    df_con = pd.read_parquet(con_data_path)
    df_ev = pd.read_parquet(ev_data_path)
    
    return df_pv, df_con, df_ev

def add_season_and_time_features(df):
    """Adds season, month_name, hour, and day_name features to a dataframe."""
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['hour'] = df['Date'].dt.hour
        df['month'] = df['Date'].dt.month
        df['month_name'] = df['Date'].dt.strftime('%b')
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['day_name'] = df['Date'].dt.strftime('%a')
        
        # Season classification (Meteorological)
        # Winter: Dec, Jan, Feb (12, 1, 2)
        # Spring: Mar, Apr, May (3, 4, 5)
        # Summer: Jun, Jul, Aug (6, 7, 8)
        # Autumn: Sep, Oct, Nov (9, 10, 11)
        season_map = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        }
        df['season'] = df['month'].map(season_map)
        
        # Order days and months logically
        season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
        df['season'] = pd.Categorical(df['season'], categories=season_order, ordered=True)
        
        day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        df['day_name'] = pd.Categorical(df['day_name'], categories=day_order, ordered=True)
        
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        df['month_name'] = pd.Categorical(df['month_name'], categories=month_order, ordered=True)
        
    return df

def generate_source_seasonality_plot(df, value_col, title_prefix, color_palette, save_filename):
    """Generates a 3-panel seasonality analysis figure for a single data source."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    season_colors = {
        'Winter': '#2b5c8f',
        'Spring': '#2ca02c',
        'Summer': '#ff7f0e',
        'Autumn': '#d62728'
    }

    # 1. Diurnal Seasonality Profile by Season (24-hour average)
    ax1 = axes[0]
    hourly_season = df.groupby(['season', 'hour'], observed=False)[value_col].mean().reset_index()
    for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
        data_s = hourly_season[hourly_season['season'] == season]
        ax1.plot(data_s['hour'], data_s[value_col], label=season, color=season_colors[season], linewidth=2.5)
    
    ax1.set_title(f'1. Diurnal Profile by Season ({title_prefix})', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Hour of Day (UTC)', fontsize=10)
    ax1.set_ylabel('Average Power (kW)', fontsize=10)
    ax1.set_xticks(range(0, 24, 3))
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(title='Season', frameon=True, facecolor='white', edgecolor='none')

    # 2. Weekly Seasonality Profile (Day of Week)
    ax2 = axes[1]
    weekly_season = df.groupby(['day_name', 'season'], observed=False)[value_col].mean().unstack()
    weekly_season.plot(kind='bar', ax=ax2, color=[season_colors[s] for s in weekly_season.columns], width=0.8)
    ax2.set_title(f'2. Weekly Profile by Day of Week ({title_prefix})', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Day of Week', fontsize=10)
    ax2.set_ylabel('Average Power (kW)', fontsize=10)
    ax2.set_xticklabels(weekly_season.index, rotation=0)
    ax2.grid(True, linestyle='--', alpha=0.6, axis='y')
    ax2.legend(title='Season', frameon=True, facecolor='white', edgecolor='none')

    # 3. Monthly Seasonality Profile (Average & Peak Power)
    ax3 = axes[2]
    monthly_stats = df.groupby('month_name', observed=False)[value_col].agg(['mean', 'max']).reset_index()
    ax3.bar(monthly_stats['month_name'], monthly_stats['mean'], label='Average Power (kW)', color=color_palette[0], alpha=0.85, width=0.6)
    ax3.plot(monthly_stats['month_name'], monthly_stats['max'], label='Peak Power (kW)', color='#d95f02', marker='o', linewidth=2)
    ax3.set_title(f'3. Monthly Distribution ({title_prefix})', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('Month', fontsize=10)
    ax3.set_ylabel('Power (kW)', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.6, axis='y')
    ax3.legend(frameon=True, facecolor='white', edgecolor='none')

    plt.suptitle(f'Seasonality Analysis — {title_prefix}', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = output_dir / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved seasonality plot: {save_path}")
    plt.close()

def generate_combined_overview_plot(df_pv, df_con, df_ev):
    """Generates an overview plot comparing seasonal profiles across all 3 data sources."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    sources = [
        (df_pv, 'PV', 'Solar PV Generation Seasonality (kW)', '#e6ab02'),
        (df_con, 'c_gen', 'Building Consumption Seasonality (kW)', '#7570b3'),
        (df_ev, 'c_ev', 'EV Charging Seasonality (kW)', '#1b9e77')
    ]

    season_colors = {
        'Winter': '#2b5c8f',
        'Spring': '#2ca02c',
        'Summer': '#ff7f0e',
        'Autumn': '#d62728'
    }

    for idx, (df, col, ylabel, base_color) in enumerate(sources):
        ax = axes[idx]
        hourly_season = df.groupby(['season', 'hour'], observed=False)[col].mean().reset_index()
        for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
            data_s = hourly_season[hourly_season['season'] == season]
            ax.plot(data_s['hour'], data_s[col], label=season, color=season_colors[season], linewidth=2.5)
        
        ax.set_title(f'Diurnal Profile: {ylabel}', fontsize=12, fontweight='bold', pad=8)
        ax.set_ylabel('Power (kW)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title='Season', loc='upper right', frameon=True, facecolor='white')

    axes[-1].set_xlabel('Hour of Day (UTC)', fontsize=11)
    axes[-1].set_xticks(range(0, 24, 1))

    plt.suptitle('Multi-Source Diurnal Seasonality Comparison (PV vs. Building vs. EV)', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()

    save_path = output_dir / "all_sources_seasonality_overview.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined seasonality overview plot: {save_path}")
    plt.close()

def main():
    print("=== Starting Seasonality Analysis ===")
    df_pv, df_con, df_ev = load_or_generate_data()

    # Preprocess date features
    df_pv = add_season_and_time_features(df_pv)
    df_con = add_season_and_time_features(df_con)
    df_ev = add_season_and_time_features(df_ev)

    # 1. PV Seasonality Plot
    generate_source_seasonality_plot(
        df_pv, 'PV', 'Solar PV Generation',
        color_palette=['#e6ab02'],
        save_filename='pv_seasonality_plot.png'
    )

    # 2. Building Consumption Seasonality Plot
    generate_source_seasonality_plot(
        df_con, 'c_gen', 'Building Consumption',
        color_palette=['#7570b3'],
        save_filename='consumption_seasonality_plot.png'
    )

    # 3. EV Charging Seasonality Plot
    generate_source_seasonality_plot(
        df_ev, 'c_ev', 'EV Charging Demand',
        color_palette=['#1b9e77'],
        save_filename='ev_seasonality_plot.png'
    )

    # 4. Multi-Source Overview Seasonality Plot
    generate_combined_overview_plot(df_pv, df_con, df_ev)

    print(f"\nAll seasonality plots successfully exported to: {output_dir}")

if __name__ == "__main__":
    main()
