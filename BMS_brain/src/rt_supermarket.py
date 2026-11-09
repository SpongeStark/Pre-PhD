"""
Standalone Real-Time Secondary Controller Runner: Supermarket Facility (24-Hour / 96-Step)
"""
import sys
import argparse
from pathlib import Path

SRC_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SRC_DIR))

from rt_controller import run_rt_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--horizon", type=int, default=96, help="Horizon steps (default: 96 = 24h)")
    args = parser.parse_args()
    print("\n--- Launching 24-Hour RT Controller: Supermarket Facility ---")
    run_rt_pipeline(mode="supermarket", start_date=args.date, horizon=args.horizon)
