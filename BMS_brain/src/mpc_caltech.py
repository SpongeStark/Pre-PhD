import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from mpc_optimizer import run_mpc_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--horizon", type=int, default=96, help="Horizon steps (default: 96 = 24h)")
    args = parser.parse_args()
    run_mpc_pipeline(mode="caltech_ev", start_date=args.date, horizon=args.horizon)
