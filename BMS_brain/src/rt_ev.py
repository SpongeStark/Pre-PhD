"""
Standalone Real-Time Secondary Controller Runner: Lidl Commercial EV Chargers
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SRC_DIR))

from rt_controller import run_rt_pipeline

if __name__ == "__main__":
    print("\n--- Launching 2-Day RT Controller: Lidl Commercial EV Chargers ---")
    run_rt_pipeline(mode="ev", horizon=192)
