"""
Standalone Real-Time Secondary Controller Runner: Caltech Campus EV Fleet
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SRC_DIR))

from rt_controller import run_rt_pipeline

if __name__ == "__main__":
    print("\n--- Launching 2-Day RT Controller: Caltech Campus EV Fleet ---")
    run_rt_pipeline(mode="caltech_ev", horizon=192)
