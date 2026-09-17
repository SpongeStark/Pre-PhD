import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from mpc_optimizer import run_mpc_pipeline

if __name__ == "__main__":
    run_mpc_pipeline(mode="supermarket")
