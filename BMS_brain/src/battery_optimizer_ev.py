"""
Standalone Entrypoint: Battery Optimizer with EV Charging Demand
Executes the battery optimizer with the EV charging station consumption profile
loaded from data/master_dataset_ev.parquet added to the supermarket load.
"""
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    src_dir = Path(__file__).parent.resolve()
    target_script = src_dir / "battery_optimizer.py"
    
    cmd = [sys.executable, "-u", str(target_script), "--with-ev"] + sys.argv[1:]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)
