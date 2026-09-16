"""
Standalone Entrypoint: Battery Optimizer for Caltech ACN EV Charging Data
Executes the battery optimizer with the Caltech campus EV charging profile
loaded from data/master_dataset_caltech_ev.parquet.
"""
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    src_dir = Path(__file__).parent.resolve()
    target_script = src_dir / "battery_optimizer.py"
    
    cmd = [sys.executable, "-u", str(target_script), "--mode", "caltech_ev"] + sys.argv[1:]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)
