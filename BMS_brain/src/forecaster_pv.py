"""
Standalone Entrypoint: Solar PV Forecaster
Executes the 24-hour ahead forecasting engine for solar PV generation
using data from data/master_dataset.parquet.
"""
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    src_dir = Path(__file__).parent.resolve()
    engine_script = src_dir / "forecaster_engine.py"
    
    cmd = [sys.executable, "-u", str(engine_script), "--target", "pv"] + sys.argv[1:]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)
