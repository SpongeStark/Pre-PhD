"""
Standalone Entrypoint: Supermarket Consumption Forecaster
Executes the 24-hour ahead forecasting engine for commercial supermarket load
using data from data/master_dataset_con.parquet.
"""
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    src_dir = Path(__file__).parent.resolve()
    engine_script = src_dir / "forecaster_engine.py"
    
    cmd = [sys.executable, "-u", str(engine_script), "--target", "con"] + sys.argv[1:]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)
