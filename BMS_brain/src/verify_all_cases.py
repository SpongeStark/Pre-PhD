"""
Wrapper script that invokes the centralized OpenDSS-G Time-Mode runner (`opendss_runner.py`).
All OpenDSS circuit operations are consolidated in `opendss_runner.py`.
"""
from opendss_runner import run_all_cases

if __name__ == "__main__":
    run_all_cases(source="rt", enforce_zero_export=True)
