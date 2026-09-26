"""One-command reproduction for Problem 4, including Q1 recalibration."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    # Rebuild the previously missing calibrated-parameter file from the two
    # supplied workbooks before running Problem 4.
    subprocess.run([sys.executable, str(ROOT / "问题1" / "run.py")], check=True)
    subprocess.run([sys.executable, str(HERE / "optimize.py")], check=True)
    subprocess.run([sys.executable, str(HERE / "verify.py")], check=True)
    subprocess.run([sys.executable, str(HERE / "build_report.py")], check=True)
    subprocess.run([sys.executable, str(HERE / "build_manifest.py")], check=True)


if __name__ == "__main__":
    raise SystemExit(main())
