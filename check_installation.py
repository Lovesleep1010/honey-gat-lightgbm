"""Check data and entry points; --tests also runs protocol/regression fixtures."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SUITES = {
    "main": ["test_protocol", "test_early_stopping"],
    "analysis": ["test_protocol"],
    "baselines": ["test_protocol", "test_baselines"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", action="store_true")
    args = parser.parse_args()
    for entry, tests in SUITES.items():
        print(f"Checking {entry}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "run.py"),
                        "--experiment", entry, "--check"], check=True, cwd=ROOT)
        if args.tests:
            subprocess.run([sys.executable, "-m", "unittest", "-q", *tests],
                           check=True, cwd=ROOT / "experiments" / entry)
    print("All requested checks passed.")


if __name__ == "__main__":
    main()
