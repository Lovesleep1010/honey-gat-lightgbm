"""Run the remaining revision experiments in PyCharm."""

from pathlib import Path
import argparse
import sys


# Leave blank to select the completed main-model results folder when running.
MAIN_RESULTS_FOLDER = ""
OUTPUT_FOLDER = ""
DEVICE = "auto"
CPU_THREADS = 4


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=MAIN_RESULTS_FOLDER)
    parser.add_argument("--output", default=OUTPUT_FOLDER)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEVICE)
    parser.add_argument("--threads", type=int, default=CPU_THREADS)
    parser.add_argument("--check", action="store_true", help="Check data and settings without training")
    args = parser.parse_args(argv)
    sys.path[:0] = [str(root / "experiments" / "main"), str(root / "experiments" / "analysis")]
    from revision import run
    run(root, args)


if __name__ == "__main__":
    main()
