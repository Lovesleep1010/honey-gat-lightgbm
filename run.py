from pathlib import Path
import argparse
import importlib.util
import sys

EXPERIMENT = "main"
CHECK_ONLY = False
ENTRIES = ("main", "analysis", "baselines", "graph_structure")
ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=ENTRIES, default=EXPERIMENT)
    parser.add_argument("--check", action="store_true", default=CHECK_ONLY)
    args = parser.parse_args(argv)
    directory = ROOT / "experiments" / args.experiment
    sys.path[:0] = [str(directory), str(ROOT / "experiments" / "main")]
    spec = importlib.util.spec_from_file_location("selected_experiment_launcher", directory / "config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.check:
        module.CHECK_ONLY = True
    module.main()


if __name__ == "__main__":
    main()
