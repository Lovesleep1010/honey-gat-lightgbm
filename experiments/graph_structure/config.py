"""Graph-structure ablation settings for PyCharm."""

from pathlib import Path
from types import SimpleNamespace
import sys


SEEDS = [2026, 2027, 2028, 2029, 2030]
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
DEVICE = "auto"
CPU_THREADS = 4
CHECK_ONLY = False
DATA_FILE = ""
OUTPUT_FOLDER = ""

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    sys.path[:0] = [str(HERE), str(HERE.parent / "main")]
    from ablation import run
    data = Path(DATA_FILE).expanduser() if DATA_FILE else ROOT / "data/adulteration_dataset_26_08_2021.csv"
    output = Path(OUTPUT_FOLDER).expanduser() if OUTPUT_FOLDER else HERE / "results_graph_structure"
    if not data.is_absolute():
        data = ROOT / data
    if not output.is_absolute():
        output = ROOT / output
    args = SimpleNamespace(seeds=SEEDS, epochs=EPOCHS, patience=EARLY_STOPPING_PATIENCE,
                           device=DEVICE, threads=CPU_THREADS, check=CHECK_ONLY,
                           data=data.resolve(), output=output.resolve())
    return run(ROOT, args)


if __name__ == "__main__":
    main()
