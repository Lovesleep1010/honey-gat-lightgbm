from pathlib import Path
import sys


EXPERIMENTS = ["all"]
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
SCHEDULER_EPOCHS = 50
SEEDS = [42, 123, 2024, 3407, 9999]
K_VALUES = [8, 10, 12, 14, 16]
BAND_COUNTS = [16, 32, 48]
SPARSE_K = 8
RANKING_SEED = 42
PERMUTATION_REPEATS = 5
DEVICE = "auto"
CPU_THREADS = 4
STAGE = "all"
RESUME = True
CHECK_ONLY = False


DATA_FILE = ""

OUTPUT_FOLDER = ""


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "main"))


def build_argv():
    data = Path(DATA_FILE) if DATA_FILE else HERE.parents[1] / "data" / "adulteration_dataset_26_08_2021.csv"
    output = Path(OUTPUT_FOLDER) if OUTPUT_FOLDER else HERE / f"results_max{EPOCHS}_patience{EARLY_STOPPING_PATIENCE}"
    if not data.is_absolute():
        data = HERE / data
    if not output.is_absolute():
        output = HERE / output
    args = [
        "--data", str(data), "--output", str(output),
        "--experiments", *EXPERIMENTS,
        "--epochs", str(EPOCHS), "--scheduler-epochs", str(SCHEDULER_EPOCHS),
        "--early-stopping-patience", str(EARLY_STOPPING_PATIENCE),
        "--seeds", *map(str, SEEDS),
        "--k-values", *map(str, K_VALUES),
        "--band-counts", *map(str, BAND_COUNTS),
        "--sparse-k", str(SPARSE_K), "--ranking-seed", str(RANKING_SEED),
        "--permutation-repeats", str(PERMUTATION_REPEATS),
        "--device", DEVICE, "--threads", str(CPU_THREADS), "--stage", STAGE,
    ]
    if RESUME:
        args.append("--resume")
    if CHECK_ONLY:
        args.append("--check")
    return args


def main():
    from pipeline import main as run
    run(build_argv())


if __name__ == "__main__":
    main()
