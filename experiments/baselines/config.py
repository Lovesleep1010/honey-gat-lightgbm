from pathlib import Path
import sys


MODELS = ["graphsage", "gcn", "gin", "gat"]
SEEDS = [2026, 2027, 2028, 2029, 2030]
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
SCHEDULER_EPOCHS = None
DEVICE = "auto"
CPU_THREADS = 4
STAGE = "all"
RESUME = True
CHECK_ONLY = False

DATA_FILE = ""
OUTPUT_FOLDER = ""


HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "main"))


def build_argv():
    data = Path(DATA_FILE) if DATA_FILE else HERE.parents[1] / "data" / "adulteration_dataset_26_08_2021.csv"
    output = Path(OUTPUT_FOLDER) if OUTPUT_FOLDER else HERE / f"results_max{EPOCHS}_patience{EARLY_STOPPING_PATIENCE}"
    if not data.is_absolute():
        data = HERE / data
    if not output.is_absolute():
        output = HERE / output
    argv = ["--data", str(data), "--output", str(output), "--models", *MODELS,
            "--seeds", *map(str, SEEDS), "--epochs", str(EPOCHS), "--device", DEVICE,
            "--early-stopping-patience", str(EARLY_STOPPING_PATIENCE),
            "--threads", str(CPU_THREADS), "--stage", STAGE]
    if SCHEDULER_EPOCHS is not None:
        argv.extend(["--scheduler-epochs", str(SCHEDULER_EPOCHS)])
    if RESUME:
        argv.append("--resume")
    if CHECK_ONLY:
        argv.append("--check")
    return argv


def main():
    from compare import main as run
    run(build_argv())


if __name__ == "__main__":
    main()
