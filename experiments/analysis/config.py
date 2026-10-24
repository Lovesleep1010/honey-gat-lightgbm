
from pathlib import Path


EXPERIMENTS = ["all"]     #  ["ablation"] / ["k"] / ["bands"]
EPOCHS = 50                
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

# 留空时自动读取仓库 data/ 下的 CSV；也可填写自己的路径。
DATA_FILE = ""

OUTPUT_FOLDER = ""
# ======================================================

HERE = Path(__file__).resolve().parent


def build_argv():
    data = Path(DATA_FILE) if DATA_FILE else HERE.parents[1] / "data" / "adulteration_dataset_26_08_2021.csv"
    output = Path(OUTPUT_FOLDER) if OUTPUT_FOLDER else HERE / f"results_{EPOCHS}epochs_c4filtered_multiband"
    if not data.is_absolute():
        data = HERE / data
    if not output.is_absolute():
        output = HERE / output
    args = [
        "--data", str(data), "--output", str(output),
        "--experiments", *EXPERIMENTS,
        "--epochs", str(EPOCHS), "--scheduler-epochs", str(SCHEDULER_EPOCHS),
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
