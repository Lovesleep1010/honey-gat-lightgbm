"""在 PyCharm 中打开这个文件，右键 Run / 点击绿色运行按钮。

先把项目解释器设为原来能运行 PyTorch / PyG / LightGBM 的环境。
配置都在下面；无需填写运行参数，无需指定工作目录。
"""
from pathlib import Path

# ==================== 在这里改配置 ====================
EXPERIMENTS = ["baseline"]  # 默认全谱主模型；全部补实验改为 ["all"]
EPOCHS = 50                # 最大训练轮数
EARLY_STOPPING_PATIENCE = 5 # Acq5下游Bal-Acc连续5轮未提高则停止
SCHEDULER_EPOCHS = 50      # 保持原来的余弦调度周期
SEEDS = [42, 123, 2024, 3407, 9999]
K_VALUES = [8, 10, 12, 14, 16]
BAND_COUNTS = [16, 32, 48]  # 预先固定全部报告；含48时自动补Vis/NIR和McNemar
SPARSE_K = 8
RANKING_SEED = 42
PERMUTATION_REPEATS = 5
DEVICE = "auto"          # 有 CUDA 用 GPU；也可填 "cuda" 或 "cpu"
CPU_THREADS = 4
STAGE = "all"            # 全流程；或 "fit" / "evaluate"
RESUME = True            # 跳过已完成任务；未完成任务从头运行
CHECK_ONLY = False       # 改为 True 只检查环境和数据，不训练

# 留空时自动读取仓库 data/ 下的 CSV；也可填写自己的路径。
DATA_FILE = ""
# 留空自动用 results_max50_patience5_c4filtered 等新目录。
# 改种子/k/波段数量/设备等配置后，请换一个新结果目录。
OUTPUT_FOLDER = ""
# ======================================================

HERE = Path(__file__).resolve().parent


def build_argv():
    data = Path(DATA_FILE) if DATA_FILE else HERE.parents[1] / "data" / "adulteration_dataset_26_08_2021.csv"
    output = Path(OUTPUT_FOLDER) if OUTPUT_FOLDER else HERE / f"results_max{EPOCHS}_patience{EARLY_STOPPING_PATIENCE}_c4filtered"
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
