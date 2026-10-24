"""Windows / PyCharm: select the existing experiment interpreter, then click Run.

All paths default to this file's folder; no working-directory setting is needed.
"""
from pathlib import Path

# ==================== 在这里改配置 ====================
MODELS = ["graphsage", "gcn", "gin", "gat"]  # gat 是普通 GAT，无残差连接
SEEDS = [42, 123, 2024, 3407, 9999]
EPOCHS = 5                  # 沿用本次短训练预算：每个种子训练5轮
SCHEDULER_EPOCHS = None     # None保留原配置：SAGE/GCN=60，GIN/GAT=50
DEVICE = "auto"            # 有 CUDA 用 GPU；否则 CPU；可填 "cuda" / "cpu"
CPU_THREADS = 4
STAGE = "all"              # "fit" 只训练并锁定；"evaluate" 只评估已锁定模型
RESUME = True              # 完成的任务跳过；未完成的任务从头运行
CHECK_ONLY = False         # True 只检查环境、数据和任务数量

DATA_FILE = ""             # 留空时自动读取仓库 data/ 下的 CSV
OUTPUT_FOLDER = ""         # 留空用 results_table1_gnn_original_5epochs
# 改模型、种子、轮数、设备、线程或代码后，请使用新结果目录。
# ======================================================

HERE = Path(__file__).resolve().parent


def build_argv():
    data = Path(DATA_FILE) if DATA_FILE else HERE.parents[1] / "data" / "adulteration_dataset_26_08_2021.csv"
    output = Path(OUTPUT_FOLDER) if OUTPUT_FOLDER else HERE / f"results_table1_gnn_original_{EPOCHS}epochs"
    if not data.is_absolute():
        data = HERE / data
    if not output.is_absolute():
        output = HERE / output
    argv = ["--data", str(data), "--output", str(output), "--models", *MODELS,
            "--seeds", *map(str, SEEDS), "--epochs", str(EPOCHS), "--device", DEVICE,
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
