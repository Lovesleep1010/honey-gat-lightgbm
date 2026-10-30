"""Reuse the accepted main models and run matched auxiliary experiments."""

from pathlib import Path
from types import SimpleNamespace
import contextlib
import shutil
import sys
import time
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch_geometric.loader import DataLoader

import pipeline as shared


SEEDS = [2026, 2027, 2028, 2029, 2030]
MAIN_FILES = ["encoder.pt", "encoder_cache.npz", "lightgbm_train_only.txt",
              "lightgbm_final.txt", "validation_calibration.npz",
              "test_predictions.npz", "test_metrics.json"]


def main_folder(path):
    path = Path(path).expanduser().resolve()
    if path.name == "tasks":
        path = path.parent
    return path


def has_main_tasks(path):
    return all((path / "tasks" / f"full_k12_seed{seed}" / "encoder.pt").is_file()
               for seed in SEEDS)


def find_source(project, requested):
    if requested:
        path = Path(requested).expanduser()
        return main_folder(path if path.is_absolute() else project / path)
    candidates = set()
    for pattern in ["results*", "experiments/main/results*", "main/results*",
                    "*/experiments/main/results*", "*/main/results*"]:
        for parent in [project, project.parent]:
            for path in parent.glob(pattern):
                if path.is_dir() and path.name != "results_revision" and has_main_tasks(path):
                    candidates.add(path.resolve())
    if len(candidates) == 1:
        return candidates.pop()
    try:
        from tkinter import Tk, filedialog
        window = Tk()
        window.withdraw()
        window.attributes("-topmost", True)
        try:
            selected = filedialog.askdirectory(
                title="Select the completed 2026-2030 main-model results folder (contains tasks)")
        finally:
            window.destroy()
        if selected:
            return main_folder(selected)
    except ImportError:
        pass
    raise RuntimeError("Set MAIN_RESULTS_FOLDER in run_revision.py to the completed main-model results folder.")


def read_selection(folder):
    for name in ["selection.json", "locked.json"]:
        if (folder / name).is_file():
            return shared.read_json(folder / name)
    raise FileNotFoundError(f"Missing selection.json in {folder}")


def validate_main(source, tv, reference):
    validated = {}
    expected = reference.set_index("seed")
    for seed in SEEDS:
        task = f"full_k12_seed{seed}"
        folder = source / "tasks" / task
        missing = [name for name in MAIN_FILES if not (folder / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{folder}: missing {', '.join(missing)}. Select the complete results folder.")
        info = read_selection(folder)
        state = torch.load(folder / "encoder.pt", map_location="cpu", weights_only=True)
        if tuple(state["enc.weight"].shape) != (64, 3):
            raise ValueError(f"{task}: the saved encoder does not have three input features")
        for key, value in [("seed", seed), ("k", 12), ("n_bands", 128),
                           ("epochs", 50), ("early_stopping_patience", 5)]:
            if info.get(key) != value:
                raise ValueError(f"{task}: unexpected {key}={info.get(key)}; expected {value}")
        if info.get("features") != "embedding_global" or info.get("bands") != list(range(128)):
            raise ValueError(f"{task}: expected the full-spectrum embedding + global-statistics model")
        with np.load(folder / "encoder_cache.npz", allow_pickle=False) as cache:
            for key, value in [("y_train", tv.train.y), ("y_val", tv.val.y),
                               ("train_rows", tv.train.rows), ("val_rows", tv.val.rows)]:
                if not np.array_equal(cache[key], value):
                    raise ValueError(f"{task}: cached {key} differs from the included dataset split")
            if cache["train_embedding"].shape != (len(tv.train.y), 128):
                raise ValueError(f"{task}: unexpected embedding dimensions")
            if cache["train_global"].shape != (len(tv.train.y), 6):
                raise ValueError(f"{task}: expected six raw-spectrum statistics")
        saved_metrics = shared.read_json(folder / "test_metrics.json")["metrics"]
        with np.load(folder / "test_predictions.npz", allow_pickle=False) as prediction:
            scores = shared.metrics(prediction["y_true"], prediction["predictions"])
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "recall0"]:
            if not np.isclose(scores[metric], expected.loc[seed, metric], rtol=0, atol=1e-10):
                raise ValueError(f"{task}: {metric} does not match the supplied Table 1 run")
            if not np.isclose(saved_metrics[metric], scores[metric], rtol=0, atol=1e-10):
                raise ValueError(f"{task}: saved metrics differ from the saved predictions")
        for key in ["selected_epoch", "gbm_best_iteration", "tau0"]:
            if not np.isclose(info[key], expected.loc[seed, key], rtol=0, atol=1e-10):
                raise ValueError(f"{task}: saved {key} does not match the supplied main-model run")
        validated[task] = dict(info, node_features=3)
    return validated


def import_main(source, output, validated):
    for task, info in validated.items():
        source_folder = source / "tasks" / task
        destination = output / "tasks" / task
        destination.mkdir(parents=True, exist_ok=True)
        for name in MAIN_FILES + ["epoch_history.json"]:
            path = source_folder / name
            if path.exists():
                shutil.copy2(path, destination / name)
        shared.save_task(destination, info)
        print(f"[Reuse main model] {task}", flush=True)


def test_attribution(output, cfg, data_file):
    directory = output / "attribution"
    directory.mkdir(exist_ok=True)
    result = directory / "acq6_importance.csv"
    if result.exists():
        return result
    df, columns, wavelengths = shared.load_frame(data_file)
    test = shared.get_split(df, columns, [6])
    task = f"full_k12_seed{cfg['ranking_seed']}"
    info = shared.load_task(shared.task_folder(output, task))
    device = torch.device(cfg["device"])
    encoder, edges, mean, std = shared.load_model_encoder(output, info, device)
    graphs = shared.make_graphs(test.raw, test.y, np.arange(128), edges, wavelengths)
    loader = DataLoader(graphs, batch_size=256, shuffle=False)
    global_features, _, _ = shared.global_stats(test.raw, mean, std)
    booster = lgb.Booster(model_file=str(shared.task_folder(output, task) / "lightgbm_final.txt"))

    def graph_ba():
        embedding, labels = shared.core.extract_embeddings(encoder, loader, device)
        if not np.array_equal(labels, test.y):
            raise ValueError("Unexpected test sample order")
        probabilities = booster.predict(np.hstack([embedding, global_features]), num_threads=cfg["threads"])
        return shared.metrics(test.y, shared.core.apply_tau0(probabilities, info["tau0"]))["balanced_accuracy"]

    baseline = graph_ba()
    gat_scores = []
    for band in range(128):
        original = [graph.x[band].clone() for graph in graphs]
        try:
            for graph in graphs:
                graph.x[band] = 0
            gat_scores.append(baseline - graph_ba())
        finally:
            for graph, values in zip(graphs, original):
                graph.x[band] = values
        if (band + 1) % 8 == 0:
            print(f"[Acq6 GAT attribution] {band + 1}/128", flush=True)
    raw_info = shared.load_task(shared.task_folder(output, "raw_full128"))
    raw_booster = lgb.Booster(model_file=str(shared.task_folder(output, "raw_full128") / "lightgbm_final.txt"))
    x = shared.core.snv(test.raw)
    raw_baseline = shared.metrics(test.y, shared.core.apply_tau0(
        raw_booster.predict(x, num_threads=cfg["threads"]), raw_info["tau0"]))["balanced_accuracy"]
    rng = np.random.default_rng(cfg["ranking_seed"])
    raw_scores = np.empty((128, cfg["permutation_repeats"]), dtype=np.float64)
    for band in range(128):
        for repeat in range(cfg["permutation_repeats"]):
            perturbed = x.copy()
            perturbed[:, band] = rng.permutation(perturbed[:, band])
            predictions = shared.core.apply_tau0(
                raw_booster.predict(perturbed, num_threads=cfg["threads"]), raw_info["tau0"])
            raw_scores[band, repeat] = raw_baseline - shared.metrics(test.y, predictions)["balanced_accuracy"]
        if (band + 1) % 16 == 0:
            print(f"[Acq6 LightGBM attribution] {band + 1}/128", flush=True)
    pd.DataFrame({"band_index": np.arange(128), "wavelength_nm": wavelengths,
                  "gat_node_occlusion_delta_ba": gat_scores,
                  "raw_snv_permutation_delta_ba_mean": raw_scores.mean(axis=1),
                  "raw_snv_permutation_delta_ba_sd": raw_scores.std(axis=1, ddof=0),
                  "gat_baseline_balanced_accuracy": baseline,
                  "raw_baseline_balanced_accuracy": raw_baseline,
                  "encoder_seed": cfg["ranking_seed"], "node_features": 3}).to_csv(
                      result, index=False, encoding="utf-8-sig")
    return result


def class_summary(output):
    matrices, class_metrics = [], []
    reference = None
    for seed in SEEDS:
        folder = shared.task_folder(output, f"full_k12_seed{seed}")
        with np.load(folder / "test_predictions.npz", allow_pickle=False) as saved:
            true, pred, rows = saved["y_true"], saved["predictions"], saved["row_indices"]
            if reference is None:
                reference = (true.copy(), rows.copy())
            if not np.array_equal(true, reference[0]) or not np.array_equal(rows, reference[1]):
                raise ValueError("The five main runs contain different test samples")
            cm = confusion_matrix(true, pred, labels=range(5))
            matrices.append(cm / cm.sum(axis=1, keepdims=True) * 100)
            class_metrics.append(np.array(precision_recall_fscore_support(
                true, pred, labels=range(5), zero_division=0)[:3]))
    matrices, class_metrics = np.array(matrices), np.array(class_metrics)
    for label, values in [("mean", matrices.mean(axis=0)), ("sd", matrices.std(axis=0, ddof=1))]:
        pd.DataFrame(values, index=shared.CLASSES, columns=shared.CLASSES).rename_axis(
            "true_adulteration_percent").to_csv(output / f"main_confusion_{label}_percent.csv", encoding="utf-8-sig")
    rows = []
    for column, level in enumerate(shared.CLASSES):
        row = {"adulteration_percent": level}
        for index, metric in enumerate(["precision", "recall", "f1"]):
            row[metric + "_mean"] = class_metrics[:, index, column].mean()
            row[metric + "_sd"] = class_metrics[:, index, column].std(ddof=1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "main_class_metrics.csv", index=False, encoding="utf-8-sig")


def export_results(output, reference_file):
    selection = shared.load_task(output / "band_selection")
    rows = []
    for family in ["selected_bands", "raw_ranked_bands"]:
        for count, bands in selection[family].items():
            for band, wavelength in zip(bands, selection[
                    "selected_wavelengths" if family == "selected_bands" else "raw_ranked_wavelengths"][count]):
                rows.append({"selection": family, "n_bands": int(count), "band_index": band,
                             "wavelength_nm": wavelength, "region": "Vis" if wavelength < 700 else "NIR"})
    pd.DataFrame(rows).to_csv(output / "selected_bands.csv", index=False, encoding="utf-8-sig")
    class_summary(output)
    archive = output / "results_for_revision.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.glob("plan_*_*.csv")) + sorted(output.glob("main_*.csv")) + [output / "selected_bands.csv"]:
            bundle.write(path, path.name)
        bundle.write(reference_file, "table1_main_reference.csv")
        bundle.write(output / "band_selection/acq5_importance.csv", "acq5_importance.csv")
        bundle.write(output / "attribution/acq6_importance.csv", "acq6_importance.csv")
        for seed in SEEDS:
            path = shared.task_folder(output, f"full_k12_seed{seed}") / "test_predictions.npz"
            bundle.write(path, f"predictions/full_k12_seed{seed}.npz")
    return archive


def run(project, args):
    data_file = project / "data/adulteration_dataset_26_08_2021.csv"
    output = Path(args.output).expanduser() if args.output else project / "results_revision"
    if not output.is_absolute():
        output = project / output
    tv = shared.load_trainval(data_file)
    if args.check:
        print(f"Train={len(tv.train.y)}, Val={len(tv.val.y)}, spectral channels=128, node features=3")
        print(f"Seeds={SEEDS}; maximum epochs=50; early-stopping patience=5")
        print("Reuse 5 main models; train 20 k-sensitivity and 15 reduced-band encoders.")
        print("CHECK ONLY: no training, no test evaluation, no output folder created.")
        return
    source = find_source(project, args.source)
    if source == output.resolve():
        raise ValueError("Use a different output folder from the completed main-model results")
    reference_file = project / "data/table1_main_reference.csv"
    validated = validate_main(source, tv, pd.read_csv(reference_file))
    settings = SimpleNamespace(data=data_file, device=args.device, epochs=50, scheduler_epochs=50,
                               early_stopping_patience=5, seeds=SEEDS,
                               k_values=[8, 10, 12, 14, 16], band_counts=[16, 32, 48],
                               sparse_k=8, ranking_seed=2026, permutation_repeats=5,
                               threads=args.threads, resume=True)
    cfg = shared.configure_run(settings, output)
    started = time.perf_counter()
    with (output / "run.log").open("a", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(shared.Tee(sys.stdout, log)), contextlib.redirect_stderr(shared.Tee(sys.stderr, log)):
            print(f"[Revision] source={source}; output={output}; device={cfg['device']}", flush=True)
            import_main(source, output, validated)
            plan_path = shared.fit_requested(output, tv, cfg, ["ablation", "k", "bands"])
            shared.evaluate_requested(output, cfg, plan_path, data_file)
            test_attribution(output, cfg, data_file)
            archive = export_results(output, reference_file)
            print(f"[Complete] {time.perf_counter() - started:.1f} seconds", flush=True)
            print(f"Return this file: {archive}", flush=True)
