"""Matched graph-structure experiments using the accepted main-model pipeline."""

from pathlib import Path
import contextlib
import sys
import time
import traceback
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

import pipeline as shared
from graph_structures import GRAPH_LABELS, GRAPH_ORDER, build_edge_index


METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "recall0"]
CONTROLS = ["self_only", "adjacent_only"]


def task_id(graph_mode, seed):
    return f"full_k12_seed{seed}" if graph_mode == "full" else f"{graph_mode}_seed{seed}"


def settings(args):
    if not args.seeds or len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0:
        raise ValueError("Seeds must be nonnegative and unique")
    if not 1 <= args.epochs <= 50 or args.patience < 1 or args.threads < 1:
        raise ValueError("Require 1..50 maximum epochs, positive patience and positive CPU threads")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment; select cpu or use your CUDA PyTorch environment")
    return dict(epochs=args.epochs, scheduler_epochs=50, early_stopping_patience=args.patience,
                seeds=list(args.seeds), gbm_seed=42, device=device, threads=args.threads,
                node_features=3, n_bands=128,
                full_graph_k=12, graph_modes=GRAPH_ORDER,
                train_acquisitions=[1, 2, 3, 4], validation_acquisitions=[5], test_acquisitions=[6],
                data_file=str(args.data))


def configure_output(output, cfg):
    path = output / "settings.json"
    if path.exists():
        if shared.read_json(path) != cfg:
            raise RuntimeError("This results folder uses different settings; choose a new OUTPUT_FOLDER")
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("The output folder is not empty; choose a new OUTPUT_FOLDER")
        output.mkdir(parents=True, exist_ok=True)
        shared.write_json(path, cfg)


def result_row(info, graph_mode, origin, scores):
    fields = ["task_id", "seed", "selected_epoch", "epochs", "epochs_completed",
              "early_stopping_patience", "gbm_best_iteration", "tau0"]
    row = {field: info[field] for field in fields}
    row.update(graph_mode=graph_mode, graph=GRAPH_LABELS[graph_mode], source=origin,
               node_features=3, n_bands=128, k=12 if graph_mode == "full" else 0)
    row.update({metric: float(scores[metric]) for metric in METRICS})
    return row


def evaluate_task(output, data, wavelengths, cfg, graph_mode, seed):
    task = task_id(graph_mode, seed)
    folder = shared.task_folder(output, task)
    info = shared.load_task(folder)
    if info.get("graph_mode") != graph_mode or info.get("node_features") != 3:
        raise ValueError(f"{task}: incompatible saved model")
    prediction_file = folder / "test_predictions.npz"
    if prediction_file.exists():
        with np.load(prediction_file, allow_pickle=False) as saved:
            if not np.array_equal(saved["y_true"], data.y) or not np.array_equal(saved["row_indices"], data.rows):
                raise ValueError(f"{task}: saved test samples differ from this dataset")
            scores = shared.metrics(data.y, saved["predictions"])
    else:
        encoder, edges, mean, std = shared.load_model_encoder(output, info, torch.device(cfg["device"]))
        graphs = shared.make_graphs(data.raw, data.y, np.arange(128), edges, wavelengths)
        embedding, labels = shared.core.extract_embeddings(
            encoder, DataLoader(graphs, batch_size=256, shuffle=False), torch.device(cfg["device"]))
        if not np.array_equal(labels, data.y):
            raise ValueError(f"{task}: unexpected test sample order")
        global_features, _, _ = shared.global_stats(data.raw, mean, std)
        booster = lgb.Booster(model_file=str(folder / "lightgbm_final.txt"))
        probabilities = booster.predict(np.hstack([embedding, global_features]), num_threads=cfg["threads"])
        predictions = shared.core.apply_tau0(probabilities, info["tau0"])
        np.savez_compressed(prediction_file, y_true=data.y, predictions=predictions,
                            probabilities=probabilities, row_indices=data.rows)
        scores = shared.metrics(data.y, predictions)
        del encoder, graphs, embedding
        if cfg["device"] == "cuda":
            torch.cuda.empty_cache()
    shared.write_json(folder / "test_metrics.json", dict(task_id=task, acquisition=6, metrics=scores))
    return result_row(info, graph_mode, "new_training", scores), prediction_file


def export_results(output, rows, prediction_files):
    frame = pd.DataFrame(rows)
    order = {name: i for i, name in enumerate(GRAPH_ORDER)}
    frame["_order"] = frame.graph_mode.map(order)
    frame = frame.sort_values(["_order", "seed"]).drop(columns="_order").reset_index(drop=True)
    if frame.duplicated(["graph_mode", "seed"]).any():
        raise ValueError("Duplicate graph/seed results")
    expected_seeds = set(frame.loc[frame.graph_mode.eq("full"), "seed"])
    summary = []
    paper_rows = []
    for graph_mode in GRAPH_ORDER:
        group = frame.loc[frame.graph_mode.eq(graph_mode)]
        if set(group.seed) != expected_seeds or not len(group):
            raise ValueError("All three graph structures must use the same seeds")
        row = dict(graph_mode=graph_mode, graph=GRAPH_LABELS[graph_mode], n_runs=len(group))
        for metric in METRICS:
            row[metric + "_mean"] = float(group[metric].mean())
            row[metric + "_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan
        summary.append(row)
        cells = {label: f"{row[metric + '_mean']:.4f} ± {row[metric + '_sd']:.4f}" if len(group) > 1
                 else f"{row[metric + '_mean']:.4f}"
                 for metric, label in [("accuracy", "Accuracy"), ("balanced_accuracy", "Bal-Acc"), ("macro_f1", "Macro-F1")]}
        paper_rows.append(dict(Graph=GRAPH_LABELS[graph_mode], n=len(group), **cells))
    indexed = frame.set_index(["graph_mode", "seed"])
    differences = []
    for left, right in [("full", "self_only"), ("full", "adjacent_only"), ("adjacent_only", "self_only")]:
        for seed in sorted(expected_seeds):
            a, b = indexed.loc[(left, seed)], indexed.loc[(right, seed)]
            row = dict(comparison=f"{left} minus {right}", seed=int(seed))
            for metric in METRICS:
                row["delta_" + metric] = float(a[metric] - b[metric])
            row["delta_balanced_accuracy_pp"] = row["delta_balanced_accuracy"] * 100
            differences.append(row)
    files = {
        "graph_ablation_metrics.csv": frame,
        "graph_ablation_summary.csv": pd.DataFrame(summary),
        "graph_ablation_paired_deltas.csv": pd.DataFrame(differences),
        "graph_ablation_for_paper.csv": pd.DataFrame(paper_rows),
    }
    for name, table in files.items():
        table.to_csv(output / name, index=False, encoding="utf-8-sig")
    archive = output / "graph_ablation_results.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in files:
            bundle.write(output / name, name)
        for task, path in sorted(prediction_files.items()):
            bundle.write(path, f"predictions/{task}.npz")
    print(files["graph_ablation_for_paper.csv"].to_string(index=False), flush=True)
    return archive


def run(project, args):
    cfg = settings(args)
    data_file = args.data
    tv = shared.load_trainval(data_file)
    normalized = shared.core.snv(tv.train.raw)
    print(f"Train={len(tv.train.y)}, Acq5={len(tv.val.y)}, channels=128, node features=3")
    print(f"Seeds={cfg['seeds']}; maximum epochs={cfg['epochs']}; patience={cfg['early_stopping_patience']}")
    for graph_mode in GRAPH_ORDER:
        edges = build_edge_index(normalized, graph_mode)
        print(f"{GRAPH_LABELS[graph_mode]}: {edges.shape[1]} directed edges including self-loops")
    workload = len(cfg["seeds"]) * 3
    print(f"Device={cfg['device']}; encoders to train={workload}")
    if args.check:
        print("CHECK ONLY: no training, no test evaluation, no output folder created.")
        return None
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = project / output
    output = output.resolve()
    configure_output(output, cfg)
    torch.set_num_threads(cfg["threads"])
    torch.set_num_interop_threads(min(cfg["threads"], 2))
    modes = CONTROLS + ["full"]
    start = time.perf_counter()
    with (output / "run.log").open("a", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(shared.Tee(sys.stdout, log)), contextlib.redirect_stderr(shared.Tee(sys.stderr, log)):
            try:
                shared.write_json(output / "status.json", {"status": "training"})
                for seed in cfg["seeds"]:
                    for graph_mode in modes:
                        task = task_id(graph_mode, seed)
                        print(f"\n[Train] {GRAPH_LABELS[graph_mode]}, seed={seed}", flush=True)
                        shared.fit_encoder(output, tv, cfg, seed, 12, np.arange(128), task, graph_mode=graph_mode)
                # Acq6 is evaluated after all encoder and LightGBM selections are finished.
                shared.write_json(output / "status.json", {"status": "evaluating"})
                frame, columns, wavelengths = shared.load_frame(data_file)
                test = shared.get_split(frame, columns, [6])
                rows, predictions = [], {}
                for graph_mode in GRAPH_ORDER:
                    for seed in cfg["seeds"]:
                        row, path = evaluate_task(output, test, wavelengths, cfg, graph_mode, seed)
                        rows.append(row)
                        predictions[task_id(graph_mode, seed)] = path
                        print(f"[Test] {graph_mode}, seed={seed}: Bal-Acc={row['balanced_accuracy']:.6f}, Macro-F1={row['macro_f1']:.6f}", flush=True)
                archive = export_results(output, rows, predictions)
                shared.write_json(output / "status.json", dict(status="complete", elapsed_seconds=time.perf_counter()-start))
                print(f"\n[Complete] Return this file: {archive}", flush=True)
                return archive
            except BaseException:
                shared.write_json(output / "status.json", {"status": "failed", "error": traceback.format_exc()})
                raise
