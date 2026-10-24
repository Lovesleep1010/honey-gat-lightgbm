from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import traceback

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1] / "main"))

import model as core
import pipeline as shared
import training as training
import baseline_config as protocol
from models import MODEL_LABELS, MODEL_ORDER, build_encoder, model_spec

HERE = Path(__file__).resolve().parent

DEFAULT_SEEDS = [42, 123, 2024, 3407, 9999]
PLAN_FILE = "plan_table1_gnn.json"
METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1", "recall0")


def task_id(architecture, seed):
    return f"{architecture}_bands128_k12_seed{seed}"


def planned_tasks(cfg):
    return [(architecture, seed) for architecture in cfg["models"] for seed in cfg["seeds"]]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=HERE.parents[1] / "data" / "adulteration_dataset_26_08_2021.csv")
    p.add_argument("--output", type=Path)
    p.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS.copy())
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument("--scheduler-epochs", type=int, default=None,
                   help="Override original cosine periods (SAGE/GCN=60; GIN/GAT=50)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--stage", choices=("all", "fit", "evaluate"), default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check", action="store_true", help="Check data/environment; do not train or create results")
    return p


def resolve_arguments(argv=None):
    args = parser().parse_args(argv)
    periods = [protocol.settings(name, args.scheduler_epochs)["scheduler_epochs"] for name in args.models]
    if args.epochs < 1 or any(args.epochs > period for period in periods):
        raise ValueError("Require 1 <= epochs <= each selected model's scheduler period")
    if args.early_stopping_patience < 1:
        raise ValueError("early-stopping-patience must be positive")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        raise ValueError("Seeds must be unique integers in 0..2**32-1")
    if len(set(args.models)) != len(args.models):
        raise ValueError("Models must be unique")
    args.models = [name for name in MODEL_ORDER if name in args.models]
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select CPU or the existing CUDA interpreter")
    if not args.data.is_absolute():
        args.data = HERE / args.data
    args.data = args.data.resolve()
    if args.output is None:
        args.output = HERE / f"results_max{args.epochs}_patience{args.early_stopping_patience}"
    elif not args.output.is_absolute():
        args.output = HERE / args.output
    args.output = args.output.resolve()
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    return args


def make_config(args):
    return dict(data_file=str(args.data), models=args.models,
                model_specs={name: model_spec(name) for name in args.models},
                comparator_settings={name: protocol.settings(name, args.scheduler_epochs) for name in args.models},
                seeds=args.seeds, gbm_seed=42, epochs=args.epochs,
                scheduler_epochs=args.scheduler_epochs, early_stopping_patience=args.early_stopping_patience,
                k=12, n_bands=128, device=args.device, threads=args.threads)


def configure_run(args, cfg):
    root = args.output
    path = root / "config.json"
    if path.exists():
        if not args.resume or shared.read_json(path) != cfg:
            raise RuntimeError("This output folder contains another run. Use a new OUTPUT_FOLDER.")
    else:
        if root.exists() and any(root.iterdir()):
            raise RuntimeError("The output folder is not empty. Use a new OUTPUT_FOLDER.")
        root.mkdir(parents=True, exist_ok=True)
        shared.write_json(path, cfg)


@dataclass
class Prepared:
    train_graphs: dict
    val_graphs: dict
    edges: torch.Tensor


def prepare_trainval(tv, cfg):
    normalized = core.snv(tv.train.raw)
    if not np.isfinite(np.corrcoef(normalized.T)).all():
        raise ValueError("Training graph correlation contains NaN; check constant channels")
    edges = core.build_hybrid_edge_index(normalized, top_k=cfg["k"])
    train_graphs, val_graphs = {}, {}
    for specification in cfg["comparator_settings"].values():
        window = specification["sg_window"]
        if window not in train_graphs:
            train_graphs[window] = protocol.make_graphs(tv.train.raw, tv.train.y, edges, specification)
            val_graphs[window] = protocol.make_graphs(tv.val.raw, tv.val.y, edges, specification)
    return Prepared(train_graphs, val_graphs, edges)


def read_task(root, cfg, architecture, seed):
    return shared.load_task(root / "tasks" / task_id(architecture, seed))


def fit_one(root, tv, prepared, cfg, architecture, seed):
    name = task_id(architecture, seed)
    folder = root / "tasks" / name
    if (folder / "selection.json").exists():
        print(f"[Reuse fit] {name}", flush=True)
        return read_task(root, cfg, architecture, seed)
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    shared.write_json(folder / "status.json", {"status": "training", "task_id": name})
    core.seed_everything(seed)
    device = torch.device(cfg["device"])
    specification = cfg["comparator_settings"][architecture]
    weights = core.compute_class_weights(tv.train.y, boost0=specification["boost0"], mode="effective")
    window = specification["sg_window"]
    train_graphs, val_graphs = prepared.train_graphs[window], prepared.val_graphs[window]
    train_loader = DataLoader(train_graphs, batch_size=128, shuffle=True, num_workers=0)
    embed_loader = DataLoader(train_graphs, batch_size=256, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_graphs, batch_size=256, shuffle=False, num_workers=0)
    model = build_encoder(architecture).to(device)
    history = []

    def log(record):
        history.append(dict(record, elapsed_seconds=time.perf_counter() - start))
        shared.write_json(folder / "epoch_history.json", history)
        shared.write_json(folder / "status.json", dict(
            status="training", task_id=name, epoch=record["epoch"], epochs=cfg["epochs"]))
        print(f"[Epoch] {name} {record['epoch']}/{cfg['epochs']} loss={record['loss']:.5f}", flush=True)

    print(f"\n[Fit] {MODEL_LABELS[architecture]}, seed={seed}", flush=True)
    checkpoint = folder / "encoder.pt"
    ep, probe_ba = training.train_encoder_downstream_selection(
        model, train_loader, embed_loader, val_loader, device,
        torch.tensor(weights, dtype=torch.float32, device=device),
        tv.train.y, tv.val.y, specification,
        epochs=specification["scheduler_epochs"], train_epochs=cfg["epochs"],
        ckpt_path=str(checkpoint), check_interval=1, n_jobs=cfg["threads"],
        gbm_seed=cfg["gbm_seed"], log_callback=log,
        early_stopping_patience=cfg["early_stopping_patience"])
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    etr, ytr = core.extract_embeddings(model, embed_loader, device)
    ev, yval = core.extract_embeddings(model, val_loader, device)
    if not (np.array_equal(ytr, tv.train.y) and np.array_equal(yval, tv.val.y)):
        raise RuntimeError("Embedding rows were reordered")
    if etr.shape[1] != cfg["model_specs"][architecture]["embedding_dim"]:
        raise RuntimeError("Unexpected embedding width")
    np.savez_compressed(folder / "encoder_cache.npz", train_embedding=etr, val_embedding=ev,
                        y_train=ytr, y_val=yval, edges=prepared.edges.numpy(),
                        train_rows=tv.train.rows, val_rows=tv.val.rows)
    info = dict(task_id=name, architecture=architecture, label=MODEL_LABELS[architecture], seed=seed,
                model_spec=cfg["model_specs"][architecture],
                parameter_count=sum(p.numel() for p in model.parameters()),
                embedding_dim=int(etr.shape[1]), downstream_input_dim=int(etr.shape[1]),
                features="embedding_only", node_features=3, sg_window=window,
                comparator_settings=specification, bands=list(range(128)), k=12, n_bands=128,
                selected_epoch=int(ep), selected_probe_val_ba=float(probe_ba),
                epochs=cfg["epochs"], scheduler_epochs=specification["scheduler_epochs"],
                epochs_completed=int(history[-1]["epoch"]),
                early_stopping_patience=cfg["early_stopping_patience"],
                early_stopping_triggered=bool(history[-1]["early_stopping_triggered"]),
                encoder_fit_acquisitions=[1, 2, 3, 4], selection_acquisition=5,
                test_used_for_selection=False)
    info = protocol.fit_gbm(folder, etr, ytr, ev, yval, weights, cfg, info)
    info["elapsed_fit_seconds"] = time.perf_counter() - start
    info = shared.save_task(folder, info)
    shared.write_json(folder / "status.json", {"status": "trained", "task_id": name})
    del model
    if cfg["device"] == "cuda":
        torch.cuda.empty_cache()
    return info


def fit_all(root, cfg, data_path):
    tv = shared.load_trainval(data_path)
    print(f"[Split] Train={len(tv.train.y)}, Val={len(tv.val.y)}, bands=128", flush=True)
    prepared = prepare_trainval(tv, cfg)
    tasks = planned_tasks(cfg)
    for number, (architecture, seed) in enumerate(tasks, 1):
        print(f"[Task {number}/{len(tasks)}] {task_id(architecture, seed)}", flush=True)
        fit_one(root, tv, prepared, cfg, architecture, seed)
    plan = dict(tasks=[task_id(a, seed) for a, seed in tasks])
    shared.write_json(root / PLAN_FILE, plan)
    return plan


def load_plan(root, cfg):
    plan = shared.read_json(root / PLAN_FILE)
    names = [task_id(architecture, seed) for architecture, seed in planned_tasks(cfg)]
    if plan["tasks"] != names:
        raise RuntimeError("The experiment plan differs from the selected models/seeds.")
    return [shared.load_task(root / "tasks" / name) for name in names]


def evaluate_one(root, cfg, info, test, wavelengths):
    folder = root / "tasks" / info["task_id"]
    if (folder / "test_metrics.json").exists():
        print(f"[Reuse evaluation] {info['task_id']}", flush=True)
        return shared.read_json(folder / "test_metrics.json")
    device = torch.device(cfg["device"])
    model = build_encoder(info["architecture"]).to(device)
    model.load_state_dict(torch.load(folder / "encoder.pt", map_location=device, weights_only=True))
    with np.load(folder / "encoder_cache.npz", allow_pickle=False) as saved:
        edges = torch.from_numpy(saved["edges"].copy())
    graphs = protocol.make_graphs(test.raw, test.y, edges, info["comparator_settings"])
    emb, labels = core.extract_embeddings(model, DataLoader(
        graphs, batch_size=256, shuffle=False, num_workers=0), device)
    if not np.array_equal(labels, test.y):
        raise RuntimeError("Test embedding rows were reordered")
    x = emb
    booster = lgb.Booster(model_file=str(folder / "lightgbm_final.txt"))
    if x.shape[1] != info["downstream_input_dim"] or booster.num_feature() != x.shape[1]:
        raise RuntimeError("Encoder/LightGBM feature dimensions differ")
    probabilities = booster.predict(x, num_threads=cfg["threads"])
    if probabilities.shape != (len(test.y), 5) or not np.isfinite(probabilities).all():
        raise RuntimeError("Invalid test probabilities")
    predictions = protocol.predict_classes(probabilities, info)
    np.savez_compressed(folder / "test_predictions.npz", y_true=test.y, predictions=predictions,
                        probabilities=probabilities, row_indices=test.rows)
    evaluated = dict(task_id=info["task_id"], acquisition=6,
                     metrics=shared.metrics(test.y, predictions))
    shared.write_json(folder / "test_metrics.json", evaluated)
    del model, graphs
    if cfg["device"] == "cuda":
        torch.cuda.empty_cache()
    return evaluated


def summarize(root, cfg, rows):
    frame = pd.DataFrame(rows)
    expected = {task_id(a, s) for a, s in planned_tasks(cfg)}
    if len(frame) != len(expected) or set(frame.task_id) != expected:
        raise RuntimeError("Incomplete/duplicate runs; refusing to summarize a selected subset")
    summaries, paper = [], []
    for architecture in cfg["models"]:
        selected = frame[frame.architecture == architecture]
        if set(selected.seed) != set(cfg["seeds"]):
            raise RuntimeError(f"Missing seeds for {architecture}")
        row = dict(architecture=architecture, model=MODEL_LABELS[architecture], n_runs=len(selected),
                   parameter_count=int(selected.iloc[0].parameter_count),
                   embedding_dim=int(selected.iloc[0].embedding_dim))
        display = dict(model=MODEL_LABELS[architecture], n_runs=len(selected))
        for metric in METRIC_NAMES:
            mean = float(selected[metric].mean())
            sd = float(selected[metric].std(ddof=1)) if len(selected) > 1 else None
            row[metric + "_mean"], row[metric + "_sd"] = mean, sd
            display[metric + "_percent"] = (f"{100 * mean:.2f}% ± {100 * sd:.2f}%"
                                             if sd is not None else f"{100 * mean:.2f}% (single run)")
        summaries.append(row)
        paper.append(display)
    frame.to_csv(root / "table1_gnn_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(root / "table1_gnn_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paper).to_csv(root / "table1_gnn_for_paper.csv", index=False, encoding="utf-8-sig")


def evaluate_all(root, cfg, data_path):
    infos = load_plan(root, cfg)
    frame, columns, wavelengths = shared.load_frame(data_path)
    test = shared.get_split(frame, columns, [6])
    rows = []
    for info in infos:
        evaluated = evaluate_one(root, cfg, info, test, wavelengths)
        row = {key: info[key] for key in (
            "task_id", "architecture", "seed", "n_bands", "k", "selected_epoch",
            "selected_probe_val_ba", "gbm_best_iteration", "prediction_rule", "parameter_count",
            "node_features", "sg_window", "features", "epochs", "scheduler_epochs",
            "embedding_dim", "downstream_input_dim", "elapsed_fit_seconds",
            "epochs_completed", "early_stopping_patience", "early_stopping_triggered")}
        row.update({name: evaluated["metrics"][name] for name in METRIC_NAMES})
        rows.append(row)
        print(f"[Test] {info['task_id']}: BA={row['balanced_accuracy']:.6f}, "
              f"Macro-F1={row['macro_f1']:.6f}", flush=True)
    summarize(root, cfg, rows)


def check_only(args):
    frame, columns, _ = shared.load_frame(args.data)
    counts = {acq: len(shared.get_split(frame, columns, [acq]).y) for acq in range(1, 7)}
    print(f"Data filter: {shared.DATA_FILTER['rule']}; excluded={frame.attrs['excluded_c4_rows']}")
    print(f"Train={sum(counts[a] for a in [1, 2, 3, 4])}, Val={counts[5]}, Test={counts[6]}, bands=128")
    print(f"PyTorch={torch.__version__}, device={args.device}, CUDA available={torch.cuda.is_available()}")
    if args.device == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)}")
    for architecture in args.models:
        model = build_encoder(architecture)
        specification = protocol.settings(architecture, args.scheduler_epochs)
        print(f"{MODEL_LABELS[architecture]}: nodes=3, SG={specification['sg_window']}, "
              f"loss={specification['loss']}, scheduler={specification['scheduler_epochs']}, "
              f"hidden={model_spec(architecture)['hidden']}, "
              f"embedding={model_spec(architecture)['embedding_dim']}, "
              f"parameters={sum(p.numel() for p in model.parameters())}")
    print(f"Seeds={args.seeds}; GNN fits={len(args.models) * len(args.seeds)}; "
          f"max epochs={args.epochs}; patience={args.early_stopping_patience}; GBM seed=42; downstream=embedding only; prediction=argmax")
    print("CHECK ONLY: no training, no test prediction, no result folder created")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = resolve_arguments(argv)
    if args.check:
        check_only(args)
        return
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(min(args.threads, 2))
    cfg = make_config(args)
    configure_run(args, cfg)
    root = args.output
    start = time.perf_counter()
    with (root / "run.log").open("a", encoding="utf-8", buffering=1) as logfile, \
         contextlib.redirect_stdout(shared.Tee(sys.stdout, logfile)), \
         contextlib.redirect_stderr(shared.Tee(sys.stderr, logfile)):
        try:
            shared.write_json(root / "status.json", dict(status="running", stage=args.stage))
            print(f"[Run] device={cfg['device']}; output={root}", flush=True)
            print(f"[Workload] {len(planned_tasks(cfg))} GNN fits; residual GAT is not rerun", flush=True)
            if args.stage in ("fit", "all"):
                fit_all(root, cfg, args.data)
            if args.stage in ("evaluate", "all"):
                evaluate_all(root, cfg, args.data)
            shared.write_json(root / "status.json", dict(
                status="complete", stage=args.stage, elapsed_seconds=time.perf_counter() - start))
            print(f"[Complete] elapsed={time.perf_counter() - start:.1f}s", flush=True)
        except BaseException:
            shared.write_json(root / "status.json", dict(status="failed", error=traceback.format_exc()))
            raise


if __name__ == "__main__":
    main()
