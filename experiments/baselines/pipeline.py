"""CILS revision experiments. Selection sees Acq1-5; evaluation sees Acq6.

See the repository README for setup, experiment entry points, and outputs.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from scipy.stats import binomtest
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import model as core

HERE = Path(__file__).resolve().parent
CLASSES = [0, 5, 10, 25, 50]
PROTOCOL_VERSION = "cils-best-revision-v3-c4filtered-multiband"
DATA_FILTER = {
    "rule": "exclude C4 / ManukaUMF10 / Concentration_Class 50",
    "Brand": "C4", "Class": "ManukaUMF10", "Concentration_Class": 50,
    "basis": "Fixed cohort exclusion applied before fitting",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=range(5), average="macro", zero_division=0)),
        "recall0": float(core.recall_of_class(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(5)).tolist(),
    }


@dataclass
class Split:
    raw: np.ndarray
    y: np.ndarray
    rows: np.ndarray


@dataclass
class TrainVal:
    train: Split
    val: Split
    wavelengths: np.ndarray
    columns: list


def load_frame(path):
    df = pd.read_csv(path)
    for name in ("Acquisition", "Concentration_Class", "Brand", "Class"):
        if name not in df:
            raise ValueError(f"Missing column: {name}")
    labels = pd.to_numeric(df["Concentration_Class"], errors="coerce")
    five_class = labels.isin(CLASSES)
    excluded = (df["Brand"].astype(str).str.strip().eq(DATA_FILTER["Brand"])
                & df["Class"].astype(str).str.strip().eq(DATA_FILTER["Class"])
                & labels.eq(DATA_FILTER["Concentration_Class"]))
    keep = five_class & ~excluded
    excluded_count = int((five_class & excluded).sum())
    df = df.loc[keep].copy().reset_index(drop=True)
    df["Concentration_Class"] = labels.loc[keep].astype(int).to_numpy()
    df["Acquisition"] = pd.to_numeric(df["Acquisition"], errors="raise").astype(int)
    df.attrs["excluded_c4_rows"] = excluded_count
    cols = sorted([c for c in df if str(c).endswith("nm")], key=lambda c: float(c[:-2]))
    if len(cols) != 128:
        raise ValueError(f"Expected 128 spectral columns; got {len(cols)}")
    wl = np.array([float(c[:-2]) for c in cols], dtype=np.float64)
    if not np.all(np.diff(wl) > 0):
        raise ValueError("Wavelengths must be unique and ascending")
    return df, cols, wl


def get_split(df, cols, acquisitions):
    mask = df["Acquisition"].isin(acquisitions).to_numpy()
    x = df.loc[mask, cols].to_numpy(np.float32)
    y = df.loc[mask, "Concentration_Class"].map({v: i for i, v in enumerate(CLASSES)}).to_numpy(np.int64)
    if not len(y) or set(np.unique(y)) != set(range(5)):
        raise ValueError(f"All five classes are required in acquisitions {acquisitions}")
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite spectra in acquisitions {acquisitions}")
    return Split(x, y, np.flatnonzero(mask))


def load_trainval(path):
    # No Acq6 feature/label array is exposed to any fitting or ranking function.
    df, cols, wl = load_frame(path)
    return TrainVal(get_split(df, cols, [1, 2, 3, 4]), get_split(df, cols, [5]), wl, cols)


def global_stats(raw, mean=None, std=None):
    g = core.make_global_features_from_raw(raw)
    if mean is None:
        mean = g.mean(axis=0, keepdims=True)
        std = g.std(axis=0, keepdims=True) + 1e-8
    return (g - mean) / std, mean, std


def select_raw(raw, bands):
    # Preserve full-spectrum memory layout/reduction order from best.py.
    if np.array_equal(bands, np.arange(raw.shape[1])):
        return raw
    return raw[:, bands]


def local_polynomial_matrices(wavelengths, full_wavelengths, window=15, degree=3):
    """Local polynomial smoothing/derivatives on physically irregular wavelengths.

    Derivative units are median full-spectrum channel spacing, for compatibility
    with the original d1*10/d2*100 scales. No unselected measurements are used.
    """
    n = len(wavelengths)
    if n < degree + 1:
        raise ValueError("At least four selected bands are required")
    window = min(window, n)
    coord = (wavelengths - full_wavelengths[0]) / np.median(np.diff(full_wavelengths))
    matrices = np.zeros((3, n, n), dtype=np.float64)
    for i in range(n):
        neighbors = np.argsort(np.abs(coord - coord[i]), kind="stable")[:window]
        design = np.vander(coord[neighbors] - coord[i], N=degree + 1, increasing=True)
        inverse = np.linalg.pinv(design)
        matrices[0, i, neighbors] = inverse[0]
        matrices[1, i, neighbors] = inverse[1]
        matrices[2, i, neighbors] = 2 * inverse[2]
    return matrices


def make_graphs(raw, y, bands, edges, wavelengths):
    # Slice BEFORE SNV. Sparse models never consume unselected intensities.
    selected = select_raw(raw, bands)
    normalized = core.snv(selected)
    if np.array_equal(bands, np.arange(len(wavelengths))):
        return [core.spectrum_to_graph_sg3(x, int(label), edges, use_pos=True)
                for x, label in zip(normalized, y)]
    mats = local_polynomial_matrices(wavelengths[bands], wavelengths)
    pos = ((wavelengths[bands] - wavelengths[0]) / (wavelengths[-1] - wavelengths[0])).astype(np.float32)
    graphs = []
    for x, label in zip(normalized, y):
        smooth, d1, d2 = (m @ x for m in mats)
        feats = np.stack([smooth, d1 * 10.0, d2 * 100.0, pos], axis=1).astype(np.float32)
        graphs.append(Data(x=torch.from_numpy(feats), edge_index=edges, y=torch.tensor(int(label), dtype=torch.long)))
    return graphs


def gbm_params(cfg, class_weights, quick=False):
    common = dict(objective="multiclass", num_class=5, num_leaves=31,
                  random_state=cfg["gbm_seed"], n_jobs=cfg["threads"], verbosity=-1,
                  class_weight={i: float(w) for i, w in enumerate(class_weights)})
    if quick:
        return dict(common, n_estimators=500, learning_rate=0.05)
    return dict(common, n_estimators=3000, learning_rate=0.01, max_depth=-1,
                min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=5.0)


def seal_task(folder, info, artifacts):
    info = dict(info, artifact_sha256={name: sha(folder / name) for name in artifacts}, status="locked")
    write_json(folder / "locked.json", info)
    return info


def read_locked(folder):
    info = read_json(folder / "locked.json")
    for name, expected in info["artifact_sha256"].items():
        if sha(folder / name) != expected:
            raise RuntimeError(f"Artifact has changed: {folder / name}")
    return info


def fit_gbm(folder, xtr, ytr, xval, yval, class_weights, cfg, info):
    """Tree count and tau are selected BEFORE refitting on Acq1-5."""
    folder.mkdir(parents=True, exist_ok=True)
    params = gbm_params(cfg, class_weights)
    train_model = lgb.LGBMClassifier(**params)
    train_model.fit(xtr, ytr, eval_set=[(xval, yval)], eval_metric="multi_logloss",
                    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    trees = int(train_model.best_iteration_ or params["n_estimators"])
    pval = train_model.predict_proba(xval)
    tau = float(core.tune_tau0_on_val_strict(train_model, xval, yval))
    before_refit = metrics(yval, core.apply_tau0(pval, tau))
    train_model.booster_.save_model(str(folder / "lightgbm_train_only.txt"))
    final_params = dict(params, n_estimators=trees)
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(np.vstack([xtr, xval]), np.concatenate([ytr, yval]))
    final_model.booster_.save_model(str(folder / "lightgbm_final.txt"))
    np.savez_compressed(folder / "validation_calibration.npz", y_val=yval, probabilities=pval)
    payload = dict(info, gbm_seed=cfg["gbm_seed"], gbm_best_iteration=trees, tau0=tau,
                   validation_before_refit=before_refit, gbm_parameters=params,
                   gbm_fit_acquisitions=[1, 2, 3, 4], refit_acquisitions=[1, 2, 3, 4, 5],
                   threshold_priority=["Macro-F1", "balanced_accuracy", "recall0"],
                   threshold_locked_before_refit=True)
    return payload


def task_folder(root, task_id):
    return root / "tasks" / task_id


def fit_encoder(root, tv, cfg, seed, k, bands, task_id):
    folder = task_folder(root, task_id)
    if (folder / "locked.json").exists():
        print(f"[Reuse] {task_id}", flush=True)
        return read_locked(folder)
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    write_json(folder / "status.json", {"status": "training", "task": task_id})
    core.seed_everything(seed)
    device = torch.device(cfg["device"])
    class_weights = core.compute_class_weights(tv.train.y, boost0=2.0, mode="effective")
    gtr, mu, sd = global_stats(select_raw(tv.train.raw, bands))
    gval, _, _ = global_stats(select_raw(tv.val.raw, bands), mu, sd)
    normalized_train = core.snv(select_raw(tv.train.raw, bands))
    if not np.isfinite(np.corrcoef(normalized_train.T)).all():
        raise ValueError("Training-only graph correlations contain NaN; check constant channels")
    edges = core.build_hybrid_edge_index(normalized_train, top_k=k)
    train_graphs = make_graphs(tv.train.raw, tv.train.y, bands, edges, tv.wavelengths)
    val_graphs = make_graphs(tv.val.raw, tv.val.y, bands, edges, tv.wavelengths)
    sampler = WeightedRandomSampler(class_weights[tv.train.y].astype(np.float64),
                                    num_samples=len(tv.train.y), replacement=True)
    train_loader = DataLoader(train_graphs, batch_size=128, sampler=sampler, shuffle=False)
    embed_loader = DataLoader(train_graphs, batch_size=256, shuffle=False)
    val_loader = DataLoader(val_graphs, batch_size=256, shuffle=False)
    model = core.GATEncoder(in_dim=4, hid=64, heads=4, dropout=0.5, num_classes=5).to(device)
    history = []

    def log(record):
        history.append(dict(record, elapsed_seconds=time.perf_counter() - start))
        write_json(folder / "epoch_history.json", history)
        write_json(folder / "status.json", {"status": "training", "epoch": record["epoch"], "epochs": cfg["epochs"]})
        print(f"[Epoch] {task_id} {record['epoch']}/{cfg['epochs']} loss={record['loss']:.5f}", flush=True)

    checkpoint = folder / "encoder.pt"
    ep, probe_ba = core.train_encoder_downstream_selection(
        model, train_loader, embed_loader, val_loader, device,
        torch.tensor(class_weights, dtype=torch.float32, device=device),
        gtr, gval, tv.train.y, tv.val.y,
        epochs=cfg["scheduler_epochs"], train_epochs=cfg["epochs"],
        ckpt_path=str(checkpoint), check_interval=5, n_jobs=cfg["threads"],
        gbm_seed=cfg["gbm_seed"], log_callback=log)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    etr, ytr = core.extract_embeddings(model, embed_loader, device)
    eval_, yval = core.extract_embeddings(model, val_loader, device)
    assert np.array_equal(ytr, tv.train.y) and np.array_equal(yval, tv.val.y)
    np.savez_compressed(folder / "encoder_cache.npz", train_embedding=etr, val_embedding=eval_,
                        train_global=gtr, val_global=gval, y_train=ytr, y_val=yval,
                        global_mean=mu, global_std=sd, edges=edges.numpy(),
                        train_rows=tv.train.rows, val_rows=tv.val.rows)
    info = dict(task_id=task_id, kind="gat", features="embedding_global", seed=seed,
                k=k, n_bands=len(bands), bands=bands.tolist(), selected_epoch=int(ep),
                selected_probe_val_ba=float(probe_ba), epochs=cfg["epochs"],
                scheduler_epochs=cfg["scheduler_epochs"], encoder_task=task_id,
                preprocess="original_sg15_index" if len(bands) == 128 else "irregular_local_polynomial15",
                global_features="raw selected-band mean/std/range/three channel-segment means")
    info = fit_gbm(folder, np.hstack([etr, gtr]), ytr, np.hstack([eval_, gval]), yval,
                   class_weights, cfg, info)
    info["elapsed_fit_seconds"] = time.perf_counter() - start
    info = seal_task(folder, info, ["encoder.pt", "encoder_cache.npz", "epoch_history.json",
                                   "lightgbm_train_only.txt", "lightgbm_final.txt", "validation_calibration.npz"])
    write_json(folder / "status.json", {"status": "locked"})
    return info


def fit_ablation(root, tv, cfg, seed):
    encoder_id = f"full_k12_seed{seed}"
    full = fit_encoder(root, tv, cfg, seed, 12, np.arange(128), encoder_id)
    task_id = f"embedding_only_seed{seed}"
    folder = task_folder(root, task_id)
    if (folder / "locked.json").exists():
        read_locked(folder)
        return [encoder_id, task_id]
    with np.load(task_folder(root, encoder_id) / "encoder_cache.npz") as cache:
        folder.mkdir(parents=True, exist_ok=True)
        info = dict(full)
        for key in ("artifact_sha256", "status", "elapsed_fit_seconds"):
            info.pop(key, None)
        info.update(task_id=task_id, kind="frozen_encoder_ablation", features="embedding_only",
                    ablation_scope="Fixed encoder selected by embedding+global probe; conditional feature contribution")
        info = fit_gbm(folder, cache["train_embedding"], cache["y_train"],
                       cache["val_embedding"], cache["y_val"],
                       core.compute_class_weights(tv.train.y), cfg, info)
    seal_task(folder, info, ["lightgbm_train_only.txt", "lightgbm_final.txt", "validation_calibration.npz"])
    return [encoder_id, task_id]


def fit_global_only(root, tv, cfg):
    # GAT seed cannot affect these six input features; do not fake five repetitions.
    task_id = "global_only_fixed_gbm42"
    folder = task_folder(root, task_id)
    if (folder / "locked.json").exists():
        read_locked(folder)
        return task_id
    folder.mkdir(parents=True, exist_ok=True)
    gtr, mu, sd = global_stats(tv.train.raw)
    gval, _, _ = global_stats(tv.val.raw, mu, sd)
    np.savez_compressed(folder / "global_transform.npz", mean=mu, std=sd)
    info = fit_gbm(folder, gtr, tv.train.y, gval, tv.val.y, core.compute_class_weights(tv.train.y), cfg,
                   dict(task_id=task_id, kind="global", features="global_only", seed=None,
                        k=None, n_bands=128, bands=list(range(128)), independent_repeats=1))
    seal_task(folder, info, ["global_transform.npz", "lightgbm_train_only.txt", "lightgbm_final.txt", "validation_calibration.npz"])
    return task_id


def fit_raw(root, tv, cfg, bands, task_id, extra_info=None):
    folder = task_folder(root, task_id)
    if (folder / "locked.json").exists():
        read_locked(folder)
        return task_id
    info = fit_gbm(folder, core.snv(select_raw(tv.train.raw, bands)), tv.train.y,
                   core.snv(select_raw(tv.val.raw, bands)), tv.val.y, core.compute_class_weights(tv.train.y), cfg,
                   dict(task_id=task_id, kind="raw", features="raw_snv", seed=None, k=None,
                        n_bands=len(bands), bands=bands.tolist(), independent_repeats=1,
                        preprocess="selected raw intensities -> sample-wise SNV", **(extra_info or {})))
    seal_task(folder, info, ["lightgbm_train_only.txt", "lightgbm_final.txt", "validation_calibration.npz"])
    return task_id


def stable_rank(scores):
    # Explicit tie-breaking by original band index.
    return np.lexsort((np.arange(len(scores)), -np.asarray(scores)))


def hybrid_bands(gat_scores, raw_scores, count):
    gat_rank, raw_rank = stable_rank(gat_scores), stable_rank(raw_scores)
    n_gat = count // 2
    selected = set(gat_rank[:n_gat].tolist() + raw_rank[:count - n_gat].tolist())
    for band in raw_rank:
        if len(selected) >= count:
            break
        selected.add(int(band))
    return np.array(sorted(selected), dtype=np.int64)


def selection_sets(gat_scores, raw_scores, counts, wavelengths):
    """Two fixed selection families; regional subsets inherit raw-ranked 48."""
    hybrid = {str(n): hybrid_bands(gat_scores, raw_scores, n).tolist() for n in counts}
    raw_rank = stable_rank(raw_scores)
    raw = {str(n): np.sort(raw_rank[:n]).tolist() for n in counts}
    regions = {}
    if 48 in counts:
        parent = np.array(raw["48"], dtype=np.int64)
        for region, mask in (("vis", wavelengths[parent] < 700), ("nir", wavelengths[parent] >= 700)):
            bands = parent[mask].tolist()
            regions[region] = dict(bands=bands, n_bands=len(bands),
                                   status="planned" if len(bands) >= 2 else "skipped",
                                   reason=None if len(bands) >= 2 else "Fewer than two selected channels; sample SNV is degenerate")
    return dict(selected_bands=hybrid, raw_ranked_bands=raw, raw48_regions=regions,
                region_boundary_nm=700, regional_parent="raw_ranked_bands48",
                region_counts_fixed_after_acq5_before_test=True)


def rank_bands(root, tv, cfg):
    """Training-only models, Acq5 attribution; no test split accepted here."""
    folder = root / "band_selection"
    if (folder / "locked.json").exists():
        return read_locked(folder)
    folder.mkdir(parents=True, exist_ok=True)
    seed = cfg["ranking_seed"]
    encoder_id = f"full_k12_seed{seed}"
    fit_encoder(root, tv, cfg, seed, 12, np.arange(128), encoder_id)
    raw_id = fit_raw(root, tv, cfg, np.arange(128), "raw_full128")
    gat_info = read_locked(task_folder(root, encoder_id))
    raw_info = read_locked(task_folder(root, raw_id))
    device = torch.device(cfg["device"])
    model = core.GATEncoder().to(device)
    model.load_state_dict(torch.load(task_folder(root, encoder_id) / "encoder.pt", map_location=device, weights_only=True))
    model.eval()
    with np.load(task_folder(root, encoder_id) / "encoder_cache.npz") as cache:
        edges = torch.from_numpy(cache["edges"])
        gval = cache["val_global"].copy()
    graphs = make_graphs(tv.val.raw, tv.val.y, np.arange(128), edges, tv.wavelengths)
    loader = DataLoader(graphs, batch_size=256, shuffle=False)
    gat_booster = lgb.Booster(model_file=str(task_folder(root, encoder_id) / "lightgbm_train_only.txt"))

    def gat_val_ba():
        embedding, labels = core.extract_embeddings(model, loader, device)
        assert np.array_equal(labels, tv.val.y)
        proba = gat_booster.predict(np.hstack([embedding, gval]), num_threads=cfg["threads"])
        return balanced_accuracy_score(tv.val.y, core.apply_tau0(proba, gat_info["tau0"]))

    base = gat_val_ba()
    gat_scores = []
    for band in range(128):
        original = [g.x[band].clone() for g in graphs]
        try:
            for graph in graphs:
                graph.x[band] = 0  # Original node-feature occlusion, including position.
            gat_scores.append(float(base - gat_val_ba()))
        finally:
            for graph, values in zip(graphs, original):
                graph.x[band] = values
        if (band + 1) % 8 == 0:
            print(f"[Acq5 GAT occlusion] {band + 1}/128", flush=True)
        write_json(folder / "progress.json", {"stage": "gat_occlusion", "done": band + 1, "total": 128})
    raw_booster = lgb.Booster(model_file=str(task_folder(root, raw_id) / "lightgbm_train_only.txt"))
    xval = core.snv(tv.val.raw)
    base_raw = balanced_accuracy_score(tv.val.y, core.apply_tau0(raw_booster.predict(xval, num_threads=cfg["threads"]), raw_info["tau0"]))
    rng = np.random.default_rng(cfg["ranking_seed"])
    raw_scores = np.empty((128, cfg["permutation_repeats"]), dtype=np.float64)
    for band in range(128):
        for rep in range(cfg["permutation_repeats"]):
            perturbed = xval.copy()
            perturbed[:, band] = rng.permutation(perturbed[:, band])
            pred = core.apply_tau0(raw_booster.predict(perturbed, num_threads=cfg["threads"]), raw_info["tau0"])
            raw_scores[band, rep] = base_raw - balanced_accuracy_score(tv.val.y, pred)
        if (band + 1) % 16 == 0:
            print(f"[Acq5 raw GBM permutation] {band + 1}/128", flush=True)
    raw_mean = raw_scores.mean(axis=1)
    frame = pd.DataFrame({"band_index": np.arange(128), "wavelength_nm": tv.wavelengths,
                          "gat_node_occlusion_delta_ba": gat_scores,
                          "raw_snv_permutation_delta_ba_mean": raw_mean,
                          "raw_snv_permutation_delta_ba_sd": raw_scores.std(axis=1, ddof=0)})
    frame.to_csv(folder / "acq5_importance.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(folder / "importance.npz", gat=gat_scores, raw_mean=raw_mean, raw_repeats=raw_scores)
    sets = selection_sets(gat_scores, raw_mean, cfg["band_counts"], tv.wavelengths)
    info = dict(ranking_acquisition=5, ranking_model_fit_acquisitions=[1, 2, 3, 4],
                ranking_encoder_seed=seed, encoder_task=encoder_id, raw_task=raw_id,
                gat_attribution="zero all node features at one band; global features and graph unchanged",
                raw_attribution="permute one SNV channel; fixed original sample SNV normalization",
                ranking_is_sensor_selection_heuristic=True, permutation_repeats=cfg["permutation_repeats"],
                selection_rule="union of top half GAT and top half raw; fill overlaps from raw rank; ties use band index",
                **sets,
                selected_wavelengths={n: tv.wavelengths[b].tolist() for n, b in sets["selected_bands"].items()},
                raw_ranked_wavelengths={n: tv.wavelengths[b].tolist() for n, b in sets["raw_ranked_bands"].items()},
                raw_selection_rule="top N raw SNV GBM Acq5 mean permutation delta BA; ties use band index",
                band_counts_fixed_before_test=True, test_used_for_selection=False,
                sparse_k=cfg["sparse_k"])
    return seal_task(folder, info, ["acq5_importance.csv", "importance.npz"])


def fit_requested(root, tv, cfg, experiments):
    tasks, groups = [], {"baseline": [], "ablation": [], "k": [], "bands": [],
                        "raw_ranked_bands": [], "raw48_regions": []}

    def add(group, task_id):
        groups[group].append(task_id)
        if task_id not in tasks:
            tasks.append(task_id)

    # These references are always recomputed under this package's protocol.
    for seed in cfg["seeds"]:
        task = f"full_k12_seed{seed}"
        fit_encoder(root, tv, cfg, seed, 12, np.arange(128), task)
        add("baseline", task)
        if "ablation" in experiments:
            for task in fit_ablation(root, tv, cfg, seed):
                add("ablation", task)
        if "k" in experiments:
            for k in cfg["k_values"]:
                task = f"full_k{k}_seed{seed}"
                fit_encoder(root, tv, cfg, seed, k, np.arange(128), task)
                add("k", task)
    if "ablation" in experiments:
        add("ablation", fit_global_only(root, tv, cfg))
    if "bands" in experiments:
        selection = rank_bands(root, tv, cfg)
        add("bands", "raw_full128")
        add("raw_ranked_bands", "raw_full128")
        for task in groups["baseline"]:
            add("bands", task)
        for count in cfg["band_counts"]:
            bands = np.array(selection["selected_bands"][str(count)], dtype=np.int64)
            add("bands", fit_raw(root, tv, cfg, bands, f"raw_bands{count}"))
            for seed in cfg["seeds"]:
                task = f"gat_bands{count}_k{cfg['sparse_k']}_seed{seed}"
                fit_encoder(root, tv, cfg, seed, cfg["sparse_k"], bands, task)
                add("bands", task)
            raw_bands = np.array(selection["raw_ranked_bands"][str(count)], dtype=np.int64)
            task = fit_raw(root, tv, cfg, raw_bands, f"raw_ranked_bands{count}",
                           dict(selection_family="raw_gbm_rank", spectral_region="mixed"))
            add("raw_ranked_bands", task)
        if "48" in selection["raw_ranked_bands"]:
            add("raw48_regions", "raw_ranked_bands48")
            for region, spec in selection["raw48_regions"].items():
                if spec["status"] == "skipped":
                    print(f"[Region skipped] {region}: {spec['reason']}", flush=True)
                    continue
                task = fit_raw(root, tv, cfg, np.array(spec["bands"], dtype=np.int64),
                               f"raw_ranked48_{region}_only",
                               dict(selection_family="raw_gbm_rank", spectral_region=region,
                                    parent_task="raw_ranked_bands48", region_boundary_nm=700))
                add("raw48_regions", task)
    # This seal is written only after ALL requested conditions are fixed.
    artifacts = {task: sha(task_folder(root, task) / "locked.json") for task in tasks}
    plan = dict(experiments=experiments, tasks=tasks, groups=groups, task_lock_sha256=artifacts,
                selection_finished_before_test=True, protocol_sha256=sha(root / "protocol.json"))
    if "bands" in experiments:
        plan["band_selection_sha256"] = sha(root / "band_selection" / "locked.json")
    plan_id = "_".join(sorted(experiments))
    path = root / f"plan_{plan_id}.json"
    write_json(path, plan)
    return path


def load_model_encoder(root, info, device):
    folder = task_folder(root, info["encoder_task"])
    read_locked(folder)
    model = core.GATEncoder().to(device)
    model.load_state_dict(torch.load(folder / "encoder.pt", map_location=device, weights_only=True))
    model.eval()
    with np.load(folder / "encoder_cache.npz") as cache:
        return model, torch.from_numpy(cache["edges"]), cache["global_mean"].copy(), cache["global_std"].copy()


def evaluate_requested(root, cfg, plan_path, data_path):
    plan = read_json(plan_path)
    if plan["protocol_sha256"] != sha(root / "protocol.json"):
        raise RuntimeError("Protocol changed since the experiment plan was sealed")
    if "band_selection_sha256" in plan:
        if plan["band_selection_sha256"] != sha(root / "band_selection" / "locked.json"):
            raise RuntimeError("Band selection changed after sealing")
        read_locked(root / "band_selection")
    for task in plan["tasks"]:
        if plan["task_lock_sha256"][task] != sha(task_folder(root, task) / "locked.json"):
            raise RuntimeError(f"Task settings changed after sealing: {task}")
        read_locked(task_folder(root, task))
    # First place in the workflow where an Acq6 feature/label array is created.
    df, cols, wavelengths = load_frame(data_path)
    test = get_split(df, cols, [6])
    results = []
    for task in plan["tasks"]:
        folder = task_folder(root, task)
        info = read_locked(folder)
        if (folder / "test_metrics.json").exists():
            evaluated = read_json(folder / "test_metrics.json")
            if evaluated["task_lock_sha256"] != sha(folder / "locked.json"):
                raise RuntimeError(f"Stored evaluation belongs to a different model: {task}")
            if sha(folder / "test_predictions.npz") != evaluated["predictions_sha256"]:
                raise RuntimeError(f"Stored predictions changed: {task}")
        else:
            bands = np.array(info["bands"], dtype=np.int64)
            if info["kind"] == "raw":
                xtest = core.snv(select_raw(test.raw, bands))
            elif info["kind"] == "global":
                with np.load(folder / "global_transform.npz") as transform:
                    xtest, _, _ = global_stats(select_raw(test.raw, bands), transform["mean"], transform["std"])
            else:
                model, edges, mu, sd = load_model_encoder(root, info, torch.device(cfg["device"]))
                graphs = make_graphs(test.raw, test.y, bands, edges, wavelengths)
                embedding, labels = core.extract_embeddings(model, DataLoader(graphs, batch_size=256, shuffle=False), torch.device(cfg["device"]))
                assert np.array_equal(labels, test.y)
                if info["features"] == "embedding_only":
                    xtest = embedding
                else:
                    gtest, _, _ = global_stats(select_raw(test.raw, bands), mu, sd)
                    xtest = np.hstack([embedding, gtest])
                del model, graphs
                if cfg["device"].startswith("cuda"):
                    torch.cuda.empty_cache()
            booster = lgb.Booster(model_file=str(folder / "lightgbm_final.txt"))
            probabilities = booster.predict(xtest, num_threads=cfg["threads"])
            predictions = core.apply_tau0(probabilities, info["tau0"])
            np.savez_compressed(folder / "test_predictions.npz", y_true=test.y, predictions=predictions,
                                probabilities=probabilities, row_indices=test.rows)
            evaluated = dict(task_id=task, task_lock_sha256=sha(folder / "locked.json"),
                             predictions_sha256=sha(folder / "test_predictions.npz"),
                             acquisition=6, metrics=metrics(test.y, predictions))
            write_json(folder / "test_metrics.json", evaluated)
        row = {key: info.get(key) for key in ("task_id", "kind", "features", "seed", "k", "n_bands", "selected_epoch", "gbm_best_iteration", "tau0", "selection_family", "spectral_region")}
        row["selection_family"] = info.get("selection_family", "full_spectrum" if info["n_bands"] == 128 else "hybrid_gat_raw_rank")
        selected_wls = wavelengths[np.array(info["bands"], dtype=np.int64)]
        row.update(n_vis=int((selected_wls < 700).sum()), n_nir=int((selected_wls >= 700).sum()),
                   band_reduction_fraction=1 - info["n_bands"] / 128)
        row.update({k: v for k, v in evaluated["metrics"].items() if k != "confusion_matrix"})
        results.append(row)
        print(f"[Test] {task}: BA={row['balanced_accuracy']:.6f}, Macro-F1={row['macro_f1']:.6f}", flush=True)
    summarize(root, results, plan, plan_path.stem)


def summarize(root, results, plan, name):
    frame = pd.DataFrame(results)
    frame.to_csv(root / f"{name}_metrics.csv", index=False, encoding="utf-8-sig")
    summary_rows, paired_rows = [], []
    lookup = {row["task_id"]: row for row in results}
    for group, task_ids in plan["groups"].items():
        if not task_ids:
            continue
        sub = frame[frame.task_id.isin(task_ids)].copy()
        def condition_name(row):
            base = f"{row['features']}_bands{int(row['n_bands'])}_k{int(row['k']) if pd.notna(row['k']) else 'NA'}"
            if pd.notna(row.get("spectral_region")):
                base += f"_{row['spectral_region']}"
            return base
        sub["condition"] = sub.apply(condition_name, axis=1)
        for condition, rows in sub.groupby("condition", sort=False):
            record = dict(group=group, condition=condition, n_runs=len(rows),
                          selection_family=rows.iloc[0]["selection_family"],
                          spectral_region=rows.iloc[0]["spectral_region"],
                          n_bands=int(rows.iloc[0]["n_bands"]), n_vis=int(rows.iloc[0]["n_vis"]),
                          n_nir=int(rows.iloc[0]["n_nir"]),
                          band_reduction_fraction=float(rows.iloc[0]["band_reduction_fraction"]),
                          variability="GAT training seed; GBM seed fixed at 42" if len(rows) > 1 else "single fixed control; SD not estimable")
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "recall0"):
                record[metric + "_mean"] = float(rows[metric].mean())
                record[metric + "_sd"] = float(rows[metric].std(ddof=1)) if len(rows) > 1 else None
            summary_rows.append(record)
        if group in ("ablation", "k", "bands"):
            for task in task_ids:
                item = lookup[task]
                if item["seed"] is None:
                    continue
                if group == "bands":
                    raw_task = "raw_full128" if item["n_bands"] == 128 else f"raw_bands{int(item['n_bands'])}"
                    raw = lookup[raw_task]
                    paired_rows.append(dict(group=group, task_id=task, seed=item["seed"],
                        comparison=f"GAT+GBM minus raw SNV GBM on identical {int(item['n_bands'])} bands",
                        delta_balanced_accuracy=item["balanced_accuracy"] - raw["balanced_accuracy"],
                        delta_macro_f1=item["macro_f1"] - raw["macro_f1"]))
                baseline = lookup.get(f"full_k12_seed{int(item['seed'])}")
                if baseline is None or task == baseline["task_id"]:
                    continue
                paired_rows.append(dict(group=group, task_id=task, seed=item["seed"],
                    comparison="condition minus same-seed full128 k12 embedding+global",
                    delta_balanced_accuracy=item["balanced_accuracy"] - baseline["balanced_accuracy"],
                    delta_macro_f1=item["macro_f1"] - baseline["macro_f1"]))
        if group in ("raw_ranked_bands", "raw48_regions"):
            baseline_id = "raw_full128" if group == "raw_ranked_bands" else "raw_ranked_bands48"
            baseline = lookup[baseline_id]
            for task in task_ids:
                if task == baseline_id:
                    continue
                item = lookup[task]
                paired_rows.append(dict(group=group, task_id=task, seed=None,
                    comparison=f"condition minus {baseline_id}; fixed GBM seed 42",
                    delta_balanced_accuracy=item["balanced_accuracy"] - baseline["balanced_accuracy"],
                    delta_macro_f1=item["macro_f1"] - baseline["macro_f1"]))
    pd.DataFrame(summary_rows).to_csv(root / f"{name}_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paired_rows).to_csv(root / f"{name}_paired_deltas.csv", index=False, encoding="utf-8-sig")
    mcnemar_rows = []
    if "raw_ranked_bands48" in lookup:
        with np.load(task_folder(root, "raw_ranked_bands48") / "test_predictions.npz") as a, \
             np.load(task_folder(root, "raw_full128") / "test_predictions.npz") as b:
            mcnemar_rows.append(dict(task_a="raw_ranked_bands48", task_b="raw_full128",
                                     **paired_mcnemar(a, b)))
    pd.DataFrame(mcnemar_rows, columns=["task_a", "task_b", "n_samples", "a_only_correct", "b_only_correct",
        "both_correct", "both_wrong", "delta_accuracy", "p_value", "method", "scope"]).to_csv(
        root / f"{name}_mcnemar.csv", index=False, encoding="utf-8-sig")
    write_json(root / f"{name}_summary.json", {"summary": summary_rows, "paired_deltas": paired_rows,
        "mcnemar": mcnemar_rows,
        "sd_definition": "sample standard deviation (ddof=1), not a confidence interval",
        "all_seeds_retained": True, "ablation_scope": "fixed encoder; conditional feature contribution",
        "acq6_role": "held-out acquisition evaluation, not unseen honey-product evaluation"})


def paired_mcnemar(a, b):
    if not (np.array_equal(a["row_indices"], b["row_indices"]) and np.array_equal(a["y_true"], b["y_true"])):
        raise RuntimeError("McNemar requires identical test rows and labels in identical order")
    ca, cb = a["predictions"] == a["y_true"], b["predictions"] == b["y_true"]
    a_only, b_only = int((ca & ~cb).sum()), int((cb & ~ca).sum())
    discordant = a_only + b_only
    return dict(n_samples=len(ca), a_only_correct=a_only, b_only_correct=b_only,
                both_correct=int((ca & cb).sum()), both_wrong=int((~ca & ~cb).sum()),
                delta_accuracy=float(ca.mean() - cb.mean()),
                p_value=float(binomtest(a_only, discordant, 0.5).pvalue) if discordant else 1.0,
                method="exact two-sided McNemar (binomial discordant pairs)",
                scope="sample-level accuracy; not BA/F1 or acquisition-level generalization")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def configure_protocol(args, root):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install matching GPU PyTorch or use --device cpu.")
    horizon = args.scheduler_epochs if args.scheduler_epochs is not None else args.epochs
    if not 1 <= args.epochs <= horizon:
        raise ValueError("Require 1 <= epochs <= scheduler-epochs")
    if args.threads < 1 or args.permutation_repeats < 1:
        raise ValueError("Threads and permutation repeats must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError("Seeds must be nonnegative and unique")
    if len(set(args.k_values)) != len(args.k_values) or any(k < 1 or k >= 128 for k in args.k_values):
        raise ValueError("k values must be unique and within 1..127")
    if len(set(args.band_counts)) != len(args.band_counts) or any(n < 4 or n >= 128 for n in args.band_counts):
        raise ValueError("Band counts must be unique and within 4..127")
    if not 1 <= args.sparse_k < min(args.band_counts):
        raise ValueError("sparse-k must be smaller than every selected band count")
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "torch-geometric", "lightgbm", "numpy", "pandas", "scipy", "scikit-learn"]}
    cfg = dict(protocol=PROTOCOL_VERSION, data_filter=DATA_FILTER,
               band_selection_families=["hybrid_gat_raw_rank", "raw_gbm_rank"],
               raw48_region_rule="Vis <700nm, NIR >=700nm; nested raw-ranked 48; skip regions with fewer than 2 channels",
               mcnemar_comparison="raw-ranked 48 vs raw full128; exact two-sided sample-accuracy test",
               epochs=args.epochs, scheduler_epochs=horizon,
               seeds=args.seeds, gbm_seed=42, k_values=args.k_values, band_counts=args.band_counts,
               sparse_k=args.sparse_k, ranking_seed=args.ranking_seed,
               permutation_repeats=args.permutation_repeats, device=device, threads=args.threads,
               dataset_sha256=sha(args.data), source_sha256={n: sha(HERE / n) for n in
                   ["pipeline.py", "model.py", "reference_model.py"]}, versions=versions,
               python_version=platform.python_version(), split={"train": [1, 2, 3, 4], "validation": [5], "test": [6]})
    path = root / "protocol.json"
    if path.exists():
        if not args.resume:
            raise RuntimeError("Output already exists. Use --resume with identical settings, or a new --output folder.")
        previous = read_json(path)
        if previous != cfg:
            keys = [key for key in cfg if cfg[key] != previous.get(key)]
            raise RuntimeError(f"Resume settings changed: {keys}. Use a new output folder.")
    else:
        root.mkdir(parents=True, exist_ok=True)
        write_json(path, cfg)
        write_json(root / "runtime.json", dict(executable=sys.executable, platform=platform.platform(),
                   data_path=str(args.data.resolve()), started_at=datetime.now().isoformat(),
                   cuda_device=torch.cuda.get_device_name(0) if device.startswith("cuda") else None))
        (root / "source").mkdir(exist_ok=True)
        for name in cfg["source_sha256"]:
            (root / "source" / name).write_bytes((HERE / name).read_bytes())
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(min(args.threads, 2))
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE.parents[1] / "data" / "adulteration_dataset_26_08_2021.csv")
    parser.add_argument("--output", type=Path, default=HERE / "results_5epochs_c4filtered_multiband")
    parser.add_argument("--experiments", nargs="+", choices=["all", "baseline", "ablation", "k", "bands"], default=["all"])
    parser.add_argument("--stage", choices=["fit", "evaluate", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--scheduler-epochs", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024, 3407, 9999])
    parser.add_argument("--k-values", nargs="+", type=int, default=[8, 10, 12, 14, 16])
    parser.add_argument("--band-counts", nargs="+", type=int, default=[16, 32, 48])
    parser.add_argument("--sparse-k", type=int, default=8)
    parser.add_argument("--ranking-seed", type=int, default=42)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true", help="Validate data/environment and print workload; no training")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    experiments = ["ablation", "k", "bands"] if "all" in args.experiments else sorted(set(args.experiments))
    if args.check:
        tv = load_trainval(args.data)
        frame, _, _ = load_frame(args.data)
        print(f"Train={len(tv.train.y)}, Val={len(tv.val.y)}, "
              f"Test rows={int(frame['Acquisition'].eq(6).sum())}, bands={len(tv.wavelengths)}")
        print(f"Data filter: {DATA_FILTER['rule']}; excluded={frame.attrs['excluded_c4_rows']} rows")
        print(f"PyTorch={torch.__version__}, CUDA available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU={torch.cuda.get_device_name(0)}")
        print(f"GAT seeds={args.seeds}; GBM seed stays 42; epochs={args.epochs}")
        print(f"Requested={experiments}; k={args.k_values}; reduced bands={args.band_counts}")
        print("CHECK ONLY: no training, no test evaluation, no result folder created.")
        return
    root = args.output.resolve()
    cfg = configure_protocol(args, root)
    plan_path = root / ("plan_" + "_".join(sorted(experiments)) + ".json")
    log = (root / "run.log").open("a", encoding="utf-8", buffering=1)
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            write_json(root / "status.json", dict(status="running", stage=args.stage, experiments=experiments))
            print(f"[Protocol] {PROTOCOL_VERSION}; device={cfg['device']}; output={root}", flush=True)
            print(f"[Data filter] {DATA_FILTER['rule']}", flush=True)
            if args.stage in ("fit", "all"):
                tv = load_trainval(args.data)
                print(f"[Split] Train={len(tv.train.y)}, Val={len(tv.val.y)}", flush=True)
                plan_path = fit_requested(root, tv, cfg, experiments)
            if args.stage in ("evaluate", "all"):
                if not plan_path.exists():
                    raise RuntimeError("No sealed plan. Run the same experiment set with --stage fit first.")
                evaluate_requested(root, cfg, plan_path, args.data)
            write_json(root / "status.json", dict(status="complete", stage=args.stage,
                       experiments=experiments, elapsed_seconds=time.perf_counter() - start))
            print(f"[Complete] elapsed={time.perf_counter() - start:.1f}s", flush=True)
    except BaseException:
        write_json(root / "status.json", dict(status="failed", error=traceback.format_exc()))
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
