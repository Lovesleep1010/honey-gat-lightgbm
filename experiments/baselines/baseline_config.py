"""Original comparator settings; independent selection precedes final refitting."""
from copy import deepcopy

import lightgbm as lgb
import numpy as np
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

import model as core
import pipeline as shared


ORIGINAL_SETTINGS = {
    "graphsage": dict(sg_window=11, sg_polyorder=3, loss="focal", gamma=3.0,
                      label_smoothing=0.0, boost0=2.5, scheduler_epochs=60,
                      original_train_epochs=60, min_child_samples=20,
                      reg_alpha=0.0, reg_lambda=0.0),
    "gcn": dict(sg_window=11, sg_polyorder=3, loss="focal", gamma=3.0,
                label_smoothing=0.0, boost0=2.5, scheduler_epochs=60,
                original_train_epochs=60, min_child_samples=20,
                reg_alpha=0.0, reg_lambda=0.0),
    "gin": dict(sg_window=15, sg_polyorder=3, loss="cross_entropy", gamma=None,
                label_smoothing=0.1, boost0=2.0, scheduler_epochs=50,
                original_train_epochs=50, min_child_samples=30,
                reg_alpha=0.0, reg_lambda=0.0),
    "gat": dict(sg_window=15, sg_polyorder=3, loss="cross_entropy", gamma=None,
                label_smoothing=0.05, boost0=2.0, scheduler_epochs=50,
                original_train_epochs=50, min_child_samples=30,
                reg_alpha=0.1, reg_lambda=5.0),
}


def settings(architecture, scheduler_epochs=None):
    result = deepcopy(ORIGINAL_SETTINGS[architecture])
    if scheduler_epochs is not None:
        result["scheduler_epochs"] = scheduler_epochs
    return result


def make_graphs(raw, labels, edges, specification):
    """Three original SG features, retaining the full-spectrum channel order."""
    graphs = []
    for spectrum, label in zip(core.snv(raw), labels):
        spectrum = spectrum.astype(np.float32)
        smoothed, first, second = (
            savgol_filter(spectrum, window_length=specification["sg_window"],
                          polyorder=specification["sg_polyorder"], deriv=derivative)
            for derivative in (0, 1, 2))
        nodes = np.stack([smoothed, first * 10.0, second * 100.0], axis=1).astype(np.float32)
        graphs.append(Data(x=torch.from_numpy(nodes), edge_index=edges,
                           y=torch.tensor(int(label), dtype=torch.long)))
    return graphs


class FocalLoss(nn.Module):
    """The original SAGE/GCN formula uses weighted CE inside exp(-CE)."""
    def __init__(self, alpha, gamma=3.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        return (((1.0 - torch.exp(-ce)) ** self.gamma) * ce).mean()


def criterion(class_weights, specification):
    if specification["loss"] == "focal":
        return FocalLoss(class_weights, gamma=specification["gamma"])
    if specification["loss"] == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights,
                                   label_smoothing=specification["label_smoothing"])
    raise ValueError(f"Unsupported loss: {specification['loss']}")


def gbm_params(specification, cfg, class_weights):
    return dict(objective="multiclass", num_class=5, n_estimators=3000,
                learning_rate=0.01, num_leaves=31, max_depth=-1,
                min_child_samples=specification["min_child_samples"],
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=specification["reg_alpha"], reg_lambda=specification["reg_lambda"],
                random_state=cfg["gbm_seed"], n_jobs=cfg["threads"], verbosity=-1,
                class_weight={i: float(weight) for i, weight in enumerate(class_weights)})


def predict_classes(probabilities, info):
    if info["prediction_rule"] != "argmax" or info["tau0"] is not None:
        raise RuntimeError("Original comparators require argmax with no tuned threshold")
    return np.asarray(probabilities).argmax(axis=1)


def fit_gbm(folder, xtr, ytr, xval, yval, class_weights, cfg, info):
    """Select tree count on Acq5, lock it, then refit; no added tau calibration."""
    folder.mkdir(parents=True, exist_ok=True)
    specification = cfg["comparator_settings"][info["architecture"]]
    params = gbm_params(specification, cfg, class_weights)
    train_model = lgb.LGBMClassifier(**params)
    train_model.fit(xtr, ytr, eval_set=[(xval, yval)], eval_metric="multi_logloss",
                    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    trees = int(train_model.best_iteration_ or params["n_estimators"])
    pval = train_model.predict_proba(xval)
    before_refit = shared.metrics(yval, pval.argmax(axis=1))
    train_model.booster_.save_model(str(folder / "lightgbm_train_only.txt"))
    np.savez_compressed(folder / "validation_selection.npz", y_val=yval, probabilities=pval)
    selection = dict(gbm_best_iteration=trees, prediction_rule="argmax", tau0=None,
                     validation_before_refit=before_refit,
                     tree_count_locked_before_refit=True)
    shared.write_json(folder / "gbm_selection.json", selection)
    # Keep selected GBM settings at refit, including GIN's min_child_samples.
    final_params = dict(params, n_estimators=trees)
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(np.vstack([xtr, xval]), np.concatenate([ytr, yval]))
    final_model.booster_.save_model(str(folder / "lightgbm_final.txt"))
    return dict(info, **selection, gbm_seed=cfg["gbm_seed"], gbm_parameters=params,
                gbm_final_parameters=final_params, gbm_fit_acquisitions=[1, 2, 3, 4],
                refit_acquisitions=[1, 2, 3, 4, 5], threshold_tuned=False)
