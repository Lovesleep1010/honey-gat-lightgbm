import os
import random
import warnings
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, classification_report
import lightgbm as lgb

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# =========================
# 0) Utilities
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


def snv(X: np.ndarray):
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True) + 1e-8
    return (X - m) / s


def compute_class_weights(y_train: np.ndarray, num_classes: int = 5,
                          boost0: float = 2.0, mode: str = "effective") -> np.ndarray:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)

    if mode == "inverse":
        N = counts.sum()
        w = N / (num_classes * counts)
    else:
        beta = 0.9999
        eff_num = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / eff_num

    w = w / (w.mean() + 1e-8)
    w[0] *= boost0
    w = w / (w.mean() + 1e-8)
    return w.astype(np.float32)


def recall_of_class(y_true, y_pred, cls=0):
    mask = (y_true == cls)
    if mask.sum() == 0:
        return 0.0
    return (y_pred[mask] == cls).mean()


# =========================
# 1) Graph Construction
# =========================
def build_hybrid_edge_index(x_train: np.ndarray, top_k=12):
    print(f"[Graph] Building Hybrid Graph on Train Set {x_train.shape} ...")
    corr = np.corrcoef(x_train.T)
    abs_corr = np.abs(corr)
    num_nodes = x_train.shape[1]
    edges = set()

    for i in range(num_nodes):
        top_idx = np.argsort(abs_corr[i])[-(top_k + 1):][::-1]
        neigh = [j for j in top_idx if j != i][:top_k]
        for j in neigh:
            edges.add((i, j))
            edges.add((j, i))

    for i in range(num_nodes - 1):
        edges.add((i, i + 1))
        edges.add((i + 1, i))

    for i in range(num_nodes):
        edges.add((i, i))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    print(f"[Graph] nodes={num_nodes}, edges={edge_index.shape[1]}")
    return edge_index


def spectrum_to_graph_sg3(x_128: np.ndarray, y_int: int, edge_index: torch.Tensor,
                          use_pos: bool = True) -> Data:
    s = x_128.astype(np.float32)
    win = 15
    poly = 3
    s0 = savgol_filter(s, window_length=win, polyorder=poly, deriv=0).astype(np.float32)
    d1 = savgol_filter(s, window_length=win, polyorder=poly, deriv=1).astype(np.float32)
    d2 = savgol_filter(s, window_length=win, polyorder=poly, deriv=2).astype(np.float32)

    feats = [s0, d1 * 10.0, d2 * 100.0]
    if use_pos:
        pos = np.linspace(0.0, 1.0, num=s.shape[0], dtype=np.float32)
        feats.append(pos)

    node_feat = np.stack(feats, axis=1).astype(np.float32)

    return Data(
        x=torch.from_numpy(node_feat),
        edge_index=edge_index,
        y=torch.tensor(y_int, dtype=torch.long),
    )


def make_global_features_from_raw(X_raw: np.ndarray):
    mean_ = X_raw.mean(axis=1)
    std_ = X_raw.std(axis=1)
    ptp_ = X_raw.max(axis=1) - X_raw.min(axis=1)

    n = X_raw.shape[1]
    a = n // 3
    seg1 = X_raw[:, :a].mean(axis=1)
    seg2 = X_raw[:, a:2 * a].mean(axis=1)
    seg3 = X_raw[:, 2 * a:].mean(axis=1)

    G = np.stack([mean_, std_, ptp_, seg1, seg2, seg3], axis=1).astype(np.float32)
    return G


def standardize_by_train(G_train, G_val, G_test):
    mu = G_train.mean(axis=0, keepdims=True)
    sd = G_train.std(axis=0, keepdims=True) + 1e-8
    return (G_train - mu) / sd, (G_val - mu) / sd, (G_test - mu) / sd


# =========================
# 2) GAT Model
# =========================
class GATEncoder(nn.Module):
    def __init__(self, in_dim=4, hid=64, heads=4, dropout=0.5, num_classes=5):
        super().__init__()
        self.dropout = dropout
        self.hid = hid

        self.enc = nn.Linear(in_dim, hid)

        self.gat1 = GATConv(hid, hid, heads=heads, concat=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hid * heads)
        self.res_proj1 = nn.Linear(hid, hid * heads)

        self.gat2 = GATConv(hid * heads, hid, heads=1, concat=False, dropout=dropout)
        self.norm2 = nn.LayerNorm(hid)
        self.res_proj2 = nn.Linear(hid * heads, hid)

        self.head = nn.Sequential(
            nn.Linear(hid * 2, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, num_classes)
        )

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.enc(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        identity1 = self.res_proj1(x)
        x = self.gat1(x, edge_index)
        x = self.norm1(x)
        x = x + identity1
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        identity2 = self.res_proj2(x)
        x = self.gat2(x, edge_index)
        x = self.norm2(x)
        x = x + identity2
        x = F.relu(x)

        g_mean = global_mean_pool(x, batch)
        g_max = global_max_pool(x, batch)
        emb = torch.cat([g_mean, g_max], dim=1)

        logits = self.head(emb)

        return emb, logits


def evaluate_loader_logits(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _, logits = model(data)
            pred = logits.argmax(dim=1).cpu().numpy()
            y = data.y.cpu().numpy()
            preds.append(pred)
            ys.append(y)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    bal = balanced_accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    r0 = recall_of_class(y_true, y_pred, cls=0)
    return bal, mf1, r0


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embs, ys = [], []
    for data in loader:
        data = data.to(device)
        emb, _ = model(data)
        embs.append(emb.cpu().numpy())
        ys.append(data.y.cpu().numpy())
    return np.concatenate(embs, axis=0), np.concatenate(ys, axis=0)


# =========================
# 3) ✅ 核心：下游导向的训练
# =========================
def train_encoder_downstream_selection(
        model, train_loader_train, train_loader_embed, val_loader, device,
        class_weights_t, Gtr, Gval, ytr, yval,
        epochs=50, lr=1e-3, label_smoothing=0.1,
        ckpt_path="best_gat_downstream.pt",
        check_interval=5):
    """
    ✅ 用下游LightGBM性能选择GAT checkpoint
    这才是two-stage learning的正确方式
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=label_smoothing)

    best_downstream_acc = -1.0
    best_ep = None

    lgb_class_weight = {i: float(class_weights_t[i].cpu()) for i in range(5)}

    print("\n" + "=" * 70)
    print(" GAT Training: Downstream-Guided Checkpoint Selection ".center(70))
    print("=" * 70)
    print(f"Strategy: Select checkpoint by LightGBM Val performance (not GAT metrics)")
    print(f"Check interval: Every {check_interval} epochs")
    print("=" * 70)

    for ep in range(1, epochs + 1):
        # === Train GAT ===
        model.train()
        total_loss = 0.0
        seen = 0
        for data in train_loader_train:
            data = data.to(device)
            optimizer.zero_grad()
            _, logits = model(data)
            loss = criterion(logits, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            seen += data.num_graphs

        scheduler.step()
        train_loss = total_loss / max(seen, 1)

        # === GAT validation metrics (just for monitoring) ===
        bal, mf1, r0 = evaluate_loader_logits(model, val_loader, device)

        # === Downstream evaluation ===
        if ep % check_interval == 0 or ep == 1 or ep == epochs:
            print(f"\n{'─' * 70}")
            print(f"Epoch {ep:03d} | GAT Loss={train_loss:.4f}")
            print(f"  GAT Val Metrics: r0={r0:.4f} mF1={mf1:.4f} bal={bal:.4f}")
            print(f"  → Evaluating downstream LightGBM performance...")

            # Extract embeddings
            model.eval()
            with torch.no_grad():
                Xtr_emb, _ = extract_embeddings(model, train_loader_embed, device)
                Xval_emb, _ = extract_embeddings(model, val_loader, device)

            # Combine with global features
            Xtr_lgb = np.hstack([Xtr_emb, Gtr])
            Xval_lgb = np.hstack([Xval_emb, Gval])

            # Quick LightGBM training
            lgb_quick = lgb.LGBMClassifier(
                objective="multiclass", num_class=5,
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                class_weight=lgb_class_weight
            )
            lgb_quick.fit(Xtr_lgb, ytr)

            # Evaluate on Val
            pred_val = lgb_quick.predict(Xval_lgb)
            downstream_acc = balanced_accuracy_score(yval, pred_val)
            downstream_mf1 = f1_score(yval, pred_val, average='macro')
            downstream_r0 = recall_of_class(yval, pred_val, cls=0)

            print(f"  → Downstream Val: BalAcc={downstream_acc:.4f} mF1={downstream_mf1:.4f} r0={downstream_r0:.4f}")

            # Save if best
            if downstream_acc > best_downstream_acc:
                best_downstream_acc = downstream_acc
                best_ep = ep
                torch.save(model.state_dict(), ckpt_path)
                print(f"  ✅ NEW BEST! Saved checkpoint to {ckpt_path}")
            else:
                print(f"  ⚪ Not better than best ({best_downstream_acc:.4f} at Ep {best_ep})")

        elif ep % 10 == 0:
            print(f"Ep {ep:03d} | Loss={train_loss:.4f} | GAT Val: r0={r0:.4f} mF1={mf1:.4f} bal={bal:.4f}")

    print(f"\n{'=' * 70}")
    print(f"🏆 Best Checkpoint: Epoch {best_ep}")
    print(f"   Downstream Val BalAcc: {best_downstream_acc:.4f}")
    print(f"{'=' * 70}\n")

    return best_ep, best_downstream_acc


# =========================
# 4) Tau0 Tuning
# =========================
def apply_tau0(proba: np.ndarray, tau0: float):
    pred = proba.argmax(axis=1)
    pred = pred.copy()
    pred[proba[:, 0] > tau0] = 0
    return pred


def tune_tau0_on_val_strict(lgbm_model, Xval, yval, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.80, 76)

    proba_val = lgbm_model.predict_proba(Xval)
    best = (-1.0, -1.0, -1.0, None)

    for tau in grid:
        pred = apply_tau0(proba_val, tau)
        bal = balanced_accuracy_score(yval, pred)
        mf1 = f1_score(yval, pred, average="macro", zero_division=0)
        r0 = recall_of_class(yval, pred, cls=0)
        cur = (mf1, bal, r0, float(tau))

        if cur[:3] > best[:3]:
            best = cur

    print(f"[Tau0 Tuning] Best tau={best[3]:.3f}: mF1={best[0]:.4f} bal={best[1]:.4f} r0={best[2]:.4f}")
    return best[3]


# =========================
# 5) Complete Pipeline
# =========================
def run_full_pipeline_downstream(base_config):
    """完整pipeline，用下游性能选checkpoint"""

    seed_everything(42)

    X_raw_all = base_config['X_raw_all']
    X_all = base_config['X_all']
    y_all = base_config['y_all']
    train_idx = base_config['train_idx']
    val_idx = base_config['val_idx']
    test_idx = base_config['test_idx']
    class_w = base_config['class_w']
    device = base_config['device']
    EDGE_INDEX = base_config['EDGE_INDEX']

    # Prepare global features FIRST
    G_all = make_global_features_from_raw(X_raw_all)
    Gtr = G_all[train_idx]
    Gval = G_all[val_idx]
    Gte = G_all[test_idx]
    Gtr, Gval, Gte = standardize_by_train(Gtr, Gval, Gte)

    ytr = y_all[train_idx]
    yval = y_all[val_idx]
    yte = y_all[test_idx]

    # Build graphs
    print("\n[Building PyG graphs...]")
    graphs_train = [spectrum_to_graph_sg3(X_all[i], int(y_all[i]), EDGE_INDEX, use_pos=True) for i in train_idx]
    graphs_val = [spectrum_to_graph_sg3(X_all[i], int(y_all[i]), EDGE_INDEX, use_pos=True) for i in val_idx]
    graphs_test = [spectrum_to_graph_sg3(X_all[i], int(y_all[i]), EDGE_INDEX, use_pos=True) for i in test_idx]

    # Loaders
    sw = class_w[y_all[train_idx]].astype(np.float64)
    sampler = WeightedRandomSampler(weights=sw, num_samples=len(sw), replacement=True)
    train_loader_train = DataLoader(graphs_train, batch_size=128, sampler=sampler, shuffle=False)
    train_loader_embed = DataLoader(graphs_train, batch_size=256, shuffle=False)
    val_loader = DataLoader(graphs_val, batch_size=256, shuffle=False)
    test_loader = DataLoader(graphs_test, batch_size=256, shuffle=False)

    # ===== Stage 1: Train GAT with downstream selection =====
    class_weights_t = torch.tensor(class_w, dtype=torch.float32, device=device)
    model = GATEncoder(in_dim=4, hid=64, heads=4, dropout=0.5, num_classes=5).to(device)

    ckpt_path = "best_gat_downstream.pt"
    saved_ep, best_val_acc = train_encoder_downstream_selection(
        model, train_loader_train, train_loader_embed, val_loader, device,
        class_weights_t, Gtr, Gval, ytr, yval,
        epochs=50, lr=1e-3, label_smoothing=0.1,
        ckpt_path=ckpt_path,
        check_interval=5
    )

    # Load best checkpoint
    print(f"\n[Loading best checkpoint: Epoch {saved_ep}...]")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ===== Stage 2: Extract final embeddings =====
    print("[Extracting final embeddings...]")
    Xtr_emb, _ = extract_embeddings(model, train_loader_embed, device)
    Xval_emb, _ = extract_embeddings(model, val_loader, device)
    Xte_emb, _ = extract_embeddings(model, test_loader, device)

    Xtr_lgb = np.hstack([Xtr_emb, Gtr])
    Xval_lgb = np.hstack([Xval_emb, Gval])
    Xte_lgb = np.hstack([Xte_emb, Gte])

    print(f"[Feature dimensions] Embedding: {Xtr_emb.shape[1]}, Global: {Gtr.shape[1]}, Total: {Xtr_lgb.shape[1]}")

    # ===== Stage 3: Train LightGBM =====
    lgb_class_weight = {i: float(class_w[i]) for i in range(5)}

    print("\n[Training LightGBM on Train set...]")
    lgbm_train = lgb.LGBMClassifier(
        objective="multiclass", num_class=5, n_estimators=3000,
        learning_rate=0.01, num_leaves=31, max_depth=-1,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=5.0, random_state=42,
        n_jobs=-1, verbosity=-1, class_weight=lgb_class_weight
    )

    lgbm_train.fit(Xtr_lgb, ytr, eval_set=[(Xval_lgb, yval)],
                   eval_metric="multi_logloss",
                   callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])

    best_iter = getattr(lgbm_train, "best_iteration_", 500)
    print(f"[LightGBM] Best iteration: {best_iter}")

    # ===== Stage 4: Tune tau0 =====
    print(f"\n[Tuning tau0 on Val set...]")
    tau0 = tune_tau0_on_val_strict(lgbm_train, Xval_lgb, yval)

    # ===== Stage 5: Refit on Train+Val =====
    print(f"\n[Refitting LightGBM on Train+Val with {best_iter} trees...]")
    X_trainval = np.vstack([Xtr_lgb, Xval_lgb])
    y_trainval = np.concatenate([ytr, yval])

    lgbm_final = lgb.LGBMClassifier(
        objective="multiclass", num_class=5, n_estimators=int(best_iter),
        learning_rate=0.01, num_leaves=31, max_depth=-1,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=5.0, random_state=42,
        n_jobs=-1, verbosity=-1, class_weight=lgb_class_weight
    )
    lgbm_final.fit(X_trainval, y_trainval)

    # ===== Stage 6: Final blind test =====
    print("\n" + "=" * 70)
    print(" FINAL BLIND TEST (Acquisition 6) ".center(70))
    print("=" * 70)
    print(f"Using fixed tau0={tau0:.3f} (tuned on Val only)")
    print("=" * 70)

    proba_te = lgbm_final.predict_proba(Xte_lgb)
    pred_te = apply_tau0(proba_te, tau0)

    test_bal = balanced_accuracy_score(yte, pred_te)
    test_mf1 = f1_score(yte, pred_te, average="macro")

    print(f"\n{'🎯 PERFORMANCE METRICS':^70}")
    print("─" * 70)
    print(f"  Test Balanced Accuracy: {test_bal:.4f}")
    print(f"  Test Macro F1-Score:    {test_mf1:.4f}")
    print("─" * 70)

    print(f"\n{'CONFUSION MATRIX':^70}")
    cm = confusion_matrix(yte, pred_te)
    print(cm)

    print(f"\n{'PER-CLASS RECALL':^70}")
    classes_cc = ["0%", "5%", "10%", "25%", "50%"]
    for cls_idx, cls_name in enumerate(classes_cc):
        rec = recall_of_class(yte, pred_te, cls=cls_idx)
        print(f"  {cls_name:>4s}: {rec:.4f}")

    print("\n" + "=" * 70)
    print(f"\n{'DETAILED CLASSIFICATION REPORT':^70}")
    print("=" * 70)
    print(classification_report(yte, pred_te, target_names=classes_cc, digits=4))
    print("=" * 70)

    # Save model
    try:
        lgbm_final.booster_.save_model("lgbm_final_downstream.txt")
        print("\n✅ Saved LightGBM model to: lgbm_final_downstream.txt")
    except:
        pass

    return saved_ep, test_bal, test_mf1


# =========================
# 6) Main Function
# =========================
def main():
    seed_everything(42)

    PATH = r"adulteration_dataset_26_08_2021.csv"

    if not os.path.exists(PATH):
        print(f"❌ Error: File not found at {PATH}")
        return

    print("=" * 70)
    print(" GraphLight-Plus: Downstream-Guided Training ".center(70))
    print("=" * 70)
    print(f"\nLoading data from:\n  {PATH}")

    df = pd.read_csv(PATH)
    df["Concentration_Class"] = df["Concentration_Class"].astype(str)

    keep = {"0", "5", "10", "25", "50"}
    df = df[df["Concentration_Class"].isin(keep)].copy()
    df = df.reset_index(drop=True)

    spec_cols = [c for c in df.columns if str(c).endswith("nm")]
    spec_cols.sort(key=lambda x: float(str(x).replace("nm", "")))

    X_raw_all = df[spec_cols].to_numpy(np.float32)
    X_all = snv(X_raw_all)

    classes_cc = ["0", "5", "10", "25", "50"]
    cc2id = {c: i for i, c in enumerate(classes_cc)}
    y_all = df["Concentration_Class"].map(cc2id).to_numpy(np.int64)

    acq = df["Acquisition"].to_numpy()
    train_idx = np.where(np.isin(acq, [1, 2, 3, 4]))[0]
    val_idx = np.where(np.isin(acq, [5]))[0]
    test_idx = np.where(np.isin(acq, [6]))[0]

    print(f"\n{'DATA SPLIT':^70}")
    print("─" * 70)
    print(f"  Train (Acq 1-4): {len(train_idx):>4d} samples")
    print(f"  Val   (Acq 5):   {len(val_idx):>4d} samples")
    print(f"  Test  (Acq 6):   {len(test_idx):>4d} samples (BLIND)")
    print("─" * 70)

    y_train_only = y_all[train_idx]
    class_w = compute_class_weights(y_train_only, num_classes=5, boost0=2.0, mode="effective")
    print(f"\nClass weights: {class_w}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    EDGE_INDEX = build_hybrid_edge_index(X_all[train_idx], top_k=12)

    base_config = {
        'X_raw_all': X_raw_all,
        'X_all': X_all,
        'y_all': y_all,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'class_w': class_w,
        'device': device,
        'EDGE_INDEX': EDGE_INDEX
    }

    # ✅ Run pipeline
    saved_ep, test_bal, test_mf1 = run_full_pipeline_downstream(base_config)

    print("\n" + "=" * 70)
    print(" 🏆 FINAL SUMMARY ".center(70))
    print("=" * 70)
    print(f"  Selected GAT Checkpoint: Epoch {saved_ep}")
    print(f"  Test Balanced Accuracy:  {test_bal:.4f}")
    print(f"  Test Macro F1-Score:     {test_mf1:.4f}")
    print("=" * 70)
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
