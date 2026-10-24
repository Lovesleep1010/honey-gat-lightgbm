import lightgbm as lgb
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
import torch

import baseline_config as protocol
from model import evaluate_loader_logits, extract_embeddings, recall_of_class


def train_encoder_downstream_selection(
        model, train_loader_train, train_loader_embed, val_loader, device,
        class_weights_t, ytr, yval, specification,
        epochs=50, lr=1e-3, ckpt_path="best_encoder_downstream.pt",
        check_interval=1, train_epochs=None, n_jobs=-1, gbm_seed=42, log_callback=None,
        early_stopping_patience=5):
    if check_interval != 1:
        raise ValueError("Early stopping requires validation after every epoch")
    if early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = protocol.criterion(class_weights_t, specification)
    best_ba, best_epoch = -1.0, None
    epochs_without_improvement = 0
    lgb_weights = {i: float(class_weights_t[i].cpu()) for i in range(5)}
    actual_epochs = epochs if train_epochs is None else train_epochs
    for epoch in range(1, actual_epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
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
        head_ba, head_f1, head_r0 = evaluate_loader_logits(model, val_loader, device)
        candidate = None
        if epoch == 1 or epoch % check_interval == 0 or epoch == actual_epochs:
            etr, labels_tr = extract_embeddings(model, train_loader_embed, device)
            ev, labels_val = extract_embeddings(model, val_loader, device)
            if not (np.array_equal(labels_tr, ytr) and np.array_equal(labels_val, yval)):
                raise RuntimeError("Probe embedding rows were reordered")

            probe = lgb.LGBMClassifier(
                objective="multiclass", num_class=5, n_estimators=500,
                learning_rate=0.05, num_leaves=31, random_state=gbm_seed,
                n_jobs=n_jobs, verbosity=-1, class_weight=lgb_weights)
            probe.fit(etr, ytr)
            prediction = probe.predict(ev)
            ba = balanced_accuracy_score(yval, prediction)
            candidate = dict(epoch=epoch, probe_val_ba=float(ba),
                             probe_val_f1=float(f1_score(yval, prediction, average="macro")),
                             probe_val_recall0=float(recall_of_class(yval, prediction, cls=0)))
            if ba > best_ba:
                best_ba, best_epoch = ba, epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                epochs_without_improvement += 1
            print(f"[Selection] epoch={epoch}, downstream BA={ba:.6f}, selected={best_epoch}", flush=True)
        stop_training = epochs_without_improvement >= early_stopping_patience
        if log_callback is not None:
            log_callback(dict(epoch=epoch, loss=float(total_loss / max(seen, 1)),
                              head_val_ba=float(head_ba), head_val_f1=float(head_f1),
                              head_val_recall0=float(head_r0), candidate=candidate,
                              selected_epoch_so_far=best_epoch,
                              epochs_without_improvement=epochs_without_improvement,
                              early_stopping_triggered=stop_training))
        if stop_training:
            print(f"[Early stopping] epoch={epoch}, best epoch={best_epoch}", flush=True)
            break
    return best_epoch, best_ba
