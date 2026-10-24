"""Protocol/integration checks; real training uses only a tiny synthetic fixture.

No honey training or manuscript-result experiment is launched by these tests.
"""
import ast
import copy
import contextlib
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from torch.utils.data import RandomSampler
from torch_geometric.data import Batch, Data

import model as core
import models as models
import training as training
import pipeline as shared
import compare as exp
import config as launcher
import baseline_config as protocol


class ComparatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_class_definitions_and_provenance(self):
        current = ast.parse((exp.HERE / "models.py").read_text(encoding="utf-8"))
        definitions = {n.name: n for n in current.body if isinstance(n, ast.ClassDef)}
        for record in shared.read_json(exp.HERE / "architecture_sources.json"):
            snapshot = exp.HERE / record["class_snapshot"]
            self.assertEqual(shared.sha(snapshot), record["class_snapshot_sha256"])
            old = ast.parse(snapshot.read_text(encoding="utf-8")).body[0]
            self.assertEqual(ast.dump(old), ast.dump(definitions[record["class_name"]]))
            self.assertEqual(shared.sha(exp.HERE / record["original_snapshot"]), record["source_file_sha256"])
            self.assertEqual(models.model_spec(record["architecture"])["in_dim"], 3)
        # The GIN matching Table 1's recorded BA/F1 is main_GIN_1.py, not main_GIN.py.
        self.assertEqual(models.model_spec("gin")["source_file"], "main_GIN_1.py")
        self.assertEqual(models.model_spec("gin")["hidden"], 64)

    def test_features_class_weights_losses_and_gbm_match_archived_sources(self):
        rng = np.random.default_rng(59)
        raw = rng.normal(size=(5, 128)).astype(np.float32)
        labels = np.arange(5)
        edges = torch.tensor([[0, 1], [1, 0]])
        logits = torch.from_numpy(rng.normal(size=(5, 5)).astype(np.float32))
        for record in shared.read_json(exp.HERE / "architecture_sources.json"):
            architecture = record["architecture"]
            specification = protocol.settings(architecture)
            tree = ast.parse((exp.HERE / record["original_snapshot"]).read_text(encoding="utf-8"))
            # Load only pure feature/weight/loss definitions. Never execute main or source I/O.
            definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                           and n.name in ("spectrum_to_graph_sg3", "compute_class_weights", "FocalLoss")]
            namespace = dict(np=np, torch=torch, nn=torch.nn, F=torch.nn.functional,
                             Data=Data, savgol_filter=savgol_filter)
            exec(compile(ast.Module(body=definitions, type_ignores=[]), record["source_file"], "exec"), namespace)
            graphs = protocol.make_graphs(raw, labels, edges, specification)
            for graph, spectrum, label in zip(graphs, core.snv(raw), labels):
                original = namespace["spectrum_to_graph_sg3"](spectrum, int(label), edges)
                self.assertEqual(tuple(graph.x.shape), (128, 3))
                torch.testing.assert_close(graph.x, original.x, rtol=0, atol=0)
            y = np.repeat(labels, [1, 3, 5, 7, 9])
            weights = core.compute_class_weights(y, boost0=specification["boost0"])
            np.testing.assert_array_equal(weights, namespace["compute_class_weights"](y, boost0=specification["boost0"]))
            weights = torch.from_numpy(weights)
            if architecture in ("graphsage", "gcn"):
                original_loss = namespace["FocalLoss"](alpha=weights, gamma=3.0)(logits, torch.from_numpy(labels))
            else:
                ce_call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                               and isinstance(n.func, ast.Attribute) and n.func.attr == "CrossEntropyLoss")
                smoothing = next(ast.literal_eval(k.value) for k in ce_call.keywords if k.arg == "label_smoothing")
                original_loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(labels),
                                                                 weight=weights, label_smoothing=smoothing)
            torch.testing.assert_close(protocol.criterion(weights, specification)(logits, torch.from_numpy(labels)),
                                       original_loss, rtol=0, atol=0)
            old_gbm = min((n for n in ast.walk(tree) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Attribute) and n.func.attr == "LGBMClassifier"),
                          key=lambda n: n.lineno)
            params = protocol.gbm_params(specification, dict(gbm_seed=42, threads=2), weights)
            for keyword in old_gbm.keywords:
                if keyword.arg not in ("class_weight", "n_jobs", "verbosity"):
                    self.assertEqual(params[keyword.arg], ast.literal_eval(keyword.value))

    def test_tree_selection_precedes_refit_without_added_threshold(self):
        labels = np.arange(5)
        xtr, xval = np.zeros((5, 128)), np.ones((5, 128))
        calls, parameters = [], []
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)

            class FakeBooster:
                def save_model(self, filename):
                    Path(filename).write_text("fixture")

            class FakeGBM:
                def __init__(self, **kwargs):
                    self.index = len(parameters)
                    parameters.append(kwargs)
                    self.best_iteration_ = 7
                    self.booster_ = FakeBooster()
                    if self.index == 1:
                        selected = shared.read_json(folder / "gbm_selection.json")
                        self_test.assertEqual(selected["gbm_best_iteration"], 7)
                        self_test.assertIsNone(selected["tau0"])
                def fit(self, x, y, **kwargs):
                    calls.append((self.index, len(x), kwargs))
                    return self
                def predict_proba(self, x):
                    if self.index != 0:
                        raise AssertionError("Refit must not be evaluated or tuned on Acq5")
                    return np.eye(5)

            self_test = self
            cfg = dict(gbm_seed=42, threads=2, comparator_settings={"gin": protocol.settings("gin")})
            with patch.object(protocol.lgb, "LGBMClassifier", FakeGBM), \
                 patch.object(core, "tune_tau0_on_val_strict", side_effect=AssertionError("No added threshold")):
                info = protocol.fit_gbm(folder, xtr, labels, xval, labels, np.ones(5), cfg, {"architecture": "gin"})
            self.assertEqual([(i, n) for i, n, _ in calls], [(0, 5), (1, 10)])
            self.assertIn("eval_set", calls[0][2])
            self.assertFalse(calls[1][2])
            expected = dict(parameters[0], n_estimators=7)
            self.assertEqual(parameters[1], expected)
            self.assertEqual(parameters[1]["min_child_samples"], 30)
            self.assertTrue(info["tree_count_locked_before_refit"])
            self.assertFalse(info["threshold_tuned"])
            np.testing.assert_array_equal(protocol.predict_classes(np.eye(5), info), labels)

    def test_all_architectures_forward_backward_and_checkpoint_roundtrip(self):
        core.seed_everything(42)
        edges = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3], [1, 2, 3, 0, 0, 1, 2, 3]])
        batch = Batch.from_data_list([Data(x=torch.randn(4, 3), edge_index=edges, y=torch.tensor(i))
                                     for i in range(5)])
        for architecture in models.MODEL_ORDER:
            with self.subTest(architecture=architecture):
                model = models.build_encoder(architecture)
                model.train()
                embedding, logits = model(batch)
                self.assertEqual(tuple(embedding.shape), (5, models.model_spec(architecture)["embedding_dim"]))
                self.assertEqual(tuple(logits.shape), (5, 5))
                loss = torch.nn.functional.cross_entropy(logits, batch.y)
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                model.eval()
                with torch.no_grad():
                    expected, _ = model(batch)
                    restored = models.build_encoder(architecture)
                    restored.load_state_dict(model.state_dict())
                    restored.eval()
                    actual, _ = restored(batch)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_plain_gat_has_no_residual_projection(self):
        ordinary = models.build_encoder("gat")
        residual = core.GATEncoder(in_dim=4, hid=64, heads=4, dropout=0.5, num_classes=5)
        self.assertFalse(any("res_proj" in key for key in ordinary.state_dict()))
        self.assertTrue(any("res_proj" in key for key in residual.state_dict()))
        with self.assertRaises(RuntimeError):
            ordinary.load_state_dict(residual.state_dict())
        with self.assertRaises(ValueError):
            models.build_encoder("residual_gat")

    def test_launcher_scope_and_defaults(self):
        args = exp.resolve_arguments(launcher.build_argv())
        self.assertEqual(args.models, list(models.MODEL_ORDER))
        self.assertEqual(args.seeds, exp.DEFAULT_SEEDS)
        self.assertEqual((args.epochs, args.scheduler_epochs), (5, None))
        self.assertEqual(len(exp.planned_tasks(exp.make_config(args))), 20)
        self.assertTrue(args.data.is_absolute())
        self.assertTrue(args.output.is_absolute())
        self.assertEqual(args.output.name, "results_table1_gnn_original_5epochs")
        cfg = exp.make_config(args)
        self.assertEqual([cfg["comparator_settings"][name]["scheduler_epochs"] for name in models.MODEL_ORDER],
                         [60, 60, 50, 50])
        self.assertFalse(cfg["preprocessing"]["global_features"])
        self.assertFalse(cfg["preprocessing"]["position_feature"])

    def test_checkpoint_selection_follows_downstream_ba_despite_head_scores(self):
        labels = np.arange(5)
        probe_predictions = iter([np.zeros(5, dtype=int), labels])
        parameters, records, dimensions = [], [], []

        class FakeGBM:
            def __init__(self, **kwargs):
                parameters.append(kwargs)
            def fit(self, x, y, **kwargs):
                dimensions.append(x.shape[1])
                return self
            def predict(self, *args):
                return next(probe_predictions)

        head_scores = iter([(1., 1., 1.), (0., 0., 0.), (0., 0., 0.), (0., 0., 0.), (0., 0., 0.)])
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(training.lgb, "LGBMClassifier", FakeGBM), \
             patch.object(training, "evaluate_loader_logits", side_effect=lambda *a: next(head_scores)), \
             patch.object(training, "extract_embeddings", return_value=(np.ones((5, 128)), labels)):
            epoch, ba = training.train_encoder_downstream_selection(
                torch.nn.Linear(1, 1), [], [], [], "cpu", torch.ones(5),
                labels, labels, protocol.settings("gin"),
                epochs=50, train_epochs=5, ckpt_path=str(Path(tmp) / "encoder.pt"),
                gbm_seed=42, n_jobs=2, log_callback=records.append)
        self.assertEqual((epoch, ba), (5, 1.))
        self.assertEqual(dimensions, [128, 128])
        self.assertEqual([r["epoch"] for r in records if r["candidate"]], [1, 5])
        self.assertTrue(all(p["n_estimators"] == 500 and p["learning_rate"] == .05
                            and p["random_state"] == 42 for p in parameters))

    def test_resume_rejects_changed_configuration_and_task_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = exp.resolve_arguments(["--output", str(Path(tmp) / "out"), "--device", "cpu", "--resume"])
            cfg = exp.make_config(args)
            exp.configure_protocol(args, cfg)
            exp.configure_protocol(args, cfg)  # JSON roundtrip must preserve exact configuration.
            for key, value in [("epochs", 50), ("dataset_sha256", "changed"),
                               ("models", ["gat"]), ("source_sha256", {})]:
                changed = dict(cfg, **{key: value})
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    exp.configure_protocol(args, changed)
            folder = args.output / "tasks" / exp.task_id("gat", 42)
            folder.mkdir(parents=True)
            (folder / "fixture").write_text("fixture")
            info = dict(task_id=exp.task_id("gat", 42), architecture="gcn", seed=42,
                        model_spec=cfg["model_specs"]["gat"], k=12, n_bands=128,
                        protocol_sha256=shared.sha(args.output / "protocol.json"))
            shared.seal_task(folder, info, ["fixture"])
            with self.assertRaisesRegex(RuntimeError, "architecture"):
                exp.read_task(args.output, cfg, "gat", 42)

    def test_incomplete_plan_is_rejected_before_test_data_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = exp.resolve_arguments(["--output", str(Path(tmp) / "out"), "--device", "cpu"])
            cfg = exp.make_config(args)
            exp.configure_protocol(args, cfg)
            shared.write_json(args.output / exp.PLAN_FILE, dict(
                protocol_sha256=shared.sha(args.output / "protocol.json"),
                all_fits_locked=True, task_locks={}))
            with patch.object(shared, "load_frame", side_effect=AssertionError("Test must remain inaccessible")):
                with self.assertRaises(RuntimeError):
                    exp.evaluate_all(args.output, cfg, args.data)

    def test_summaries_keep_every_requested_seed(self):
        cfg = dict(models=["gat"], seeds=[42, 123])
        rows = [dict(task_id=exp.task_id("gat", seed), architecture="gat", seed=seed,
                     parameter_count=10, embedding_dim=128,
                     accuracy=value, balanced_accuracy=value, macro_f1=value, recall0=value)
                for seed, value in [(42, .8), (123, 1.)]]
        with tempfile.TemporaryDirectory() as tmp:
            exp.summarize(Path(tmp), cfg, rows)
            summary = shared.read_json(Path(tmp) / "table1_gnn_summary.json")
            self.assertEqual(summary["summary"][0]["n_runs"], 2)
            self.assertAlmostEqual(summary["summary"][0]["balanced_accuracy_mean"], .9)
            self.assertAlmostEqual(summary["summary"][0]["balanced_accuracy_sd"], np.sqrt(.02))
            with self.assertRaises(RuntimeError):
                exp.summarize(Path(tmp), cfg, rows[:1])
            with self.assertRaises(RuntimeError):
                exp.summarize(Path(tmp), cfg, rows + rows[:1])

    def test_real_synthetic_lifecycle_all_four_architectures(self):
        # One actual epoch per comparator on 20 synthetic train graphs, NOT honey.
        # This tests actual PyG training, real LightGBM save/load, all four encoder
        # checkpoint types, locked evaluation, resume, and summary generation.
        rng = np.random.default_rng(6)
        columns = [f"{400 + i * 5:.2f}nm" for i in range(128)]
        records = []
        for acq in (1, 5, 6):
            for label in shared.CLASSES:
                for _ in range(4):
                    raw = rng.normal(size=128) + label / 10
                    records.append(dict(Acquisition=acq, Concentration_Class=label,
                                        Brand="fixture", Class="fixture", **dict(zip(columns, raw))))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp) / "results"
            data = Path(tmp) / "synthetic.csv"
            pd.DataFrame(records).to_csv(data, index=False)
            args = exp.resolve_arguments(["--data", str(data), "--output", str(root), "--epochs", "1",
                                          "--seeds", "7", "--threads", "2", "--device", "cpu", "--resume"])
            cfg = exp.make_config(args)
            exp.configure_protocol(args, cfg)
            # Fitting functions receive only TrainVal; no function opening the
            # test split can be called during this stage.
            split_fn = shared.get_split
            def guard_split(frame, cols, acquisitions):
                if 6 in acquisitions:
                    raise AssertionError("Acq6 is forbidden during fitting")
                return split_fn(frame, cols, acquisitions)
            trainer = training.train_encoder_downstream_selection
            def checked_trainer(model, train, embed, val, *args, **kwargs):
                self.assertIsInstance(train.sampler, RandomSampler)
                self.assertFalse(train.sampler.replacement)
                self.assertEqual(train.dataset[0].x.shape[1], 3)
                return trainer(model, train, embed, val, *args, **kwargs)
            with patch.object(shared, "get_split", side_effect=guard_split), \
                 patch.object(shared, "global_stats", side_effect=AssertionError("No global statistics")), \
                 patch.object(training, "train_encoder_downstream_selection", side_effect=checked_trainer):
                exp.fit_all(root, cfg, data)
            self.assertFalse(list(root.glob("tasks/*/test_predictions.npz")))
            infos = exp.locked_plan(root, cfg)
            self.assertEqual(len(infos), 4)
            for info in infos:
                self.assertFalse(info["test_used_for_selection"])
                self.assertEqual(info["encoder_fit_acquisitions"], [1, 2, 3, 4])
                self.assertEqual(info["selected_epoch"], 1)
                self.assertTrue(info["tree_count_locked_before_refit"])
                self.assertEqual(info["prediction_rule"], "argmax")
                self.assertIsNone(info["tau0"])
                self.assertEqual(info["embedding_dim"], info["downstream_input_dim"])
            exp.evaluate_all(root, cfg, data)
            summary = pd.read_csv(root / "table1_gnn_summary.csv")
            self.assertEqual(len(summary), 4)
            self.assertTrue(summary.balanced_accuracy_sd.isna().all())
            self.assertEqual(len(list(root.glob("tasks/*/test_predictions.npz"))), 4)
            # A normal resume must not retrain, re-extract or choose new settings.
            with patch.object(training, "train_encoder_downstream_selection", side_effect=AssertionError("No retraining")), \
                 patch.object(core, "extract_embeddings", side_effect=AssertionError("No repeated inference")):
                exp.fit_all(root, cfg, data)
                exp.evaluate_all(root, cfg, data)
            # Recalculation detects edited metric files, not just edited weights.
            first = infos[0]
            folder = root / "tasks" / first["task_id"]
            saved = shared.read_json(folder / "test_metrics.json")
            saved["metrics"]["balanced_accuracy"] += .01
            shared.write_json(folder / "test_metrics.json", saved)
            with self.assertRaisesRegex(RuntimeError, "metrics disagree"):
                exp.evaluate_all(root, cfg, data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
