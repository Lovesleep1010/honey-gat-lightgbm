"""Fast protocol checks: no real experiment or GAT training is launched."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import model as core
import pipeline as exp


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_bundled_data_matches_confirmed_manuscript_population(self):
        df, cols, _ = exp.load_frame(exp.HERE.parents[1] / "data" / "adulteration_dataset_26_08_2021.csv")
        self.assertEqual(len(df), 8400)
        self.assertEqual(df.groupby("Acquisition").size().to_dict(), {i: 1400 for i in range(1, 7)})
        self.assertEqual(df.attrs["excluded_c4_rows"], 125)
        self.assertEqual(len(exp.get_split(df, cols, [1, 2, 3, 4]).y), 5600)
        self.assertEqual(len(exp.get_split(df, cols, [5]).y), 1400)
        self.assertEqual(len(exp.get_split(df, cols, [6]).y), 1400)
        # The separate Acq6 concentration-field conflict is NOT relabelled.
        block = df[(df.Brand == "C10") & (df.Class == "ManukaUMF20")
                   & (df.Acquisition == 6) & (df.Concentration == 50.0866)]
        self.assertEqual(len(block), 25)
        self.assertTrue(block.Concentration_Class.eq(5).all())

    def test_filter_preserves_other_products_and_concentration_classes(self):
        cols = [f"{400 + i * 5:.2f}nm" for i in range(128)]
        cases = [("C4", "ManukaUMF10", 50), ("C4", "ManukaUMF10", 5),
                 ("C10", "ManukaUMF10", 50), ("C4", "ManukaUMF20", 50),
                 ("C4", "ManukaUMF10", 100)]
        records = [dict(Brand=brand, Class=species, Concentration_Class=label,
                        Acquisition=1, case_id=i, **{c: float(i) for c in cols})
                   for i, (brand, species, label) in enumerate(cases)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.csv"
            pd.DataFrame(records).to_csv(path, index=False)
            df, _, _ = exp.load_frame(path)
        self.assertEqual(df.case_id.tolist(), [1, 2, 3])
        self.assertEqual(df.attrs["excluded_c4_rows"], 1)

    def test_unchanged_original_definitions(self):
        original = ast.parse((exp.HERE / "reference_model.py").read_text(encoding="utf-8"))
        derived = ast.parse((exp.HERE / "model.py").read_text(encoding="utf-8"))
        old = {n.name: n for n in original.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        new = {n.name: n for n in derived.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name in new:
            if name == "train_encoder_downstream_selection":
                continue
            self.assertEqual(ast.dump(old[name]), ast.dump(new[name]), name)
        # Normalize only the explicitly documented trainer instrumentation.
        text = (exp.HERE / "model.py").read_text(encoding="utf-8")
        text = text.replace('check_interval=5, train_epochs=None, n_jobs=-1, gbm_seed=42, log_callback=None):', 'check_interval=5):')
        text = text.replace('    actual_epochs = epochs if train_epochs is None else train_epochs\n    for ep in range(1, actual_epochs + 1):\n        candidate = None', '    for ep in range(1, epochs + 1):')
        text = text.replace('ep == actual_epochs:', 'ep == epochs:')
        text = text.replace('random_state=gbm_seed,', 'random_state=42,').replace('n_jobs=n_jobs,', 'n_jobs=-1,')
        text = text.replace('\n            candidate = {"epoch": ep, "probe_val_ba": float(downstream_acc), "probe_val_f1": float(downstream_mf1), "probe_val_recall0": float(downstream_r0)}', '')
        text = text.replace('        if log_callback is not None:\n            log_callback({"epoch": ep, "loss": float(train_loss), "head_val_ba": float(bal), "head_val_f1": float(mf1), "head_val_recall0": float(r0), "candidate": candidate, "selected_epoch_so_far": best_ep})\n\n', '')
        restored = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'train_encoder_downstream_selection')
        self.assertEqual(ast.dump(old[restored.name]), ast.dump(restored))

    def test_full_spectrum_preprocessing_matches_original(self):
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(3, 128)).astype(np.float32)
        edges = torch.tensor([[0, 1, 1], [1, 0, 1]])
        wl = np.linspace(399.4, 1063.79, 128)
        new = exp.make_graphs(raw, np.array([0, 1, 2]), np.arange(128), edges, wl)
        for i, graph in enumerate(new):
            old = core.spectrum_to_graph_sg3(core.snv(raw)[i], i, edges)
            self.assertTrue(torch.equal(graph.x, old.x))
            self.assertTrue(torch.equal(graph.edge_index, old.edge_index))

    def test_sparse_features_ignore_unselected_intensities(self):
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(3, 128)).astype(np.float32)
        bands = np.array([1, 4, 7, 11, 22, 27, 34, 40, 51, 59, 64, 73, 88, 95, 111, 124])
        changed = raw.copy()
        unused = np.setdiff1d(np.arange(128), bands)
        changed[:, unused] = 1e7
        edges = torch.tensor([[0, 1, 1], [1, 0, 1]])
        wl = np.linspace(399.4, 1063.79, 128)
        a = exp.make_graphs(raw, np.array([0, 1, 2]), bands, edges, wl)
        b = exp.make_graphs(changed, np.array([0, 1, 2]), bands, edges, wl)
        for left, right in zip(a, b):
            self.assertTrue(torch.equal(left.x, right.x))
        np.testing.assert_array_equal(core.make_global_features_from_raw(raw[:, bands]),
                                      core.make_global_features_from_raw(changed[:, bands]))
        np.testing.assert_array_equal(core.snv(raw[:, bands]), core.snv(changed[:, bands]))

    def test_irregular_polynomial_derivatives(self):
        full = np.linspace(400, 1040, 129)
        wl = full[[0, 1, 4, 8, 12, 20, 30, 33, 49, 58, 62, 80, 90, 102, 113, 127]]
        x = (wl - full[0]) / np.median(np.diff(full))
        y = 2 + 3 * x + 0.01 * x ** 2
        m = exp.local_polynomial_matrices(wl, full)
        np.testing.assert_allclose(m[0] @ y, y, atol=1e-7)
        np.testing.assert_allclose(m[1] @ y, 3 + 0.02 * x, atol=1e-7)
        np.testing.assert_allclose(m[2] @ y, np.full(len(x), 0.02), atol=1e-7)

    def test_hybrid_selection_is_fixed_size_and_deterministic(self):
        scores = np.zeros(128)
        np.testing.assert_array_equal(exp.hybrid_bands(scores, scores, 16), np.arange(16))
        rng = np.random.default_rng(42)
        g, r = rng.normal(size=128), rng.normal(size=128)
        for count in [16, 20, 24, 32, 48]:
            bands = exp.hybrid_bands(g, r, count)
            self.assertEqual(len(bands), count)
            self.assertEqual(len(set(bands)), count)
            self.assertTrue(np.all(np.diff(bands) > 0))

    def test_raw_ranked_counts_and_nested_regions(self):
        wl = np.linspace(400, 1040, 128)
        order = np.concatenate([np.arange(0, 128, 2), np.arange(1, 128, 2)])
        scores = np.zeros(128)
        scores[order] = np.arange(128, 0, -1)
        sets = exp.selection_sets(-scores, scores, [16, 32, 48], wl)
        for count in (16, 32, 48):
            self.assertEqual(sets['raw_ranked_bands'][str(count)], sorted(order[:count].tolist()))
            self.assertEqual(len(sets['selected_bands'][str(count)]), count)
        parent = set(sets['raw_ranked_bands']['48'])
        vis = sets['raw48_regions']['vis']
        nir = sets['raw48_regions']['nir']
        self.assertEqual(vis['status'], 'planned')
        self.assertEqual(nir['status'], 'planned')
        self.assertEqual(set(vis['bands']) | set(nir['bands']), parent)
        self.assertFalse(set(vis['bands']) & set(nir['bands']))
        self.assertTrue((wl[vis['bands']] < 700).all())
        self.assertTrue((wl[nir['bands']] >= 700).all())
        self.assertNotIn('48', exp.selection_sets(-scores, scores, [16], wl)['raw_ranked_bands'])
        self.assertEqual(exp.selection_sets(scores, scores, [16], wl)['raw48_regions'], {})
        tied = exp.selection_sets(np.zeros(128), np.zeros(128), [48], wl)
        self.assertEqual(tied['raw48_regions']['nir']['status'], 'skipped')

    def test_mcnemar_counts_effect_and_alignment(self):
        y = np.arange(6) % 5
        a = dict(y_true=y, row_indices=np.arange(6), predictions=y.copy())
        b = dict(y_true=y, row_indices=np.arange(6), predictions=(y + 1) % 5)
        outcome = exp.paired_mcnemar(a, b)
        self.assertEqual(outcome['a_only_correct'], 6)
        self.assertEqual(outcome['b_only_correct'], 0)
        self.assertEqual(outcome['delta_accuracy'], 1.)
        self.assertAlmostEqual(outcome['p_value'], 0.03125)
        self.assertEqual(exp.paired_mcnemar(a, a)['p_value'], 1.)
        b['row_indices'] = b['row_indices'][::-1]
        with self.assertRaises(RuntimeError):
            exp.paired_mcnemar(a, b)

    def test_trainval_arrays_independent_of_test_values(self):
        cols = [f"{400 + i * 5:.2f}nm" for i in range(128)]
        data = []
        for acq in (1, 5, 6):
            for cls in exp.CLASSES:
                data.append(dict(Acquisition=acq, Concentration_Class=cls, Brand="fixture", Class="fixture",
                                 **{c: float(i + acq + cls) for i, c in enumerate(cols)}))
        df = pd.DataFrame(data)
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
            df.to_csv(a, index=False)
            df.loc[df.Acquisition == 6, cols] = -1e9
            df.loc[df.Acquisition == 6, "Concentration_Class"] = 50
            df.to_csv(b, index=False)
            left, right = exp.load_trainval(a), exp.load_trainval(b)
            for name in ("train", "val"):
                np.testing.assert_array_equal(getattr(left, name).raw, getattr(right, name).raw)
                np.testing.assert_array_equal(getattr(left, name).y, getattr(right, name).y)
            self.assertFalse(hasattr(left, "test"))

    def test_checkpoint_selected_by_gbm_not_gat_head(self):
        y = np.array([0, 1, 2, 3, 4])
        predictions = iter([np.zeros(5, dtype=int), y])
        class FakeGBM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def fit(self, *args, **kwargs):
                return self
            def predict(self, *args, **kwargs):
                return next(predictions)
        heads = iter([(1., 1., 1.), (0., 0., 0.), (0., 0., 0.), (0., 0., 0.), (0., 0., 0.)])
        records = []
        with tempfile.TemporaryDirectory() as tmp, patch.object(core.lgb, 'LGBMClassifier', FakeGBM), \
             patch.object(core, 'evaluate_loader_logits', side_effect=lambda *a: next(heads)), \
             patch.object(core, 'extract_embeddings', return_value=(np.ones((5, 128)), y)):
            model = torch.nn.Linear(1, 1)
            ep, ba = core.train_encoder_downstream_selection(model, [], [], [], 'cpu',
                torch.ones(5), np.ones((5, 6)), np.ones((5, 6)), y, y,
                epochs=50, train_epochs=5, ckpt_path=str(Path(tmp) / 'encoder.pt'), log_callback=records.append)
            self.assertEqual(ep, 5)
            self.assertEqual(ba, 1.0)
            self.assertEqual(len(records), 5)
            self.assertEqual([r['candidate']['epoch'] for r in records if r['candidate']], [1, 5])

    def test_threshold_and_tree_count_locked_before_refit(self):
        y = np.tile(np.arange(5), 3)
        x = np.eye(5)[y]
        events, models = [], []
        class FakeBooster:
            def save_model(self, path):
                Path(path).write_text('fixture', encoding='utf-8')
        class FakeGBM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.best_iteration_ = 7
                self.booster_ = FakeBooster()
                models.append(self)
            def fit(self, features, labels, **kwargs):
                events.append(('fit', len(labels), kwargs))
                return self
            def predict_proba(self, features):
                events.append(('predict_val', len(features)))
                return features * 0.9 + 0.02
        cfg = {'gbm_seed': 42, 'threads': 1}
        with tempfile.TemporaryDirectory() as tmp, patch.object(exp.lgb, 'LGBMClassifier', FakeGBM):
            result = exp.fit_gbm(Path(tmp), x, y, x[:5], y[:5], np.ones(5), cfg, {})
        self.assertEqual(events[0][0:2], ('fit', 15))
        self.assertEqual(events[-1][0:2], ('fit', 20))
        self.assertEqual(models[-1].kwargs['n_estimators'], 7)
        self.assertTrue(result['threshold_locked_before_refit'])
        self.assertNotIn('eval_set', events[-1][2])

    def test_artifact_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'model.txt').write_text('original')
            exp.seal_task(folder, {'task_id': 'fixture'}, ['model.txt'])
            exp.read_locked(folder)
            (folder / 'model.txt').write_text('changed')
            with self.assertRaises(RuntimeError):
                exp.read_locked(folder)

    def test_tiny_fixture_lifecycle_without_gat_training(self):
        # Synthetic orchestration test only. GAT training and embedding inference
        # are mocked; this does not run the honey experiment or publish results.
        rng = np.random.default_rng(1)
        cols = [f"{400 + i * 5:.2f}nm" for i in range(128)]
        records = []
        for acq in (1, 5, 6):
            for cls in exp.CLASSES:
                for rep in range(4):
                    raw = rng.normal(size=128) + cls / 10
                    records.append(dict(Acquisition=acq, Concentration_Class=cls,
                                        Brand="fixture", Class="fixture", **dict(zip(cols, raw))))

        def mock_train(model, *args, ckpt_path, log_callback, **kwargs):
            torch.save(model.state_dict(), ckpt_path)
            log_callback(dict(epoch=1, loss=0., head_val_ba=0., head_val_f1=0., head_val_recall0=0.,
                              candidate=dict(epoch=1, probe_val_ba=0.2), selected_epoch_so_far=1))
            return 1, 0.2

        def mock_embeddings(model, loader, device):
            features, labels = [], []
            for graph in loader.dataset:
                features.append(np.tile(graph.x.numpy().mean(axis=0), 32))
                labels.append(int(graph.y))
            return np.array(features, dtype=np.float32), np.array(labels)

        # Deterministic synthetic importance fixture spans both regions, allowing
        # all new orchestration branches to run despite a tiny constant-tree GBM.
        selection_function = exp.selection_sets
        def fixture_selection(gat_scores, raw_scores, counts, wavelengths):
            ranked = np.full(128, -1.)
            chosen = np.linspace(0, 127, 48, dtype=int)
            ranked[chosen] = np.arange(48, 0, -1)
            return selection_function(gat_scores, ranked, counts, wavelengths)

        cfg = dict(epochs=5, scheduler_epochs=50, seeds=[42], gbm_seed=42, k_values=[8, 12],
                   band_counts=[16, 32, 48], sparse_k=8, ranking_seed=42, permutation_repeats=1,
                   threads=1, device='cpu')
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(core, 'train_encoder_downstream_selection', side_effect=mock_train), \
             patch.object(core, 'extract_embeddings', side_effect=mock_embeddings), \
             patch.object(exp, 'selection_sets', side_effect=fixture_selection):
            root = Path(tmp)
            data_path = root / 'fixture.csv'
            pd.DataFrame(records).to_csv(data_path, index=False)
            exp.write_json(root / 'protocol.json', cfg)
            tv = exp.load_trainval(data_path)
            plan_path = exp.fit_requested(root, tv, cfg, ['ablation', 'k', 'bands'])
            self.assertFalse(list(root.glob('tasks/*/test_metrics.json')))
            selection = exp.read_locked(root / 'band_selection')
            self.assertFalse(selection['test_used_for_selection'])
            for count in cfg['band_counts']:
                sparse = exp.read_locked(root / f'tasks/gat_bands{count}_k8_seed42')
                raw = exp.read_locked(root / f'tasks/raw_bands{count}')
                self.assertEqual(sparse['bands'], raw['bands'])
                raw_ranked = exp.read_locked(root / f'tasks/raw_ranked_bands{count}')
                self.assertEqual(raw_ranked['bands'], selection['raw_ranked_bands'][str(count)])
            parent = exp.read_locked(root / 'tasks/raw_ranked_bands48')
            regions = [exp.read_locked(root / f'tasks/raw_ranked48_{region}_only') for region in ('vis', 'nir')]
            self.assertEqual(set(regions[0]['bands']) | set(regions[1]['bands']), set(parent['bands']))
            exp.evaluate_requested(root, cfg, plan_path, data_path)
            summary = exp.read_json(root / 'plan_ablation_bands_k_summary.json')
            self.assertTrue(summary['all_seeds_retained'])
            rows = pd.read_csv(root / 'plan_ablation_bands_k_metrics.csv')
            self.assertEqual(len(rows), 16)
            self.assertEqual(len(list(root.glob('tasks/*/test_metrics.json'))), 16)
            self.assertTrue((rows.n_vis + rows.n_nir == rows.n_bands).all())
            self.assertEqual(len(summary['mcnemar']), 1)
            self.assertEqual(summary['mcnemar'][0]['n_samples'], 20)
            tab = pd.read_csv(root / 'plan_ablation_bands_k_summary.csv')
            self.assertEqual(len(tab[tab.group == 'raw_ranked_bands']), 4)
            self.assertEqual(len(tab[tab.group == 'raw48_regions']), 3)
            with patch.object(core, 'extract_embeddings', side_effect=AssertionError('should reuse stored evaluation')):
                exp.evaluate_requested(root, cfg, plan_path, data_path)
            exp.write_json(root / 'protocol.json', dict(cfg, epochs=99))
            with patch.object(exp, 'load_frame', side_effect=AssertionError('must reject before loading Test')):
                with self.assertRaises(RuntimeError):
                    exp.evaluate_requested(root, cfg, plan_path, data_path)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
