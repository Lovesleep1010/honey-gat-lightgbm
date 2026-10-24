"""Exercise stopping and checkpoint selection with simulated validation scores.

No honey training, gradient updates, or LightGBM fitting is performed here.
"""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import model as core
import config as launcher


class SimulatedEncoder:
    def __init__(self):
        self.epoch = 0

    def parameters(self):
        return []

    def train(self):
        self.epoch += 1

    def eval(self):
        pass

    def state_dict(self):
        return {"simulated_epoch": torch.tensor(self.epoch)}


class EarlyStoppingTests(unittest.TestCase):
    def run_scores(self, correct_counts, maximum=50, patience=5, interval=1):
        y = np.arange(5)
        counts = iter(correct_counts)
        model = SimulatedEncoder()
        records = []

        class SimulatedGBM:
            def __init__(self, **kwargs):
                pass

            def fit(self, *args, **kwargs):
                return self

            def predict(self, *args, **kwargs):
                correct = next(counts)
                prediction = (y + 1) % 5
                prediction[:correct] = y[:correct]
                return prediction

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "encoder.pt"
            with patch.object(core.torch.optim, "AdamW", return_value=object()), \
                 patch.object(core.torch.optim.lr_scheduler, "CosineAnnealingLR",
                              return_value=SimpleNamespace(step=lambda: None)), \
                 patch.object(core.lgb, "LGBMClassifier", SimulatedGBM), \
                 patch.object(core, "evaluate_loader_logits", return_value=(1., 1., 1.)), \
                 patch.object(core, "extract_embeddings", return_value=(np.ones((5, 128)), y)):
                selected, ba = core.train_encoder_downstream_selection(
                    model, [], [], [], "cpu", torch.ones(5),
                    np.ones((5, 6)), np.ones((5, 6)), y, y,
                    epochs=50, train_epochs=maximum, check_interval=interval,
                    early_stopping_patience=patience, ckpt_path=str(checkpoint),
                    log_callback=records.append)
            saved_epoch = int(torch.load(checkpoint, weights_only=True)["simulated_epoch"])
        return selected, ba, records, model.epoch, saved_epoch

    def test_first_epoch_best_stops_after_five_more_epochs(self):
        selected, ba, records, completed, saved = self.run_scores([3] * 50)
        self.assertEqual((selected, completed, saved), (1, 6, 1))
        self.assertAlmostEqual(ba, .6)
        self.assertEqual(records[-1]["epochs_without_improvement"], 5)
        self.assertTrue(records[-1]["early_stopping_triggered"])
        self.assertEqual([r["candidate"]["epoch"] for r in records], list(range(1, 7)))

    def test_improvement_on_fifth_epoch_resets_wait(self):
        selected, _, records, completed, saved = self.run_scores([1] * 4 + [2] * 6)
        self.assertEqual((selected, completed, saved), (5, 10, 5))
        self.assertEqual(records[4]["epochs_without_improvement"], 0)

    def test_worse_later_scores_do_not_overwrite_best_checkpoint(self):
        selected, ba, records, completed, saved = self.run_scores([1, 4, 3, 2, 1, 1, 1])
        self.assertEqual((selected, completed, saved), (2, 7, 2))
        self.assertAlmostEqual(ba, .8)
        self.assertTrue(records[-1]["early_stopping_triggered"])

    def test_training_stops_at_maximum_when_scores_keep_improving(self):
        selected, ba, records, completed, saved = self.run_scores([1, 2, 3, 4, 5], maximum=5)
        self.assertEqual((selected, completed, saved), (5, 5, 5))
        self.assertEqual(ba, 1.)
        self.assertFalse(records[-1]["early_stopping_triggered"])

    def test_ties_count_as_no_improvement(self):
        _, _, records, _, _ = self.run_scores([3] * 50)
        self.assertEqual([r["epochs_without_improvement"] for r in records], [0, 1, 2, 3, 4, 5])

    def test_legacy_selection_is_available_when_stopping_is_disabled(self):
        selected, _, records, completed, saved = self.run_scores(
            [1, 4], maximum=5, patience=None, interval=5)
        self.assertEqual((selected, completed, saved), (5, 5, 5))
        self.assertEqual([r["candidate"]["epoch"] for r in records if r["candidate"]], [1, 5])

    def test_epoch_patience_rejects_validation_every_five_epochs(self):
        with self.assertRaisesRegex(ValueError, "check_interval=1"):
            self.run_scores([3] * 50, interval=5)

    def test_zero_patience_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self.run_scores([3] * 50, patience=0)

    def test_pycharm_entry_forwards_stopping_settings_and_new_output(self):
        arguments = launcher.build_argv()
        self.assertEqual(arguments[arguments.index("--epochs") + 1], "50")
        self.assertEqual(arguments[arguments.index("--early-stopping-patience") + 1], "5")
        self.assertEqual(arguments[arguments.index("--experiments") + 1], "baseline")
        self.assertEqual(Path(arguments[arguments.index("--output") + 1]).name,
                         "results_max50_patience5_c4filtered")


if __name__ == "__main__":
    unittest.main(verbosity=2)
