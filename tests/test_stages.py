import unittest

import torch

from snn_structural_evolution.metrics import spike_statistics
from snn_structural_evolution.stages import get_model


class StageSmokeTest(unittest.TestCase):
    def test_all_stages_return_class_logits(self):
        inputs = torch.rand(2, 1, 28, 28)
        for stage in range(5):
            with self.subTest(stage=stage):
                model = get_model(stage=stage, time_steps=3)
                outputs = model(inputs)
                self.assertEqual(tuple(outputs.shape), (2, 10))

    def test_invalid_stage_raises(self):
        with self.assertRaises(ValueError):
            get_model(stage=9)

    def test_stage4_spike_statistics_are_bounded(self):
        torch.manual_seed(35)
        model = get_model(stage=4, hidden_dim=8, time_steps=3)
        inputs = torch.rand(4, 1, 28, 28)
        targets = torch.zeros(4, dtype=torch.long)

        stats = spike_statistics(model, [(inputs, targets)], torch.device("cpu"))

        self.assertIsNotNone(stats["spike_rate"])
        self.assertGreaterEqual(stats["spike_rate"], 0.0)
        self.assertLessEqual(stats["spike_rate"], 1.0)
        self.assertAlmostEqual(stats["activation_sparsity"], 1.0 - stats["spike_rate"])
        self.assertEqual(stats["possible_spikes"], 3 * 4 * 8)
        self.assertEqual(stats["event_ops_proxy"], stats["total_spikes"] * 10)


if __name__ == "__main__":
    unittest.main()
