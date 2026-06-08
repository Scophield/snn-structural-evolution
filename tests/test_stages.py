import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
